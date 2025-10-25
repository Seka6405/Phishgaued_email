# app/main.py
import joblib
import numpy as np
import json
from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt
from pydantic import BaseModel
from sqlmodel import select
from app.db import create_db_and_tables, get_session, ENGINE
from app.models import User, EmailRecord, Feedback, ModelVersion
from typing import List, Dict, Any
import shap
from contextlib import contextmanager
from sqlmodel import Session

# --- config (change in production) ---
SECRET = "CHANGE_THIS_SECRET_TO_SOMETHING_SAFE"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

app = FastAPI(title="PhishGuard Prototype")

# In-memory websockets clients
clients: List[WebSocket] = []

# Load model artifacts (train_model.py must be run first)
MODEL_PATH = "models/model.joblib"
VEC_PATH = "models/vectorizer.joblib"
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# shap explainer (linear)
explainer = shap.LinearExplainer(model, vectorizer.transform([""]).toarray(), feature_perturbation="correlation")

# Helpers
def verify_password(plain, hashed):
    return pwd_ctx.verify(plain, hashed)

def get_password_hash(p):
    return pwd_ctx.hash(p)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid auth token")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid auth token")
    with Session(ENGINE) as session:
        user = session.exec(select(User).where(User.email == email)).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user

def require_role(user: User = Depends(get_current_user), role: str = "user"):
    if role == "admin":
        if user.role != "admin":
            raise HTTPException(status_code=403, detail="Admin required")
    return user

# Create DB
create_db_and_tables()

# --- Pydantic schemas ---
class RegisterIn(BaseModel):
    email: str
    password: str
    name: str = None

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class ClassifyIn(BaseModel):
    text: str
    subject: str = None

class ClassifyOut(BaseModel):
    label: str
    confidence: float
    explanation: Dict[str, Any]

# --- Auth endpoints ---
@app.post("/api/register")
def register(payload: RegisterIn):
    with Session(ENGINE) as session:
        existing = session.exec(select(User).where(User.email == payload.email)).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email exists")
        user = User(email=payload.email, password_hash=get_password_hash(payload.password), name=payload.name)
        session.add(user)
        session.commit()
        session.refresh(user)
        return {"status": "ok", "user_id": user.id}

@app.post("/api/token", response_model=TokenOut)
def token(form_data: OAuth2PasswordRequestForm = Depends()):
    with Session(ENGINE) as session:
        user = session.exec(select(User).where(User.email == form_data.username)).first()
        if not user or not verify_password(form_data.password, user.password_hash):
            raise HTTPException(status_code=400, detail="Incorrect credentials")
        access = create_access_token({"sub": user.email, "role": user.role})
        return {"access_token": access}

# --- Classification endpoint ---
@app.post("/api/classify", response_model=ClassifyOut)
def classify(payload: ClassifyIn, user: User = Depends(get_current_user)):
    text = (payload.subject or "") + " " + payload.text
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    label_idx = int(np.argmax(proba))
    label = "phish" if model.classes_[label_idx] == 1 else "legit"
    confidence = float(np.max(proba))

    # SHAP explanation (top features)
    try:
        # shap wants dense array
        sv = explainer.shap_values(X.toarray())[1] if len(model.classes_)>1 else explainer.shap_values(X.toarray())
        sv = sv[0] if isinstance(sv, list) else sv
        feature_names = vectorizer.get_feature_names_out()
        arr = sv[0] if sv.ndim==2 else sv
        top_idx = np.argsort(np.abs(arr))[-6:][::-1]
        explanation = [{"feature": feature_names[i], "shap": float(arr[i])} for i in top_idx]
    except Exception as e:
        explanation = {"error": "explainability failed", "detail": str(e)}

    # persist record
    with Session(ENGINE) as session:
        rec = EmailRecord(user_id=user.id, subject=payload.subject, body=payload.text,
                          predicted_label=label, confidence=confidence, explanation={"items": explanation})
        session.add(rec)
        session.commit()
        session.refresh(rec)

    # send websocket alert if high confidence phish
    if label == "phish" and confidence > 0.85:
        payload_ws = {"type":"phish_alert", "email_id": rec.id, "summary": (payload.subject or "")[:120], "confidence": confidence}
        # broadcast (async) using background tasks
        background_broadcast(payload_ws)

    return {"label": label, "confidence": confidence, "explanation": {"items": explanation}}

# --- Feedback endpoint ---
class FeedbackIn(BaseModel):
    email_id: int
    label: str  # 'phish' | 'legit'
    comment: str = None

@app.post("/api/feedback")
def feedback(payload: FeedbackIn, user: User = Depends(get_current_user)):
    with Session(ENGINE) as session:
        rec = session.get(EmailRecord, payload.email_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Email not found")
        fb = Feedback(email_id=payload.email_id, user_id=user.id, label=payload.label, comment=payload.comment)
        session.add(fb)
        # optional: update record predicted label or mark as reviewed
        session.commit()
        session.refresh(fb)

        # check whether to retrain (simple heuristic: if feedbacks >= threshold)
        count = session.exec(select(Feedback)).all()
        if len(count) >= 3:  # threshold for demo; tune in prod
            # launch retrain in background
            background_retrain()

    return {"status": "saved", "feedback_id": fb.id}

# --- Simple model retrain (background) ---
def retrain_model():
    # Very small demo retrain: re-fit on feedback combined with original toy data
    print("Starting retrain job...")
    with Session(ENGINE) as session:
        fb_items = session.exec(select(Feedback)).all()
        if not fb_items:
            print("No feedback to retrain on.")
            return
        # Build small dataset
        texts = []
        labels = []
        for f in fb_items:
            rec = session.get(EmailRecord, f.email_id)
            if rec:
                texts.append(rec.body)
                labels.append(1 if f.label=="phish" else 0)
        # fallback: require at least 3 labeled examples
        if len(texts) < 3:
            print("Not enough feedback samples; abort retrain.")
            return
        # load base training data from train_model.py dataset if available
        # For demo we use the same vectorizer and only partial re-fit
        global model, vectorizer, explainer
        X = vectorizer.transform(texts)
        try:
            model.partial_fit(X, labels)
        except Exception:
            # if model doesn't support partial_fit, retrain logistic from scratch on combined dataset
            from sklearn.linear_model import LogisticRegression
            import joblib, os
            # compose small dataset
            model_new = LogisticRegression(solver='liblinear')
            X_all = X
            y_all = labels
            model_new.fit(X_all, y_all)
            model = model_new
            joblib.dump(model, MODEL_PATH)
        # update explainer
        try:
            explainer = shap.LinearExplainer(model, vectorizer.transform([""]).toarray(), feature_perturbation="correlation")
        except Exception:
            pass
        print("Retrain finished.")

def background_retrain():
    import threading
    threading.Thread(target=retrain_model, daemon=True).start()

# --- Websocket endpoints ---
@app.websocket("/ws/notifications")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            data = await ws.receive_text()  # in demo we don't consume client messages; keep connection open
    except WebSocketDisconnect:
        clients.remove(ws)

def background_broadcast(payload: dict):
    import asyncio
    async def _send():
        for c in list(clients):
            try:
                await c.send_json(payload)
            except:
                try:
                    clients.remove(c)
                except:
                    pass
    # schedule
    asyncio.get_event_loop().create_task(_send())

# --- Utility endpoints for demo ---
@app.get("/api/emails")
def list_emails(user: User = Depends(get_current_user)):
    with Session(ENGINE) as session:
        items = session.exec(select(EmailRecord).where(EmailRecord.user_id==user.id)).all()
        return items

@app.get("/api/me")
def me(user: User = Depends(get_current_user)):
    return {"email": user.email, "name": user.name, "role": user.role}
