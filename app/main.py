"""
PhishGuard Backend API - Phase 2
"""

from fastapi import FastAPI, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import numpy as np
from datetime import datetime
import shap
import os
import pandas as pd
import csv

# ------------------ Paths ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app folder
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # root folder
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.joblib")
VEC_PATH = os.path.join(PROJECT_ROOT, "models", "vectorizer.joblib")
FEEDBACK_CSV = os.path.join(PROJECT_ROOT, "feedback_history.csv")
ADMIN_KEY = "phishguard_admin_123"  # simple API key for retrain/admin

# ------------------ FastAPI ------------------
app = FastAPI(title="PhishGuard - Real-Time Phishing Detection API")

# Serve frontend files
app.mount("/frontend", StaticFiles(directory=os.path.join(PROJECT_ROOT, "frontend")), name="frontend")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Load Model & SHAP ------------------
try:
    print("🔹 Loading model and vectorizer...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    print("✅ Model & Vectorizer loaded successfully")

    # SHAP explainer
    print("🔹 Initializing SHAP explainer...")
    background = vectorizer.transform(["dummy text"])
    explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")
    print("✅ SHAP explainer ready")
except Exception as e:
    print("❌ Error loading model/vectorizer:", e)
    raise e

# ------------------ Routes ------------------

@app.get("/")
def root():
    return FileResponse(os.path.join(PROJECT_ROOT, "frontend", "index.html"))

@app.get("/dashboard")
def dashboard():
    return FileResponse(os.path.join(PROJECT_ROOT, "frontend", "dashboard.html"))

# -------- Classification Endpoint --------
@app.post("/api/classify")
def classify(text: str = Form(...), subject: str = Form("")):
    try:
        full_text = f"{subject} {text}"
        X = vectorizer.transform([full_text])
        proba = model.predict_proba(X)[0]
        label_idx = np.argmax(proba)
        label = "phish" if model.classes_[label_idx] == 1 else "legit"
        confidence = float(np.max(proba))

        # SHAP
        shap_values = explainer.shap_values(X)
        feature_names = vectorizer.get_feature_names_out()
        if isinstance(shap_values, list):
            sv = shap_values[label_idx][0]
        else:
            sv = shap_values[0]
        top_idx = np.argsort(np.abs(sv))[-8:][::-1]
        explanation = [{"feature": feature_names[i], "impact": float(sv[i])} for i in top_idx]

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "label": label,
            "confidence": round(confidence, 4),
            "explanation": explanation,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------- Feedback Endpoint --------
@app.post("/api/feedback")
def feedback(email_text: str = Form(...), user_label: str = Form(...)):
    timestamp = datetime.utcnow().isoformat()
    row = [timestamp, email_text, user_label]
    file_exists = os.path.isfile(FEEDBACK_CSV)
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "email_text", "user_label"])
        writer.writerow(row)
    print(f"📝 Feedback received: Label={user_label}")
    return {"status": "success", "message": "Feedback recorded"}

# -------- History Endpoint --------
@app.get("/api/history")
def get_history(limit: int = 50):
    if not os.path.isfile(FEEDBACK_CSV):
        return []
    df = pd.read_csv(FEEDBACK_CSV)
    df = df.tail(limit)
    return df.to_dict(orient="records")

# -------- Retrain Endpoint --------
@app.post("/api/retrain")
def retrain_model(api_key: str = Header(...)):
    if api_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not os.path.isfile(FEEDBACK_CSV):
        return {"status": "error", "message": "Invalid feedback file"}
    df = pd.read_csv(FEEDBACK_CSV)
    if df.empty:
        return {"status": "error", "message": "Feedback CSV is empty"}

    X_text = df['email_text'].astype(str).tolist()
    y = df['user_label'].map(lambda x: 1 if x=='phish' else 0).tolist()

    X = vectorizer.transform(X_text)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print("✅ Model retrained successfully")
    return {"status": "success", "message": "Model retrained"}










# """
# PhishGuard Backend API - Phase 2
# -------------------------------
# Features:
# ✅ Real-time phishing detection
# ✅ SHAP explainability
# ✅ Feedback storage (CSV)
# ✅ Admin dashboard + retrain
# ✅ API key authentication
# """

# import os
# import csv
# from fastapi import FastAPI, Form, HTTPException, Request, Depends
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, JSONResponse
# import joblib
# import numpy as np
# from datetime import datetime
# import shap
# import pandas as pd

# # ------------------- Configuration -------------------
# API_KEY = "admin123"  # Simple API key for admin routes
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")
# FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.csv")

# # ------------------- App Init -------------------
# app = FastAPI(title="PhishGuard - Real-Time Phishing Detection API")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ------------------- Load Model -------------------
# import os

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
# VEC_PATH = os.path.join(BASE_DIR, "models", "vectorizer.joblib")


# print("🔹 Loading model and vectorizer...")
# model = joblib.load(MODEL_PATH)
# vectorizer = joblib.load(VEC_PATH)
# print("✅ Model & Vectorizer loaded successfully")

# # ------------------- SHAP Explainer -------------------
# print("🔹 Initializing SHAP explainer...")
# background = vectorizer.transform(["test"])
# explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")
# print("✅ SHAP explainer ready")

# # ------------------- Helper: API Key -------------------
# def verify_api_key(request: Request):
#     key = request.headers.get("x-api-key")
#     if key != API_KEY:
#         raise HTTPException(status_code=403, detail="Unauthorized")
#     return True

# # ------------------- Routes -------------------

# @app.get("/", response_class=FileResponse)
# def root():
#     return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# @app.get("/dashboard", response_class=FileResponse)
# def dashboard_page():
#     return FileResponse(os.path.join(FRONTEND_DIR, "dashboard.html"))

# @app.post("/api/classify")
# def classify(text: str = Form(...), subject: str = Form("")):
#     full_text = f"{subject} {text}"
#     X = vectorizer.transform([full_text])
#     proba = model.predict_proba(X)[0]
#     label_idx = np.argmax(proba)
#     label = "phish" if model.classes_[label_idx] == 1 else "legit"
#     confidence = float(np.max(proba))

#     # SHAP
#     try:
#         shap_values = explainer.shap_values(X)
#         feature_names = vectorizer.get_feature_names_out()
#         sv = shap_values[label_idx][0] if isinstance(shap_values, list) else shap_values[0]
#         top_idx = np.argsort(np.abs(sv))[-8:][::-1]
#         explanation = [{"feature": feature_names[i], "impact": float(sv[i])} for i in top_idx]
#     except Exception as e:
#         explanation = {"error": str(e)}

#     return {
#         "timestamp": datetime.utcnow().isoformat(),
#         "label": label,
#         "confidence": round(confidence, 4),
#         "explanation": explanation,
#     }

# @app.post("/api/feedback")
# def feedback(email_text: str = Form(...), user_label: str = Form(...)):
#     """Save feedback to CSV"""
#     fieldnames = ["timestamp", "email_text", "user_label"]
#     row = {"timestamp": datetime.utcnow().isoformat(), "email_text": email_text, "user_label": user_label}

#     file_exists = os.path.exists(FEEDBACK_FILE)
#     with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         if not file_exists:
#             writer.writeheader()
#         writer.writerow(row)

#     print(f"📝 Feedback received: Label={user_label}")
#     return {"status": "success", "user_label": user_label, "received_at": datetime.utcnow().isoformat()}

# # ------------------- Admin: History -------------------
# @app.get("/api/history")
# def history(limit: int = 50, authorized: bool = Depends(verify_api_key)):
#     if not os.path.exists(FEEDBACK_FILE):
#         return []
#     df = pd.read_csv(FEEDBACK_FILE)
#     df = df.tail(limit)
#     return df.to_dict(orient="records")

# # ------------------- Admin: Retrain -------------------
# @app.post("/api/retrain")
# def retrain_model(authorized: bool = Depends(verify_api_key)):
#     """Retrain Logistic Regression with feedback CSV"""
#     if not os.path.exists(FEEDBACK_FILE):
#         return {"message": "No feedback to retrain"}
#     df = pd.read_csv(FEEDBACK_FILE)
#     if "email_text" not in df.columns or "user_label" not in df.columns:
#         return {"message": "Invalid feedback file"}
    
#     X_text = df["email_text"].tolist()
#     y = df["user_label"].apply(lambda x: 1 if x=="phish" else 0).values
#     X_vec = vectorizer.transform(X_text)

#     from sklearn.linear_model import LogisticRegression
#     new_model = LogisticRegression(max_iter=500)
#     new_model.fit(X_vec, y)

#     # Save updated model
#     joblib.dump(new_model, MODEL_PATH)
#     global model, explainer
#     model = new_model
#     explainer = shap.LinearExplainer(model, X_vec, feature_perturbation="interventional")

#     return {"message": f"Model retrained on {len(df)} feedback entries"}

# # ------------------- Health -------------------
# @app.get("/health")
# def health_check():
#     return {"status": "ok", "model_loaded": True, "time": datetime.utcnow().isoformat()}









# """
# PhishGuard Backend + Frontend (Advanced)
# ----------------------------------------
# Features:
# - Real-time phishing detection
# - SHAP explainability with Chart.js frontend
# - Feedback stored in CSV
# - On-demand retraining endpoint
# - Simple API key authentication
# - SHAP performance improved with Independent masker
# """

# from fastapi import FastAPI, Form, HTTPException, Depends, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# import joblib
# import numpy as np
# from datetime import datetime
# import shap
# import traceback
# import os
# import pandas as pd

# # ----------------- Config -----------------
# API_KEY = "phishguard123"  # simple API key for endpoints
# FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback.csv")
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
# MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
# VEC_PATH = os.path.join(BASE_DIR, "models", "vectorizer.joblib")

# # ----------------- App -----------------
# app = FastAPI(title="PhishGuard - Advanced Email Detection")

# # Allow CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ----------------- API Key Dependency -----------------
# def verify_api_key(request: Request):
#     key = request.headers.get("x-api-key")
#     if key != API_KEY:
#         raise HTTPException(status_code=401, detail="Unauthorized")

# # ----------------- Load Model & Vectorizer -----------------
# try:
#     print("🔹 Loading model and vectorizer...")
#     model = joblib.load(MODEL_PATH)
#     vectorizer = joblib.load(VEC_PATH)
#     print("✅ Model & Vectorizer loaded successfully")

#     print("🔹 Initializing SHAP explainer...")
#     background = vectorizer.transform(["sample email text"])
#     masker = shap.maskers.Independent(background)
#     explainer = shap.LinearExplainer(model, masker)
#     print("✅ SHAP explainer ready")
# except Exception as e:
#     print("❌ Error loading model/vectorizer:", e)
#     raise e

# # ----------------- Health Check -----------------
# @app.get("/health")
# def health_check():
#     return {"status": "ok", "model_loaded": True, "time": datetime.utcnow().isoformat()}

# # ----------------- Classification -----------------
# @app.post("/api/classify", dependencies=[Depends(verify_api_key)])
# def classify(text: str = Form(...), subject: str = Form("")):
#     try:
#         full_text = f"{subject} {text}".strip()
#         if not full_text:
#             raise HTTPException(status_code=400, detail="Empty email content")

#         X = vectorizer.transform([full_text])
#         proba = model.predict_proba(X)[0]
#         label_idx = np.argmax(proba)
#         label = "phish" if model.classes_[label_idx] == 1 else "legit"
#         confidence = float(np.max(proba))

#         # SHAP explanation
#         try:
#             shap_values = explainer.shap_values(X)
#             feature_names = vectorizer.get_feature_names_out()
#             sv = shap_values[label_idx][0] if isinstance(shap_values, list) else shap_values[0]
#             top_idx = np.argsort(np.abs(sv))[-8:][::-1]
#             explanation = [{"feature": feature_names[i], "impact": float(sv[i])} for i in top_idx]
#         except Exception as shap_err:
#             explanation = {"error": "Explainability failed", "detail": str(shap_err)}

#         return {
#             "timestamp": datetime.utcnow().isoformat(),
#             "label": label,
#             "confidence": round(confidence, 4),
#             "explanation": explanation,
#         }

#     except HTTPException as he:
#         raise he
#     except Exception as e:
#         print("❌ Classification error:", traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"Server error: {e}")

# # ----------------- Feedback -----------------
# @app.post("/api/feedback", dependencies=[Depends(verify_api_key)])
# def feedback(email_text: str = Form(...), user_label: str = Form(...)):
#     df = pd.DataFrame([[datetime.utcnow().isoformat(), email_text, user_label]], columns=["timestamp","email_text","label"])
#     if os.path.exists(FEEDBACK_FILE):
#         df.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
#     else:
#         df.to_csv(FEEDBACK_FILE, index=False)
#     print(f"📝 Feedback received: Label={user_label}")
#     return {"status":"success","message":"Feedback recorded","user_label":user_label}

# # ----------------- Retrain Model -----------------
# @app.post("/api/retrain", dependencies=[Depends(verify_api_key)])
# def retrain_model():
#     if not os.path.exists(FEEDBACK_FILE):
#         raise HTTPException(status_code=404, detail="No feedback found for retraining")

#     df = pd.read_csv(FEEDBACK_FILE)
#     if df.empty:
#         raise HTTPException(status_code=400, detail="Feedback CSV is empty")

#     # Simple TF-IDF + Logistic Regression retrain
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.linear_model import LogisticRegression

#     print("🔹 Retraining model from feedback CSV...")
#     X_text = df['email_text'].tolist()
#     y = df['label'].apply(lambda x: 1 if x=='phish' else 0).values

#     new_vectorizer = TfidfVectorizer(max_features=5000)
#     X = new_vectorizer.fit_transform(X_text)
#     new_model = LogisticRegression(max_iter=500)
#     new_model.fit(X, y)

#     # Save updated model/vectorizer
#     joblib.dump(new_model, MODEL_PATH)
#     joblib.dump(new_vectorizer, VEC_PATH)

#     # Reload global model & explainer
#     global model, vectorizer, explainer
#     model = new_model
#     vectorizer = new_vectorizer
#     masker = shap.maskers.Independent(vectorizer.transform(["sample text"]))
#     explainer = shap.LinearExplainer(model, masker)

#     print("✅ Retraining complete")
#     return {"status":"success","message":"Model retrained from feedback CSV"}

# # ----------------- Serve Frontend -----------------
# app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")



