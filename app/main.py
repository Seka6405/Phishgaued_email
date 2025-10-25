"""
PhishGuard Backend + Frontend
-----------------------------
- Real-time phishing detection API
- SHAP explainability
- Serves frontend at '/'
"""

from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
from datetime import datetime
import shap
import traceback
import os

# ------------------------------------------------------------
# 1️⃣ App Configuration
# ------------------------------------------------------------
app = FastAPI(title="PhishGuard - Real-Time Phishing Detection API")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# 2️⃣ Load Model & Vectorizer
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
VEC_PATH = os.path.join(BASE_DIR, "models", "vectorizer.joblib")

try:
    print("🔹 Loading model and vectorizer...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    print("✅ Model & Vectorizer loaded successfully")

    print("🔹 Initializing SHAP explainer...")
    background = vectorizer.transform(["sample email text"])
    explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")
    print("✅ SHAP explainer ready")

except Exception as e:
    print("❌ Error loading model/vectorizer:", e)
    raise e

# ------------------------------------------------------------
# 3️⃣ API Endpoints
# ------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True, "time": datetime.utcnow().isoformat()}


@app.post("/api/classify")
def classify(text: str = Form(...), subject: str = Form("")):
    try:
        text = str(text)
        subject = str(subject)
        full_text = f"{subject} {text}".strip()
        if not full_text:
            raise HTTPException(status_code=400, detail="Empty email content")

        try:
            X = vectorizer.transform([full_text])
        except Exception as vec_err:
            raise HTTPException(status_code=500, detail=f"Vectorization failed: {vec_err}")

        proba = model.predict_proba(X)[0]
        label_idx = np.argmax(proba)
        label = "phish" if model.classes_[label_idx] == 1 else "legit"
        confidence = float(np.max(proba))

        # SHAP explanation
        try:
            shap_values = explainer.shap_values(X)
            feature_names = vectorizer.get_feature_names_out()
            sv = shap_values[label_idx][0] if isinstance(shap_values, list) else shap_values[0]
            top_idx = np.argsort(np.abs(sv))[-8:][::-1]
            explanation = [{"feature": feature_names[i], "impact": float(sv[i])} for i in top_idx]
        except Exception as shap_err:
            explanation = {"error": "Explainability failed", "detail": str(shap_err)}

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "label": label,
            "confidence": round(confidence, 4),
            "explanation": explanation,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print("❌ Classification error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Server error: {e}")


@app.post("/api/feedback")
def feedback(email_text: str = Form(...), user_label: str = Form(...)):
    """
    Optional: store feedback for adaptive learning
    """
    print(f"📝 Feedback received: Label={user_label}")
    return {
        "status": "success",
        "message": "Feedback recorded",
        "user_label": user_label,
        "received_at": datetime.utcnow().isoformat(),
    }

# ------------------------------------------------------------
# 4️⃣ Serve Frontend (last!)
# ------------------------------------------------------------
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")















# """
# PhishGuard Backend + Frontend
# -----------------------------
# - Serves real-time phishing detection API
# - SHAP explainability
# - Serves responsive frontend at '/'
# """

# from fastapi import FastAPI, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# import joblib
# import numpy as np
# from datetime import datetime
# import shap
# import traceback
# import os

# # ------------------------------------------------------------
# # 1️⃣ App Configuration
# # ------------------------------------------------------------
# app = FastAPI(title="PhishGuard - Real-Time Phishing Detection API")

# # Allow frontend access (adjust origins for production)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Serve frontend static files
# FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
# app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# # ------------------------------------------------------------
# # 2️⃣ Load Model & Vectorizer
# # ------------------------------------------------------------
# try:
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
#     VEC_PATH = os.path.join(BASE_DIR, "models", "vectorizer.joblib")

#     print("🔹 Loading model and vectorizer...")
#     model = joblib.load(MODEL_PATH)
#     vectorizer = joblib.load(VEC_PATH)
#     print("✅ Model & Vectorizer loaded successfully")

#     print("🔹 Initializing SHAP explainer...")
#     background = vectorizer.transform(["sample email text"])
#     explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")
#     print("✅ SHAP explainer ready")

# except Exception as e:
#     print("❌ Error loading model/vectorizer:", e)
#     raise e

# # ------------------------------------------------------------
# # 3️⃣ API Endpoints
# # ------------------------------------------------------------

# @app.get("/health")
# def health_check():
#     return {"status": "ok", "model_loaded": True, "time": datetime.utcnow().isoformat()}


# @app.post("/api/classify")
# def classify(text: str = Form(...), subject: str = Form("")):
#     try:
#         full_text = f"{subject} {text}"
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

#     except Exception as e:
#         print("❌ Classification error:", traceback.format_exc())
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/api/feedback")
# def feedback(email_text: str = Form(...), user_label: str = Form(...)):
#     """
#     Optional: store feedback for adaptive learning
#     """
#     print(f"📝 Feedback received: Label={user_label}")
#     return {
#         "status": "success",
#         "message": "Feedback recorded",
#         "user_label": user_label,
#         "received_at": datetime.utcnow().isoformat(),
#     }
