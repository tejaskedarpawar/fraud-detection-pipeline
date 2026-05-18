import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

# ── App setup ──────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction fraud scoring using XGBoost",
    version="1.0.0"
)

# ── Load model once at startup ──────────────────────────────
model = joblib.load("fraud_model.pkl")
FEATURE_COLS = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']

# ── Track basic metrics ─────────────────────────────────────
stats = {
    "total_requests": 0,
    "total_fraud_flagged": 0,
    "started_at": datetime.now().isoformat()
}

# ── Request schema ──────────────────────────────────────────
class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float
    V5: float; V6: float; V7: float; V8: float
    V9: float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float
    scaled_amount: float
    scaled_time: float
    transaction_id: Optional[str] = None

# ── Response schema ─────────────────────────────────────────
class PredictionResponse(BaseModel):
    transaction_id: Optional[str]
    fraud_probability: float
    is_fraud: bool
    decision: str
    threshold: float
    timestamp: str

# ── Endpoints ───────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "fraud_model.pkl",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(txn: Transaction):
    try:
        stats["total_requests"] += 1

        # Build feature dataframe in correct column order
        data = pd.DataFrame([{col: getattr(txn, col) for col in FEATURE_COLS}])

        # Score the transaction
        fraud_prob = float(model.predict_proba(data)[0][1])
        is_fraud   = fraud_prob >= 0.5

        if is_fraud:
            stats["total_fraud_flagged"] += 1

        return PredictionResponse(
            transaction_id   = txn.transaction_id,
            fraud_probability= round(fraud_prob, 6),
            is_fraud         = is_fraud,
            decision         = "FRAUD" if is_fraud else "LEGIT",
            threshold        = 0.5,
            timestamp        = datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    fraud_rate = (
        stats["total_fraud_flagged"] / stats["total_requests"] * 100
        if stats["total_requests"] > 0 else 0
    )
    return {
        **stats,
        "fraud_rate_percent": round(fraud_rate, 4)
    }