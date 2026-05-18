# Real-Time Fraud Detection System

An end-to-end production-grade fraud detection pipeline that processes financial transactions in real-time using Apache Kafka, XGBoost/LightGBM, FastAPI, and Docker.

---

## Architecture

```
Transaction Producer → Kafka Topic → Fraud Detector (ML Model) → Fraud Alerts Topic
                                                                 ↓
                                                         FastAPI REST API
```

- **Producer** — Simulates real-time bank transactions streaming into Kafka
- **Kafka** — Message broker handling real-time transaction streams
- **Fraud Detector** — Consumes transactions, scores with ML model, publishes alerts
- **FastAPI** — REST API for on-demand fraud scoring with Swagger UI
- **Docker** — Full system containerized and orchestrated with docker-compose

---

## Tech Stack

| Layer | Technology |
|---|---|
| Streaming | Apache Kafka (KRaft mode) |
| ML Models | XGBoost, LightGBM, Isolation Forest |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| API | FastAPI + Uvicorn |
| Containerization | Docker + docker-compose |
| Data Processing | Pandas, NumPy, Scikit-learn |

---

## Project Structure

```
fraud-detection-pipeline/
│
├── fraud_eda.py              # Phase 1: EDA and data preprocessing
├── fraud_model.py            # Phase 2: Model training and evaluation
├── producer.py               # Phase 3: Kafka transaction producer
├── consumer.py               # Phase 3: Kafka fraud detection consumer
├── alert_monitor.py          # Phase 3: Real-time fraud alert monitor
├── main.py                   # Phase 4: FastAPI REST API
├── test_api.py               # Phase 4: API test suite
├── Dockerfile.api            # Phase 5: FastAPI container
├── Dockerfile.consumer       # Phase 5: Consumer container
├── docker-compose.yml        # Phase 5: Full system orchestration
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ML Pipeline

### Dataset
- **Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraud rate**: 0.17% (highly imbalanced)

### Approach
1. **EDA** — Analyzed class imbalance, feature distributions, time/amount patterns
2. **Imbalance Handling** — Applied SMOTE to synthetically oversample fraud cases
3. **Models Trained**:
   - XGBoost (primary)
   - LightGBM (comparison)
   - Isolation Forest (unsupervised anomaly detection)
4. **Evaluation** — Used Precision-Recall AUC and F1-score (not accuracy)

### Results

| Metric | Score |
|---|---|
| ROC-AUC | ~0.98 |
| Precision-Recall AUC | ~0.80 |
| Fraud Recall | ~87% |
| Fraud Precision | ~88% |

---

## Getting Started

### Prerequisites
- Docker Desktop
- Python 3.9+
- Git

### Run with Docker (recommended)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/fraud-detection-pipeline.git
cd fraud-detection-pipeline

# Train the model first (requires creditcard.csv from Kaggle)
python fraud_eda.py
python fraud_model.py

# Start the full pipeline
docker compose up

# In a new terminal — start the transaction producer
python producer.py
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/predict` | POST | Score a transaction |
| `/metrics` | GET | API usage statistics |

### Test the API

```bash
python test_api.py
```

Or visit the interactive Swagger UI at:
```
http://localhost:8000/docs
```

### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37,
    "V5": -0.33, "V6": 0.46, "V7": 0.23, "V8": 0.09,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.61,
    "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47,
    "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25,
    "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06,
    "V25": 0.12, "V26": -0.18, "V27": 0.13, "V28": -0.02,
    "scaled_amount": 0.24,
    "scaled_time": -0.99,
    "transaction_id": "txn_001"
  }'
```

### Example Response

```json
{
  "transaction_id": "txn_001",
  "fraud_probability": 0.032,
  "is_fraud": false,
  "decision": "LEGIT",
  "threshold": 0.5,
  "timestamp": "2026-05-18T10:30:00"
}
```

---

## Key Learnings

- Accuracy is a misleading metric for imbalanced datasets — Precision-Recall AUC is far more informative
- SMOTE significantly improves recall on fraud cases vs training on raw imbalanced data
- Kafka enables sub-second transaction scoring latency at scale
- Containerizing ML pipelines with Docker makes deployment environment-agnostic

---

## License

MIT License
