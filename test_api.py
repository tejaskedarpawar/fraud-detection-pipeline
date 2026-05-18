import requests
import numpy as np
import random

API_URL = "http://localhost:8000"

def make_transaction(is_fraud=False):
    if is_fraud:
        v = {f'V{i}': round(np.random.normal(0, 3), 4) for i in range(1, 29)}
        v['V14'] = round(np.random.normal(-10, 2), 4)
        v['V17'] = round(np.random.normal(-8, 2), 4)
    else:
        v = {f'V{i}': round(np.random.normal(0, 1), 4) for i in range(1, 29)}

    return {
        **v,
        "scaled_amount": round(random.uniform(-1, 3), 4),
        "scaled_time":   round(random.uniform(-2, 2), 4),
        "transaction_id": f"txn_{random.randint(1000, 9999)}"
    }

# Test 1 — health check
print("=== Health Check ===")
r = requests.get(f"{API_URL}/health")
print(r.json())

# Test 2 — legit transaction
print("\n=== Legit Transaction ===")
r = requests.post(f"{API_URL}/predict", json=make_transaction(is_fraud=False))
print(r.json())

# Test 3 — fraudulent transaction
print("\n=== Fraud Transaction ===")
r = requests.post(f"{API_URL}/predict", json=make_transaction(is_fraud=True))
print(r.json())

# Test 4 — metrics
print("\n=== API Metrics ===")
r = requests.get(f"{API_URL}/metrics")
print(r.json())