import os
import json
import joblib
import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from datetime import datetime

# Load the model
model = joblib.load('fraud_model.pkl')
print("Model loaded successfully.")

# Feature order must match training
FEATURE_COLS = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']

# Use env var so it works both locally and inside Docker
KAFKA_SERVER = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')

# Consumer
consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers=KAFKA_SERVER,        # ← no quotes
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='latest',
    group_id='fraud-detection-group'
)

# Alert producer
alert_producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,        # ← no quotes
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

total_seen  = 0
total_fraud = 0
THRESHOLD   = 0.5

print(f"Consumer started — connected to {KAFKA_SERVER}")
print(f"\n{'Time':<25} {'TxnID':<10} {'Score':<10} {'Decision'}")
print("-" * 60)

try:
    for message in consumer:
        txn = message.value
        total_seen += 1

        features   = pd.DataFrame([{col: txn.get(col, 0) for col in FEATURE_COLS}])
        fraud_prob = model.predict_proba(features)[0][1]
        is_fraud   = fraud_prob >= THRESHOLD
        decision   = "FRAUD" if is_fraud else "legit"
        timestamp  = txn.get('timestamp', datetime.now().isoformat())[:19]

        print(f"{timestamp:<25} #{txn['transaction_id']:<9} {fraud_prob:.4f}    {decision}")

        if is_fraud:
            total_fraud += 1
            alert = {
                'transaction_id': txn['transaction_id'],
                'timestamp':      timestamp,
                'fraud_score':    round(float(fraud_prob), 4),
                'scaled_amount':  txn.get('scaled_amount'),
                'alert_time':     datetime.now().isoformat()
            }
            alert_producer.send('fraud-alerts', value=alert)
            print(f"  *** ALERT PUBLISHED: txn #{txn['transaction_id']} (score={fraud_prob:.4f}) ***")

        if total_seen % 100 == 0:
            print(f"\n--- Stats: {total_seen} processed | {total_fraud} flagged "
                  f"({total_fraud/total_seen*100:.2f}%) ---\n")

except KeyboardInterrupt:
    print(f"\nStopped. Processed: {total_seen} | Fraud flagged: {total_fraud}")
    consumer.close()
    alert_producer.close()