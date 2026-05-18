import json
import time
import random
import numpy as np
from kafka import KafkaProducer
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_transaction(transaction_id):
    """
    Generate a fake transaction that mimics the structure of our
    creditcard.csv dataset — 28 V features + scaled amount + scaled time.
    Occasionally inject a 'fraudy' transaction with extreme V values.
    """
    is_fraud = random.random() < 0.002   # 0.2% fraud rate — realistic

    if is_fraud:
        # Fraud transactions have extreme values in key features
        v_features = {f'V{i}': round(np.random.normal(0, 3), 4) for i in range(1, 29)}
        v_features['V14'] = round(np.random.normal(-10, 2), 4)  # V14 is a strong signal
        v_features['V17'] = round(np.random.normal(-8, 2), 4)
        amount = round(random.uniform(1, 300), 2)
    else:
        v_features = {f'V{i}': round(np.random.normal(0, 1), 4) for i in range(1, 29)}
        amount = round(random.uniform(5, 2000), 2)

    transaction = {
        'transaction_id': transaction_id,
        'timestamp': datetime.now().isoformat(),
        'scaled_amount': round((amount - 88.35) / 250.12, 4),
        'scaled_time': round(random.uniform(-2, 2), 4),
        **v_features,
        '_is_actually_fraud': is_fraud   # ground truth for testing only
    }
    return transaction

print("Producer started — sending transactions to Kafka...")
print("Press Ctrl+C to stop\n")

transaction_id = 0
try:
    while True:
        txn = generate_transaction(transaction_id)
        producer.send('transactions', value=txn)

        label = "FRAUD" if txn['_is_actually_fraud'] else "legit"
        print(f"[{txn['timestamp']}] txn #{transaction_id:05d} | "
              f"Amount scaled: {txn['scaled_amount']:>7.4f} | {label}")

        transaction_id += 1
        time.sleep(0.5)   # one transaction every 0.5 seconds

except KeyboardInterrupt:
    print("\nProducer stopped.")
    producer.close()