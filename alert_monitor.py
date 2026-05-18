import json
from kafka import KafkaConsumer
from datetime import datetime

consumer = KafkaConsumer(
    'fraud-alerts',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='earliest',
    group_id='alert-monitor-group'
)

print("=" * 60)
print("  FRAUD ALERT MONITOR — listening for flagged transactions")
print("=" * 60)

alert_count = 0

for message in consumer:
    alert = message.value
    alert_count += 1

    print(f"\n[ALERT #{alert_count}]")
    print(f"  Transaction ID : {alert['transaction_id']}")
    print(f"  Time           : {alert['timestamp']}")
    print(f"  Fraud Score    : {alert['fraud_score']:.4f}")
    print(f"  Amount (scaled): {alert['scaled_amount']}")
    print(f"  Alert fired at : {alert['alert_time']}")
    print("-" * 40)