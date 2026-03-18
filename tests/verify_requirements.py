import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_api():
    print("Starting API Verification...")
    
    # 1. Health check
    try:
        resp = requests.get(f"{BASE_URL}/health")
        print(f"Health Check: {resp.status_code} - {resp.json()}")
    except Exception as e:
        print(f"API NOT RUNNING: {e}")
        return

    # 2. Test Fraud Score (Anomaly)
    print("\nTesting Fraud Score (High deviation value)...")
    huge_txn = {
        "transaction_id": "TEST_ANOMALY_001",
        "transaction_time": "2026-03-18 10:00:00",
        "transaction_amount": 10000000.0, # Huge amount to trigger deviation
        "pan": "507872XXXXXX9999",
        "terminal_id": "T001",
        "merchant_id": "M001",
        "scheme": "VERVE",
        "terminal_country": "NG",
        "status": "200"
    }
    resp = requests.post(f"{BASE_URL}/fraud_score", json=huge_txn)
    print(f"Fraud Score Response: {resp.json().get('is_flagged')} | Reason: {resp.json().get('trigger_reason')}")

    # 3. Test Insights
    endpoints = ["churn", "geography", "peak_times", "arpu", "settlement"]
    for ep in endpoints:
        print(f"\nTesting /insights/{ep}...")
        resp = requests.get(f"{BASE_URL}/insights/{ep}")
        print(f"Response ({ep}): {json.dumps(resp.json(), indent=2)[:200]}...")

    # 4. Test Past Insights
    print("\nTesting /past_insights...")
    resp = requests.get(f"{BASE_URL}/past_insights")
    print(f"Past Insights Count: {len(resp.json())}")

if __name__ == "__main__":
    test_api()
