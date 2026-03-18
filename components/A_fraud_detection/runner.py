import argparse
import sys
import os
import time
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.loader import load_transactions
from components.A_fraud_detection.features import RealTimeFeatureEngine
from components.A_fraud_detection.model import FraudModel
from components.A_fraud_detection.alerts import AlertDispatcher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="transactions snapshot.csv", help="Input dataset path")
    parser.add_argument("--limit", type=int, default=5000, help="Number of rows to process")
    args = parser.parse_args()

    # Support absolute paths or relative to project root
    input_path = args.input
    if not os.path.exists(input_path):
        import config
        input_path = config.BASE_DIR / input_path

    print(f"Loading data from {input_path}...")
    try:
        df = load_transactions(input_path)
    except Exception as e:
        print(f"Error loading transactions: {e}")
        sys.exit(1)

    if args.limit:
        df = df.head(args.limit)
        
    print(f"Data loaded. Simulating stream for {len(df)} transactions...")
    
    # Init components
    feature_engine = RealTimeFeatureEngine()
    fraud_model = FraudModel()
    alert_dispatcher = AlertDispatcher()
    
    # 1. We need a small batch to fit the Isolation Forest first
    # In production this would be offline trained daily by Airflow
    batch_size = max(50, int(len(df) * 0.2))
    print(f"Fitting Isolation Forest on first {batch_size} rows of stream...")
    
    historical_features = []
    for i, row in df.head(batch_size).iterrows():
        enriched, num_vector = feature_engine.compute_features(row.to_dict())
        historical_features.append(num_vector)
        
    fraud_model.fit(pd.DataFrame(historical_features))
    print("Isolation Forest trained.")
    
    # 2. Streaming execution
    alerts_triggered = 0
    start_time = time.time()
    
    # Continue streaming from the dataset
    print("Starting real-time evaluation...")
    for i, row in df.iloc[batch_size:].iterrows():
        # Feature Engineering (Fast stateful updates)
        enriched_row, numeric_vector = feature_engine.compute_features(row.to_dict())
        
        # Scoring (ML + Rules)
        scored_txn = fraud_model.score(enriched_row, numeric_vector)
        
        # Alerting
        if scored_txn.get('is_flagged'):
            alert_dispatcher.dispatch(scored_txn)
            alerts_triggered += 1
            
    elapsed = time.time() - start_time
    print(f"\nSimulation complete: {len(df)-batch_size} txns evaluated in {elapsed:.2f}s.")
    print(f"Total Alerts Triggered: {alerts_triggered}")

if __name__ == "__main__":
    main()
