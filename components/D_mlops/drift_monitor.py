import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

def generate_drift_report(ref_data: pd.DataFrame, curr_data: pd.DataFrame, output_path: str):
    """
    Given a reference dataset (e.g. last week) and current dataset (e.g. yesterday),
    computes drift detection via Kolmogrov-Smirnov and Wasserstein distance.
    """
    
    # We want to monitor critical columns for the transaction stream
    important_columns = [
        'transaction_amount',
        'status',
        'terminal_owner',
        'department'
    ]
    
    ref_subset = ref_data[important_columns]
    curr_subset = curr_data[important_columns]

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_subset, current_data=curr_subset)
    
    # In production, snapshot.dict() is parsed by Airflow to trigger alerts if dataset drift > threshold
    snapshot.save_html(output_path)
    return snapshot.dict()

if __name__ == "__main__":
    from data.loader import load_transactions
    df = load_transactions()
    
    # Split the dataset in half artificially since we only have 1 day of data
    midpoint = len(df) // 2
    ref = df.iloc[:midpoint]
    curr = df.iloc[midpoint:]
    
    out_file = config.REPORTS_DIR / "data_drift.html"
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    summary = generate_drift_report(ref, curr, str(out_file))
    
    # Extract results from the first metric (DriftedColumnsCount)
    first_metric = summary["metrics"][0]["value"]
    drift_share = first_metric["share"]
    drifting_cols = int(first_metric["count"])
    # In Evidently 0.7.21, the drifted columns count is the first metric.
    # The number of columns can be inferred or hardcoded for the print.
    total_cols = 4 # transaction_amount, status, terminal_owner, department
    
    print(f"Data Drift Report saved to {out_file}")
    print(f"Overall Drift Share: {drift_share:.2%}")
    print(f"Drifted Columns: {drifting_cols} out of {total_cols}")
