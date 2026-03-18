import pandas as pd
import numpy as np
from datetime import timedelta

def predict_churn(df: pd.DataFrame, recency_threshold_days: int = 30) -> pd.DataFrame:
    """
    Identifies unique customers (PANs) lost over a period.
    FRD: "Monitor customers' churn (total number of unique customers lost over a period) 
    rate and suggest retention strategies."
    """
    if 'pan' not in df.columns or 'transaction_time' not in df.columns:
        return pd.DataFrame()
        
    last_txn = df.groupby('pan')['transaction_time'].max().reset_index()
    snapshot_date = df['transaction_time'].max()
    
    last_txn['days_since_last'] = (snapshot_date - last_txn['transaction_time']).dt.days
    
    # Label as churned if recency > threshold
    last_txn['is_churned'] = last_txn['days_since_last'] > recency_threshold_days
    
    return last_txn

def calculate_churn_rate(churn_df: pd.DataFrame) -> float:
    if churn_df.empty:
        return 0.0
    return churn_df['is_churned'].mean()

def get_churn_alert(current_rate: float, benchmark_rate: float):
    """
    FRD: "Critical Alert: Unprecedented Increase in Customer Churn Detected - Immediate Action Recommended!"
    """
    diff_pct = (current_rate - benchmark_rate) / benchmark_rate if benchmark_rate > 0 else 0
    
    if current_rate > benchmark_rate:
        return {
            "prompt": "Critical Alert: Unprecedented Increase in Customer Churn Detected - Immediate Action Recommended!",
            "title": "Increase in Customer Churn Rate",
            "description": "Our system has flagged an uptick in customer churn rate for your esteemed financial institution.",
            "data_snapshot": {
                "Recent Churn Rate Increase": f"{diff_pct:+.1%}",
                "Affected Customer Base": "Estimated counts unavailable in mock",
                "Time Period of Concern": "Last 30 days"
            },
            "recommendation": "We recommend conducting customer outreach and implementing tailored retention initiatives to mitigate the churn risk."
        }
    elif current_rate < benchmark_rate:
         return {
            "prompt": "Notification: Significant Drop in Customer Churn Rate - Positive Trend Detected",
            "title": "Decrease in Customer Churn Rate",
            "description": "Our system has flagged a positive trend in customer churn rate.",
            "data_snapshot": {
                "Recent Churn Rate Decrease": f"{diff_pct:+.1%}"
            },
            "recommendation": "Continue your current retention strategies."
        }
    return None
