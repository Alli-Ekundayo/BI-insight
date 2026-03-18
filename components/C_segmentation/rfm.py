import pandas as pd
import numpy as np

def compute_rfm(df: pd.DataFrame, entity_col: str = 'merchant_id') -> pd.DataFrame:
    """
    Computes RFM (Recency, Frequency, Monetary) + categorical modes 
    for a given entity (e.g., 'merchant_id' or 'pan').
    """
    # Max date as relative "today"
    max_date = df['transaction_time'].max()
    
    # Calculate R, F, M
    rfm = df.groupby(entity_col).agg(
        last_txn_date=('transaction_time', 'max'),
        frequency=('id', 'count'),
        monetary=('transaction_amount', 'sum'),
        total_msc=('msc', 'sum'),
        dispute_count=('dispute_id', lambda x: x.notna().sum()),
        # Mode for categorical context
        primary_department=('department', lambda x: x.mode()[0] if not x.mode().empty else 'UNKNOWN')
    ).reset_index()
    
    # Recency in days
    rfm['recency_days'] = (max_date - rfm['last_txn_date']).dt.days
    rfm['recency_days'] = rfm['recency_days'].fillna(0)
    rfm['recency'] = rfm['recency_days'] # Alias for notebook plots
    
    # Ratios
    rfm['dispute_rate'] = rfm['dispute_count'] / rfm['frequency']
    
    return rfm

# Alias for compatibility
build_rfm_features = compute_rfm

