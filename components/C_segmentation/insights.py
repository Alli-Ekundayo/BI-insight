import pandas as pd
import numpy as np

def analyze_geography(df: pd.DataFrame) -> pd.DataFrame:
    """
    FRD: "Analyse where transactions are originating to optimize ATM and POS placement."
    Uses 'terminal_id' or 'merchant_name' as a proxy for location if 'terminal_country' is empty.
    """
    # Group by terminal_id (assuming first few digits are region-coded)
    geo_df = df.groupby('terminal_id').agg(
        volume=('id', 'count'),
        value=('transaction_amount', 'sum')
    ).reset_index()
    
    # Calculate averages for benchmarking
    avg_vol = geo_df['volume'].mean()
    geo_df['deviation_pct'] = (geo_df['volume'] - avg_vol) / avg_vol
    
    return geo_df

def analyze_peak_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    FRD: "Determine peak transaction times to optimise staffing and resources."
    Uses hourly bins.
    """
    df_copy = df.copy()
    df_copy['hour'] = df_copy['transaction_time'].dt.hour
    
    peak_df = df_copy.groupby('hour').agg(
        txn_count=('id', 'count')
    ).reset_index()
    
    return peak_df

def detect_cross_sell(df: pd.DataFrame) -> list:
    """
    FRD: "Identify opportunities to cross-sell financial products to existing customers."
    Rule: Customers with high volume but low variety or high revenue potential.
    """
    customer_stats = df.groupby('pan').agg(
        volume=('id', 'count'),
        avg_amt=('transaction_amount', 'mean'),
        schemes=('scheme', 'nunique')
    ).reset_index()
    
    # Logic: High volume, single scheme -> cross-sell other schemes or premium cards
    opportunities = customer_stats[
        (customer_stats['volume'] > 10) & (customer_stats['schemes'] == 1)
    ]
    
    return opportunities['pan'].tolist()
