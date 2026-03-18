import pandas as pd
import numpy as np

def build_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given historical transactions, aggregate by day and engineer features
    suitable for XGBoost and Prophet.
    Target: sum of amount_due_merchant
    """
    df_clean = df.copy()
    
    # 1. Daily Aggregations
    # Set transaction_time as index to resample
    df_clean.set_index('transaction_time', inplace=True)
    
    daily = df_clean.resample('D').agg(
        total_amount_due=('transaction_amount', 'sum'),
        total_msc=('msc', 'sum'),
        txn_count=('id', 'count'),
        unique_terminals=('terminal_id', 'nunique'),
        education_txn_count=('department', lambda x: (x == 'cipa').sum()),
        foreign_card_count=('issuer_country', lambda x: x.notna().sum()) # simplistic proxy
    ).reset_index()
    
    # Fill any missing days with 0
    daily.fillna(0, inplace=True)
    
    # rename date column to 'ds', target to 'y' for Prophet compatibility later
    daily.rename(columns={'transaction_time': 'ds', 'total_amount_due': 'y'}, inplace=True)
    
    # 2. Time Engineering
    daily['day_of_week'] = daily['ds'].dt.dayofweek
    daily['day_of_month'] = daily['ds'].dt.day
    daily['month'] = daily['ds'].dt.month
    daily['is_weekend'] = daily['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical features for day of week
    daily['dow_sin'] = np.sin(2 * np.pi * daily['day_of_week'] / 7.0)
    daily['dow_cos'] = np.cos(2 * np.pi * daily['day_of_week'] / 7.0)
    
    # 3. Lag Features
    # Shift daily target by 1, 3, 7 days
    daily['y_lag_1'] = daily['y'].shift(1)
    daily['y_lag_3'] = daily['y'].shift(3)
    daily['y_lag_7'] = daily['y'].shift(7)
    
    # Fill NAs caused by shifting
    daily.fillna(0, inplace=True)
    
    return daily
