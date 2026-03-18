import pandas as pd

def calculate_arpu(df: pd.DataFrame) -> float:
    """
    FRD: "Total revenue divided by total number of users"
    Uses 'transaction_amount' as revenue and 'pan' as unique users.
    """
    total_rev = df['transaction_amount'].sum()
    total_users = df['pan'].nunique()
    if total_users == 0:
        return 0.0
    return total_rev / total_users

def predict_card_replacements(df: pd.DataFrame, target_month: int = None) -> pd.DataFrame:
    """
    FRD: "Predict when customers are likely to request card replacements."
    Uses 'due_date' column.
    """
    if 'due_date' not in df.columns:
        return pd.DataFrame()
        
    df['due_date'] = pd.to_datetime(df['due_date'], errors='coerce')
    expiring = df[df['due_date'].notna()]
    
    if target_month:
        expiring = expiring[expiring['due_date'].dt.month == target_month]
        
    return expiring.groupby('pan')['due_date'].max().reset_index()

def track_merchant_onboarding(df: pd.DataFrame) -> pd.DataFrame:
    """
    FRD: "Evaluate merchant onboarding rate. Count of total merchants onboarded over a period."
    Finds first transaction date for each merchant.
    """
    onboarding = df.groupby('merchant_id')['transaction_time'].min().reset_index()
    onboarding.columns = ['merchant_id', 'onboarding_date']
    
    # Group by month
    monthly = onboarding.groupby(onboarding['onboarding_date'].dt.to_period('M')).size()
    return monthly.reset_index()
