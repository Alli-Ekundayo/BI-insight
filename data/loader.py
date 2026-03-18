import pandas as pd
import sys
import os

# Ensure we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_transactions(filepath=config.DATA_FILE):
    """
    Loads and preprocesses the transactions snapshot.
    Handles data types predictably.
    """
    df = pd.read_csv(filepath)
    
    # Date parsing
    df['transaction_time'] = pd.to_datetime(df['transaction_time'], format='mixed')
    if 'settlement_date' in df.columns:
        df['settlement_date'] = pd.to_datetime(df['settlement_date'], errors='coerce')
    
    # Numeric casting
    df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce')
    df['amount_due_merchant'] = pd.to_numeric(df['amount_due_merchant'], errors='coerce')
    df['msc'] = pd.to_numeric(df['msc'], errors='coerce')
    
    # Sort chronologically, critical for real-time simulation
    df = df.sort_values('transaction_time').reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = load_transactions()
    print(f"Loaded {len(df)} rows. Range: {df['transaction_time'].min()} to {df['transaction_time'].max()}")
    print("Core Data Types:")
    print(df[['transaction_time', 'transaction_amount', 'status', 'terminal_owner', 'pan']].dtypes)
