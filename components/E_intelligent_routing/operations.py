import pandas as pd

def analyze_settlement_accuracy(df: pd.DataFrame) -> dict:
    """
    FRD: "Track the number of settled transactions. Compare data from transactions 
    and settlement tables to ensure precision and reliability."
    Uses 'settlement_date' (if NULL, considered unsettled).
    """
    total = len(df)
    settled = df['settlement_date'].notna().sum()
    unsettled = total - settled
    
    accuracy = settled / total if total > 0 else 0
    
    return {
        "status": "Reduced Settlement Accuracy" if accuracy < 0.95 else "Healthy",
        "settled_count": int(settled),
        "unsettled_count": int(unsettled),
        "accuracy": float(accuracy)
    }

def analyze_chargebacks(df: pd.DataFrame) -> pd.DataFrame:
    """
    FRD: "Track the percentage of transactions resulting in chargebacks or disputes."
    Uses 'dispute_id'.
    """
    total_txns = len(df)
    chargebacks = df[df['dispute_id'].notna()]
    
    cb_volume = len(chargebacks)
    cb_value = chargebacks['transaction_amount'].sum()
    
    cb_rate = cb_volume / total_txns if total_txns > 0 else 0
    
    return {
        "volume": cb_volume,
        "value": cb_value,
        "rate": cb_rate
    }

def analyze_response_time(df: pd.DataFrame) -> float:
    """
    FRD: "Monitor and reduce authorization latency... tracking the time taken 
    for a transaction to be authorized from initiation."
    Note: Standard CSV doesn't have initiation vs auth time, 
    so we return a dummy latency metric for the prototype.
    """
    # Mock latency: 200ms - 1.5s
    import random
    return random.uniform(200.0, 1500.0)
