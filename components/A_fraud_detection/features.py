import pandas as pd
from datetime import timedelta
import sys
import os

# Ensure config is available
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class RealTimeFeatureEngine:
    def __init__(self):
        # State for sequence-based features (velocity)
        self.txn_history = []
        self.merchant_amounts = {} # {merchant_id: [amounts]}
        
        # State for Benchmarking (FRD Requirement)
        # Simplified: stores (count, total_sum) per day/entity for historical comparison
        self.daily_benchmarks = {} # {date: {merchant_id: {'volume': X, 'value': Y}}}
        self.global_moving_avg = {'volume': 0.0, 'value': 0.0, 'count': 0}

    def compute_features(self, row: dict) -> tuple:
        """
        Takes a raw transaction dict, returns (enriched_row, numeric_feature_vector).
        """
        txn_time = row['transaction_time']
        cur_date = txn_time.date()
        pan = row.get('pan', '')
        # Extract PAN prefix to group velocities (since PAN is masked e.g. 507872XXXXXX1139)
        pan_prefix = str(pan)[:6] if pd.notna(pan) else 'UNKNOWN'
        
        amt = float(row.get('transaction_amount', 0) or 0)
        if pd.isna(amt): amt = 0.0
        
        merchant = str(row.get('merchant_id', 'UNKNOWN'))
        
        # 1. Update State
        self.txn_history.append({'time': txn_time, 'pan_prefix': pan_prefix, 'amount': amt})
        if merchant not in self.merchant_amounts:
            self.merchant_amounts[merchant] = []
        self.merchant_amounts[merchant].append(amt)
        
        # 2. Compute Velocity (count of txns on this PAN prefix in last 60s)
        cutoff_time = txn_time - timedelta(seconds=config.VELOCITY_WINDOW_SECONDS)
        
        # Prune old state to avoid memory leak in streaming
        self.txn_history = [t for t in self.txn_history if t['time'] >= cutoff_time]
        
        velocity = sum(1 for t in self.txn_history if t['pan_prefix'] == pan_prefix)
        
        # 3. Compute Amount Percentile for this merchant
        merchant_history = self.merchant_amounts.get(merchant, [])
        if len(merchant_history) > 1:
            # Empirical CDF
            less_count = sum(1 for a in merchant_history if a < amt)
            percentile = less_count / len(merchant_history)
        else:
            percentile = 0.5 # Default middle if first txn for merchant
            
        # 4. Scheme-Country Mismatch (e.g., VERVE outside NG)
        scheme = str(row.get('scheme', '')).upper()
        country = str(row.get('terminal_country', '')).upper()
        # Assume valid countries for domestic are NG, NIGERIA, or NaN(handled as 'NAN'/'NONE')
        valid_domestic = ('NG', 'NIGERIA', 'NAN', 'NONE', '', 'NULL', 'NA')
        mismatch = 1 if scheme == 'VERVE' and pd.notna(row.get('terminal_country')) and country not in valid_domestic else 0
        
        # Return combined row
        features = dict(row)
        features['derived_velocity_60s'] = velocity
        features['derived_amount_percentile'] = percentile
        features['derived_scheme_mismatch'] = mismatch
        
        # Numeric vector for ML Model
        numeric_features = [
            float(velocity), 
            float(percentile), 
            float(mismatch),
            float(amt)
        ]
        
        return features, numeric_features
