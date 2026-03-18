import sys
import os
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class FraudModel:
    def __init__(self):
        self.iso_forest = IsolationForest(
            contamination=config.ANOMALY_CONTAMINATION,
            random_state=42,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, numeric_features_df: pd.DataFrame):
        """
        Trains the Isolation Forest on a historical batch of features.
        """
        self.iso_forest.fit(numeric_features_df)
        self.is_fitted = True

    def score(self, row: dict, numeric_features: list) -> dict:
        """
        Evaluate a single transaction with ML + Rules.
        Returns the original row with 'risk_score', 'is_flagged', and 'trigger_reason'.
        """
        res = dict(row)
        reasons = []
        is_flagged = False
        
        # 1. ML Score (Isolation Forest)
        # IF returns 1 for inliers, -1 for outliers
        if self.is_fitted:
            # Predict expects a 2D array
            pred = self.iso_forest.predict([numeric_features])[0]
            # Convert decision_function to a ~0-1 risk score (lower is more anomalous in sklearn)
            raw_score = self.iso_forest.decision_function([numeric_features])[0]
            # Normalize it loosely: typically range is [-0.5, 0.5]
            risk_score = 0.5 - raw_score
            
            if pred == -1:
                is_flagged = True
                reasons.append(f"ML Anomaly (Score: {risk_score:.2f})")
        else:
            risk_score = 0.0

        # 2. Rule Booster
        status = str(row.get('status', ''))
        velocity = row.get('derived_velocity_60s', 0)
        percentile = row.get('derived_amount_percentile', 0)
        mismatch = row.get('derived_scheme_mismatch', 0)
        amt = float(row.get('transaction_amount', 0) or 0)
        
        # Rule 1: High Velocity
        if velocity > config.VELOCITY_THRESHOLD:
            is_flagged = True
            reasons.append(f"Velocity Breach ({velocity} txns/60s)")
            
        # Rule 2: Benchmark Deviation (FRD requirement)
        vol_dev = row.get('derived_vol_deviation', 0)
        val_dev = row.get('derived_val_deviation', 0)
        threshold = config.DEFAULT_THRESHOLD_PERCENT / 100.0
        
        if abs(vol_dev) > threshold:
            is_flagged = True
            reasons.append(f"Volume Anomaly ({vol_dev:+.1%})")
            
        if abs(val_dev) > threshold:
            is_flagged = True
            reasons.append(f"Value Anomaly ({val_dev:+.1%})")

        # Rule 3: Failed high-value + mismatched scheme
        if status != '200' and percentile > config.AMOUNT_PERCENTILE_THRESHOLD and mismatch == 1:
            is_flagged = True
            reasons.append(f"Failed high-value foreign mismatch (amount={amt})")
            
        res['risk_score'] = risk_score
        res['is_flagged'] = is_flagged
        res['trigger_reason'] = " | ".join(reasons) if is_flagged else ""
        
        return res
