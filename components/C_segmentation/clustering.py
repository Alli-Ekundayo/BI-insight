import pandas as pd
import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class SegmentationEngine:
    def __init__(self, min_cluster_size=5):
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_cols = ['recency_days', 'frequency', 'monetary', 'dispute_rate']

    def fit_predict(self, rfm_df: pd.DataFrame) -> np.ndarray:
        """
        Calculates clusters. Returns and appends labels.
        """
        labels, _ = self.fit_predict_full(rfm_df)
        return labels

    def fit_predict_full(self, rfm_df: pd.DataFrame) -> tuple:
        """
        Scales RFM metrics and computes HDBSCAN clusters.
        """
        # Handle NAs
        df_clean = rfm_df[self.feature_cols].copy()
        df_clean = df_clean.fillna(0)
        
        # Log transform skewed features for better distance metrics
        df_clean['frequency'] = np.log1p(df_clean['frequency'])
        # Handle negative monetary safely
        df_clean['monetary'] = np.log1p(np.maximum(df_clean['monetary'], 0))
        
        # Scale
        X_scaled = self.scaler.fit_transform(df_clean)
        
        # Cluster
        labels = self.model.fit_predict(X_scaled)
        
        self.is_fitted = True
        return labels, X_scaled

# Alias for compatibility
SegmentationModel = SegmentationEngine

