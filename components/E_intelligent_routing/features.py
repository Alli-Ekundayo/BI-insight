import pandas as pd
import numpy as np
from typing import Dict, List

class RoutingFeatureEngine:
    def __init__(self):
        self.stats = None

    def compute_routing_stats(self, df: pd.DataFrame):
        """
        Aggregates performance metrics per Acquirer, Issuer, and Scheme.
        """
        # 1. Clean data: only learn from txns that actually were processed
        df = df.dropna(subset=['acquirer_institution_id', 'status']).copy()
        
        # 2. Map NaNs in grouping keys to string placeholders
        df['issuer_institution_id'] = df['issuer_institution_id'].fillna("UNKNOWN")
        df['scheme'] = df['scheme'].fillna("UNKNOWN")

        # Ensure status is numeric (200 = Success)
        df['is_success'] = (df['status'] == '200') | (df['status'] == 200)
        
        # Group by the routing triplet (don't drop NaNs, they represent unknown issuers)
        grouped = df.groupby(['acquirer_institution_id', 'issuer_institution_id', 'scheme'], dropna=False).agg(
            total_txns=('status', 'count'),
            success_count=('is_success', 'sum'),
            avg_msc=('msc', 'mean')
        ).reset_index()
        
        # Calculate Success Rate
        grouped['success_rate'] = grouped['success_count'] / grouped['total_txns']
        
        # Fill NaNs for avg_msc if any (MSC is 0 if not provided)
        grouped['avg_msc'] = grouped['avg_msc'].fillna(0.0)
        
        self.stats = grouped
        return grouped

    def get_best_acquirer(self, issuer_id: str, scheme: str, alpha: float = 0.7) -> Dict:
        """
        Returns the best acquirer for a given issuer and scheme based on:
        Score = alpha * SuccessRate - (1-alpha) * NormalizedCost
        """
        # Map input NaNs or None to the internal string placeholder
        if issuer_id is None or (isinstance(issuer_id, float) and np.isnan(issuer_id)) or str(issuer_id).lower() == 'nan':
            issuer_id = "UNKNOWN"
        if scheme is None or (isinstance(scheme, float) and np.isnan(scheme)) or str(scheme).lower() == 'nan':
            scheme = "UNKNOWN"

        if self.stats is None:
            return {"error": "Stats not computed. Run compute_routing_stats first."}
            
        # Filter for the specific issuer and scheme
        relevant = self.stats[
            (self.stats['issuer_institution_id'] == issuer_id) & 
            (self.stats['scheme'] == scheme)
        ].copy()
        
        if relevant.empty:
            # Fallback to global best acquirer for this scheme
            relevant = self.stats[self.stats['scheme'] == scheme].copy()
            if relevant.empty:
                 return {"error": f"No data found for scheme {scheme}"}
        
        # Normalize costs (msc) to [0, 1] range within the filtered set
        if len(relevant) > 1:
            max_msc = relevant['avg_msc'].max()
            min_msc = relevant['avg_msc'].min()
            if max_msc > min_msc:
                relevant['norm_cost'] = (relevant['avg_msc'] - min_msc) / (max_msc - min_msc)
            else:
                relevant['norm_cost'] = 0.5
        else:
            relevant['norm_cost'] = 0.0

        # Calculate Score
        relevant['routing_score'] = (alpha * relevant['success_rate']) - ((1 - alpha) * relevant['norm_cost'])
        
        # Pick the best
        best = relevant.sort_values(by='routing_score', ascending=False).iloc[0]
        
        return {
            "recommended_acquirer": best['acquirer_institution_id'],
            "expected_success_rate": float(best['success_rate']),
            "avg_msc": float(best['avg_msc']),
            "score": float(best['routing_score'])
        }
