import pandas as pd
import numpy as np

def assign_personas(rfm_df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """
    Given an RFM dataframe and HDBSCAN cluster labels, applies a rule-based
    engine to name the clusters (personas) by analysing their centroids.
    """
    df = rfm_df.copy()
    df['cluster'] = cluster_labels
    
    # -1 is HDBSCAN's noise label
    persona_mapping = {-1: "Outliers & Noise"}
    
    # Compute cluster centroids
    valid_clusters = [c for c in np.unique(cluster_labels) if c != -1]
    
    for c in valid_clusters:
        cluster_data = df[df['cluster'] == c]
        
        avg_freq = cluster_data['frequency'].mean()
        avg_monetary = cluster_data['monetary'].mean()
        avg_recency = cluster_data['recency_days'].mean()
        
        # Check primary department mode
        # get most common department in this cluster
        mode_dept = cluster_data['primary_department'].mode()
        dept = mode_dept[0] if not mode_dept.empty else ""
        
        # Rule-based persona labeller
        if avg_freq > 10 and avg_monetary < 5000:
            persona = "High-Freq Micro-Transactors"
        elif dept == 'cipa' and avg_monetary > 10000:
            persona = "Exam-fee Bulk Payers"
        elif avg_recency > 14:
            persona = "Churn-Risk / Dormant"
        elif avg_monetary > 50000:
            persona = "Whales / VIPs"
        else:
            persona = f"Standard Cluster {c}"
            
        persona_mapping[c] = persona
        
    df['persona'] = df['cluster'].map(persona_mapping)
    return df
