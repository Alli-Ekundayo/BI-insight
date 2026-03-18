import plotly.express as px
import pandas as pd
import numpy as np

def plot_clusters(rfm_df: pd.DataFrame, output_path: str):
    """
    Generates a 3D scatter plot of the clusters since we have R, F, and M.
    If UMAP was used, we would plot 2D embeddings here.
    """
    # Just standardise 'persona' to string if not present
    if 'persona' not in rfm_df.columns:
        rfm_df['persona'] = rfm_df.get('cluster', 'Unknown').astype(str)
        
    # We plot log(monetary) and log(frequency) for better visual spread
    plot_df = rfm_df.copy()
    plot_df['log_frequency'] = plot_df['frequency'].apply(lambda x: np.log1p(x))
    plot_df['log_monetary'] = plot_df['monetary'].apply(lambda x: np.log1p(max(x, 0)))
    
    fig = px.scatter_3d(
        plot_df,
        x='recency_days', 
        y='log_frequency', 
        z='log_monetary',
        color='persona',
        hover_data=['frequency', 'monetary', 'primary_department'],
        title="Merchant RFM Segmentation"
    )
    
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    fig.write_html(output_path)
    return output_path
