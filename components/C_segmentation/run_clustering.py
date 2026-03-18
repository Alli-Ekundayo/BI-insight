import argparse
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.loader import load_transactions
from components.C_segmentation.rfm import build_rfm_features
from components.C_segmentation.clustering import SegmentationModel
from components.C_segmentation.personas import assign_personas
from components.C_segmentation.visualise import plot_clusters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="transactions snapshot.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        import config
        input_path = config.BASE_DIR / input_path

    print("Loading transaction data...")
    df = load_transactions(input_path)
    if args.limit:
        df = df.head(args.limit)

    print("Step 1: Aggregating RFM features per Merchant (terminal_owner)...")
    rfm_df = build_rfm_features(df, 'terminal_owner')
    print(f"Generated RFM profiles for {len(rfm_df)} unique merchants.")

    # We need at least a few merchants to cluster
    if len(rfm_df) < 5:
        print("Not enough unique merchants to form meaningful clusters. Exiting.")
        sys.exit(0)

    print("Step 2: Fitting HDBSCAN Clustering Model...")
    # Adjust min_cluster_size if dataset is very small
    min_size = min(5, max(2, int(len(rfm_df) * 0.05)))
    seg_model = SegmentationModel(min_cluster_size=min_size)
    
    labels, _ = seg_model.fit_predict(rfm_df)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Model identified {n_clusters} distinct clusters + 1 noise class.")

    print("Step 3: Assigning Personas...")
    persona_df = assign_personas(rfm_df, labels)
    
    # Print a summary
    summary = persona_df.groupby('persona').agg(
        merchant_count=('terminal_owner', 'count'),
        avg_revenue=('monetary', 'mean'),
        avg_freq=('frequency', 'mean')
    ).round(2).sort_values('merchant_count', ascending=False)
    
    print("\n--- Persona Summary ---")
    print(summary.to_string())

    print("\nStep 4: Generating Interactive Visualisation...")
    import config
    out_file = config.REPORTS_DIR / "merchant_clusters.html"
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    plot_clusters(persona_df, str(out_file))
    print(f"✅ Dashboard saved to: {out_file}")

if __name__ == "__main__":
    main()
