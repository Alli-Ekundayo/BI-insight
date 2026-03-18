import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.E_intelligent_routing.model import SmartRouter
from data.loader import load_transactions

def verify_routing():
    print("--- Verifying Intelligent Routing (Component E) ---")
    
    # 1. Load Data
    df = load_transactions()
    
    # 2. Initialize Router
    router = SmartRouter()
    router.train(df)
    
    # 3. Test Recommendation
    # Pick a row that has valid acquirer and status
    valid_rows = df.dropna(subset=['acquirer_institution_id', 'status'])
    if valid_rows.empty:
        print("❌ Error: No valid rows for routing evaluation in the dataset.")
        return
        
    sample = valid_rows.iloc[0]
    issuer = sample['issuer_institution_id']
    scheme = sample['scheme']
    
    print(f"\nRequesting route for Issuer: {issuer}, Scheme: {scheme}")
    rec = router.recommend(issuer, scheme)
    
    if "error" in rec:
        print(f"❌ Error: {rec['error']}")
    else:
        print(f"✅ Recommended Acquirer: {rec['recommended_acquirer']}")
        print(f"📊 Expected Success Rate: {rec['expected_success_rate']:.2%}")
        print(f"💰 Avg MSC: {rec['avg_msc']:.4f}")
        print(f"⭐ Routing Score: {rec['score']:.4f}")

    # 4. Test API (assuming it auto-reloaded)
    print("\nAPI Integration Hint: Try hitting the endpoint:")
    print(f"curl \"http://localhost:8000/recommend_route?issuer_id={issuer}&scheme={scheme}\"")

if __name__ == "__main__":
    verify_routing()
