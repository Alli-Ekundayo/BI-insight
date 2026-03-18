from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from components.A_fraud_detection.features import RealTimeFeatureEngine
from components.A_fraud_detection.model import FraudModel
from components.A_fraud_detection.alerts import AlertDispatcher

from components.C_segmentation.churn import predict_churn, calculate_churn_rate, get_churn_alert
from components.C_segmentation.insights import analyze_geography, analyze_peak_times
from components.C_segmentation.clustering import SegmentationModel

from components.B_revenue_forecast.performance import calculate_arpu, track_merchant_onboarding
from components.E_intelligent_routing.operations import analyze_settlement_accuracy, analyze_chargebacks

from components.D_mlops.logger import InsightLogger
from data.loader import load_transactions

import config
import pandas as pd

app = FastAPI(title="APCB ML Insights API", description="Schema-Mapped Intelligence Endpoints")

# Globals
os.makedirs(config.REPORTS_DIR, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(config.REPORTS_DIR)), name="reports")

insight_logger = InsightLogger()
alert_dispatcher = AlertDispatcher()
feature_engine = RealTimeFeatureEngine()
fraud_model = FraudModel()
fraud_model.is_fitted = True

# Cache loaded data for insights
df_cache = None
try:
    df_cache = load_transactions()
except Exception as e:
    print(f"Error loading data for API: {e}")

@app.get("/")
def read_root():
    return RedirectResponse(url="/reports/merchant_clusters.html")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/fraud_score")
def score_transaction(txn_data: dict):
    """
    FRD: "Trigger prompt and configurable notifications for detected insights."
    """
    try:
        # Convert string timestamp to datetime for the feature engine
        if isinstance(txn_data.get('transaction_time'), str):
            txn_data['transaction_time'] = pd.to_datetime(txn_data['transaction_time'])
            
        enriched, num_vec = feature_engine.compute_features(txn_data)
        scored = fraud_model.score(enriched, num_vec)
        
        # Convert datetime back to string for JSON responsiveness
        scored['transaction_time'] = str(scored['transaction_time'])
        
        if scored.get('is_flagged'):
            # Determine category for FRD template
            category = "GENERAL"
            if scored.get('terminal_id'): category = "TERMINALS"
            elif scored.get('pan'): category = "CARDS"
            
            alert_dispatcher.dispatch(scored, alert_type="VOLUME", category=category)
            insight_logger.log_insight({"type": "FRAUD_ANOMALY", "data": scored})
            
        return scored
    except Exception as e:
        print(f"Error in fraud_score: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/insights/churn")
def get_churn_insight():
    """
    FRD: "Monitor customers' churn rate and suggest retention strategies."
    """
    if df_cache is None: raise HTTPException(status_code=500, detail="Data not loaded")
    churn_df = predict_churn(df_cache)
    rate = calculate_churn_rate(churn_df)
    alert = get_churn_alert(rate, 0.05) # Compare against 5% benchmark
    if alert: insight_logger.log_insight({"type": "CHURN_ALERT", "data": alert})
    return {"churn_rate": rate, "insight": alert}

@app.get("/insights/geography")
def get_geo_insight():
    """
    FRD: "Analyse where transactions are originating to optimize ATM and POS placement."
    """
    if df_cache is None: raise HTTPException(status_code=500, detail="Data not loaded")
    geo_data = analyze_geography(df_cache)
    return geo_data.to_dict(orient='records')

@app.get("/insights/peak_times")
def get_peak_insight():
    """
    FRD: "Determine peak transaction times to optimise staffing and resources."
    """
    if df_cache is None: raise HTTPException(status_code=500, detail="Data not loaded")
    peak_data = analyze_peak_times(df_cache)
    return peak_data.to_dict(orient='records')

@app.get("/insights/arpu")
def get_arpu():
    """
    FRD: "Analyse average revenue per user."
    """
    if df_cache is None: raise HTTPException(status_code=500, detail="Data not loaded")
    return {"arpu": calculate_arpu(df_cache)}

@app.get("/insights/settlement")
def get_settlement():
    """
    FRD: "Track the number of settled transactions and accuracy."
    """
    if df_cache is None: raise HTTPException(status_code=500, detail="Data not loaded")
    return analyze_settlement_accuracy(df_cache)

@app.get("/past_insights")
def get_past_insights():
    """
    FRD: "Provide a page for users to review past insights and suggestions."
    """
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
