import pandas as pd
import numpy as np

class EnsembleForecastModel:
    def __init__(self, xgb_client, prophet_client, xgb_weight=0.6, prophet_weight=0.4):
        self.xgb_model = xgb_client
        self.prophet_model = prophet_client
        self.w_xgb = xgb_weight
        self.w_prophet = prophet_weight

    def fit(self, df: pd.DataFrame):
        self.xgb_model.fit(df)
        self.prophet_model.fit(df)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        xgb_preds = self.xgb_model.predict(df)
        prophet_preds = self.prophet_model.predict(df)
        
        # Weighted average
        ensemble_preds = (xgb_preds * self.w_xgb) + (prophet_preds * self.w_prophet)
        
        # Revenue cannot be negative
        ensemble_preds = np.maximum(ensemble_preds, 0)
        return ensemble_preds
        
    def generate_narrative(self, df_predict: pd.DataFrame, preds: np.ndarray) -> str:
        """
        Generates a human-readable insight based on tomorrow's forecast.
        """
        target_date = df_predict['ds'].iloc[-1].strftime('%A, %b %d')
        forecast_val = preds[-1]
        
        # Check if education txns are unusually high in recent history
        edu_count = df_predict['education_txn_count'].iloc[-1]
        
        narrative = f"📊 *Daily Revenue Forecast for {target_date}*\n"
        narrative += f"Expected volume: ₦{forecast_val:,.2f}.\n"
        
        if edu_count > 10: # arbitrary heuristic for the narrative demo
            narrative += "📈 Education segment transactions are historically elevated, boosting tomorrow's forecast."
        
        return narrative

# Alias for notebook compatibility
ForecastEnsemble = EnsembleForecastModel

