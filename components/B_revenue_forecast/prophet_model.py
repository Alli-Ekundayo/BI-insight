from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

class ProphetForecastModel:
    def __init__(self):
        self.model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        # Add regressors that match our feature layer
        self.model.add_regressor('education_txn_count')
        self.model.add_regressor('foreign_card_count')
        self.model.add_regressor('y_lag_1')
        self.model.add_regressor('y_lag_7')
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Prophet expects 'ds' and 'y', which our feature layer provides.
        """
        self.model.fit(df)
        self.is_fitted = True

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict.")
        forecast = self.model.predict(df)
        return forecast['yhat'].values

    def evaluate(self, df: pd.DataFrame) -> dict:
        preds = self.predict(df)
        y_true = df['y'].values
        
        mask = y_true != 0
        if not np.any(mask):
            return {"mape": 0.0, "rmse": 0.0}
            
        mape = mean_absolute_percentage_error(y_true[mask], preds[mask])
        rmse = np.sqrt(np.mean((y_true - preds)**2))
        return {
            "mape": round(float(mape), 4),
            "rmse": round(float(rmse), 2)
        }
