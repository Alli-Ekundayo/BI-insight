import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

class XGBoostForecastModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.features = [
            'total_msc', 'txn_count', 'unique_terminals',
            'education_txn_count', 'foreign_card_count', 'day_of_week',
            'day_of_month', 'month', 'is_weekend', 'dow_sin', 'dow_cos',
            'y_lag_1', 'y_lag_3', 'y_lag_7'
        ]
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        X = df[self.features]
        y = df['y']
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict.")
        X = df[self.features]
        return self.model.predict(X)

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Evaluate on the provided dataframe (assumed to be holdout).
        """
        preds = self.predict(df)
        y_true = df['y'].values
        
        # Avoid div by zero in MAPE
        mask = y_true != 0
        if not np.any(mask):
            return {"mape": 0.0, "rmse": 0.0}
            
        mape = mean_absolute_percentage_error(y_true[mask], preds[mask])
        rmse = np.sqrt(np.mean((y_true - preds)**2))
        return {
            "mape": round(float(mape), 4),
            "rmse": round(float(rmse), 2)
        }
