import argparse
import sys
import os
import time
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.loader import load_transactions
from components.B_revenue_forecast.features import build_forecast_features
from components.B_revenue_forecast.xgboost_model import XGBoostForecastModel
from components.B_revenue_forecast.prophet_model import ProphetForecastModel
from components.B_revenue_forecast.ensemble import EnsembleForecastModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true", help="Run model on holdout set")
    parser.add_argument("--input", default="transactions snapshot.csv")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        import config
        input_path = config.BASE_DIR / input_path

    print("Loading data...")
    df = load_transactions(input_path)
    
    # We need multiple days to make sense of time-series forecasting.
    # Our snapshot might only span a few hours! Let's check.
    span = df['transaction_time'].max() - df['transaction_time'].min()
    print(f"Dataset span: {span}")

    print("Engineering daily features...")
    daily_df = build_forecast_features(df)
    print(f"Aggregated into {len(daily_df)} days of data.")
    
    if len(daily_df) < 5:
        print("\n[WARNING] This dataset spans less than 5 days. A daily forecasting model ")
        print("needs weeks/months of data to find patterns. We will run it as a test, ")
        print("but the metrics will be overfitted/meaningless.\n")
        print("--> Mocking 14 days of history to allow Prophet and XGBoost to run...")
        
        # Duplicate the single day backwards for 14 days with some noise
        base_row = daily_df.iloc[-1].to_dict()
        mocked_rows = []
        for i in range(14, 0, -1):
            row = base_row.copy()
            row['ds'] = row['ds'] - pd.Timedelta(days=i)
            row['y'] = row['y'] * np.random.uniform(0.8, 1.2)
            row['txn_count'] = int(row['txn_count'] * np.random.uniform(0.8, 1.2))
            mocked_rows.append(row)
        
        mock_df = pd.DataFrame(mocked_rows)
        daily_df = pd.concat([mock_df, daily_df]).reset_index(drop=True)
        # Re-apply lags and time features
        daily_df['day_of_week'] = daily_df['ds'].dt.dayofweek
        daily_df['day_of_month'] = daily_df['ds'].dt.day
        daily_df['month'] = daily_df['ds'].dt.month
        daily_df['is_weekend'] = daily_df['day_of_week'].isin([5, 6]).astype(int)
        daily_df['dow_sin'] = np.sin(2 * np.pi * daily_df['day_of_week'] / 7.0)
        daily_df['dow_cos'] = np.cos(2 * np.pi * daily_df['day_of_week'] / 7.0)
        daily_df['y_lag_1'] = daily_df['y'].shift(1).fillna(0)
        daily_df['y_lag_3'] = daily_df['y'].shift(3).fillna(0)
        daily_df['y_lag_7'] = daily_df['y'].shift(7).fillna(0)

    # Split for backtest (last 20% of days, minimum 1)
    split_idx = max(int(len(daily_df) * 0.8), len(daily_df) - 1)
    train_df = daily_df.iloc[:split_idx]
    test_df = daily_df.iloc[split_idx:]
    
    # Train
    xgb_model = XGBoostForecastModel()
    prophet_model = ProphetForecastModel()
    ensemble = EnsembleForecastModel(xgb_model, prophet_model)
    
    print(f"Training on {len(train_df)} days...")
    start = time.time()
    ensemble.fit(train_df)
    print(f"Training completed in {time.time()-start:.2f}s")
    
    if args.backtest and len(test_df) > 0:
        print("\n--- BACKTEST RESULTS ---")
        xgb_eval = xgb_model.evaluate(test_df)
        prop_eval = prophet_model.evaluate(test_df)
        print(f"XGBoost MAPE: {xgb_eval['mape']:.2%} | RMSE: {xgb_eval['rmse']:.2f}")
        print(f"Prophet MAPE: {prop_eval['mape']:.2%} | RMSE: {prop_eval['rmse']:.2f}")
        
        preds = ensemble.predict(test_df)
        actual = test_df['y'].values
        from sklearn.metrics import mean_absolute_percentage_error
        mask = actual != 0
        if np.any(mask):
            ens_mape = mean_absolute_percentage_error(actual[mask], preds[mask])
            print(f"Ensemble MAPE: {ens_mape:.2%}")

    # Generate next day forecast based on the last row of test_df
    print("\n--- NEXT DAY FORECAST ---")
    if len(test_df) > 0:
        forecast_input = test_df.tail(1)
        # Shift date by 1 to represent tomorrow
        forecast_input = forecast_input.copy()
        forecast_input['ds'] = forecast_input['ds'] + pd.Timedelta(days=1)
        
        preds = ensemble.predict(forecast_input)
        narrative = ensemble.generate_narrative(forecast_input, preds)
        
        print(narrative)

if __name__ == "__main__":
    main()
