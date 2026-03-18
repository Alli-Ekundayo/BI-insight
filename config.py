import os
from pathlib import Path

# Project Paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = BASE_DIR / "data/transactions snapshot.csv"
REPORTS_DIR = BASE_DIR / "reports"

# Fraud Rules Configuration
VELOCITY_WINDOW_SECONDS = 60
VELOCITY_THRESHOLD = 3  # Alert if > 3 txns per PAN in 60s
AMOUNT_PERCENTILE_THRESHOLD = 0.95  # Alert if > 95th percentile for merchant


# Anomaly Thresholds & Benchmarks
MEASUREMENT_PERIODS = ["daily", "weekly", "monthly"]
DEFAULT_THRESHOLD_PERCENT = 20.0  # Alert if > 20% deviation from benchmark

# ML Settings
ANOMALY_CONTAMINATION = 0.01  # Expected proportion of outliers in IF
