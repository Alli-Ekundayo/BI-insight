import pandas as pd
from datetime import datetime, timedelta
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.A_fraud_detection.features import RealTimeFeatureEngine
import config

def test_velocity_computation():
    engine = RealTimeFeatureEngine()
    
    # Base transaction 1
    t1 = datetime(2026, 3, 17, 10, 0, 0)
    row1 = {'transaction_time': t1, 'pan': '507872_1', 'transaction_amount': 100}
    feats1, _ = engine.compute_features(row1)
    
    assert feats1['derived_velocity_60s'] == 1
    
    # Fast forward 10 seconds, same PAN prefix
    t2 = t1 + timedelta(seconds=10)
    row2 = {'transaction_time': t2, 'pan': '507872_2', 'transaction_amount': 200}
    feats2, _ = engine.compute_features(row2)
    
    assert feats2['derived_velocity_60s'] == 2
    
    # Fast forward 61 seconds (beyond window)
    t3 = t2 + timedelta(seconds=61)
    row3 = {'transaction_time': t3, 'pan': '507872_3', 'transaction_amount': 300}
    feats3, _ = engine.compute_features(row3)
    
    # Should be 1 because the first two fell out of the 60s window relative to t3
    assert feats3['derived_velocity_60s'] == 1

def test_scheme_mismatch():
    engine = RealTimeFeatureEngine()
    
    row1 = {'transaction_time': datetime.now(), 'scheme': 'VERVE', 'terminal_country': 'US'}
    feats1, _ = engine.compute_features(row1)
    assert feats1['derived_scheme_mismatch'] == 1

    row2 = {'transaction_time': datetime.now(), 'scheme': 'VERVE', 'terminal_country': 'NG'}
    feats2, _ = engine.compute_features(row2)
    assert feats2['derived_scheme_mismatch'] == 0

    row3 = {'transaction_time': datetime.now(), 'scheme': 'MASTERCARD', 'terminal_country': 'US'}
    feats3, _ = engine.compute_features(row3)
    assert feats3['derived_scheme_mismatch'] == 0
