"""
🚨 Anomaly Detection Model - Training Version
يكتشف الحركات الغريبة: Pump & Dump, Flash Crash, Whale Movements
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Feature names (12 features)
FEATURE_NAMES = [
    'rsi', 'macd_diff', 'volume_ratio', 'price_momentum', 'atr',
    'anomaly_score', 'statistical_outliers', 'pattern_anomalies',
    'behavioral_anomalies', 'volume_anomalies',
    'tp_accuracy', 'sell_accuracy',
]


def _extract_features(trade):
    """Extract features from a single trade. Returns list or raises."""
    data = trade.get('data', {})
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        data = {}

    def _f(key, default=0):
        try:
            return float(data.get(key, default) or default)
        except (ValueError, TypeError):
            return float(default)

    return [
        _f('rsi',                  50),
        _f('macd_diff',             0),
        _f('volume_ratio',        1.0),
        _f('price_momentum',        0),
        _f('atr',                 2.5),
        _f('anomaly_score',         0),
        _f('statistical_outliers',  0),
        _f('pattern_anomalies',     0),
        _f('behavioral_anomalies',  0),
        _f('volume_anomalies',      0),
        0.5,   # tp_accuracy
        0.5,   # sell_accuracy
    ]


def train_anomaly_model(trades, voting_scores=None):
    """
    تدريب نموذج كشف الشذوذات
    يتعلم من الصفقات التاريخية لكشف الأنماط الشاذة
    """
    print("\n🚨 Training Anomaly Detection Model...")

    if not trades or len(trades) < 50:
        print("  ⚠️ Not enough trades for anomaly model (need 50+)")
        return None

    features_list, labels_list = [], []

    for trade in trades:
        try:
            features = _extract_features(trade)

            profit = float(
                trade.get('profit_percent',
                trade.get('profit',
                trade.get('pnl', 0))) or 0
            )
            trade_quality = trade.get('trade_quality', 'OK') or 'OK'

            is_safe = 1 if (profit > 0 and trade_quality not in ['TRAP', 'RISKY']) else 0

            features_list.append(features)
            labels_list.append(is_safe)

        except Exception as e:
            print(f"  ⚠️ Skipping trade: {e}")
            continue

    if len(features_list) < 50:
        print(f"  ⚠️ Only {len(features_list)} valid samples, need 50+")
        return None

    X = pd.DataFrame(features_list, columns=FEATURE_NAMES)
    y = pd.Series(labels_list, name='target')

    stratify_param = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"  ✅ Anomaly Model: Accuracy {accuracy * 100:.2f}%")
    print(f"  📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    return model, accuracy