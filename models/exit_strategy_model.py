"""
🎯 Exit Strategy Model - Training Version
يحسن توقيت البيع ويزيد الأرباح
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
    'hours_held', 'profit_optimization_score', 'time_decay_signals',
    'opportunity_cost_exits', 'market_condition_exits',
    'tp_accuracy', 'sell_accuracy',
]


def _extract_features(trade):
    """Extract features from a single trade. Returns list or raises."""
    data = trade.get('data', {})
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        data = {}

    def _f(d, key, default=0):
        try:
            return float(d.get(key, default) or default)
        except (ValueError, TypeError):
            return float(default)

    rsi            = _f(data, 'rsi', 50)
    macd_diff      = _f(data, 'macd_diff', 0)
    volume_ratio   = _f(data, 'volume_ratio', 1.0)
    price_momentum = _f(data, 'price_momentum', 0)
    atr            = _f(data, 'atr', 2.5)

    hours_held                = _f(trade, 'hours_held', 24)
    profit_optimization_score = _f(data,  'profit_optimization_score', 0)
    time_decay_signals        = _f(data,  'time_decay_signals', 0)
    opportunity_cost_exits    = _f(data,  'opportunity_cost_exits', 0)
    market_condition_exits    = _f(data,  'market_condition_exits', 0)

    tp_accuracy   = 0.5
    sell_accuracy = 0.5

    return [
        rsi, macd_diff, volume_ratio, price_momentum, atr,
        hours_held, profit_optimization_score, time_decay_signals,
        opportunity_cost_exits, market_condition_exits,
        tp_accuracy, sell_accuracy,
    ]


def train_exit_strategy_model(trades, voting_scores=None):
    """
    تدريب نموذج استراتيجية الخروج
    يتعلم أفضل توقيت للبيع لتعظيم الأرباح
    """
    print("\n🎯 Training Exit Strategy Model...")

    if not trades or len(trades) < 50:
        print("  ⚠️ Not enough trades for exit model (need 50+)")
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

            is_good_exit = 1 if (profit > 1.0 and trade_quality in ['GREAT', 'GOOD']) else 0

            features_list.append(features)
            labels_list.append(is_good_exit)

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
    print(f"  ✅ Exit Strategy Model: Accuracy {accuracy * 100:.2f}%")
    print(f"  📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    return model, accuracy