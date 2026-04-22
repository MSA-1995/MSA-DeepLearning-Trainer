"""
🧠 Enhanced Pattern Recognition Model - Training Version
يتعلم من الأنماط بشكل أعمق ويتوقع النجاح
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Feature names (13 features)
FEATURE_NAMES = [
    'rsi', 'macd_diff', 'volume_ratio', 'price_momentum', 'atr',
    'pattern_momentum', 'harmonic_patterns_score', 'elliott_wave_signals',
    'fractal_patterns', 'cycle_patterns', 'momentum_patterns',
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

    rsi            = _f('rsi', 50)
    macd_diff      = _f('macd_diff', 0)
    volume_ratio   = _f('volume_ratio', 1.0)
    price_momentum = _f('price_momentum', 0)
    atr            = _f('atr', 2.5)

    pattern_momentum         = price_momentum
    harmonic_patterns_score  = _f('harmonic_patterns_score', 0)
    elliott_wave_signals     = _f('elliott_wave_signals', 0)
    fractal_patterns         = _f('fractal_patterns', 0)
    cycle_patterns           = _f('cycle_patterns', 0)
    momentum_patterns        = _f('momentum_patterns', 0)

    tp_accuracy   = 0.5
    sell_accuracy = 0.5

    return [
        rsi, macd_diff, volume_ratio, price_momentum, atr,
        pattern_momentum, harmonic_patterns_score, elliott_wave_signals,
        fractal_patterns, cycle_patterns, momentum_patterns,
        tp_accuracy, sell_accuracy,
    ]


def train_pattern_recognition_model(trades, voting_scores=None):
    """
    تدريب نموذج التعرف على الأنماط
    يتعلم من الأنماط الناجحة والفاشلة
    """
    print("\n🧠 Training Pattern Recognition Model...")

    if not trades or len(trades) < 50:
        print("  ⚠️ Not enough trades for pattern model (need 50+)")
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

            is_success = 1 if (profit > 0.5 and trade_quality in ['GREAT', 'GOOD', 'OK']) else 0

            features_list.append(features)
            labels_list.append(is_success)

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
    print(f"  ✅ Pattern Model: Accuracy {accuracy * 100:.2f}%")
    print(f"  📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    return model, accuracy