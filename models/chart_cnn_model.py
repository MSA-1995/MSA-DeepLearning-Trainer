"""
📊 Chart Pattern CNN Model - Training Version
يحلل أنماط الرسوم البيانية باستخدام الميزات المتقدمة
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── core path ─────────────────────────────────────────────────
_models_dir = os.path.dirname(os.path.abspath(__file__))
_core_dir   = os.path.join(_models_dir, '..', 'core')
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from features import calculate_enhanced_features, get_feature_names

# Extra feature names added on top of the 43 base features
_EXTRA_FEATURE_NAMES = [
    'bullish_chart', 'bearish_chart', 'neutral_chart',
    'attention_mechanism_score', 'multi_scale_features', 'temporal_features',
    'tp_accuracy', 'sell_accuracy',
]


def _extract_features(trade, scores):
    """Extract all features from a single trade. Returns list or raises."""
    data = trade.get('data', {})
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        data = {}

    # 43 base features
    features = calculate_enhanced_features(data, trade)

    def _f(d, key, default=0):
        try:
            return float(d.get(key, default) or default)
        except (ValueError, TypeError):
            return float(default)

    rsi  = _f(data, 'rsi',          50)
    macd = _f(data, 'macd',          0)
    vol  = _f(data, 'volume_ratio',  1)

    bullish_chart = 1 if (rsi < 40 and macd > 0 and vol > 1.2) else 0
    bearish_chart = 1 if (rsi > 65 and macd < 0)               else 0
    neutral_chart = 1 if (40 <= rsi <= 60)                      else 0

    attention_mechanism_score = _f(trade, 'attention_mechanism_score', 0)
    multi_scale_features      = _f(trade, 'multi_scale_features',      0)
    temporal_features         = _f(trade, 'temporal_features',         0)

    tp_accuracy   = float(scores.get('tp_accuracy',   0.5))
    sell_accuracy = float(scores.get('sell_accuracy', 0.5))

    features.extend([
        bullish_chart, bearish_chart, neutral_chart,
        attention_mechanism_score, multi_scale_features, temporal_features,
        tp_accuracy, sell_accuracy,
    ])

    return features


def train_chart_cnn_model(trades, voting_scores=None):
    """
    تدريب نموذج تحليل أنماط الرسوم البيانية
    يستخدم نفس الـ 43 ميزة من calculate_enhanced_features + 8 ميزات إضافية
    """
    print("\n📊 Training Chart Pattern CNN Model...")

    if not trades or len(trades) < 50:
        print("  ⚠️ Not enough trades for chart pattern model (need 50+)")
        return None

    scores = (voting_scores or {}).get('chart_cnn', {})

    features_list, labels_list = [], []

    for trade in trades:
        try:
            features = _extract_features(trade, scores)
            profit   = float(trade.get('profit_percent', 0) or 0)

            features_list.append(features)
            labels_list.append(1 if profit >= 1.0 else 0)

        except Exception as e:
            print(f"  ⚠️ Skipping trade: {e}")
            continue

    if len(features_list) < 50:
        print(f"  ⚠️ Only {len(features_list)} valid samples, need 50+")
        return None

    feature_names = get_feature_names() + _EXTRA_FEATURE_NAMES

    X = pd.DataFrame(features_list, columns=feature_names)
    y = pd.Series(labels_list, name='target')

    pos = int(y.sum())
    neg = len(y) - pos
    print(f"    Label balance: {pos} positive ({pos / len(y) * 100:.1f}%) | {neg} negative")

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
    print(f"  ✅ Chart Pattern Model: Accuracy {accuracy * 100:.2f}%")
    print(f"  📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    return model, accuracy