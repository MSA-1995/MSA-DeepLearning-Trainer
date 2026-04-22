"""
💧 Liquidity Analyzer Model - Training Version
يحلل عمق السوق والسيولة لاختيار أفضل العملات
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Feature names (25 features)
FEATURE_NAMES = [
    # Basic (10)
    'volume_ratio', 'bid_ask_spread', 'volume_trend',
    'depth_ratio', 'liquidity_score', 'price_impact', 'volume_consistency',
    'good_liquidity', 'low_impact', 'consistent_vol',
    # Advanced (5)
    'spread_volatility', 'depth_at_1pct', 'market_impact_score',
    'liquidity_trends', 'order_book_imbalance',
    # Derived (6)
    'high_depth', 'low_spread_vol', 'balanced_book',
    'liquidity_depth', 'impact_risk', 'volume_liquidity_score',
    # Additional (4)
    'spread_percent', 'spread_impact',
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

    # Basic features
    volume_ratio       = _f('volume_ratio', 1.0)
    bid_ask_spread     = _f('bid_ask_spread', 0)
    volume_consistency = _f('volume_consistency', 50)
    depth_ratio        = _f('depth_ratio', 1.0)
    liquidity_score    = _f('liquidity_score', 50)
    price_impact       = _f('price_impact', 0.5)

    # volume_trend: handle string or numeric
    _vt = data.get('volume_trend', 0)
    if _vt == 'up':
        volume_trend = 1.2
    elif _vt == 'down':
        volume_trend = 0.8
    elif _vt == 'neutral':
        volume_trend = 0.0
    else:
        try:
            volume_trend = float(_vt or 0)
        except (ValueError, TypeError):
            volume_trend = 0.0

    # Derived basic
    good_liquidity = 1 if liquidity_score  > 70  else 0
    low_impact     = 1 if price_impact     < 0.3 else 0
    consistent_vol = 1 if volume_consistency > 60 else 0

    # Advanced features
    spread_volatility    = _f('spread_volatility', 0)
    depth_at_1pct        = _f('depth_at_1pct', 0)
    market_impact_score  = _f('market_impact_score', 0)
    liquidity_trends     = _f('liquidity_trends', 0)
    order_book_imbalance = _f('order_book_imbalance', 0)

    # Derived advanced
    high_depth    = 1 if depth_at_1pct        > 100000 else 0
    low_spread_vol = 1 if spread_volatility   < 0.5    else 0
    balanced_book  = 1 if abs(order_book_imbalance) < 0.2 else 0

    # Composite features
    liquidity_depth        = depth_ratio   * liquidity_score / 100
    impact_risk            = price_impact  * (1 - liquidity_score / 100)
    volume_liquidity_score = volume_ratio  * liquidity_score / 100

    # Additional
    spread_percent = _f('spread_percent', 0.1)
    spread_impact  = spread_percent * price_impact

    tp_accuracy   = 0.5
    sell_accuracy = 0.5

    return [
        # Basic (10)
        volume_ratio, bid_ask_spread, volume_trend,
        depth_ratio, liquidity_score, price_impact, volume_consistency,
        good_liquidity, low_impact, consistent_vol,
        # Advanced (5)
        spread_volatility, depth_at_1pct, market_impact_score,
        liquidity_trends, order_book_imbalance,
        # Derived (6)
        high_depth, low_spread_vol, balanced_book,
        liquidity_depth, impact_risk, volume_liquidity_score,
        # Additional (4)
        spread_percent, spread_impact,
        tp_accuracy, sell_accuracy,
    ]


def train_liquidity_model(trades, voting_scores=None):
    """
    تدريب نموذج تحليل السيولة
    يتعلم من السيولة التاريخية لتحديد أفضل العملات للتداول
    """
    print("\n💧 Training Liquidity Analyzer Model...")

    if not trades or len(trades) < 50:
        print("  ⚠️ Not enough trades for liquidity model (need 50+)")
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

            is_good_liquidity = 1 if (profit > 0 and trade_quality in ['GREAT', 'GOOD', 'OK']) else 0

            features_list.append(features)
            labels_list.append(is_good_liquidity)

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
    print(f"  ✅ Liquidity Model: Accuracy {accuracy * 100:.2f}%")
    print(f"  📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    return model, accuracy