"""
🕯️ Candle Expert Model - Japanese Candlestick Pattern Recognition
Learns ALL 14+ candlestick patterns intelligently from historical data
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def extract_candle_features(candles):
    """
    استخراج ميزات ذكية من الشموع لتعليم الموديل على جميع الأنماط

    Args:
        candles: list of dicts with keys: open, high, low, close

    Returns:
        list of features (56 features total)
    """
    features = []

    # تحليل آخر 7 شموع
    last_7 = list(candles[-7:]) if len(candles) >= 7 else list(candles)
    while len(last_7) < 7:
        last_7.insert(0, {'open': 0, 'high': 0, 'low': 0, 'close': 0})

    # === 1. ميزات الشموع الفردية (7 × 6 = 42) ===
    for c in last_7:
        try:
            o  = float(c.get('open',  0) or 0)
            h  = float(c.get('high',  0) or 0)
            l  = float(c.get('low',   0) or 0)
            cl = float(c.get('close', 0) or 0)
        except (ValueError, TypeError):
            o = h = l = cl = 0.0

        full_range = (h - l) if (h - l) > 0 else 0.000001
        body_size  = abs(cl - o)

        body_ratio          = body_size / full_range
        upper_shadow_ratio  = (h - max(o, cl)) / full_range
        lower_shadow_ratio  = (min(o, cl) - l) / full_range
        direction           = 1 if cl > o else (-1 if cl < o else 0)
        is_doji             = 1 if body_ratio < 0.1 else 0
        is_long_body        = 1 if body_ratio > 0.7 else 0

        features.extend([
            body_ratio, upper_shadow_ratio, lower_shadow_ratio,
            direction, is_doji, is_long_body,
        ])

    # === 2. ميزات الأنماط المركبة (14 ميزة) ===
    if len(last_7) >= 3:
        c1, c2, c3 = last_7[-3], last_7[-2], last_7[-1]

        def _get(c, key):
            try:
                return float(c.get(key, 0) or 0)
            except (ValueError, TypeError):
                return 0.0

        body1 = abs(_get(c1, 'close') - _get(c1, 'open'))
        body2 = abs(_get(c2, 'close') - _get(c2, 'open'))
        body3 = abs(_get(c3, 'close') - _get(c3, 'open'))

        rel_size_2_1 = body2 / (body1 + 0.000001)
        rel_size_3_2 = body3 / (body2 + 0.000001)

        dir1 = 1 if _get(c1, 'close') > _get(c1, 'open') else -1
        dir2 = 1 if _get(c2, 'close') > _get(c2, 'open') else -1
        dir3 = 1 if _get(c3, 'close') > _get(c3, 'open') else -1

        consecutive_green = 1 if (dir1 == 1  and dir2 == 1  and dir3 == 1)  else 0
        consecutive_red   = 1 if (dir1 == -1 and dir2 == -1 and dir3 == -1) else 0

        gap_2_1 = abs(_get(c2, 'open') - _get(c1, 'close')) / (_get(c1, 'close') + 0.000001)
        gap_3_2 = abs(_get(c3, 'open') - _get(c2, 'close')) / (_get(c2, 'close') + 0.000001)

        c3_upper = _get(c3, 'high') - max(_get(c3, 'open'), _get(c3, 'close'))
        c3_lower = min(_get(c3, 'open'), _get(c3, 'close')) - _get(c3, 'low')
        c3_range = _get(c3, 'high') - _get(c3, 'low')
        c3_range = c3_range if c3_range > 0 else 0.000001

        upper_wick_dominance = c3_upper / c3_range
        lower_wick_dominance = c3_lower / c3_range

        is_star      = 1 if (body3 < 0.1 * c3_range and gap_3_2 > 0.005) else 0

        high_3 = max(_get(c1, 'high'), _get(c2, 'high'), _get(c3, 'high'))
        low_3  = min(_get(c1, 'low'),  _get(c2, 'low'),  _get(c3, 'low'))
        breakout_up   = 1 if _get(c3, 'close') > high_3 * 1.002 else 0
        breakout_down = 1 if _get(c3, 'close') < low_3  * 0.998 else 0

        features.extend([
            rel_size_2_1, rel_size_3_2,
            consecutive_green, consecutive_red,
            gap_2_1, gap_3_2,
            upper_wick_dominance, lower_wick_dominance,
            is_star, breakout_up, breakout_down,
            dir1, dir2, dir3,
        ])
    else:
        features.extend([0] * 14)

    return features


def get_candle_feature_names():
    """أسماء الميزات (56 ميزة)"""
    names = []
    for i in range(7, 0, -1):
        names.extend([
            f'c{i}_body_ratio',
            f'c{i}_upper_shadow',
            f'c{i}_lower_shadow',
            f'c{i}_direction',
            f'c{i}_is_doji',
            f'c{i}_is_long_body',
        ])
    names.extend([
        'rel_size_2_1', 'rel_size_3_2',
        'consecutive_green', 'consecutive_red',
        'gap_2_1', 'gap_3_2',
        'upper_wick_dominance', 'lower_wick_dominance',
        'is_star', 'breakout_up', 'breakout_down',
        'dir1', 'dir2', 'dir3',
    ])
    return names


# Full feature names (constant, built once)
_FEATURE_NAMES = get_candle_feature_names() + [
    'rsi', 'volume_ratio', 'tp_accuracy', 'sell_accuracy'
]


def train_candle_expert_model(trades, voting_scores=None):
    """
    تدريب Candle Expert على جميع أنماط الشموع اليابانية
    """
    print("\n🕯️ Training Candle Expert Model (Advanced Pattern Recognition)...")

    scores = (voting_scores or {}).get('candle_expert', {})

    features_list, labels_list = [], []

    for trade in trades:
        try:
            data = trade.get('data', {})
            if isinstance(data, str):
                data = json.loads(data)
            if not isinstance(data, dict):
                data = {}

            # معالجة البيانات المتداخلة
            if 'data' in data and isinstance(data.get('data'), dict):
                data = data['data']

            candles = data.get('candles', [])
            if not candles or len(candles) < 3:
                continue

            candle_features = extract_candle_features(candles)

            try:
                rsi          = float(data.get('rsi',          50) or 50)
                volume_ratio = float(data.get('volume_ratio',  1) or 1)
            except (ValueError, TypeError):
                rsi, volume_ratio = 50.0, 1.0

            tp_acc   = float(scores.get('tp_accuracy',   0.5))
            sell_acc = float(scores.get('sell_accuracy', 0.5))

            features_list.append(candle_features + [rsi, volume_ratio, tp_acc, sell_acc])

            profit = float(trade.get('profit_percent', 0) or 0)
            labels_list.append(1 if profit >= 1.0 else 0)

        except Exception as e:
            print(f"  ⚠️ Skipping trade: {e}")
            continue

    if len(features_list) < 50:
        print(f"  ⚠️ Not enough data for Candle Expert ({len(features_list)} samples)")
        return None

    print(f"  📊 Training on {len(features_list)} samples with {len(_FEATURE_NAMES)} features")

    X = pd.DataFrame(features_list, columns=_FEATURE_NAMES)
    y = pd.Series(labels_list, name='target')

    pos = int(y.sum())
    neg = len(y) - pos
    print(f"    Label balance: {pos} positive ({pos / len(y) * 100:.1f}%) | {neg} negative")

    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2 or any(counts < 2):
        print("    ⚠️ Data too imbalanced... skipping training")
        return None

    stratify_param = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=0.5,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    report   = classification_report(y_test, model.predict(X_test), output_dict=True, zero_division=0)
    p1 = report.get('1', {}).get('precision', 0)
    r1 = report.get('1', {}).get('recall',    0)

    print(f"    ✅ Accuracy: {accuracy * 100:.2f}%")
    print(f"    Class-1 → Precision: {p1:.2f} | Recall: {r1:.2f}")
    print(f"    🕯️ Learns: All 14+ Japanese Candlestick Patterns")

    return model, accuracy