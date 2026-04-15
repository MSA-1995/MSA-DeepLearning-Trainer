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
        list of features (42 features total)
    """
    features = []
    
    # تحليل آخر 7 شموع (بدلاً من 3) لاكتشاف أنماط معقدة
    last_7 = candles[-7:] if len(candles) >= 7 else candles
    while len(last_7) < 7:  # Padding
        last_7.insert(0, {'open': 0, 'high': 0, 'low': 0, 'close': 0})
    
    # === 1. ميزات الشموع الفردية (7 شموع × 6 ميزات = 42) ===
    for i, c in enumerate(last_7):
        o = float(c.get('open', 0))
        h = float(c.get('high', 0))
        l = float(c.get('low', 0))
        cl = float(c.get('close', 0))
        
        full_range = (h - l) if (h - l) > 0 else 0.000001
        body_size = abs(cl - o)
        
        # ميزات أساسية
        body_ratio = body_size / full_range
        upper_shadow_ratio = (h - max(o, cl)) / full_range
        lower_shadow_ratio = (min(o, cl) - l) / full_range
        direction = 1 if cl > o else (-1 if cl < o else 0)
        
        # ميزات متقدمة
        is_doji = 1 if body_ratio < 0.1 else 0
        is_long_body = 1 if body_ratio > 0.7 else 0
        
        features.extend([
            body_ratio, upper_shadow_ratio, lower_shadow_ratio, 
            direction, is_doji, is_long_body
        ])
    
    # === 2. ميزات الأنماط المركبة (14 ميزة) ===
    if len(last_7) >= 3:
        c1, c2, c3 = last_7[-3], last_7[-2], last_7[-1]
        
        # حجم الجسم النسبي (مهم للابتلاع Engulfing)
        body1 = abs(float(c1.get('close', 0)) - float(c1.get('open', 0)))
        body2 = abs(float(c2.get('close', 0)) - float(c2.get('open', 0)))
        body3 = abs(float(c3.get('close', 0)) - float(c3.get('open', 0)))
        
        rel_size_2_1 = body2 / (body1 + 0.000001)
        rel_size_3_2 = body3 / (body2 + 0.000001)
        
        # اتجاه متتالي (Three White Soldiers / Three Black Crows)
        dir1 = 1 if float(c1.get('close', 0)) > float(c1.get('open', 0)) else -1
        dir2 = 1 if float(c2.get('close', 0)) > float(c2.get('open', 0)) else -1
        dir3 = 1 if float(c3.get('close', 0)) > float(c3.get('open', 0)) else -1
        
        consecutive_green = 1 if (dir1 == 1 and dir2 == 1 and dir3 == 1) else 0
        consecutive_red = 1 if (dir1 == -1 and dir2 == -1 and dir3 == -1) else 0
        
        # فجوة (Gap)
        gap_2_1 = abs(float(c2.get('open', 0)) - float(c1.get('close', 0))) / (float(c1.get('close', 0)) + 0.000001)
        gap_3_2 = abs(float(c3.get('open', 0)) - float(c2.get('close', 0))) / (float(c2.get('close', 0)) + 0.000001)
        
        # هيمنة الظل (Wick Dominance)
        c3_upper = float(c3.get('high', 0)) - max(float(c3.get('open', 0)), float(c3.get('close', 0)))
        c3_lower = min(float(c3.get('open', 0)), float(c3.get('close', 0))) - float(c3.get('low', 0))
        c3_range = float(c3.get('high', 0)) - float(c3.get('low', 0))
        
        upper_wick_dominance = c3_upper / (c3_range + 0.000001)
        lower_wick_dominance = c3_lower / (c3_range + 0.000001)
        
        # نمط النجمة (Star Pattern)
        is_star = 1 if (body3 < 0.1 * c3_range and gap_3_2 > 0.005) else 0
        
        # اختراق (Breakout)
        high_3 = max(float(c1.get('high', 0)), float(c2.get('high', 0)), float(c3.get('high', 0)))
        low_3 = min(float(c1.get('low', 0)), float(c2.get('low', 0)), float(c3.get('low', 0)))
        breakout_up = 1 if float(c3.get('close', 0)) > high_3 * 1.002 else 0
        breakout_down = 1 if float(c3.get('close', 0)) < low_3 * 0.998 else 0
        
        features.extend([
            rel_size_2_1, rel_size_3_2,
            consecutive_green, consecutive_red,
            gap_2_1, gap_3_2,
            upper_wick_dominance, lower_wick_dominance,
            is_star, breakout_up, breakout_down,
            dir1, dir2, dir3
        ])
    else:
        features.extend([0] * 14)
    
    return features


def get_candle_feature_names():
    """أسماء الميزات (56 ميزة)"""
    names = []
    
    # ميزات الشموع الفردية (7 × 6 = 42)
    for i in range(7, 0, -1):
        names.extend([
            f'c{i}_body_ratio',
            f'c{i}_upper_shadow',
            f'c{i}_lower_shadow',
            f'c{i}_direction',
            f'c{i}_is_doji',
            f'c{i}_is_long_body'
        ])
    
    # ميزات الأنماط المركبة (14)
    names.extend([
        'rel_size_2_1', 'rel_size_3_2',
        'consecutive_green', 'consecutive_red',
        'gap_2_1', 'gap_3_2',
        'upper_wick_dominance', 'lower_wick_dominance',
        'is_star', 'breakout_up', 'breakout_down',
        'dir1', 'dir2', 'dir3'
    ])
    
    return names


def train_candle_expert_model(trades, voting_scores=None):
    """
    تدريب Candle Expert على جميع أنماط الشموع اليابانية
    
    Args:
        trades: list of trade records
        voting_scores: dict with accuracy scores
    
    Returns:
        (model, accuracy) or None
    """
    print("\n🕯️ Training Candle Expert Model (Advanced Pattern Recognition)...")
    
    scores = (voting_scores or {}).get('candle_expert', {})
    
    features_list = []
    labels_list = []
    
    for trade in trades:
        try:
            data = trade.get('data', {})
            if isinstance(data, str):
                data = json.loads(data)
            
            # معالجة البيانات المتداخلة
            if isinstance(data, dict) and 'data' in data and isinstance(data.get('data'), dict):
                data = data['data']
            
            candles = data.get('candles', [])
            if not candles or len(candles) < 3:
                continue
            
            # استخراج الميزات
            candle_features = extract_candle_features(candles)
            
            # إضافة ميزات إضافية من التحليل
            rsi = data.get('rsi', 50)
            volume_ratio = data.get('volume_ratio', 1.0)
            
            # دقة التصويت
            tp_acc = scores.get('tp_accuracy', 0.5)
            sell_acc = scores.get('sell_accuracy', 0.5)
            
            features_list.append(candle_features + [rsi, volume_ratio, tp_acc, sell_acc])
            
            # Label: ربح جيد = 1
            profit = float(trade.get('profit_percent', 0))
            labels_list.append(1 if profit >= 1.0 else 0)
            
        except Exception as e:
            continue
    
    if len(features_list) < 50:
        print(f"  ⚠️ Not enough data for Candle Expert ({len(features_list)} samples)")
        return None
    
    # أسماء الميزات
    feature_names = get_candle_feature_names() + ['rsi', 'volume_ratio', 'tp_accuracy', 'sell_accuracy']
    
    print(f"  📊 Training on {len(features_list)} samples with {len(feature_names)} features")
    
    # تحويل لـ DataFrame
    X = pd.DataFrame(features_list, columns=feature_names)
    y = pd.Series(labels_list, name='target')
    
    # فحص التوزيع
    pos = int(sum(y))
    neg = int(len(y) - pos)
    print(f"    Label balance: {pos} positive ({pos/len(y)*100:.1f}%) | {neg} negative")
    
    # حماية من الفئة الواحدة
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2 or any(counts < 2):
        print(f"    ⚠️ Data too imbalanced... skipping training")
        return None
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # تدريب LightGBM
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
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    # تقييم
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # تقرير مفصل
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    p1 = report.get('1', {}).get('precision', 0)
    r1 = report.get('1', {}).get('recall', 0)
    
    print(f"    ✅ Accuracy: {accuracy*100:.2f}%")
    print(f"    Class-1 → Precision: {p1:.2f} | Recall: {r1:.2f}")
    print(f"    🕯️ Learns: All 14+ Japanese Candlestick Patterns")
    
    return model, accuracy
