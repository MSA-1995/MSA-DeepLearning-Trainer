"""
📊 Chart Pattern CNN Model - Training Version
يحلل أنماط الرسوم البيانية باستخدام الميزات المتقدمة
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from features import calculate_enhanced_features, get_feature_names


def train_chart_cnn_model(trades, voting_scores=None):
    """
    تدريب نموذج تحليل أنماط الرسوم البيانية
    يستخدم نفس الـ43 ميزة من calculate_enhanced_features
    """
    print("\n📊 Training Chart Pattern CNN Model...")
    
    if not trades or len(trades) < 50:
        print("  ⚠️ Not enough trades for chart pattern model (need 50+)")
        return None
    
    scores = (voting_scores or {}).get('chart_cnn', {})
    
    features_list = []
    labels_list = []
    
    for trade in trades:
        try:
            data = trade.get('data', {})
            if isinstance(data, str):
                import json
                data = json.loads(data)
            
            # استخدام نفس الميزات من features.py
            features = calculate_enhanced_features(data, trade)
            
            # إضافة ميزات إضافية خاصة بالرسوم البيانية
            rsi = data.get('rsi', 50)
            macd = data.get('macd', 0)
            vol = data.get('volume_ratio', 1)
            
            bullish_chart = 1 if (rsi < 40 and macd > 0 and vol > 1.2) else 0
            bearish_chart = 1 if (rsi > 65 and macd < 0) else 0
            neutral_chart = 1 if (40 <= rsi <= 60) else 0
            
            attention_mechanism_score = float(trade.get('attention_mechanism_score', 0))
            multi_scale_features = float(trade.get('multi_scale_features', 0))
            temporal_features = float(trade.get('temporal_features', 0))
            
            # دمج الميزات
            features.extend([
                bullish_chart, bearish_chart, neutral_chart,
                attention_mechanism_score, multi_scale_features, temporal_features,
                scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)
            ])
            
            features_list.append(features)
            
            # Label: 1 if profitable trade
            profit = trade.get('profit_percent', 0)
            labels_list.append(1 if profit >= 1.0 else 0)
            
        except Exception as e:
            continue
    
    if len(features_list) < 50:
        print(f"  ⚠️ Only {len(features_list)} valid samples, need 50+")
        return None
    
    # أسماء الميزات
    feature_names = get_feature_names() + [
        'bullish_chart', 'bearish_chart', 'neutral_chart',
        'attention_mechanism_score', 'multi_scale_features', 'temporal_features',
        'tp_accuracy', 'sell_accuracy'
    ]
    
    X = pd.DataFrame(features_list, columns=feature_names)
    y = pd.Series(labels_list, name='target')
    
    # التحقق من التوزيع
    pos = int(sum(y))
    neg = int(len(y) - pos)
    print(f"    Label balance: {pos} positive ({pos/len(y)*100:.1f}%) | {neg} negative")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train LightGBM model
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
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"  ✅ Chart Pattern Model: Accuracy {accuracy*100:.2f}%")
    print(f"  📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return model, accuracy
