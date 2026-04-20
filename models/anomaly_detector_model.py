"""
🚨 Anomaly Detection Model - Training Version
يكتشف الحركات الغريبة: Pump & Dump, Flash Crash, Whale Movements
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_anomaly_model(trades, voting_scores=None):
    """
    تدريب نموذج كشف الشذوذات
    يتعلم من الصفقات التاريخية لكشف الأنماط الشاذة
    """
    print("\n🚨 Training Anomaly Detection Model...")
    
    if not trades or len(trades) < 50:
        print("  ⚠️ Not enough trades for anomaly model (need 50+)")
        return None
    
    # Feature names
    feature_names = [
        # Technical indicators
        'rsi', 'macd_diff', 'volume_ratio', 'price_momentum', 'atr',
        # Anomaly-specific features
        'anomaly_score', 'statistical_outliers', 'pattern_anomalies',
        'behavioral_anomalies', 'volume_anomalies',
        # Accuracy metrics
        'tp_accuracy', 'sell_accuracy'
    ]
    
    features_list = []
    labels_list = []
    
    for trade in trades:
        try:
            # Extract features from trade
            data = trade.get('data', {})
            if isinstance(data, str):
                import json
                data = json.loads(data)
            
            # Basic technical features
            rsi = data.get('rsi', 50)
            macd_diff = data.get('macd_diff', 0)
            volume_ratio = data.get('volume_ratio', 1.0)
            price_momentum = data.get('price_momentum', 0)
            atr = data.get('atr', 2.5)
            
            # Anomaly features
            anomaly_score = data.get('anomaly_score', 0)
            statistical_outliers = data.get('statistical_outliers', 0)
            pattern_anomalies = data.get('pattern_anomalies', 0)
            behavioral_anomalies = data.get('behavioral_anomalies', 0)
            volume_anomalies = data.get('volume_anomalies', 0)
            
            # Accuracy metrics
            tp_accuracy = 0.5
            sell_accuracy = 0.5
            
            features = [
                rsi, macd_diff, volume_ratio, price_momentum, atr,
                anomaly_score, statistical_outliers, pattern_anomalies,
                behavioral_anomalies, volume_anomalies,
                tp_accuracy, sell_accuracy
            ]
            
            features_list.append(features)
            
            # Label: 1 if safe trade (no anomaly), 0 if anomaly detected
            profit = trade.get('profit', trade.get('pnl', 0))
            trade_quality = trade.get('trade_quality', 'OK')
            
            # Safe trade = profitable + no trap
            is_safe = 1 if (profit > 0 and trade_quality not in ['TRAP', 'RISKY']) else 0
            labels_list.append(is_safe)
            
        except Exception as e:
            continue
    
    if len(features_list) < 50:
        print(f"  ⚠️ Only {len(features_list)} valid samples, need 50+")
        return None
    
    X = pd.DataFrame(features_list, columns=feature_names)
    y = pd.Series(labels_list, name='target')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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
    
    print(f"  ✅ Anomaly Model: Accuracy {accuracy*100:.2f}%")
    print(f"  📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return model, accuracy
