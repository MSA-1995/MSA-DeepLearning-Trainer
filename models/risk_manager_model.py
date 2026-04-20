"""
🛡️ Risk Manager Model - Training Version
إدارة مخاطر متقدمة مع Kelly Criterion و Sharpe Ratio
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_risk_model(trades, voting_scores=None):
    """
    تدريب نموذج إدارة المخاطر
    يتعلم من المخاطر التاريخية لتحديد أفضل حجم للصفقات
    """
    print("\n🛡️ Training Risk Manager Model...")
    
    if not trades or len(trades) < 50:
        print("  ⚠️ Not enough trades for risk model (need 50+)")
        return None
    
    # Feature names (12 features as per dl_client_v2.py)
    feature_names = [
        # Base features (5)
        'rsi', 'macd_diff', 'volume_ratio', 'price_momentum', 'atr',
        # Risk features (7)
        'risk_rsi', 'risk_atr', 'volatility_risk_score', 'correlation_risk',
        'gap_risk_score', 'black_swan_probability', 'behavioral_risk',
        'systemic_risk', 'tp_accuracy', 'sell_accuracy'
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
            
            # Base features
            rsi = data.get('rsi', 50)
            macd_diff = data.get('macd_diff', 0)
            volume_ratio = data.get('volume_ratio', 1.0)
            price_momentum = data.get('price_momentum', 0)
            atr = data.get('atr', 2.5)
            
            # Risk features
            risk_rsi = rsi
            risk_atr = atr
            volatility_risk_score = data.get('volatility_risk_score', 0)
            correlation_risk = data.get('correlation_risk', 0)
            gap_risk_score = data.get('gap_risk_score', 0)
            black_swan_probability = data.get('black_swan_probability', 0)
            behavioral_risk = data.get('behavioral_risk', 0)
            systemic_risk = data.get('systemic_risk', 0)
            
            # Accuracy metrics
            tp_accuracy = 0.5
            sell_accuracy = 0.5
            
            features = [
                rsi, macd_diff, volume_ratio, price_momentum, atr,
                risk_rsi, risk_atr, volatility_risk_score, correlation_risk,
                gap_risk_score, black_swan_probability, behavioral_risk,
                systemic_risk, tp_accuracy, sell_accuracy
            ]
            
            # Use only first 12 features to match training
            features = features[:12]
            
            features_list.append(features)
            
            # Label: 1 if low risk trade (profitable), 0 if high risk
            profit = trade.get('profit', trade.get('pnl', 0))
            trade_quality = trade.get('trade_quality', 'OK')
            
            is_low_risk = 1 if (profit > 0 and trade_quality in ['GREAT', 'GOOD', 'OK']) else 0
            labels_list.append(is_low_risk)
            
        except Exception as e:
            continue
    
    if len(features_list) < 50:
        print(f"  ⚠️ Only {len(features_list)} valid samples, need 50+")
        return None
    
    X = pd.DataFrame(features_list, columns=feature_names[:12])
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
    
    print(f"  ✅ Risk Model: Accuracy {accuracy*100:.2f}%")
    print(f"  📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return model, accuracy
