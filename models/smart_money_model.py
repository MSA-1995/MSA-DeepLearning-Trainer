"""
🐋 Smart Money Tracker Model - Training Version
يكشف حركة الحيتان والأموال الذكية
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_smart_money_model(trades, voting_scores=None):
    """
    تدريب نموذج تتبع الأموال الذكية
    يتعلم من حركة الحيتان والأموال الذكية
    """
    print("\n🐋 Training Smart Money Tracker Model...")
    
    if not trades or len(trades) < 50:
        print("  ⚠️ Not enough trades for smart money model (need 50+)")
        return None
    
    # Feature names (12 features as per dl_client_v2.py)
    feature_names = [
        # Base features (5)
        'rsi', 'macd_diff', 'volume_ratio', 'price_momentum', 'atr',
        # Smart money features (7)
        'whale_activity', 'exchange_inflow', 'whale_wallet_changes',
        'institutional_accumulation', 'smart_money_ratio', 'exchange_whale_flows',
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
            
            # Base features
            rsi = data.get('rsi', 50)
            macd_diff = data.get('macd_diff', 0)
            volume_ratio = data.get('volume_ratio', 1.0)
            price_momentum = data.get('price_momentum', 0)
            atr = data.get('atr', 2.5)
            
            # Smart money features
            whale_activity = data.get('whale_confidence', 0)
            exchange_inflow = data.get('exchange_inflow', 0)
            whale_wallet_changes = data.get('whale_wallet_changes', 0)
            institutional_accumulation = data.get('institutional_accumulation', 0)
            smart_money_ratio = data.get('smart_money_ratio', 0)
            exchange_whale_flows = data.get('exchange_whale_flows', 0)
            
            # Accuracy metrics
            tp_accuracy = 0.5
            sell_accuracy = 0.5
            
            features = [
                rsi, macd_diff, volume_ratio, price_momentum, atr,
                whale_activity, exchange_inflow, whale_wallet_changes,
                institutional_accumulation, smart_money_ratio, exchange_whale_flows,
                tp_accuracy, sell_accuracy
            ]
            
            # Use only first 12 features to match training
            features = features[:12]
            
            features_list.append(features)
            
            # Label: 1 if smart money was right (profitable), 0 if wrong
            profit = trade.get('profit', trade.get('pnl', 0))
            
            is_smart_money_right = 1 if profit > 0 else 0
            labels_list.append(is_smart_money_right)
            
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
    
    print(f"  ✅ Smart Money Model: Accuracy {accuracy*100:.2f}%")
    print(f"  📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return model, accuracy
