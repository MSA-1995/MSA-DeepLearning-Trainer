"""
🎯 Exit Strategy Model - Training Version
يحسن توقيت البيع ويزيد الأرباح
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_exit_strategy_model(trades, voting_scores=None):
    """
    تدريب نموذج استراتيجية الخروج
    يتعلم أفضل توقيت للبيع لتعظيم الأرباح
    """
    print("\n🎯 Training Exit Strategy Model...")
    
    if not trades or len(trades) < 50:
        print("  ⚠️ Not enough trades for exit model (need 50+)")
        return None
    
    # Feature names
    feature_names = [
        # Technical indicators
        'rsi', 'macd_diff', 'volume_ratio', 'price_momentum', 'atr',
        # Exit-specific features
        'hours_held', 'profit_optimization_score', 'time_decay_signals',
        'opportunity_cost_exits', 'market_condition_exits',
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
            
            # Exit features
            hours_held = trade.get('hours_held', 24)
            profit_optimization_score = data.get('profit_optimization_score', 0)
            time_decay_signals = data.get('time_decay_signals', 0)
            opportunity_cost_exits = data.get('opportunity_cost_exits', 0)
            market_condition_exits = data.get('market_condition_exits', 0)
            
            # Accuracy metrics
            tp_accuracy = 0.5
            sell_accuracy = 0.5
            
            features = [
                rsi, macd_diff, volume_ratio, price_momentum, atr,
                hours_held, profit_optimization_score, time_decay_signals,
                opportunity_cost_exits, market_condition_exits,
                tp_accuracy, sell_accuracy
            ]
            
            features_list.append(features)
            
            # Label: 1 if good exit (profit > 1%), 0 if bad exit
            profit = trade.get('profit', trade.get('pnl', 0))
            trade_quality = trade.get('trade_quality', 'OK')
            
            is_good_exit = 1 if (profit > 1.0 and trade_quality in ['GREAT', 'GOOD']) else 0
            labels_list.append(is_good_exit)
            
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
    
    print(f"  ✅ Exit Strategy Model: Accuracy {accuracy*100:.2f}%")
    print(f"  📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return model, accuracy
