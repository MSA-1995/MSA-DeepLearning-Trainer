"""
💧 Liquidity Analyzer Model - Training Version
يحلل عمق السوق والسيولة لاختيار أفضل العملات
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_liquidity_model(trades, voting_scores=None):
    """
    تدريب نموذج تحليل السيولة
    يتعلم من السيولة التاريخية لتحديد أفضل العملات للتداول
    """
    print("\n💧 Training Liquidity Analyzer Model...")
    
    if not trades or len(trades) < 50:
        print("  ⚠️ Not enough trades for liquidity model (need 50+)")
        return None
    
    # Feature names (26 features as per dl_client_v2.py)
    feature_names = [
        # Basic features (10)
        'volume_ratio', 'bid_ask_spread', 'volume_trend',
        'depth_ratio', 'liquidity_score', 'price_impact', 'volume_consistency',
        'good_liquidity', 'low_impact', 'consistent_vol',
        # Advanced features (5)
        'spread_volatility', 'depth_at_1pct', 'market_impact_score',
        'liquidity_trends', 'order_book_imbalance',
        # Derived features (6)
        'high_depth', 'low_spread_vol', 'balanced_book',
        'liquidity_depth', 'impact_risk', 'volume_liquidity_score',
        # Additional features (3)
        'spread_percent', 'spread_impact',
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
            
            # Basic features
            volume_ratio = data.get('volume_ratio', 1.0)
            bid_ask_spread = data.get('bid_ask_spread', 0)
            volume_trend = data.get('volume_trend', 0)
            depth_ratio = data.get('depth_ratio', 1.0)
            liquidity_score = data.get('liquidity_score', 50)
            price_impact = data.get('price_impact', 0.5)
            volume_consistency = data.get('volume_consistency', 50)
            
            # Derived basic
            good_liquidity = 1 if liquidity_score > 70 else 0
            low_impact = 1 if price_impact < 0.3 else 0
            consistent_vol = 1 if volume_consistency > 60 else 0
            
            # Advanced features
            spread_volatility = data.get('spread_volatility', 0)
            depth_at_1pct = data.get('depth_at_1pct', 0)
            market_impact_score = data.get('market_impact_score', 0)
            liquidity_trends = data.get('liquidity_trends', 0)
            order_book_imbalance = data.get('order_book_imbalance', 0)
            
            # Derived advanced
            high_depth = 1 if depth_at_1pct > 100000 else 0
            low_spread_vol = 1 if spread_volatility < 0.5 else 0
            balanced_book = 1 if abs(order_book_imbalance) < 0.2 else 0
            
            # Composite features
            liquidity_depth = depth_ratio * liquidity_score / 100
            impact_risk = price_impact * (1 - liquidity_score/100)
            volume_liquidity_score = volume_ratio * liquidity_score / 100
            
            # Additional
            spread_percent = data.get('spread_percent', 0.1)
            spread_impact = spread_percent * price_impact
            
            # Accuracy metrics
            tp_accuracy = 0.5
            sell_accuracy = 0.5
            
            features = [
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
                # Additional (3)
                spread_percent, spread_impact,
                tp_accuracy, sell_accuracy
            ]
            
            features_list.append(features)
            
            # Label: 1 if good liquidity trade, 0 if poor liquidity
            profit = trade.get('profit', trade.get('pnl', 0))
            trade_quality = trade.get('trade_quality', 'OK')
            
            is_good_liquidity = 1 if (profit > 0 and trade_quality in ['GREAT', 'GOOD', 'OK']) else 0
            labels_list.append(is_good_liquidity)
            
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
    
    print(f"  ✅ Liquidity Model: Accuracy {accuracy*100:.2f}%")
    print(f"  📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return model, accuracy
