"""
👑 Meta Model - The Intelligent King
Learns decision patterns from all trades independently
Not dependent on other models - standalone decision maker
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_meta_trading(trades, voting_scores=None, since_timestamp=None, db_manager=None):
    """
    Meta Model - learns to make decisions based on market features + symbol memory
    Trains independently on features extracted from trades
    """
    print("\n👑 Training Meta Model (Standalone + Symbol Memory)...")

    # Load symbol memory from DB
    symbol_memory = {}
    if db_manager:
        try:
            symbol_memory = db_manager.load_symbol_memory()
            print(f"  🧠 Loaded memory for {len(symbol_memory)} symbols")
        except Exception as e:
            print(f"  ⚠️ Could not load symbol memory: {e}")

    features_list, labels_list = [], []
    skipped = 0

    for trade in trades:
        try:
            raw_data = trade.get('data', {})
            if isinstance(raw_data, str):
                data = json.loads(raw_data)
            elif isinstance(raw_data, dict):
                data = raw_data
            else:
                data = {}
            
            if 'data' in data and isinstance(data.get('data'), dict):
                data = data['data']

            symbol = str(trade.get('symbol', ''))
            profit = float(trade.get('profit_percent', 0))
            hours_held = float(trade.get('hours_held', 24))

            # ========== TECHNICAL INDICATORS ==========
            rsi = float(data.get('rsi', 50))
            macd = float(data.get('macd', 0))
            volume_ratio = float(data.get('volume_ratio', 1.0))
            price_momentum = float(data.get('price_momentum', 0))
            atr = float(data.get('atr', 0))

            # ========== NEWS DATA ==========
            news = data.get('news', {})
            news_score = float(news.get('news_score', 0))
            news_pos = float(news.get('positive', 0))
            news_neg = float(news.get('negative', 0))
            news_total = float(news.get('total', 0))
            news_ratio = news_pos / (news_neg + 0.001)
            has_news = 1 if news_total > 0 else 0

            # ========== SENTIMENT ==========
            sent = data.get('sentiment', {})
            sent_score = float(sent.get('news_sentiment', 0)) if sent else 0
            fear_greed = float(sent.get('fear_greed_index', 50)) if sent else 50
            fear_greed_norm = (fear_greed - 50) / 50
            is_fearful = 1 if fear_greed < 30 else 0
            is_greedy = 1 if fear_greed > 70 else 0

            # ========== LIQUIDITY ==========
            liq = data.get('liquidity', {})
            liq_score = float(liq.get('liquidity_score', 50)) if liq else 50
            depth_ratio = float(liq.get('depth_ratio', 1.0)) if liq else 1.0
            price_impact = float(liq.get('price_impact', 0.5)) if liq else 0.5
            good_liq = 1 if liq_score > 70 else 0

            # ========== MARKET CONDITIONS ==========
            whale_activity = float(data.get('whale_activity', 0))
            exchange_inflow = float(data.get('exchange_inflow', 0))
            social_volume = float(data.get('social_volume', 0))
            market_sentiment = float(data.get('market_sentiment', 0))

            # ========== CONSULTANT VOTES ==========
            buy_votes = data.get('buy_votes', {})
            sell_votes = data.get('sell_votes', {})
            buy_count = sum(1 for v in buy_votes.values() if v == 1) if buy_votes else 0
            sell_count = sum(1 for v in sell_votes.values() if v == 1) if sell_votes else 0
            consensus = buy_count / 7.0

            # ========== SYMBOL MEMORY ==========
            mem = symbol_memory.get(symbol, {})
            sym_win_rate = float(mem.get('win_count', 0)) / max(float(mem.get('total_trades', 1)), 1)
            sym_avg_profit = float(mem.get('avg_profit', 0))
            sym_trap_count = float(mem.get('trap_count', 0))
            sym_total = float(mem.get('total_trades', 0))
            sym_is_reliable = 1 if (sym_win_rate > 0.6 and sym_total > 5) else 0
            
            # الأعمدة الجديدة - تحويل إلى float
            sym_sentiment_avg = float(mem.get('sentiment_avg', 0) or 0)
            sym_whale_avg = float(mem.get('whale_confidence_avg', 0) or 0)
            sym_profit_loss_ratio = float(mem.get('profit_loss_ratio', 1.0) or 1.0)
            sym_volume_trend = float(mem.get('volume_trend', 1.0) or 1.0)
            sym_panic_avg = float(mem.get('panic_score_avg', 0) or 0)
            sym_optimism_avg = float(mem.get('optimism_penalty_avg', 0) or 0)
            
            # الـ 4 أعمدة الجديدة - تحويل إلى float
            sym_courage_boost = float(mem.get('courage_boost', 0) or 0)
            sym_time_memory = float(mem.get('time_memory_modifier', 0) or 0)
            sym_pattern_score = float(mem.get('pattern_score', 0) or 0)
            sym_win_rate_boost = float(mem.get('win_rate_boost', 0) or 0)

            # ========== DERIVED FEATURES ==========
            risk_score = (whale_activity * 0.1 + (1 - liq_score/100) * 20 + news_neg * 5)
            opportunity = ((1 if rsi < 30 else 0) * 20 + news_pos * 5 + good_liq * 10 + buy_count * 5)
            market_quality = (liq_score / 100 + np.abs(news_ratio) / 10 + consensus) / 3
            momentum_strength = np.abs(price_momentum)
            volatility_level = float(data.get('volatility', 0))

            # BUILD FEATURE VECTOR
            features = [
                # Technical
                rsi, macd, volume_ratio, price_momentum, atr,
                # News
                news_score, news_pos, news_neg, news_total, news_ratio, has_news,
                # Sentiment
                sent_score, fear_greed, fear_greed_norm, is_fearful, is_greedy,
                # Liquidity
                liq_score, depth_ratio, price_impact, good_liq,
                # Smart Money
                whale_activity, exchange_inflow,
                # Social
                social_volume, market_sentiment,
                # Consultants
                consensus, buy_count, sell_count,
                # Derived
                risk_score, opportunity, market_quality,
                momentum_strength, volatility_level,
                # Symbol Memory - Basic
                sym_win_rate, sym_avg_profit, sym_trap_count, sym_total, sym_is_reliable,
                # Symbol Memory - New 7 columns
                sym_sentiment_avg, sym_whale_avg, sym_profit_loss_ratio, sym_volume_trend,
                sym_panic_avg, sym_optimism_avg,
                # Symbol Memory - New 4 columns  
                sym_courage_boost, sym_time_memory, sym_pattern_score, sym_win_rate_boost,
                # Context
                hours_held
            ]

            features_list.append(features)
            # Label: profitable trade
            labels_list.append(1 if profit > 0.5 else 0)

        except Exception as e:
            skipped += 1
            if skipped == 1:
                print(f"  ⚠️ First sample error: {e}")
            continue

    if len(features_list) < 100:
        print(f"  ⚠️ Not enough data for Meta Model ({len(features_list)} < 100)")
        return None

    print(f"  📊 Collected {len(features_list)} samples (skipped {skipped})")

    # Feature names
    feature_names = [
        # Technical
        'rsi', 'macd', 'volume_ratio', 'price_momentum', 'atr',
        # News
        'news_score', 'news_pos', 'news_neg', 'news_total', 'news_ratio', 'has_news',
        # Sentiment
        'sent_score', 'fear_greed', 'fear_greed_norm', 'is_fearful', 'is_greedy',
        # Liquidity
        'liq_score', 'depth_ratio', 'price_impact', 'good_liq',
        # Smart Money
        'whale_activity', 'exchange_inflow',
        # Social
        'social_volume', 'market_sentiment',
        # Consultants
        'consensus', 'buy_count', 'sell_count',
        # Derived
        'risk_score', 'opportunity', 'market_quality',
        'momentum_strength', 'volatility_level',
        # Symbol Memory - Basic
        'sym_win_rate', 'sym_avg_profit', 'sym_trap_count', 'sym_total', 'sym_is_reliable',
        # Symbol Memory - New 7
        'sym_sentiment_avg', 'sym_whale_avg', 'sym_profit_loss_ratio', 'sym_volume_trend',
        'sym_panic_avg', 'sym_optimism_avg',
        # Symbol Memory - New 4
        'sym_courage_boost', 'sym_time_memory', 'sym_pattern_score', 'sym_win_rate_boost',
        # Context
        'hours_held'
    ]

    X = pd.DataFrame(features_list, columns=feature_names)
    y = pd.Series(labels_list, name='target')

    # Class balance info
    pos = int(sum(y))
    neg = int(len(y) - pos)
    ratio = neg / max(pos, 1)
    print(f"  📈 Label balance: {pos} positive ({pos/len(y)*100:.1f}%) | {neg} negative | ratio={ratio:.1f}x")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train LightGBM model
    model = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=0.4,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    # Classification report
    from sklearn.metrics import classification_report
    report = classification_report(y_test, model.predict(X_test), output_dict=True, zero_division=0)
    p1 = report.get('1', {}).get('precision', 0)
    r1 = report.get('1', {}).get('recall', 0)
    print(f"  📊 Class-1 → Precision: {p1:.2f} | Recall: {r1:.2f}")

    print(f"👑 Meta Model: Accuracy {accuracy*100:.2f}% (with Symbol Memory)")
    return model, accuracy
