"""
🧠 Training Models - LightGBM Models with Full Feature Engineering
Each model thinks and learns according to its specific role.
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from features import calculate_enhanced_features, get_feature_names


# ========== Helper ==========

def _build_dataset(trades, feature_fn, label_fn):
    features_list, labels_list = [], []
    for trade in trades:
        try:
            data = trade.get('data', {})
            if isinstance(data, str):
                data = json.loads(data)
            features_list.append(feature_fn(data, trade))
            labels_list.append(label_fn(trade))
        except:
            continue
    return features_list, labels_list


def _train_lgb(X, y, feature_names, n_estimators=300, max_depth=7, learning_rate=0.05):
    X_df = pd.DataFrame(X, columns=feature_names)
    y_s  = pd.Series(y, name='target')
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, test_size=0.2, random_state=42, stratify=y_s)
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        num_leaves=63, min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)
    return model, accuracy_score(y_test, model.predict(X_test))


# Base feature names (38 features from calculate_enhanced_features)
BASE_NAMES = get_feature_names()


# ========== Meta-Learner ==========

def train_meta_learner_model(db_manager, trained_models=None, voting_scores=None, since_timestamp=None):
    """Meta-Learner - learns from NEW trades only (like other advisors)"""
    print("\n👑🧠 Training Meta-Learner Model...")

    print("Loading auxiliary data...")
    try:
        trap_memory    = db_manager.load_traps(limit=10000)
        trap_memory_df = pd.DataFrame(trap_memory)
        symbol_memory  = db_manager.load_symbol_memory()
        causal_data    = db_manager.load_causal_data(limit=10000)
        causal_df      = pd.DataFrame(causal_data) if causal_data else pd.DataFrame()
        total_trades   = db_manager.get_total_trades_count()
        if total_trades == 0:
            print("⚠️ No trades found.")
            return None
        print(f"📈 Found {total_trades} total trades. Processing all for the King.")
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

    consultant_models = {k: v for k, v in trained_models.items() if k != 'meta_learner' and v is not None}
    trap_counts = {}
    if not trap_memory_df.empty and 'symbol' in trap_memory_df.columns:
        trap_counts = trap_memory_df['symbol'].value_counts().to_dict()

    meta_features, final_labels = [], []

    # Load trades - NEW only if since_timestamp provided
    try:
        if since_timestamp:
            print(f"  -> Processing NEW trades since {since_timestamp}...")
            trades_batch = db_manager.load_training_data(since_timestamp=since_timestamp)
        else:
            print(f"  -> Processing ALL {total_trades} trades (first training)...")
            trades_batch = db_manager.load_training_data(limit=total_trades)
    except Exception as e:
        print(f"❌ Error loading trades: {e}")
        return None
    
    if not trades_batch:
        print("⚠️ No trades found for training")
        return None

    for trade in [dict(t) for t in trades_batch]:
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
            
            base = calculate_enhanced_features(data, trade)
            rsi = data.get('rsi', 50)
            is_oversold = 1 if rsi < 30 else 0
            tq = trade.get('trade_quality', 'OK')
            is_great = 1 if tq == 'GREAT' else 0
            is_trap = 1 if tq in ['TRAP', 'RISKY'] else 0
            hours_held = float(trade.get('hours_held', 24))
            profit = float(trade.get('profit_percent', 0))
            is_profit = 1 if profit > 0 else 0
            
            news = data.get('news', {})
            news_score = news.get('news_score', 0)
            news_pos = news.get('positive', 0)
            news_neg = news.get('negative', 0)
            news_total = news.get('total', 0)
            news_ratio = news_pos / (news_neg + 0.001)
            has_news = 1 if news_total > 0 else 0
            
            sent = data.get('sentiment', {})
            sent_score = sent.get('news_sentiment', 0)
            
            liq = data.get('liquidity', {})
            liq_score = liq.get('liquidity_score', 50)
            depth_ratio = liq.get('depth_ratio', 1.0)
            price_impact = liq.get('price_impact', 0.5)
            good_liq = 1 if liq_score > 70 else 0
            
            features = base + [is_great, is_trap, hours_held, is_profit,
                               news_score, news_pos, news_neg, news_total, news_ratio, has_news, sent_score,
                               liq_score, depth_ratio, price_impact, good_liq]
            meta_features.append(features)
            final_labels.append(1 if profit > 0.5 else 0)
        except:
            continue

    if len(meta_features) < 100:
        print(f"⚠️ Not enough data ({len(meta_features)} samples)")
        return None

    print(f"\nCollected {len(meta_features)} samples. Now training the King...")

    names = (
        BASE_NAMES +
        ['is_great', 'is_trap', 'hours_held', 'is_profit',
         'news_score', 'news_pos', 'news_neg', 'news_total', 'news_ratio', 'has_news', 'sent_score',
         'liq_score', 'depth_ratio', 'price_impact', 'good_liq',
         'trap_count', 'was_trapped', 'repeat_trap',
         'sym_win_rate', 'sym_avg_profit', 'sym_trap_count', 'sym_total', 'sym_is_reliable',
         'fear_greed', 'whale_activity', 'exchange_inflow', 'social_volume', 'market_sent',
         'fear_greed_norm', 'is_fearful', 'is_greedy',
         'consensus', 'buy_count', 'sell_count',
         'risk_score', 'opportunity', 'market_quality'] +
        [f'consultant_{n}' for n in consultant_models.keys()]
    )

    model, accuracy = _train_lgb(meta_features, np.array(final_labels), names, n_estimators=300, max_depth=6, learning_rate=0.03)
    print(f"👑🧠 Meta-Learner Model: Accuracy {accuracy*100:.2f}%")
    return model, accuracy


# ========== Models ==========

def train_smart_money_model(trades, voting_scores=None):
    """🐋 Smart Money - ي discovers smart money inflows"""
    print("\n🐋 Training Smart Money Model (LightGBM)...")
    scores = (voting_scores or {}).get('smart_money', {})

    def features(data, trade):
        base = calculate_enhanced_features(data, trade)
        # Feature Engineering خاص بالأموال الذكية
        vol   = data.get('volume_ratio', 1)
        macd  = data.get('macd', 0)
        rsi   = data.get('rsi', 50)
        smart_entry    = 1 if (vol > 1.5 and macd > 0 and rsi < 45) else 0
        accumulation   = 1 if (vol > 1.2 and data.get('price_momentum', 0) > 0) else 0
        whale_signal   = 1 if vol > 3.0 else 0
        return base + [
            data.get('confidence', 60),
            smart_entry, accumulation, whale_signal,
            scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)
        ]

    names = BASE_NAMES + ['confidence', 'smart_entry', 'accumulation', 'whale_signal', 'tp_accuracy', 'sell_accuracy']

    def label(t):
        return 1 if t.get('trade_quality') == 'GREAT' or float(t.get('profit_percent', 0)) > 1.5 else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("⚠️ Not enough data for Smart Money")
        return None
    model, acc = _train_lgb(fl, np.array(ll), names)
    print(f"🐋 Smart Money Model: Accuracy {acc*100:.2f}% | Learns: GREAT trades")
    return model, acc


def train_risk_model(trades, voting_scores=None):
    """🛡️ Risk Manager - discovers traps and risks"""
    print("\n🛡️ Training Risk Model (LightGBM)...")
    scores = (voting_scores or {}).get('risk', {})

    def features(data, trade):
        base = calculate_enhanced_features(data, trade)
        rsi  = data.get('rsi', 50)
        vol  = data.get('volume_ratio', 1)
        conf = data.get('confidence', 60)
        # Feature Engineering للمخاطر
        high_risk      = 1 if (rsi > 70 and vol > 2.0) else 0
        low_confidence = 1 if conf < 60 else 0
        pump_signal    = 1 if (vol > 3.0 and rsi > 65) else 0
        trap_pattern   = 1 if (vol > 2.0 and data.get('price_momentum', 0) < -2) else 0
        return base + [
            conf, high_risk, low_confidence, pump_signal, trap_pattern,
            scores.get('sl_accuracy', 0.5), scores.get('sell_accuracy', 0.5)
        ]

    names = BASE_NAMES + ['confidence', 'high_risk', 'low_confidence', 'pump_signal', 'trap_pattern', 'sl_accuracy', 'sell_accuracy']

    def label(t):
        return 1 if t.get('trade_quality') in ['TRAP', 'RISKY'] else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("⚠️ Not enough data for Risk")
        return None
    model, acc = _train_lgb(fl, np.array(ll), names)
    print(f"🛡️ Risk Model: Accuracy {acc*100:.2f}% | Learns: TRAP/RISKY trades")
    return model, acc


def train_anomaly_model(trades, voting_scores=None):
    """🚨 Anomaly Detector - discovers anomalies"""
    print("\n🚨 Training Anomaly Model (LightGBM)...")
    scores = (voting_scores or {}).get('anomaly', {})

    def features(data, trade):
        base = calculate_enhanced_features(data, trade)
        rsi  = data.get('rsi', 50)
        vol  = data.get('volume_ratio', 1)
        mom  = data.get('price_momentum', 0)
        # Feature Engineering للشذوذ
        flash_crash  = 1 if (vol > 4.0 and mom < -5) else 0
        whale_dump   = 1 if (vol > 5.0 and rsi > 70) else 0
        return base + [
            flash_crash, whale_dump,
            scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)
        ]

    names = BASE_NAMES + ['flash_crash', 'whale_dump', 'tp_accuracy', 'sell_accuracy']

    def label(t):
        return 1 if t.get('trade_quality') == 'RISKY' or float(t.get('profit_percent', 0)) < -0.5 else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("⚠️ Not enough data for Anomaly")
        return None
    model, acc = _train_lgb(fl, np.array(ll), names)
    print(f"🚨 Anomaly Model: Accuracy {acc*100:.2f}% | Learns: RISKY trades")
    return model, acc


def train_exit_model(trades, voting_scores=None):
    """🎯 Exit Strategy - learns optimal exit timing"""
    print("\n🎯 Training Exit Model (LightGBM)...")
    scores = (voting_scores or {}).get('exit', {})

    def features(data, trade):
        base = calculate_enhanced_features(data, trade)
        rsi  = data.get('rsi', 50)
        macd = data.get('macd', 0)
        conf = data.get('confidence', 60)
        # Feature Engineering للخروج
        peak_signal    = 1 if (rsi > 65 and macd < 0) else 0
        good_exit      = 1 if (rsi > 60 and data.get('volume_trend', 0) < 0) else 0
        early_exit     = 1 if (rsi < 50 and conf < 60) else 0
        return base + [
            conf, peak_signal, good_exit, early_exit,
            scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)
        ]

    names = BASE_NAMES + ['confidence', 'peak_signal', 'good_exit', 'early_exit', 'tp_accuracy', 'sell_accuracy']

    def label(t):
        return 1 if t.get('trade_quality') in ['GREAT', 'GOOD'] else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("⚠️ Not enough data for Exit")
        return None
    model, acc = _train_lgb(fl, np.array(ll), names)
    print(f"🎯 Exit Model: Accuracy {acc*100:.2f}% | Learns: GOOD/GREAT exits")
    return model, acc


def train_pattern_model(trades, voting_scores=None):
    """🧠 Pattern Recognition - learns successful patterns"""
    print("\n🧠 Training Pattern Model (LightGBM)...")
    scores = (voting_scores or {}).get('pattern', {})

    def features(data, trade):
        base = calculate_enhanced_features(data, trade)
        rsi  = data.get('rsi', 50)
        macd = data.get('macd', 0)
        vol  = data.get('volume_ratio', 1)
        conf = data.get('confidence', 60)
        # Feature Engineering للأنماط
        reversal_pattern = 1 if (rsi < 35 and macd > 0 and vol > 1.3) else 0
        breakout_pattern = 1 if (vol > 2.0 and macd > 0 and rsi > 50) else 0
        hammer_like      = 1 if (rsi < 30 and vol > 1.5) else 0
        return base + [
            conf, reversal_pattern, breakout_pattern, hammer_like,
            scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)
        ]

    names = BASE_NAMES + ['confidence', 'reversal_pattern', 'breakout_pattern', 'hammer_like', 'tp_accuracy', 'sell_accuracy']

    def label(t):
        return 1 if t.get('trade_quality') == 'GREAT' else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("⚠️ Not enough data for Pattern")
        return None
    model, acc = _train_lgb(fl, np.array(ll), names)
    print(f"🧠 Pattern Model: Accuracy {acc*100:.2f}% | Learns: GREAT patterns")
    return model, acc


def train_liquidity_model(trades, voting_scores=None):
    """💧 Liquidity Analyzer - learns good liquidity patterns from each trade"""
    print("\n💧 Training Liquidity Model (LightGBM)...")
    scores = (voting_scores or {}).get('liquidity', {})

    features_list, labels_list = [], []
    skipped_no_data = 0
    
    for trade in trades:
        try:
            data = trade.get('data', {})
            if isinstance(data, str):
                data = json.loads(data)
            
            liq = data.get('liquidity', {})
            liquidity_score = liq.get('liquidity_score', 50)
            
            # تخطي الصفقات بدون بيانات سيولة حقيقية (الافتراضي 50 يعني ما فيه بيانات)
            if liquidity_score == 50 and not liq:
                skipped_no_data += 1
                continue
            
            profit = float(trade.get('profit_percent', 0))
            
            # كل صفقة هي عنصر تدريب واحد
            volume_ratio = data.get('volume_ratio', 1.0)
            bid_ask_spread = data.get('bid_ask_spread', 0)
            volume_trend = data.get('volume_trend', 0)
            depth_ratio = liq.get('depth_ratio', 1.0)
            price_impact = liq.get('price_impact', 0.5)
            volume_consistency = liq.get('volume_consistency', 50)
            
            # Feature Engineering
            good_liquidity = 1 if liquidity_score > 70 else 0
            low_impact = 1 if price_impact < 0.3 else 0
            consistent_vol = 1 if volume_consistency > 60 else 0
            
            features_list.append([
                profit,
                volume_ratio, bid_ask_spread, volume_trend,
                depth_ratio, liquidity_score, price_impact, volume_consistency,
                good_liquidity, low_impact, consistent_vol,
                scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)
            ])
            
            # التصنيف: صفقة ناجحة = ربح > 0%
            labels_list.append(1 if profit > 0 else 0)
            
        except:
            continue
    
    names = [
        'profit',
        'volume_ratio', 'bid_ask_spread', 'volume_trend',
        'depth_ratio', 'liquidity_score', 'price_impact', 'volume_consistency',
        'good_liquidity', 'low_impact', 'consistent_vol',
        'tp_accuracy', 'sell_accuracy'
    ]
    
    print(f"  📊 Training samples: {len(features_list)} trades (skipped {skipped_no_data} without liquidity data)")
    
    if len(features_list) < 50:
        print("⚠️ Not enough trades with liquidity data for training")
        return None
    
    model, acc = _train_lgb(features_list, np.array(labels_list), names, n_estimators=300, max_depth=6, learning_rate=0.05)
    print(f"💧 Liquidity Model: Accuracy {acc*100:.2f}% | Learns from {len(features_list)} trades")
    return model, acc


def train_chart_cnn_model(trades, voting_scores=None):
    """📊 Chart Pattern Analyzer - learns chart patterns"""
    print("\n📊 Training Chart Pattern Model (LightGBM)...")
    scores = (voting_scores or {}).get('cnn', {})

    def features(data, trade):
        base = calculate_enhanced_features(data, trade)
        rsi  = data.get('rsi', 50)
        macd = data.get('macd', 0)
        vol  = data.get('volume_ratio', 1)
        # Feature Engineering للشارت
        bullish_chart  = 1 if (rsi < 40 and macd > 0 and vol > 1.2) else 0
        bearish_chart  = 1 if (rsi > 65 and macd < 0) else 0
        neutral_chart  = 1 if (40 <= rsi <= 60) else 0
        base.extend([
            bullish_chart, bearish_chart, neutral_chart,
            scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)
        ])
        return base

    names = BASE_NAMES + ['bullish_chart', 'bearish_chart', 'neutral_chart', 'tp_accuracy', 'sell_accuracy']

    def label(t):
        return 1 if t.get('trade_quality') in ['GREAT', 'GOOD'] else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("⚠️ Not enough data for Chart Pattern")
        return None
    model, acc = _train_lgb(fl, np.array(ll), names)
    try:
        importances = model.feature_importances_
        print(f"📊 Chart Pattern Model: Accuracy {acc*100:.2f}% | Learns: GOOD/GREAT charts")
        print(f"   Top 5 features:")
        for i in np.argsort(importances)[::-1][:5]:
            print(f"   - {names[i]}: {importances[i]:.1f}")
    except:
        print(f"📊 Chart Pattern Model: Accuracy {acc*100:.2f}% | Learns: GOOD/GREAT charts")
    return model, acc
