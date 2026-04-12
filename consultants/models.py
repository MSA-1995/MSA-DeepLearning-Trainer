"""
Training Models - LightGBM Models with Full Feature Engineering
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


def _train_lgb(X, y, feature_names, n_estimators=50, max_depth=3, learning_rate=0.12):
    X_df = pd.DataFrame(X, columns=feature_names)
    y_s  = pd.Series(y, name='target')
    
    # تحقق من التوزيع
    pos = int(sum(y_s))
    neg = int(len(y_s) - pos)
    ratio = neg / max(pos, 1)

    # ✅ حماية من خطأ الفئة الواحدة (Small Dataset Protection)
    unique, counts = np.unique(y_s, return_counts=True)
    if len(unique) < 2 or any(counts < 2):
        print(f"    ⚠️ Data too imbalanced or insufficient for {feature_names[0]}... skipping split.")
        return None

    print(f"    Label balance: {pos} positive ({pos/len(y_s)*100:.1f}%) | {neg} negative | ratio={ratio:.1f}x")
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, test_size=0.2, random_state=42)
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        num_leaves=7, min_child_samples=50, subsample=0.9, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=0.6,
        class_weight='balanced',
        random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    
    # طباعة precision/recall للتحقق من جودة التدريب
    from sklearn.metrics import classification_report
    report = classification_report(y_test, model.predict(X_test), output_dict=True, zero_division=0)
    p1 = report.get('1', {}).get('precision', 0)
    r1 = report.get('1', {}).get('recall', 0)
    print(f"    Class-1 → Precision: {p1:.2f} | Recall: {r1:.2f}")
    
    return model, acc


# Base feature names (43 features from calculate_enhanced_features)
BASE_NAMES = get_feature_names()

# =========================================================
# 🔒 LEAKY FEATURES — معروفة فقط بعد إغلاق الصفقة
# لا يجوز استخدامها كـ input للنماذج
# =========================================================
_LEAKY_FEATURES = {
    'trade_quality_score',  # يعكس trade_quality مباشرة
    'is_trap_trade',        # مشتق من trade_quality
    'profit_magnitude',     # مشتق من profit_percent
    'is_profitable',        # مشتق من profit_percent
    'hours_held_normalized',# معروف فقط بعد الإغلاق
}

def _safe_feature_names():
    """أسماء الـ features بدون الـ leaky features"""
    return [n for n in BASE_NAMES if n not in _LEAKY_FEATURES]

def _safe_features(data, trade):
    """حساب الـ features وحذف الـ leaky features"""
    all_features = calculate_enhanced_features(data, trade)
    all_names    = BASE_NAMES
    return [v for v, n in zip(all_features, all_names) if n not in _LEAKY_FEATURES]

SAFE_NAMES = _safe_feature_names()


# ========== Meta-Learner ==========

def train_meta_learner_model(db_manager, trained_models=None, voting_scores=None, since_timestamp=None):
    """Meta-Learner - learns from NEW trades only (like other advisors)"""
    print("\nTraining Meta-Learner Model...")

    print("Loading auxiliary data...")
    try:
        trap_memory    = db_manager.load_traps(limit=10000)
        trap_memory_df = pd.DataFrame(trap_memory)
        symbol_memory  = db_manager.load_symbol_memory()
        causal_data    = db_manager.load_causal_data(limit=10000)
        causal_df      = pd.DataFrame(causal_data) if causal_data else pd.DataFrame()
        total_trades   = db_manager.get_total_trades_count()
        if total_trades == 0:
            print("No trades found.")
            return None
        print(f"Found {total_trades} total trades.")
    except Exception as e:
        print(f"Error: {e}")
        return None

    consultant_models = {k: v for k, v in (trained_models or {}).items() if k != 'meta_learner' and v is not None}
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
            print(f"  -> Processing ALL trades (first training)...")
            trades_batch = db_manager.load_training_data()
    except Exception as e:
        print(f"Error loading trades: {e}")
        return None
    
    if not trades_batch:
        print("No trades found for training")
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
            
            profit = float(trade.get('profit_percent', 0))
            tq = trade.get('trade_quality', 'OK')
            hours_held = float(trade.get('hours_held', 24))
            
            rsi = data.get('rsi', 50)
            macd = data.get('macd', 0)
            volume_ratio = data.get('volume_ratio', 1.0)
            price_momentum = data.get('price_momentum', 0)
            
            is_great = 1 if tq == 'GREAT' else 0
            is_trap = 1 if tq in ['TRAP', 'RISKY'] else 0
            is_profit = 1 if profit > 0 else 0
            
            news = data.get('news', {})
            news_score = news.get('news_score', 0)
            news_pos = news.get('positive', 0)
            news_neg = news.get('negative', 0)
            news_total = news.get('total', 0)
            news_ratio = news_pos / (news_neg + 0.001)
            has_news = 1 if news_total > 0 else 0
            
            sent = data.get('sentiment', {})
            sent_score = sent.get('news_sentiment', 0) if sent else 0
            
            liq = data.get('liquidity', {})
            liq_score = liq.get('liquidity_score', 50) if liq else 50
            depth_ratio = liq.get('depth_ratio', 1.0) if liq else 1.0
            price_impact = liq.get('price_impact', 0.5) if liq else 0.5
            good_liq = 1 if liq_score > 70 else 0
            
            trap_count = trap_counts.get(symbol, 0)
            was_trapped = 1 if trap_count > 0 else 0
            repeat_trap = 1 if trap_count >= 3 else 0
            
            mem = symbol_memory.get(symbol, {})
            sym_win_rate = mem.get('win_count', 0) / max(mem.get('total_trades', 1), 1)
            sym_avg_profit = mem.get('avg_profit', 0)
            sym_trap_count = mem.get('trap_count', 0)
            sym_total = mem.get('total_trades', 0)
            sym_is_reliable = 1 if (sym_win_rate > 0.6 and sym_total > 5) else 0
            
            fear_greed = 50
            whale_activity = 0
            exchange_inflow = 0
            social_volume = 0
            market_sent = 0
            fear_greed_norm = 0
            is_fearful = 0
            is_greedy = 0
            
            buy_votes = data.get('buy_votes', {})
            sell_votes = data.get('sell_votes', {})
            buy_count = sum(1 for v in buy_votes.values() if v == 1) if buy_votes else 0
            sell_count = sum(1 for v in sell_votes.values() if v == 1) if sell_votes else 0
            consensus = buy_count / 7.0
            opinions = [data.get(f'{n}_score', 0.5) for n in consultant_models.keys()]
            
            risk_score = (trap_count * 10 + (1 - liq_score/100) * 20 + news_neg * 5)
            opportunity = ((1 if rsi < 30 else 0) * 20 + news_pos * 5 + good_liq * 10 + buy_count * 5)
            market_quality = (liq_score / 100 + news_ratio / 10 + consensus) / 3
            
            # ✅ إصلاح: حذف is_great, is_trap, is_profit من الـ features (leaky)
            features = [
                rsi, macd, volume_ratio, price_momentum,
                hours_held,
                news_score, news_pos, news_neg, news_total, news_ratio, has_news, sent_score,
                liq_score, depth_ratio, price_impact, good_liq,
                trap_count, was_trapped, repeat_trap,
                sym_win_rate, sym_avg_profit, sym_trap_count, sym_total, sym_is_reliable,
                fear_greed, whale_activity, exchange_inflow, social_volume, market_sent,
                fear_greed_norm, is_fearful, is_greedy,
                consensus, buy_count, sell_count,
                risk_score, opportunity, market_quality
            ] + opinions
            meta_features.append(features)
            final_labels.append(1 if profit > 0.5 else 0)
        except Exception as e:
            if len(meta_features) == 0:
                print(f"  First trade error: {e}")
            continue

    if len(meta_features) < 100:
        print(f"Not enough data ({len(meta_features)} samples)")
        return None

    print(f"\nCollected {len(meta_features)} samples. Now training...")

    names = (
        ['rsi', 'macd', 'volume_ratio', 'price_momentum'] +
        # ✅ إصلاح: حذف is_great, is_trap, is_profit (leaky)
        ['hours_held',
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
    print(f"Meta-Learner Model: Accuracy {accuracy*100:.2f}%")
    return model, accuracy


# ========== Models ==========

def train_smart_money_model(trades, voting_scores=None):
    """Smart Money - discovers smart money inflows"""
    print("\nTraining Smart Money Model (LightGBM)...")
    scores = (voting_scores or {}).get('smart_money', {})

    def features(data, trade):
        base = _safe_features(data, trade)  # ✅ بدون leaky features
        whale = data.get('whale_activity', 0)
        inflow = data.get('exchange_inflow', 0)
        base.extend([whale, inflow, scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)])
        return base

    names = SAFE_NAMES + ['whale_activity', 'exchange_inflow', 'tp_accuracy', 'sell_accuracy']

    def label(t):
        # ✅ توسيع: GREAT + GOOD + ربح ممتاز = smart money
        profit = float(t.get('profit_percent', 0))
        tq = t.get('trade_quality', 'OK')
        return 1 if (tq in ['GREAT', 'GOOD'] or profit >= 1.5) else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("Not enough data for Smart Money")
        return None
    model, acc = _train_lgb(fl, np.array(ll), names)
    print(f"Smart Money Model: Accuracy {acc*100:.2f}% | Learns: GREAT/GOOD trades")
    return model, acc


def train_risk_model(trades, voting_scores=None):
    """Risk Model - learns risk patterns"""
    print("\nTraining Risk Model (LightGBM)...")
    scores = (voting_scores or {}).get('risk', {})

    def features(data, trade):
        base = _safe_features(data, trade)  # ✅ بدون leaky features
        rsi = data.get('rsi', 50)
        atr = data.get('atr', 0)
        base.extend([rsi, atr, scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)])
        return base

    names = SAFE_NAMES + ['risk_rsi', 'risk_atr', 'tp_accuracy', 'sell_accuracy']

    def label(t):
        # خطر = TRAP/RISKY أو خسارة
        profit = float(t.get('profit_percent', 0))
        tq = t.get('trade_quality', 'OK')
        return 1 if (tq in ['TRAP', 'RISKY'] or profit < -0.3) else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("Not enough data for Risk")
        return None
    model, acc = _train_lgb(fl, np.array(ll), names)
    print(f"Risk Model: Accuracy {acc*100:.2f}% | Learns: TRAP/RISKY/loss trades")
    return model, acc


def train_anomaly_model(trades, voting_scores=None):
    """Anomaly Model - learns anomaly patterns"""
    print("\nTraining Anomaly Model (LightGBM)...")
    scores = (voting_scores or {}).get('anomaly', {})

    def features(data, trade):
        base = _safe_features(data, trade)  # ✅ بدون leaky features
        anomaly = data.get('anomaly_score', 0)
        base.extend([anomaly, scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)])
        return base

    names = SAFE_NAMES + ['anomaly_score', 'tp_accuracy', 'sell_accuracy']

    def label(t):
        # anomaly = RISKY/TRAP أو خسارة كبيرة
        profit = float(t.get('profit_percent', 0))
        tq = t.get('trade_quality', 'OK')
        return 1 if (tq in ['RISKY', 'TRAP'] or profit < -0.5) else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("Not enough data for Anomaly")
        return None
    model, acc = _train_lgb(fl, np.array(ll), names)
    print(f"Anomaly Model: Accuracy {acc*100:.2f}% | Learns: RISKY/TRAP/loss trades")
    return model, acc


def train_exit_model(trades, voting_scores=None):
    """Exit Model - learns good exit patterns"""
    print("\nTraining Exit Model (LightGBM)...")
    scores = (voting_scores or {}).get('exit', {})

    def features(data, trade):
        base = _safe_features(data, trade)  # ✅ بدون leaky features
        hours = float(trade.get('hours_held', 24))
        base.extend([hours, scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)])
        return base

    names = SAFE_NAMES + ['hours_held', 'tp_accuracy', 'sell_accuracy']

    def label(t):
        # ✅ إصلاح: الـ label من profit فقط — لا يعتمد على trade_quality
        profit = float(t.get('profit_percent', 0))
        return 1 if profit >= 1.0 else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("Not enough data for Exit")
        return None
    model, acc = _train_lgb(fl, np.array(ll), names)
    print(f"Exit Model: Accuracy {acc*100:.2f}% | Learns: GOOD/GREAT exits")
    return model, acc


def train_pattern_model(trades, voting_scores=None):
    """Pattern Model - learns pattern recognition"""
    print("\nTraining Pattern Model (LightGBM)...")
    scores = (voting_scores or {}).get('pattern', {})

    def features(data, trade):
        base = _safe_features(data, trade)  # ✅ بدون leaky features
        momentum = data.get('price_momentum', 0)
        base.extend([momentum, scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)])
        return base

    names = SAFE_NAMES + ['pattern_momentum', 'tp_accuracy', 'sell_accuracy']

    def label(t):
        # ✅ إصلاح: الـ label من profit فقط بدون الاعتماد على trade_quality
        profit = float(t.get('profit_percent', 0))
        return 1 if profit >= 1.0 else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("Not enough data for Pattern")
        return None
    model, acc = _train_lgb(fl, np.array(ll), names)
    print(f"Pattern Model: Accuracy {acc*100:.2f}% | Learns: GREAT/GOOD patterns")
    return model, acc


def train_liquidity_model(trades, voting_scores=None):
    """Liquidity Analyzer - learns good liquidity patterns from each trade"""
    print("\nTraining Liquidity Model (LightGBM)...")
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
            
            # تخطي الصفقات بدون بيانات سيولة حقيقية
            if liquidity_score == 50 and not liq:
                skipped_no_data += 1
                continue
            
            profit = float(trade.get('profit_percent', 0))
            
            volume_ratio = data.get('volume_ratio', 1.0)
            bid_ask_spread = data.get('bid_ask_spread', 0)
            volume_trend = data.get('volume_trend', 0)
            depth_ratio = liq.get('depth_ratio', 1.0)
            price_impact = liq.get('price_impact', 0.5)
            volume_consistency = liq.get('volume_consistency', 50)
            
            good_liquidity = 1 if liquidity_score > 70 else 0
            low_impact = 1 if price_impact < 0.3 else 0
            consistent_vol = 1 if volume_consistency > 60 else 0
            
            features_list.append([
                # ✅ إصلاح: حذف profit من الـ features (كان leakage)
                volume_ratio, bid_ask_spread, volume_trend,
                depth_ratio, liquidity_score, price_impact, volume_consistency,
                good_liquidity, low_impact, consistent_vol,
                scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)
            ])
            
            labels_list.append(1 if profit > 0 else 0)
            
        except:
            continue
    
    names = [
        # ✅ إصلاح: profit محذوف من الـ features
        'volume_ratio', 'bid_ask_spread', 'volume_trend',
        'depth_ratio', 'liquidity_score', 'price_impact', 'volume_consistency',
        'good_liquidity', 'low_impact', 'consistent_vol',
        'tp_accuracy', 'sell_accuracy'
    ]
    
    print(f"  Training samples: {len(features_list)} trades (skipped {skipped_no_data} without liquidity data)")
    
    if len(features_list) < 50:
        print("Not enough trades with liquidity data for training")
        return None
    
    model, acc = _train_lgb(features_list, np.array(labels_list), names, n_estimators=300, max_depth=6, learning_rate=0.05)
    print(f"Liquidity Model: Accuracy {acc*100:.2f}% | Learns from {len(features_list)} trades")
    return model, acc


def train_chart_cnn_model(trades, voting_scores=None):
    """Chart Pattern Analyzer - learns chart patterns"""
    print("\nTraining Chart Pattern Model (LightGBM)...")
    scores = (voting_scores or {}).get('cnn', {})

    def features(data, trade):
        base = _safe_features(data, trade)  # ✅ بدون leaky features
        rsi  = data.get('rsi', 50)
        macd = data.get('macd', 0)
        vol  = data.get('volume_ratio', 1)
        bullish_chart  = 1 if (rsi < 40 and macd > 0 and vol > 1.2) else 0
        bearish_chart  = 1 if (rsi > 65 and macd < 0) else 0
        neutral_chart  = 1 if (40 <= rsi <= 60) else 0
        base.extend([
            bullish_chart, bearish_chart, neutral_chart,
            scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)
        ])
        return base

    names = SAFE_NAMES + ['bullish_chart', 'bearish_chart', 'neutral_chart', 'tp_accuracy', 'sell_accuracy']

    def label(t):
        # ✅ إصلاح: الـ label من profit فقط
        profit = float(t.get('profit_percent', 0))
        return 1 if profit >= 1.0 else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("Not enough data for Chart Pattern")
        return None
    model, acc = _train_lgb(fl, np.array(ll), names)
    print(f"Chart Pattern Model: Accuracy {acc*100:.2f}% | Learns: GOOD/GREAT charts")
    return model, acc


def train_volume_prediction_model(trades, voting_scores=None):
    """Volume Prediction Model"""
    print("\nTraining Volume Prediction Model...")

    def features(data, trade):
        volume = data.get('volume', 0)
        volume_ratio = data.get('volume_ratio', 1)
        volume_trend = data.get('volume_trend', 0)
        rsi = data.get('rsi', 50)
        macd = data.get('macd', 0)
        price_change = data.get('price_change_1h', 0)
        volume_spike = 1 if volume_ratio > 2.0 else 0
        volume_declining = 1 if volume_ratio < 0.5 else 0
        high_momentum = 1 if abs(price_change) > 3 else 0
        rsi_extreme = 1 if rsi < 20 or rsi > 80 else 0
        bullish_volume = 1 if volume_ratio > 1.5 and price_change > 0 else 0
        bearish_volume = 1 if volume_ratio > 1.5 and price_change < 0 else 0
        volume_price_conf = volume_ratio * abs(price_change) / 100
        return [
            volume, data.get('volume_avg_1h', volume), data.get('volume_avg_4h', volume),
            data.get('volume_avg_24h', volume), volume_ratio,
            data.get('volume_ratio_4h', 1), data.get('volume_ratio_24h', 1),
            volume_trend, data.get('volume_volatility', 0),
            price_change, data.get('price_change_4h', 0), data.get('price_change_24h', 0),
            rsi, macd, data.get('atr', 0), data.get('bid_ask_spread', 0),
            data.get('volume_momentum', 0), data.get('volume_acceleration', 0),
            volume_spike, volume_declining, high_momentum, rsi_extreme,
            bullish_volume, bearish_volume, volume_price_conf
        ]

    names = [
        'volume', 'volume_avg_1h', 'volume_avg_4h', 'volume_avg_24h', 'volume_ratio',
        'volume_ratio_4h', 'volume_ratio_24h', 'volume_trend', 'volume_volatility',
        'price_change', 'price_change_4h', 'price_change_24h',
        'rsi', 'macd', 'atr', 'bid_ask_spread',
        'volume_momentum', 'volume_acceleration',
        'volume_spike', 'volume_declining', 'high_momentum', 'rsi_extreme',
        'bullish_volume', 'bearish_volume', 'volume_price_conf'
    ]

    def label(t):
        return 1 if t.get('profit_percent', 0) > 0.8 else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("Not enough data for Volume Prediction")
        return None
    model, acc = _train_lgb(fl, np.array(ll), names)
    print(f"Volume Prediction Model: Accuracy {acc*100:.2f}%")
    return model, acc
