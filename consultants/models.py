"""
Training Models - LightGBM Models with Full Feature Engineering
Each model thinks and learns according to its specific role.
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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
        except Exception:
            continue
    return features_list, labels_list


def _train_lgb(X, y, feature_names, n_estimators=50, max_depth=3, learning_rate=0.12):
    X_df = pd.DataFrame(X, columns=feature_names)
    y_s  = pd.Series(y, name='target')

    pos   = int(sum(y_s))
    neg   = int(len(y_s) - pos)
    ratio = neg / max(pos, 1)

    # حماية من خطأ الفئة الواحدة
    unique, counts = np.unique(y_s, return_counts=True)
    if len(unique) < 2 or any(counts < 2):
        print(f"    ⚠️ Data too imbalanced or insufficient... skipping split.")
        return None

    print(f"    Label balance: {pos} positive ({pos/len(y_s)*100:.1f}%) | {neg} negative | ratio={ratio:.1f}x")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_s, test_size=0.2, random_state=42
    )
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        num_leaves=7, min_child_samples=50, subsample=0.9, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=0.6,
        class_weight='balanced',
        random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)
    acc    = accuracy_score(y_test, model.predict(X_test))
    report = classification_report(
        y_test, model.predict(X_test),
        output_dict=True, zero_division=0
    )
    p1 = report.get('1', {}).get('precision', 0)
    r1 = report.get('1', {}).get('recall', 0)
    print(f"    Class-1 → Precision: {p1:.2f} | Recall: {r1:.2f}")

    return model, acc


# ========== Feature Names ==========

BASE_NAMES = get_feature_names()

# FIX: تعريف الـ leaky features مرة واحدة فقط
_LEAKY_FEATURES = {
    'trade_quality_score',
    'is_trap_trade',
    'profit_magnitude',
    'is_profitable',
    'hours_held_normalized',
}

def _safe_feature_names():
    return [n for n in BASE_NAMES if n not in _LEAKY_FEATURES]

def _safe_features(data, trade):
    all_features = calculate_enhanced_features(data, trade)
    return [v for v, n in zip(all_features, BASE_NAMES) if n not in _LEAKY_FEATURES]

SAFE_NAMES = _safe_feature_names()


# ========== Meta-Learner ==========

def train_meta_learner_model(db_manager, trained_models=None, voting_scores=None, since_timestamp=None):
    """Meta-Learner - learns from NEW trades only"""
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
        print(f"Error loading auxiliary data: {e}")
        return None

    consultant_models = {
        k: v for k, v in (trained_models or {}).items()
        if k != 'meta_learner' and v is not None
    }
    trap_counts = {}
    if not trap_memory_df.empty and 'symbol' in trap_memory_df.columns:
        trap_counts = trap_memory_df['symbol'].value_counts().to_dict()

    meta_features, final_labels = [], []

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

            symbol     = str(trade.get('symbol', ''))
            profit     = float(trade.get('profit_percent', 0))
            tq         = trade.get('trade_quality', 'OK')
            hours_held = float(trade.get('hours_held', 24))

            rsi            = data.get('rsi', 50)
            macd           = data.get('macd', 0)
            volume_ratio   = data.get('volume_ratio', 1.0)
            price_momentum = data.get('price_momentum', 0)

            news       = data.get('news', {})
            news_score = news.get('news_score', 0)
            news_pos   = news.get('positive', 0)
            news_neg   = news.get('negative', 0)
            news_total = news.get('total', 0)
            news_ratio = news_pos / (news_neg + 0.001)
            has_news   = 1 if news_total > 0 else 0

            sent       = data.get('sentiment', {})
            sent_score = sent.get('news_sentiment', 0) if sent else 0

            liq          = data.get('liquidity', {})
            liq_score    = liq.get('liquidity_score', 50) if liq else 50
            depth_ratio  = liq.get('depth_ratio', 1.0) if liq else 1.0
            price_impact = liq.get('price_impact', 0.5) if liq else 0.5
            good_liq     = 1 if liq_score > 70 else 0

            trap_count  = trap_counts.get(symbol, 0)
            was_trapped = 1 if trap_count > 0 else 0
            repeat_trap = 1 if trap_count >= 3 else 0

            mem             = symbol_memory.get(symbol, {})
            sym_win_rate    = mem.get('win_count', 0) / max(mem.get('total_trades', 1), 1)
            sym_avg_profit  = mem.get('avg_profit', 0)
            sym_trap_count  = mem.get('trap_count', 0)
            sym_total       = mem.get('total_trades', 0)
            sym_is_reliable = 1 if (sym_win_rate > 0.6 and sym_total > 5) else 0

            fear_greed       = 50
            whale_activity   = 0
            exchange_inflow  = 0
            social_volume    = 0
            market_sent      = 0
            fear_greed_norm  = 0
            is_fearful       = 0
            is_greedy        = 0

            buy_votes  = data.get('buy_votes', {})
            sell_votes = data.get('sell_votes', {})
            buy_count  = sum(1 for v in buy_votes.values() if v == 1) if buy_votes else 0
            sell_count = sum(1 for v in sell_votes.values() if v == 1) if sell_votes else 0
            consensus  = buy_count / 7.0
            opinions   = [data.get(f'{n}_score', 0.5) for n in consultant_models.keys()]

            risk_score      = (trap_count * 10 + (1 - liq_score/100) * 20 + news_neg * 5)
            opportunity     = ((1 if rsi < 30 else 0) * 20 + news_pos * 5 + good_liq * 10 + buy_count * 5)
            market_quality  = (liq_score / 100 + news_ratio / 10 + consensus) / 3

            dynamic_consultant_weights  = float(trade.get('dynamic_consultant_weights', 0))
            uncertainty_quantification  = float(trade.get('uncertainty_quantification', 0))
            context_aware_score         = float(trade.get('context_aware_score', 0))

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
                risk_score, opportunity, market_quality,
                dynamic_consultant_weights, uncertainty_quantification, context_aware_score
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
        ['rsi', 'macd', 'volume_ratio', 'price_momentum',
         'hours_held',
         'news_score', 'news_pos', 'news_neg', 'news_total', 'news_ratio', 'has_news', 'sent_score',
         'liq_score', 'depth_ratio', 'price_impact', 'good_liq',
         'trap_count', 'was_trapped', 'repeat_trap',
         'sym_win_rate', 'sym_avg_profit', 'sym_trap_count', 'sym_total', 'sym_is_reliable',
         'fear_greed', 'whale_activity', 'exchange_inflow', 'social_volume', 'market_sent',
         'fear_greed_norm', 'is_fearful', 'is_greedy',
         'consensus', 'buy_count', 'sell_count',
         'risk_score', 'opportunity', 'market_quality',
         'dynamic_consultant_weights', 'uncertainty_quantification', 'context_aware_score'] +
        [f'consultant_{n}' for n in consultant_models.keys()]
    )

    result = _train_lgb(
        meta_features, np.array(final_labels), names,
        n_estimators=300, max_depth=6, learning_rate=0.03
    )
    if result is None:
        return None
    model, accuracy = result
    print(f"Meta-Learner Model: Accuracy {accuracy*100:.2f}%")
    return model, accuracy


# ========== Models ==========

def train_smart_money_model(trades, voting_scores=None):
    """Smart Money - discovers smart money inflows"""
    print("\nTraining Smart Money Model (LightGBM)...")
    scores = (voting_scores or {}).get('smart_money', {})

    def features(data, trade):
        base = _safe_features(data, trade)
        whale                       = float(trade.get('whale_confidence') or 0)
        inflow                      = data.get('exchange_inflow', 0) or data.get('whale_activity', 0)
        whale_wallet_changes        = float(trade.get('whale_wallet_changes') or 0)
        institutional_accumulation  = float(trade.get('institutional_accumulation') or 0)
        smart_money_ratio           = float(trade.get('smart_money_ratio') or 0)
        exchange_whale_flows        = float(trade.get('exchange_whale_flows') or 0)
        base.extend([
            whale, inflow, whale_wallet_changes, institutional_accumulation,
            smart_money_ratio, exchange_whale_flows,
            scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)
        ])
        return base

    names = SAFE_NAMES + [
        'whale_activity', 'exchange_inflow', 'whale_wallet_changes',
        'institutional_accumulation', 'smart_money_ratio', 'exchange_whale_flows',
        'tp_accuracy', 'sell_accuracy'
    ]

    def label(t):
        profit = float(t.get('profit_percent', 0))
        tq     = t.get('trade_quality', 'OK')
        return 1 if (tq in ['GREAT', 'GOOD'] or profit >= 1.5) else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("Not enough data for Smart Money")
        return None
    result = _train_lgb(fl, np.array(ll), names)
    if result is None:
        return None
    model, acc = result
    print(f"Smart Money Model: Accuracy {acc*100:.2f}% | Learns: GREAT/GOOD trades")
    return model, acc


def train_risk_model(trades, voting_scores=None):
    """Risk Model - learns risk patterns"""
    print("\nTraining Risk Model (LightGBM)...")
    scores = (voting_scores or {}).get('risk', {})

    def features(data, trade):
        base                    = _safe_features(data, trade)
        rsi                     = data.get('rsi', 50)
        atr                     = data.get('atr', 0)
        volatility_risk_score   = float(trade.get('volatility_risk_score') or 0)
        correlation_risk        = float(trade.get('correlation_risk') or 0)
        gap_risk_score          = float(trade.get('gap_risk_score') or 0)
        black_swan_probability  = float(trade.get('black_swan_probability') or 0)
        behavioral_risk         = float(trade.get('behavioral_risk') or 0)
        systemic_risk           = float(trade.get('systemic_risk') or 0)
        base.extend([
            rsi, atr, volatility_risk_score, correlation_risk,
            gap_risk_score, black_swan_probability, behavioral_risk, systemic_risk,
            scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)
        ])
        return base

    names = SAFE_NAMES + [
        'risk_rsi', 'risk_atr', 'volatility_risk_score', 'correlation_risk',
        'gap_risk_score', 'black_swan_probability', 'behavioral_risk', 'systemic_risk',
        'tp_accuracy', 'sell_accuracy'
    ]

    def label(t):
        profit = float(t.get('profit_percent', 0))
        tq     = t.get('trade_quality', 'OK')
        return 1 if (tq in ['TRAP', 'RISKY'] or profit < -0.3) else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("Not enough data for Risk")
        return None
    result = _train_lgb(fl, np.array(ll), names)
    if result is None:
        return None
    model, acc = result
    print(f"Risk Model: Accuracy {acc*100:.2f}% | Learns: TRAP/RISKY/loss trades")
    return model, acc


def train_anomaly_model(trades, voting_scores=None):
    """Anomaly Model - learns anomaly patterns"""
    print("\nTraining Anomaly Model (LightGBM)...")
    scores = (voting_scores or {}).get('anomaly', {})

    def features(data, trade):
        base                  = _safe_features(data, trade)
        anomaly               = data.get('anomaly_score', 0)
        statistical_outliers  = float(trade.get('statistical_outliers') or 0)
        pattern_anomalies     = float(trade.get('pattern_anomalies') or 0)
        behavioral_anomalies  = float(trade.get('behavioral_anomalies') or 0)
        volume_anomalies      = float(trade.get('volume_anomalies') or 0)
        base.extend([
            anomaly, statistical_outliers, pattern_anomalies,
            behavioral_anomalies, volume_anomalies,
            scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)
        ])
        return base

    names = SAFE_NAMES + [
        'anomaly_score', 'statistical_outliers', 'pattern_anomalies',
        'behavioral_anomalies', 'volume_anomalies',
        'tp_accuracy', 'sell_accuracy'
    ]

    def label(t):
        profit = float(t.get('profit_percent', 0))
        tq     = t.get('trade_quality', 'OK')
        return 1 if (tq in ['RISKY', 'TRAP'] or profit < -0.5) else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("Not enough data for Anomaly")
        return None
    result = _train_lgb(fl, np.array(ll), names)
    if result is None:
        return None
    model, acc = result
    print(f"Anomaly Model: Accuracy {acc*100:.2f}% | Learns: RISKY/TRAP/loss trades")
    return model, acc


def train_exit_model(trades, voting_scores=None):
    """Exit Model - learns good exit patterns"""
    print("\nTraining Exit Model (LightGBM)...")
    scores = (voting_scores or {}).get('exit', {})

    def features(data, trade):
        base                       = _safe_features(data, trade)
        hours                      = float(trade.get('hours_held', 24))
        profit_optimization_score  = float(trade.get('profit_optimization_score') or 0)
        time_decay_signals         = float(trade.get('time_decay_signals') or 0)
        opportunity_cost_exits     = float(trade.get('opportunity_cost_exits') or 0)
        market_condition_exits     = float(trade.get('market_condition_exits') or 0)
        base.extend([
            hours, profit_optimization_score, time_decay_signals,
            opportunity_cost_exits, market_condition_exits,
            scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)
        ])
        return base

    names = SAFE_NAMES + [
        'hours_held', 'profit_optimization_score', 'time_decay_signals',
        'opportunity_cost_exits', 'market_condition_exits',
        'tp_accuracy', 'sell_accuracy'
    ]

    def label(t):
        profit = float(t.get('profit_percent', 0))
        return 1 if profit >= 1.0 else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("Not enough data for Exit")
        return None
    result = _train_lgb(fl, np.array(ll), names)
    if result is None:
        return None
    model, acc = result
    print(f"Exit Model: Accuracy {acc*100:.2f}% | Learns: GOOD/GREAT exits")
    return model, acc


def train_pattern_model(trades, voting_scores=None):
    """Pattern Model - learns pattern recognition"""
    print("\nTraining Pattern Model (LightGBM)...")
    scores = (voting_scores or {}).get('pattern', {})

    def features(data, trade):
        base                    = _safe_features(data, trade)
        momentum                = data.get('price_momentum', 0)
        harmonic_patterns_score = float(trade.get('harmonic_patterns_score') or 0)
        elliott_wave_signals    = float(trade.get('elliott_wave_signals') or 0)
        fractal_patterns        = float(trade.get('fractal_patterns') or 0)
        cycle_patterns          = float(trade.get('cycle_patterns') or 0)
        momentum_patterns       = float(trade.get('momentum_patterns') or 0)
        base.extend([
            momentum, harmonic_patterns_score, elliott_wave_signals,
            fractal_patterns, cycle_patterns, momentum_patterns,
            scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)
        ])
        return base

    names = SAFE_NAMES + [
        'pattern_momentum', 'harmonic_patterns_score', 'elliott_wave_signals',
        'fractal_patterns', 'cycle_patterns', 'momentum_patterns',
        'tp_accuracy', 'sell_accuracy'
    ]

    def label(t):
        profit = float(t.get('profit_percent', 0))
        return 1 if profit >= 1.0 else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("Not enough data for Pattern")
        return None
    result = _train_lgb(fl, np.array(ll), names)
    if result is None:
        return None
    model, acc = result
    print(f"Pattern Model: Accuracy {acc*100:.2f}% | Learns: GREAT/GOOD patterns")
    return model, acc


def train_liquidity_model(trades, voting_scores=None):
    """Liquidity Analyzer - learns good liquidity patterns"""
    print("\nTraining Liquidity Model (LightGBM)...")
    scores = (voting_scores or {}).get('liquidity', {})

    features_list, labels_list = [], []

    for trade in trades:
        try:
            data = trade.get('data', {})
            if isinstance(data, str):
                data = json.loads(data)
            if isinstance(data, dict) and 'data' in data and isinstance(data.get('data'), dict):
                data = data['data']

            liq           = data.get('liquidity', {})
            liquidity_score = liq.get('liquidity_score') or trade.get('liquidity_score') or 50

            if not liq:
                depth_ratio        = 1.0
                price_impact       = 0.5
                volume_consistency = 50
                spread_percent     = 0.1
            else:
                depth_ratio        = liq.get('depth_ratio', 1.0)
                price_impact       = liq.get('price_impact', 0.5)
                volume_consistency = liq.get('volume_consistency', 50)
                spread_percent     = liq.get('spread_percent', 0.1)

            profit             = float(trade.get('profit_percent', 0))
            volume_ratio       = data.get('volume_ratio', 1.0)
            bid_ask_spread     = data.get('bid_ask_spread', 0)
            volume_trend       = data.get('volume_trend', 0)

            spread_volatility      = float(trade.get('spread_volatility', 0))
            depth_at_1pct          = float(trade.get('depth_at_1pct', 0))
            market_impact_score    = float(trade.get('market_impact_score', 0))
            liquidity_trends       = float(trade.get('liquidity_trends', 0))
            order_book_imbalance   = float(trade.get('order_book_imbalance', 0))

            good_liquidity   = 1 if liquidity_score > 70 else 0
            low_impact       = 1 if price_impact < 0.3 else 0
            consistent_vol   = 1 if volume_consistency > 60 else 0
            high_depth       = 1 if depth_at_1pct > 100000 else 0
            low_spread_vol   = 1 if spread_volatility < 0.5 else 0
            balanced_book    = 1 if abs(order_book_imbalance) < 0.2 else 0

            liquidity_depth        = depth_ratio * liquidity_score / 100
            impact_risk            = price_impact * (1 - liquidity_score/100)
            volume_liquidity_score = volume_ratio * liquidity_score / 100
            spread_impact          = spread_percent * price_impact

            features_list.append([
                volume_ratio, bid_ask_spread, volume_trend,
                depth_ratio, liquidity_score, price_impact, volume_consistency,
                good_liquidity, low_impact, consistent_vol,
                spread_volatility, depth_at_1pct, market_impact_score,
                liquidity_trends, order_book_imbalance,
                high_depth, low_spread_vol, balanced_book,
                liquidity_depth, impact_risk, volume_liquidity_score,
                spread_percent, spread_impact,
                scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)
            ])
            labels_list.append(1 if profit > 0 else 0)

        except Exception:
            continue

    names = [
        'volume_ratio', 'bid_ask_spread', 'volume_trend',
        'depth_ratio', 'liquidity_score', 'price_impact', 'volume_consistency',
        'good_liquidity', 'low_impact', 'consistent_vol',
        'spread_volatility', 'depth_at_1pct', 'market_impact_score',
        'liquidity_trends', 'order_book_imbalance',
        'high_depth', 'low_spread_vol', 'balanced_book',
        'liquidity_depth', 'impact_risk', 'volume_liquidity_score',
        'spread_percent', 'spread_impact',
        'tp_accuracy', 'sell_accuracy'
    ]

    print(f"  Training samples: {len(features_list)} trades")

    if len(features_list) < 30:
        print("Not enough trades with liquidity data for training")
        return None

    result = _train_lgb(
        features_list, np.array(labels_list), names,
        n_estimators=300, max_depth=6, learning_rate=0.05
    )
    if result is None:
        return None
    model, acc = result
    print(f"Liquidity Model: Accuracy {acc*100:.2f}% | {len(features_list)} trades | {len(names)} features")
    return model, acc


def train_chart_cnn_model(trades, voting_scores=None):
    """Chart Pattern Analyzer - learns chart patterns"""
    print("\nTraining Chart Pattern Model (LightGBM)...")
    scores = (voting_scores or {}).get('cnn', {})

    def features(data, trade):
        base                      = _safe_features(data, trade)
        rsi                       = data.get('rsi', 50)
        macd                      = data.get('macd', 0)
        vol                       = data.get('volume_ratio', 1)
        bullish_chart             = 1 if (rsi < 40 and macd > 0 and vol > 1.2) else 0
        bearish_chart             = 1 if (rsi > 65 and macd < 0) else 0
        neutral_chart             = 1 if (40 <= rsi <= 60) else 0
        attention_mechanism_score = float(trade.get('attention_mechanism_score') or 0)
        multi_scale_features      = float(trade.get('multi_scale_features') or 0)
        temporal_features         = float(trade.get('temporal_features') or 0)
        base.extend([
            bullish_chart, bearish_chart, neutral_chart,
            attention_mechanism_score, multi_scale_features, temporal_features,
            scores.get('tp_accuracy', 0.5), scores.get('sell_accuracy', 0.5)
        ])
        return base

    names = SAFE_NAMES + [
        'bullish_chart', 'bearish_chart', 'neutral_chart',
        'attention_mechanism_score', 'multi_scale_features', 'temporal_features',
        'tp_accuracy', 'sell_accuracy'
    ]

    def label(t):
        profit = float(t.get('profit_percent', 0))
        return 1 if profit >= 1.0 else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("Not enough data for Chart Pattern")
        return None
    result = _train_lgb(fl, np.array(ll), names)
    if result is None:
        return None
    model, acc = result
    print(f"Chart Pattern Model: Accuracy {acc*100:.2f}% | Learns: GOOD/GREAT charts")
    return model, acc


def train_volume_prediction_model(trades, voting_scores=None):
    """Volume Prediction Model"""
    print("\nTraining Volume Prediction Model...")

    def features(data, trade):
        volume                = data.get('volume', 0)
        volume_ratio          = data.get('volume_ratio', 1)
        volume_trend          = data.get('volume_trend', 0)
        rsi                   = data.get('rsi', 50)
        macd                  = data.get('macd', 0)
        volume_spike          = 1 if volume_ratio > 2.0 else 0
        volume_declining      = 1 if volume_ratio < 0.5 else 0
        rsi_extreme           = 1 if rsi < 20 or rsi > 80 else 0
        volume_trend_strength = float(trade.get('volume_trend_strength') or 0)
        volume_seasonality    = float(trade.get('volume_seasonality') or 0)
        volume_correlation    = float(trade.get('volume_correlation') or 0)

        return [
            volume,
            data.get('volume_avg_1h', volume),
            data.get('volume_avg_4h', volume),
            data.get('volume_avg_24h', volume),
            volume_ratio,
            data.get('volume_ratio_4h', 1),
            data.get('volume_ratio_24h', 1),
            volume_trend,
            data.get('volume_volatility', 0),
            data.get('price_change_24h_old', 0),
            rsi, macd,
            data.get('atr', 0),
            data.get('bid_ask_spread', 0),
            data.get('volume_momentum', 0),
            data.get('volume_acceleration', 0),
            volume_trend_strength, volume_seasonality, volume_correlation,
            volume_spike, volume_declining, rsi_extreme
        ]

    names = [
        'volume', 'volume_avg_1h', 'volume_avg_4h', 'volume_avg_24h', 'volume_ratio',
        'volume_ratio_4h', 'volume_ratio_24h', 'volume_trend', 'volume_volatility',
        'price_history_lag',
        'rsi', 'macd', 'atr', 'bid_ask_spread',
        'volume_momentum', 'volume_acceleration',
        'volume_trend_strength', 'volume_seasonality', 'volume_correlation',
        'volume_spike', 'volume_declining', 'rsi_extreme'
    ]

    def label(t):
        return 1 if t.get('profit_percent', 0) > 3.0 else 0

    fl, ll = _build_dataset(trades, features, label)
    if len(fl) < 50:
        print("Not enough data for Volume Prediction")
        return None
    result = _train_lgb(fl, np.array(ll), names)
    if result is None:
        return None
    model, acc = result
    print(f"Volume Prediction Model: Accuracy {acc*100:.2f}%")
    return model, acc


def train_candle_expert_model(trades, voting_scores=None):
    """⚠️ DEPRECATED - Use models/candle_expert_model.py instead"""
    print("\n⚠️ Old candle_expert in consultants/models.py is deprecated.")
    print("   Please use: from models.candle_expert_model import train_candle_expert_model")
    return None