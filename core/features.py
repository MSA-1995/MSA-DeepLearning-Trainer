"""
🔧 Feature Engineering
Calculates technical indicators used as model input features.
"""

import json
from datetime import datetime


def calculate_enhanced_features(data, trade=None):
    """
    Feature Engineering: حساب مؤشرات إضافية (38 feature)
    
    الميزات التقليدية (15):
    - RSI, MACD, Volume, Momentum, ATR, EMA, etc.
    
    الميزات من السياق (4):
    - trade_quality_score: تقييم الجودة (1-5)  ← وقت الشراء فقط
    - advisor_vote_consensus: توافق المستشارين (0-1)
    - hours_held_normalized: ساعات الاحتفاظ (0-2)
    - is_trap_trade: هل كانت صفقة فخ (0/1)
    
    الميزات من السوق والوقت (7):
    - btc_trend_1h, is_bullish_market, hour_normalized
    - is_asian/european/us_session, optimal_hold_score
    
    الميزات من فيبوناتشي (2):
    - fib_score, fib_level_encoded
    
    الميزات من Market Regime (4):
    - regime_score, regime_adx, volatility_ratio, position_multiplier
    
    الميزات من Flash Crash Protection (4):
    - flash_risk_score, flash_crash_detected
    - whale_dump_detected, cascade_risk_score

    الميزات الإضافية (4):
    - whale_confidence, sentiment_score, panic_score, optimism_penalty
    """
    try:
        # دمج البيانات إذا كان trade متوفراً
        if trade:
            trade_data = trade.get('data', {})
            if isinstance(trade_data, str):
                trade_data = json.loads(trade_data)
            if not isinstance(trade_data, dict):
                trade_data = {}
            full_data = {**trade, **trade_data, **data}
        else:
            full_data = data

        rsi            = float(full_data.get('rsi', 50) or 50)
        macd           = float(full_data.get('macd_diff', full_data.get('macd', 0)) or 0)
        volume_ratio   = float(full_data.get('volume_ratio', 1) or 1)
        price_momentum = float(full_data.get('price_momentum', 0) or 0)

        # Bollinger Bands approximation
        bb_position = (rsi - 30) / 40

        # ATR approximation
        atr_estimate = abs(price_momentum) * volume_ratio

        # Stochastic approximation
        stochastic = rsi

        # EMA crossover signal
        ema_signal = 1 if macd > 0 else -1

        # Volume strength
        volume_strength = min(volume_ratio / 2.0, 2.0)

        # Momentum strength
        momentum_strength = abs(price_momentum) / 10.0

        # New indicators
        atr             = float(full_data.get('atr', atr_estimate) or atr_estimate)
        ema_9           = float(full_data.get('ema_9', 0) or 0)
        ema_21          = float(full_data.get('ema_21', 0) or 0)
        ema_crossover   = 1 if ema_9 > ema_21 else -1
        bid_ask_spread  = float(full_data.get('bid_ask_spread', 0) or 0)
        price_change_1h = float(full_data.get('price_change_1h', 0) or 0)

        _vt = full_data.get('volume_trend', 0)
        if _vt == 'up':
            volume_trend = 1.2
        elif _vt == 'down':
            volume_trend = 0.8
        elif _vt == 'neutral':
            volume_trend = 0.0
        else:
            try:
                volume_trend = float(_vt or 0)
            except (ValueError, TypeError):
                volume_trend = 0.0

        # =========================================================
        # ✅ ميزات السياق (لا تحتوي على نتيجة الصفقة)
        # =========================================================

        # 1. تقييم الجودة - يُحسب وقت الشراء فقط
        trade_quality       = full_data.get('trade_quality', 'OK') or 'OK'
        quality_map         = {'TRAP': 1, 'RISKY': 2, 'OK': 3, 'GOOD': 4, 'GREAT': 5}
        trade_quality_score = quality_map.get(trade_quality, 3)

        # 2. توافق المستشارين
        advisor_votes = full_data.get('advisor_votes', {})
        if isinstance(advisor_votes, str):
            try:
                advisor_votes = json.loads(advisor_votes)
            except Exception:
                advisor_votes = {}
        if advisor_votes and isinstance(advisor_votes, dict):
            vote_count  = sum(1 for v in advisor_votes.values() if v == 1)
            total_votes = len(advisor_votes)
            advisor_vote_consensus = vote_count / total_votes if total_votes > 0 else 0.5
        else:
            advisor_vote_consensus = 0.5

        # 3. ساعات الاحتفاظ
        hours_held            = float(full_data.get('hours_held', 24) or 24)
        hours_held_normalized = min(hours_held / 48.0, 2.0)

        # 4. هل الصفقة كانت فخ (يُحدد وقت الشراء)
        is_trap_trade = 1 if trade_quality in ['TRAP', 'RISKY'] else 0

        # =========================================================
        # 🌍 سياق السوق والوقت
        # =========================================================

        btc_trend            = float(full_data.get('btc_change_1h', full_data.get('btc_trend_1h', 0)) or 0)
        btc_trend_normalized = max(-1.0, min(1.0, btc_trend / 5.0))
        is_bullish_market    = 1 if btc_trend > 1.0 else 0

        timestamp = full_data.get('timestamp')
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp
                hour_of_day = dt.hour
            except Exception:
                hour_of_day = 12
        else:
            hour_of_day = 12

        hour_normalized     = hour_of_day / 24.0
        is_asian_session    = 1 if 0  <= hour_of_day <= 8  else 0
        is_european_session = 1 if 8  <  hour_of_day <= 16 else 0
        is_us_session       = 1 if 16 <  hour_of_day <= 24 else 0

        # درجة الوقت المثالي (بناءً على جودة الصفقة وقت الشراء)
        if trade_quality in ['GREAT', 'GOOD']:
            optimal_hold_score = 1.0 if hours_held > 12 else 0.5
        elif trade_quality in ['TRAP', 'RISKY']:
            optimal_hold_score = 1.0 if hours_held < 4 else 0.3
        else:
            optimal_hold_score = 0.5

        # =========================================================
        # 📊 فيبوناتشي
        # =========================================================

        fib_score        = float(full_data.get('fib_score', 0) or 0)
        decision_factors = full_data.get('decision_factors', {})
        if isinstance(decision_factors, str):
            try:
                decision_factors = json.loads(decision_factors)
            except Exception:
                decision_factors = {}
        if not isinstance(decision_factors, dict):
            decision_factors = {}
        fib_score = max(fib_score, float(decision_factors.get('fib_score', 0) or 0))

        fib_level     = full_data.get('fib_level') or decision_factors.get('fib_level')
        fib_level_map = {'0': 0, '23.6': 1, '38.2': 2, '50': 3, '61.8': 4, '78.6': 5, '100': 6}
        fib_level_encoded = fib_level_map.get(str(fib_level), 0) if fib_level else 0

        # =========================================================
        # 🎯 Market Regime
        # =========================================================

        market_regime = full_data.get('market_regime', {})
        if isinstance(market_regime, str):
            try:
                market_regime = json.loads(market_regime)
            except Exception:
                market_regime = {}
        if not isinstance(market_regime, dict):
            market_regime = {}

        regime_map = {
            'STRONG_UPTREND'  : 1.0,
            'WEAK_TREND'      : 0.7,
            'RANGING'         : 0.5,
            'LOW_VOLATILITY'  : 0.4,
            'HIGH_VOLATILITY' : 0.3,
            'STRONG_DOWNTREND': 0.0,
            'UNKNOWN'         : 0.5,
        }
        regime_score     = regime_map.get(market_regime.get('regime', 'UNKNOWN'), 0.5)
        regime_adx       = float(market_regime.get('adx', 20) or 20) / 50.0
        volatility_ratio = float(market_regime.get('volatility_ratio', 1.0) or 1.0)
        trading_advice   = market_regime.get('trading_advice', {})
        if not isinstance(trading_advice, dict):
            trading_advice = {}
        position_multiplier = float(trading_advice.get('position_size', 1.0) or 1.0)

        # =========================================================
        # 🚨 Flash Crash Protection
        # =========================================================

        flash_crash = full_data.get('flash_crash_protection', {})
        if isinstance(flash_crash, str):
            try:
                flash_crash = json.loads(flash_crash)
            except Exception:
                flash_crash = {}
        if not isinstance(flash_crash, dict):
            flash_crash = {}

        flash_risk_score     = float(flash_crash.get('risk_score', 0) or 0) / 100.0
        flash_crash_detected = 1 if flash_crash.get('flash_crash_detected', False) else 0
        whale_dump_detected  = 1 if flash_crash.get('whale_dump_detected',  False) else 0
        cascade_risk         = flash_crash.get('cascade_risk', {})
        if not isinstance(cascade_risk, dict):
            cascade_risk = {}
        cascade_risk_score = float(cascade_risk.get('score', 0) or 0) / 100.0

        # =========================================================
        # 🐋 Whale + Additional
        # =========================================================

        whale_confidence = (float(trade.get('whale_confidence', 0) or 0) / 25.0) if trade else 0.0
        sentiment_score  = float(trade.get('sentiment_score',   0) or 0) if trade else 0.0
        panic_score      = float(trade.get('panic_score',       0) or 0) if trade else 0.0
        optimism_penalty = float(trade.get('optimism_penalty',  0) or 0) if trade else 0.0

        return [
            # التقليدية (15)
            rsi, macd, volume_ratio, price_momentum,
            bb_position, atr_estimate, stochastic, ema_signal,
            volume_strength, momentum_strength,
            atr, ema_crossover, bid_ask_spread, volume_trend, price_change_1h,
            # السياق (4) — بدون profit_magnitude و is_profitable
            trade_quality_score,     # 16
            advisor_vote_consensus,  # 17
            hours_held_normalized,   # 18
            is_trap_trade,           # 19
            # السوق والوقت (7)
            btc_trend_normalized,    # 20
            is_bullish_market,       # 21
            hour_normalized,         # 22
            is_asian_session,        # 23
            is_european_session,     # 24
            is_us_session,           # 25
            optimal_hold_score,      # 26
            # فيبوناتشي (2)
            fib_score,               # 27
            fib_level_encoded,       # 28
            # Market Regime (4)
            regime_score,            # 29
            regime_adx,              # 30
            volatility_ratio,        # 31
            position_multiplier,     # 32
            # Flash Crash (4)
            flash_risk_score,        # 33
            flash_crash_detected,    # 34
            whale_dump_detected,     # 35
            cascade_risk_score,      # 36
            # Whale + Additional (4)
            whale_confidence,        # 37
            sentiment_score,         # 38
            panic_score,             # 39
            optimism_penalty,        # 40
        ]

    except Exception as e:
        print(f"⚠️ Feature calculation error: {e}")
        return [
            # Technical (15)
            50, 0, 1, 0, 0.5, 1, 50, 0, 1, 0, 1, 0, 0, 0, 0,
            # Context (4)
            3, 0.5, 0.5, 0,
            # Market/Time (7)
            0, 0, 0.5, 0, 0, 0, 0.5,
            # Fibonacci (2)
            0, 0,
            # Regime (4)
            0.5, 0.4, 1.0, 1.0,
            # Flash Crash (4)
            0, 0, 0, 0,
            # Whale + Additional (4)
            0, 0, 0, 0,
        ]


def get_feature_names():
    """أسماء الـ 40 ميزة"""
    return [
        # التقليدية (15)
        'rsi', 'macd', 'volume_ratio', 'price_momentum',
        'bb_position', 'atr_estimate', 'stochastic', 'ema_signal',
        'volume_strength', 'momentum_strength',
        'atr', 'ema_crossover', 'bid_ask_spread', 'volume_trend', 'price_change_1h',
        # السياق (4)
        'trade_quality_score',
        'advisor_vote_consensus',
        'hours_held_normalized',
        'is_trap_trade',
        # السوق والوقت (7)
        'btc_trend_normalized',
        'is_bullish_market',
        'hour_normalized',
        'is_asian_session',
        'is_european_session',
        'is_us_session',
        'optimal_hold_score',
        # فيبوناتشي (2)
        'fib_score',
        'fib_level_encoded',
        # Market Regime (4)
        'regime_score',
        'regime_adx',
        'volatility_ratio',
        'position_multiplier',
        # Flash Crash (4)
        'flash_risk_score',
        'flash_crash_detected',
        'whale_dump_detected',
        'cascade_risk_score',
        # Whale + Additional (4)
        'whale_confidence',
        'sentiment_score',
        'panic_score',
        'optimism_penalty',
    ]


def extract_features(trade, symbol_memory):
    """
    Extract features for meta trading model from trade and symbol memory.
    """
    symbol   = trade.get('symbol', '')
    sym_data = symbol_memory.get(symbol, {})

    def get_num(d, key, default=0):
        val = d.get(key, default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return float(default)

    # Technical
    rsi            = get_num(trade, 'rsi', 50)
    macd_diff      = get_num(trade, 'macd_diff', 0)
    volume_ratio   = get_num(trade, 'volume_ratio', 1)
    price_momentum = get_num(trade, 'price_momentum', 0)
    atr            = get_num(trade, 'atr', 1)

    # News
    news_score = get_num(trade, 'news_score', 0)
    news_pos   = get_num(trade, 'news_pos', 0)
    news_neg   = get_num(trade, 'news_neg', 0)
    news_total = get_num(trade, 'news_total', 0)
    news_ratio = get_num(trade, 'news_ratio', 0)
    has_news   = 1.0 if news_total > 0 else 0.0

    # Sentiment
    sent_score      = get_num(trade, 'sent_score', 0)
    fear_greed      = get_num(trade, 'fear_greed', 50)
    fear_greed_norm = fear_greed / 100.0
    is_fearful      = 1.0 if fear_greed < 40 else 0.0
    is_greedy       = 1.0 if fear_greed > 60 else 0.0

    # Liquidity
    liq_score    = get_num(trade, 'liq_score', 0)
    depth_ratio  = get_num(trade, 'depth_ratio', 1)
    price_impact = get_num(trade, 'price_impact', 0)
    good_liq     = get_num(trade, 'good_liq', 0)

    # Smart Money
    whale_activity  = get_num(trade, 'whale_activity', 0)
    exchange_inflow = get_num(trade, 'exchange_inflow', 0)

    # Social
    social_volume    = get_num(trade, 'social_volume', 0)
    market_sentiment = get_num(trade, 'market_sentiment', 0)

    # Consultants
    consensus  = get_num(trade, 'consensus', 0.5)
    buy_count  = get_num(trade, 'buy_count', 0)
    sell_count = get_num(trade, 'sell_count', 0)

    # Derived
    risk_score        = get_num(trade, 'risk_score', 0)
    opportunity       = get_num(trade, 'opportunity', 0)
    market_quality    = get_num(trade, 'market_quality', 0)
    momentum_strength = get_num(trade, 'momentum_strength', 0)
    volatility_level  = get_num(trade, 'volatility_level', 0)

    # Symbol Memory
    sym_win_rate          = get_num(sym_data, 'win_rate', 0)
    sym_avg_profit        = get_num(sym_data, 'avg_profit', 0)
    sym_trap_count        = get_num(sym_data, 'trap_count', 0)
    sym_total             = get_num(sym_data, 'total', 0)
    sym_is_reliable       = get_num(sym_data, 'is_reliable', 0)
    sym_sentiment_avg     = get_num(sym_data, 'sentiment_avg', 0)
    sym_whale_avg         = get_num(sym_data, 'whale_avg', 0)
    sym_profit_loss_ratio = get_num(sym_data, 'profit_loss_ratio', 0)
    sym_volume_trend      = get_num(sym_data, 'volume_trend', 0)
    sym_panic_avg         = get_num(sym_data, 'panic_avg', 0)
    sym_optimism_avg      = get_num(sym_data, 'optimism_avg', 0)
    sym_smart_stop_loss   = get_num(sym_data, 'smart_stop_loss', 0)
    sym_courage_boost     = get_num(sym_data, 'courage_boost', 0)
    sym_time_memory       = get_num(sym_data, 'time_memory', 0)
    sym_pattern_score     = get_num(sym_data, 'pattern_score', 0)
    sym_win_rate_boost    = get_num(sym_data, 'win_rate_boost', 0)

    # Context
    hours_held = get_num(trade, 'hours_held', 0)

    return [
        rsi, macd_diff, volume_ratio, price_momentum, atr,
        news_score, news_pos, news_neg, news_total, news_ratio, has_news,
        sent_score, fear_greed, fear_greed_norm, is_fearful, is_greedy,
        liq_score, depth_ratio, price_impact, good_liq,
        whale_activity, exchange_inflow,
        social_volume, market_sentiment,
        consensus, buy_count, sell_count,
        risk_score, opportunity, market_quality, momentum_strength, volatility_level,
        sym_win_rate, sym_avg_profit, sym_trap_count, sym_total, sym_is_reliable,
        sym_sentiment_avg, sym_whale_avg, sym_profit_loss_ratio, sym_volume_trend,
        sym_panic_avg, sym_optimism_avg, sym_smart_stop_loss,
        sym_courage_boost, sym_time_memory, sym_pattern_score, sym_win_rate_boost,
        hours_held,
    ]