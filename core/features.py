"""
🔧 Feature Engineering
Calculates technical indicators used as model input features.
"""

from datetime import datetime


def calculate_enhanced_features(data, trade=None):
    """
    Feature Engineering: حساب مؤشرات إضافية (38 feature)
    
    الميزات التقليدية (15):
    - RSI, MACD, Volume, Momentum, ATR, EMA, etc.
    
    الميزات من التعلم المباشر (6):
    - trade_quality_score: تقييم الجودة (1-5)
    - advisor_vote_consensus: توافق المستشارين (0-1)
    - is_trap_trade: هل كانت صفقة فخ (0/1)
    - profit_magnitude: حجم الربح/الخسارة
    - hours_held_normalized: ساعات الاحتفاظ (0-2)
    - is_profitable: هل الربح إيجابي (0/1)
    
    الميزات من السوق والوقت (7):
    - btc_trend_1h: اتجاه BTC في الساعة الماضية
    - is_bullish_market: هل السوق صاعد (0/1)
    - hour_of_day: ساعة الدخول (0-23)
    - is_asian/european/us_session: جلسة التداول
    - optimal_hold_score: درجة وقت الاحتفاظ المثالي
    - fib_score: قوة مستوى فيبوناتشي
    - fib_level_encoded: المستوى المشفر (0-6)
    
    الميزات من Market Regime (4):
    - regime_score: نقاط حالة السوق (0-1)
    - regime_adx: قوة الترند (0-1)
    - volatility_ratio: نسبة التقلبات
    - position_multiplier: مضاعف حجم المركز
    
    الميزات من Flash Crash Protection (4):
    - flash_risk_score: درجة خطر السقوط (0-1)
    - flash_crash_detected: تم اكتشاف سقوط مفاجئ (0/1)
    - whale_dump_detected: تم اكتشاف بيع حيتان (0/1)
    - cascade_risk_score: درجة خطر التصفية السلسلة (0-1)
    
    Args:
        data: قاموس البيانات التقنية (RSI, MACD, etc.)
        trade: بيانات الصفقة الكاملة (اختياري)
    """
    try:
        # دمج البيانات إذا كان trade متوفراً
        if trade:
            trade_data = trade.get('data', {})
            if isinstance(trade_data, str):
                import json
                trade_data = json.loads(trade_data)
            full_data = {**trade, **trade_data, **data}
        else:
            full_data = data
        
        rsi          = full_data.get('rsi', 50)
        macd         = full_data.get('macd_diff', full_data.get('macd', 0))  # 🔧 fallback
        volume_ratio = full_data.get('volume_ratio', 1)
        price_momentum = full_data.get('price_momentum', 0)

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
        atr            = full_data.get('atr', atr_estimate)
        ema_9          = full_data.get('ema_9', 0)
        ema_21         = full_data.get('ema_21', 0)
        ema_crossover  = 1 if ema_9 > ema_21 else -1
        bid_ask_spread = full_data.get('bid_ask_spread', 0)
        _vt = full_data.get('volume_trend', 0)
        volume_trend = 1.2 if _vt == 'up' else (0.8 if _vt == 'down' else (0.0 if _vt == 'neutral' else float(_vt or 0)))  # 🔧 handle string
        price_change_1h = full_data.get('price_change_1h', 0)
        
        # =========================================================
        # 🎓 الميزات الجديدة من التعلم المباشر
        # =========================================================
        
        # 1. تقييم الجودة (trade_quality)
        trade_quality = full_data.get('trade_quality', 'OK')
        quality_map = {'TRAP': 1, 'RISKY': 2, 'OK': 3, 'GOOD': 4, 'GREAT': 5}
        trade_quality_score = quality_map.get(trade_quality, 3)
        
        # 2. توافق المستشارين (advisor consensus)
        advisor_votes = full_data.get('advisor_votes', {})
        if isinstance(advisor_votes, str):
            import json
            advisor_votes = json.loads(advisor_votes)
        if advisor_votes and isinstance(advisor_votes, dict):
            vote_count = sum(1 for v in advisor_votes.values() if v == 1)
            total_votes = len(advisor_votes)
            advisor_vote_consensus = vote_count / total_votes if total_votes > 0 else 0.5
        else:
            advisor_vote_consensus = 0.5
        
        # 3. هل كانت صفقة فخ (is_trap)
        is_trap_trade = 1 if trade_quality in ['TRAP', 'RISKY'] else 0
        
        # 4. حجم الربح/الخسارة (profit magnitude)
        profit_percent = full_data.get('profit_percent', 0)
        profit_magnitude = abs(profit_percent) / 10.0
        
        # 5. ساعة الاحتفاظ (hours held)
        hours_held = full_data.get('hours_held', 24)
        hours_held_normalized = min(hours_held / 48.0, 2.0)
        
        # 6. هل الربح إيجابي (is_profitable)
        is_profitable = 1 if profit_percent > 0 else 0
        
        # =========================================================
        # 🌍 الميزات الجديدة: سياق السوق والوقت
        # =========================================================
        
        # 7. اتجاه BTC (market context)
        btc_trend = full_data.get('btc_change_1h', full_data.get('btc_trend_1h', 0))
        btc_trend_normalized = max(-1.0, min(1.0, btc_trend / 5.0))  # Normalize -5% to +5% => -1 to 1
        
        # 8. هل السوق صاعد (market mood)
        is_bullish_market = 1 if btc_trend > 1.0 else 0
        
        # 9. ساعة الدخول (time-based)
        timestamp = full_data.get('timestamp')
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp
                hour_of_day = dt.hour
            except:
                hour_of_day = 12
        else:
            hour_of_day = 12
        
        # تحويل الساعة إلى قيمة م нормализована (0-1 cycle)
        hour_normalized = hour_of_day / 24.0
        
        # 10. جلسة التداول (session)
        is_asian_session = 1 if 0 <= hour_of_day <= 8 else 0
        is_european_session = 1 if 8 < hour_of_day <= 16 else 0
        is_us_session = 1 if 16 < hour_of_day <= 24 else 0
        
        # 11. درجة الوقت المثالي للبيع (optimal hold)
        # الصفقات الناجحة تستمر أكثر، الخاسرة تبيع بسرعة
        if trade_quality in ['GREAT', 'GOOD']:
            optimal_hold_score = 1.0 if hours_held > 12 else 0.5
        elif trade_quality in ['TRAP', 'RISKY']:
            optimal_hold_score = 1.0 if hours_held < 4 else 0.3
        else:
            optimal_hold_score = 0.5
        
        # =========================================================
        # 📊 الميزات الجديدة: فيبوناتشي
        # =========================================================
        
        # 12. Fibonacci score (قوة مستوى فيبوناتشي)
        fib_score = full_data.get('fib_score', 0) or 0
        decision_factors = full_data.get('decision_factors', {})
        if isinstance(decision_factors, str):
            import json
            decision_factors = json.loads(decision_factors)
        fib_score_from_decision = decision_factors.get('fib_score', 0) if decision_factors else 0
        fib_score = max(fib_score, fib_score_from_decision)
        
        # 13. Fibonacci level (المستوى المشفر)
        fib_level = full_data.get('fib_level') or (decision_factors.get('fib_level') if decision_factors else None)
        fib_level_map = {'0': 0, '23.6': 1, '38.2': 2, '50': 3, '61.8': 4, '78.6': 5, '100': 6}
        fib_level_encoded = fib_level_map.get(fib_level, 0) if fib_level else 0

        # =========================================================
        # 🎯 الميزات الجديدة: Market Regime Detection
        # =========================================================
        
        # 14. Market Regime (حالة السوق)
        market_regime = full_data.get('market_regime', {})
        if isinstance(market_regime, str):
            import json
            market_regime = json.loads(market_regime)
        
        regime_map = {
            'STRONG_UPTREND': 1.0,
            'WEAK_TREND': 0.7,
            'RANGING': 0.5,
            'LOW_VOLATILITY': 0.4,
            'HIGH_VOLATILITY': 0.3,
            'STRONG_DOWNTREND': 0.0,
            'UNKNOWN': 0.5
        }
        regime_score = regime_map.get(market_regime.get('regime', 'UNKNOWN'), 0.5)
        regime_adx = market_regime.get('adx', 20) / 50.0  # Normalize 0-50 to 0-1
        volatility_ratio = market_regime.get('volatility_ratio', 1.0)
        position_multiplier = market_regime.get('trading_advice', {}).get('position_size', 1.0)
        
        # =========================================================
        # 🚨 الميزات الجديدة: Flash Crash Protection
        # =========================================================
        
        # 15. Flash Crash Protection (حماية السقوط المفاجئ)
        flash_crash = full_data.get('flash_crash_protection', {})
        if isinstance(flash_crash, str):
            import json
            flash_crash = json.loads(flash_crash)
        
        flash_risk_score = flash_crash.get('risk_score', 0) / 100.0  # Normalize 0-100 to 0-1
        flash_crash_detected = 1 if flash_crash.get('flash_crash_detected', False) else 0
        whale_dump_detected = 1 if flash_crash.get('whale_dump_detected', False) else 0
        cascade_risk_score = flash_crash.get('cascade_risk', {}).get('score', 0) / 100.0

        # Whale Confidence (normalize -25 to 25 to -1 to 1)
        whale_confidence = (trade.get('whale_confidence', 0) / 25.0) if trade else 0

        # Additional Data
        atr_value = trade.get('atr_value', 0) if trade else 0
        sentiment_score = trade.get('sentiment_score', 0) if trade else 0
        panic_score = trade.get('panic_score', 0) if trade else 0
        optimism_penalty = trade.get('optimism_penalty', 0) if trade else 0
        
        return [
            # 15 الميزات التقليدية
            rsi, macd, volume_ratio, price_momentum,
            bb_position, atr_estimate, stochastic, ema_signal,
            volume_strength, momentum_strength,
            atr, ema_crossover, bid_ask_spread, volume_trend, price_change_1h,
            # 6 ميزات من التعلم المباشر
            trade_quality_score,    # 16
            advisor_vote_consensus, # 17
            is_trap_trade,          # 18
            profit_magnitude,       # 19
            hours_held_normalized,  # 20
            is_profitable,          # 21
            # 7 ميزات جديدة: السوق والوقت
            btc_trend_normalized,   # 22
            is_bullish_market,      # 23
            hour_normalized,        # 24
            is_asian_session,       # 25
            is_european_session,    # 26
            is_us_session,          # 27
            optimal_hold_score,     # 28
            # 2 ميزات فيبوناتشي
            fib_score,              # 29
            fib_level_encoded,      # 30
            # 4 ميزات Market Regime (جديد)
            regime_score,           # 31
            regime_adx,             # 32
            volatility_ratio,       # 33
            position_multiplier,    # 34
            # 4 ميزات Flash Crash (جديد)
            flash_risk_score,       # 35
            flash_crash_detected,   # 36
            whale_dump_detected,    # 37
            cascade_risk_score,     # 38
            whale_confidence,      # 39
            atr_value,             # 40
            sentiment_score,       # 41
            panic_score,           # 42
            optimism_penalty       # 43
        ]
    except Exception as e:
        print(f"⚠️ Feature calculation error: {e}")
        # 43 features fallback (must match return list above exactly)
        return [
            # Technical (15)
            50, 0, 1, 0, 0.5, 1, 50, 0, 1, 0, 1, 0, 0, 0, 0,
            # Learning (6)
            3, 0.5, 0, 0, 0.5, 1,
            # Market/Time (7)
            0, 0, 0.5, 0, 0, 0, 0.5,
            # Fibonacci (2)
            0, 0,
            # Regime (4)
            0.5, 0.4, 1.0, 1.0,
            # Flash Crash (4)
            0, 0, 0, 0,
            # Whale + Additional (5)
            0, 0, 0, 0, 0
        ]


def get_feature_names():
    """أسماء الميزات للـ model interpretability"""
    return [
        # التقليدية (15)
        'rsi', 'macd', 'volume_ratio', 'price_momentum',
        'bb_position', 'atr_estimate', 'stochastic', 'ema_signal',
        'volume_strength', 'momentum_strength',
        'atr', 'ema_crossover', 'bid_ask_spread', 'volume_trend', 'price_change_1h',
        # التعلم المباشر (6)
        'trade_quality_score',
        'advisor_vote_consensus',
        'is_trap_trade',
        'profit_magnitude',
        'hours_held_normalized',
        'is_profitable',
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
        # Whale Confidence (1)
        'whale_confidence',
        # Additional Data (4)
        'atr_value',
        'sentiment_score',
        'panic_score',
        'optimism_penalty'
    ]
