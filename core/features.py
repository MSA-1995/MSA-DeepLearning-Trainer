"""
🔧 Feature Engineering
Calculates technical indicators used as model input features.
"""


def calculate_enhanced_features(data):
    """Feature Engineering: حساب مؤشرات إضافية (15 feature)"""
    try:
        rsi          = data.get('rsi', 50)
        macd         = data.get('macd', 0)
        volume_ratio = data.get('volume_ratio', 1)
        price_momentum = data.get('price_momentum', 0)

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
        atr            = data.get('atr', atr_estimate)
        ema_9          = data.get('ema_9', 0)
        ema_21         = data.get('ema_21', 0)
        ema_crossover  = 1 if ema_9 > ema_21 else -1
        bid_ask_spread = data.get('bid_ask_spread', 0)
        volume_trend   = data.get('volume_trend', 0)
        price_change_1h = data.get('price_change_1h', 0)

        return [
            rsi, macd, volume_ratio, price_momentum,
            bb_position, atr_estimate, stochastic, ema_signal,
            volume_strength, momentum_strength,
            atr, ema_crossover, bid_ask_spread, volume_trend, price_change_1h
        ]
    except:
        return [50, 0, 1, 0, 0.5, 1, 50, 0, 1, 0, 1, 0, 0, 0, 0]
