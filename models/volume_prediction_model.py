"""
📊 Volume Prediction Model - نموذج التنبؤ بالحجم
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class VolumePredictor:
    def __init__(self):
        self.model = None

    def extract_features(self, data):
        """استخراج ميزات الحجم مع Feature Engineering"""
        if not isinstance(data, dict):
            data = {}

        volume       = float(data.get('volume', 0) or 0)
        volume_ratio = float(data.get('volume_ratio', 1) or 1)
        price_change = float(data.get('price_change_1h', 0) or 0)
        rsi          = float(data.get('rsi', 50) or 50)
        macd         = float(data.get('macd', 0) or 0)

        _vt = data.get('volume_trend', 0)
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

        # Feature Engineering
        volume_spike      = 1 if volume_ratio > 2.0 else 0
        volume_declining  = 1 if volume_ratio < 0.5 else 0
        high_momentum     = 1 if abs(price_change) > 3 else 0
        rsi_extreme       = 1 if rsi < 20 or rsi > 80 else 0
        bullish_volume    = 1 if volume_ratio > 1.5 and price_change > 0 else 0
        bearish_volume    = 1 if volume_ratio > 1.5 and price_change < 0 else 0
        volume_price_conf = volume_ratio * abs(price_change) / 100

        def _f(key, default=0):
            try:
                return float(data.get(key, default) or default)
            except (ValueError, TypeError):
                return float(default)

        return [
            volume,
            _f('volume_avg_1h',       volume),
            _f('volume_avg_4h',       volume),
            _f('volume_avg_24h',      volume),
            volume_ratio,
            _f('volume_ratio_4h',     1),
            _f('volume_ratio_24h',    1),
            volume_trend,
            _f('volume_volatility',   0),
            price_change,
            _f('price_change_4h',     0),
            _f('price_change_24h',    0),
            rsi,
            macd,
            _f('atr',                 0),
            _f('bid_ask_spread',      0),
            _f('volume_momentum',     0),
            _f('volume_acceleration', 0),
            volume_spike,
            volume_declining,
            high_momentum,
            rsi_extreme,
            bullish_volume,
            bearish_volume,
            volume_price_conf,
        ]

    @property
    def feature_names(self):
        return [
            'volume_current', 'volume_avg_1h', 'volume_avg_4h', 'volume_avg_24h',
            'volume_ratio_1h', 'volume_ratio_4h', 'volume_ratio_24h',
            'volume_trend', 'volume_volatility', 'price_change_1h',
            'price_change_4h', 'price_change_24h', 'rsi', 'macd',
            'atr', 'bid_ask_spread', 'volume_momentum', 'volume_acceleration',
            'volume_spike', 'volume_declining', 'high_momentum', 'rsi_extreme',
            'bullish_volume', 'bearish_volume', 'volume_price_conf',
        ]

    def train(self, trades, voting_scores=None):
        """تدريب نموذج التنبؤ بالحجم"""
        print("\n📊 Training Volume Prediction Model...")

        features_list, labels_list = [], []

        for trade in trades:
            try:
                data = trade.get('data', {})
                if isinstance(data, str):
                    data = json.loads(data)
                if not isinstance(data, dict):
                    data = {}

                profit = float(trade.get('profit_percent', 0) or 0)

                features_list.append(self.extract_features(data))
                labels_list.append(1 if profit > 0.8 else 0)
            except Exception as e:
                print(f"  ⚠️ Skipping trade: {e}")
                continue

        print(f"  Training samples: {len(features_list)} trades")

        if len(features_list) < 2:
            print("  ⚠️ Not enough data for Volume Prediction Model")
            return None

        X = pd.DataFrame(features_list, columns=self.feature_names)
        y = pd.Series(labels_list, name='target')

        pos = int(y.sum())
        neg = len(y) - pos
        print(f"    Label balance: {pos} positive | {neg} negative")

        stratify_param = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )

        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, self.model.predict(X_test))
        print(f"  ✅ Volume Prediction Model: Accuracy {accuracy * 100:.2f}%")
        return self.model, accuracy

    def predict(self, data):
        """Predict volume spike probability (0.0 - 1.0)."""
        if self.model is None:
            return 0.5
        if not isinstance(data, dict):
            return 0.5
        try:
            X     = pd.DataFrame([self.extract_features(data)], columns=self.feature_names)
            proba = self.model.predict_proba(X)[0]
            return float(proba[1]) if len(proba) > 1 else 0.5
        except Exception as e:
            print(f"  ⚠️ Volume predict error: {e}")
            return 0.5

    def get_volume_spike_probability(self, data):
        """Return spike probability as percentage (0-100)."""
        if not data:
            return 0.0
        return self.predict(data) * 100

    def detect_volume_anomaly(self, data):
        """Return True if an unusual volume anomaly is detected."""
        if not isinstance(data, dict):
            return False
        return (
            float(data.get('volume_ratio',     1) or 1) > 2.0 and
            float(data.get('volume_volatility', 0) or 0) > 0.5 and
            self.predict(data) > 0.7
        )


def train_volume_prediction_model(trades, voting_scores=None):
    predictor = VolumePredictor()
    result    = predictor.train(trades, voting_scores)
    if result:
        return result[0], result[1]
    return None