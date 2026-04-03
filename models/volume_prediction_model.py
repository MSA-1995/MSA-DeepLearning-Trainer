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
        volume        = data.get('volume', 0)
        volume_ratio  = data.get('volume_ratio', 1)
        volume_trend  = data.get('volume_trend', 0)
        rsi           = data.get('rsi', 50)
        macd          = data.get('macd', 0)
        price_change  = data.get('price_change_1h', 0)

        # Feature Engineering
        volume_spike      = 1 if volume_ratio > 2.0 else 0
        volume_declining  = 1 if volume_ratio < 0.5 else 0
        high_momentum     = 1 if abs(price_change) > 3 else 0
        rsi_extreme       = 1 if rsi < 20 or rsi > 80 else 0
        bullish_volume    = 1 if volume_ratio > 1.5 and price_change > 0 else 0
        bearish_volume    = 1 if volume_ratio > 1.5 and price_change < 0 else 0
        volume_price_conf = volume_ratio * abs(price_change) / 100

        return [
            volume, data.get('volume_avg_1h', volume), data.get('volume_avg_4h', volume),
            data.get('volume_avg_24h', volume), volume_ratio,
            data.get('volume_ratio_4h', 1), data.get('volume_ratio_24h', 1),
            volume_trend, data.get('volume_volatility', 0),
            price_change, data.get('price_change_4h', 0), data.get('price_change_24h', 0),
            rsi, macd, data.get('atr', 0), data.get('bid_ask_spread', 0),
            data.get('volume_momentum', 0), data.get('volume_acceleration', 0),
            # Feature Engineering
            volume_spike, volume_declining, high_momentum, rsi_extreme,
            bullish_volume, bearish_volume, volume_price_conf
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
            'bullish_volume', 'bearish_volume', 'volume_price_conf'
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
                features_list.append(self.extract_features(data))
                volume_ratio = data.get('volume_ratio', 1)
                profit = float(trade.get('profit_percent', 0))
                labels_list.append(1 if (volume_ratio > 1.5 and profit > 0.8) else 0)
            except:
                continue

        if len(features_list) < 50:
            print("⚠️ Not enough data for Volume Prediction Model")
            return None

        X = pd.DataFrame(features_list, columns=self.feature_names)
        y = pd.Series(labels_list, name='target')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        self.model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, self.model.predict(X_test))
        print(f"📊 Volume Prediction Model: Accuracy {accuracy*100:.2f}%")
        return self.model, accuracy

    def predict(self, data):
        if self.model is None:
            return 0.5
        X = pd.DataFrame([self.extract_features(data)], columns=self.feature_names)
        proba = self.model.predict_proba(X)[0]
        return proba[1] if len(proba) > 1 else 0.5

    def get_volume_spike_probability(self, data):
        return self.predict(data) * 100 if data else 0

    def detect_volume_anomaly(self, data):
        if not data:
            return False
        return (data.get('volume_ratio', 1) > 2.0 and
                data.get('volume_volatility', 0) > 0.5 and
                self.predict(data) > 0.7)


def train_volume_prediction_model(trades, voting_scores=None):
    predictor = VolumePredictor()
    result = predictor.train(trades, voting_scores)
    if result:
        return result[0], result[1]
    return None
