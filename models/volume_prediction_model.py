"""
📊 Volume Prediction Model - نموذج التنبؤ بالحجم
يتنبأ بحجم التداول المستقبلي لاكتشاف الانفجارات
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from datetime import datetime, timedelta


class VolumePredictor:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'volume_current', 'volume_avg_1h', 'volume_avg_4h', 'volume_avg_24h',
            'volume_ratio_1h', 'volume_ratio_4h', 'volume_ratio_24h',
            'volume_trend', 'volume_volatility', 'price_change_1h',
            'price_change_4h', 'price_change_24h', 'rsi', 'macd',
            'atr', 'bid_ask_spread', 'volume_momentum', 'volume_acceleration'
        ]
    
    def extract_features(self, data):
        """استخراج ميزات الحجم"""
        features = []
        
        # 1. الحجم الحالي
        volume_current = data.get('volume', 0)
        features.append(volume_current)
        
        # 2. متوسط الحجم لساعة واحدة
        volume_avg_1h = data.get('volume_avg_1h', volume_current)
        features.append(volume_avg_1h)
        
        # 3. متوسط الحجم لـ 4 ساعات
        volume_avg_4h = data.get('volume_avg_4h', volume_current)
        features.append(volume_avg_4h)
        
        # 4. متوسط الحجم لـ 24 ساعة
        volume_avg_24h = data.get('volume_avg_24h', volume_current)
        features.append(volume_avg_24h)
        
        # 5. نسبة الحجم لساعة واحدة
        volume_ratio_1h = volume_current / volume_avg_1h if volume_avg_1h > 0 else 1
        features.append(volume_ratio_1h)
        
        # 6. نسبة الحجم لـ 4 ساعات
        volume_ratio_4h = volume_current / volume_avg_4h if volume_avg_4h > 0 else 1
        features.append(volume_ratio_4h)
        
        # 7. نسبة الحجم لـ 24 ساعة
        volume_ratio_24h = volume_current / volume_avg_24h if volume_avg_24h > 0 else 1
        features.append(volume_ratio_24h)
        
        # 8. اتجاه الحجم
        volume_trend = data.get('volume_trend', 0)
        features.append(volume_trend)
        
        # 9. تقلب الحجم
        volume_volatility = data.get('volume_volatility', 0)
        features.append(volume_volatility)
        
        # 10. تغير السعر لساعة واحدة
        price_change_1h = data.get('price_change_1h', 0)
        features.append(price_change_1h)
        
        # 11. تغير السعر لـ 4 ساعات
        price_change_4h = data.get('price_change_4h', 0)
        features.append(price_change_4h)
        
        # 12. تغير السعر لـ 24 ساعة
        price_change_24h = data.get('price_change_24h', 0)
        features.append(price_change_24h)
        
        # 13. RSI
        rsi = data.get('rsi', 50)
        features.append(rsi)
        
        # 14. MACD
        macd = data.get('macd', 0)
        features.append(macd)
        
        # 15. ATR
        atr = data.get('atr', 0)
        features.append(atr)
        
        # 16. Bid-Ask Spread
        bid_ask_spread = data.get('bid_ask_spread', 0)
        features.append(bid_ask_spread)
        
        # 17. زخم الحجم
        volume_momentum = data.get('volume_momentum', 0)
        features.append(volume_momentum)
        
        # 18. تسارع الحجم
        volume_acceleration = data.get('volume_acceleration', 0)
        features.append(volume_acceleration)
        
        return features
    
    def train(self, trades, voting_scores=None):
        """تدريب نموذج التنبؤ بالحجم"""
        print("\n📊 Training Volume Prediction Model...")
        
        features_list = []
        labels_list = []
        
        for trade in trades:
            try:
                data = trade.get('data', {})
                if isinstance(data, str):
                    data = json.loads(data)
                
                # استخراج ميزات الحجم
                features = self.extract_features(data)
                features_list.append(features)
                
                # التسمية: 1 إذا كان هناك زيادة حجم (>1.5x) وربح > 0.8%
                volume_ratio = data.get('volume_ratio', 1)
                profit = float(trade.get('profit_percent', 0))
                
                # نموذج التصنيف: هل سيكون هناك انفجار حجم؟
                is_volume_spike = 1 if (volume_ratio > 1.5 and profit > 0.8) else 0
                labels_list.append(is_volume_spike)
                
            except Exception as e:
                continue
        
        if len(features_list) < 50:
            print("⚠️ Not enough data for Volume Prediction Model")
            return None
        
        # تحويل إلى DataFrame
        X = pd.DataFrame(features_list, columns=self.feature_names)
        y = pd.Series(labels_list, name='target')
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # تدريب النموذج
        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # تقييم الأداء
        accuracy = accuracy_score(y_test, self.model.predict(X_test))
        print(f"📊 Volume Prediction Model: Accuracy {accuracy*100:.2f}%")
        
        return self.model, accuracy
    
    def predict(self, data):
        """التنبؤ بحجم التداول المستقبلي"""
        if self.model is None:
            return 0.5
        
        features = self.extract_features(data)
        X = pd.DataFrame([features], columns=self.feature_names)
        
        # احتمالية زيادة الحجم
        proba = self.model.predict_proba(X)[0]
        return proba[1] if len(proba) > 1 else 0.5
    
    def get_volume_spike_probability(self, data):
        """حساب احتمالية زيادة الحجم (0-100)"""
        if not data:
            return 0
        
        prob = self.predict(data)
        return prob * 100
    
    def detect_volume_anomaly(self, data):
        """كشف الشذوذ في الحجم"""
        if not data:
            return False
        
        volume_ratio = data.get('volume_ratio', 1)
        volume_volatility = data.get('volume_volatility', 0)
        
        # شروط الشذوذ:
        # 1. نسبة الحجم > 2x
        # 2. تقلب الحجم > 0.5
        # 3. احتمالية النموذج > 70%
        
        is_anomaly = (
            volume_ratio > 2.0 and
            volume_volatility > 0.5 and
            self.predict(data) > 0.7
        )
        
        return is_anomaly


def train_volume_prediction_model(trades, voting_scores=None):
    """دالة تدريب نموذج التنبؤ بالحجم"""
    predictor = VolumePredictor()
    result = predictor.train(trades, voting_scores)
    
    if result:
        model, accuracy = result
        return model, accuracy
    return None
