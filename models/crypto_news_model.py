"""
Crypto News Analysis Model - reads from trades_history data
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class CryptoNewsAnalyzer:
    def __init__(self):
        self.model = None

    def extract_features(self, data):
        """Extract news features from trade data"""
        news = data.get('news', {})
        
        news_count = news.get('news_count_24h', news.get('total', 0))
        positive = news.get('positive_news_count', news.get('positive', 0))
        negative = news.get('negative_news_count', news.get('negative', 0))
        neutral = news.get('neutral_news_count', news.get('neutral', 0))
        sentiment_avg = news.get('news_sentiment_avg', news.get('news_score', 0))

        # Feature Engineering
        pos_neg_ratio = positive / (negative + 0.001)
        news_sentiment = (positive - negative) / (news_count + 0.001)
        high_news_volume = 1 if news_count > 5 else 0
        strong_positive = 1 if positive > negative * 2 else 0
        strong_negative = 1 if negative > positive * 2 else 0

        return [
            news_count, positive, negative, neutral, sentiment_avg,
            pos_neg_ratio, news_sentiment,
            high_news_volume, strong_positive, strong_negative
        ]

    @property
    def feature_names(self):
        return [
            'news_count_24h', 'positive_news_count', 'negative_news_count',
            'neutral_news_count', 'news_sentiment_avg',
            'pos_neg_ratio', 'news_sentiment',
            'high_news_volume', 'strong_positive', 'strong_negative'
        ]

    def train(self, trades, voting_scores=None, since_timestamp=None):
        """Train crypto news model from trades data"""
        print("\nTraining Crypto News Model (from trades)...")

        features_list, labels_list = [], []
        skipped = 0
        
        for trade in trades:
            try:
                data = trade.get('data', {})
                if isinstance(data, str):
                    data = json.loads(data)
                
                # ✅ معالجة البيانات المتداخلة (Nested JSON Fix)
                if isinstance(data, dict) and 'data' in data and isinstance(data.get('data'), dict):
                    data = data['data']
                
                # ✅ التحقق من وجود بيانات الأخبار في الـ JSON المتداخل أو الأعمدة
                news = data.get('news', {}) or data.get('news_analysis', {})
                
                features_list.append(self.extract_features(data))
                profit = float(trade.get('profit_percent', 0))
                labels_list.append(1 if profit > 0.8 else 0)
            except:
                continue
        
        print(f"  Training samples: {len(features_list)} (skipped {skipped} without news)")
        
        if len(features_list) < 30:
            print("Not enough data for Crypto News Model")
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
        print(f"Crypto News Model: Accuracy {accuracy*100:.2f}%")
        return self.model, accuracy

    def predict(self, news_data):
        if self.model is None:
            return 0.5
        X = pd.DataFrame([self.extract_features(news_data)], columns=self.feature_names)
        proba = self.model.predict_proba(X)[0]
        return proba[1] if len(proba) > 1 else 0.5


def train_crypto_news_model(trades, voting_scores=None, since_timestamp=None):
    analyzer = CryptoNewsAnalyzer()
    result = analyzer.train(trades, voting_scores, since_timestamp=since_timestamp)
    if result:
        return result[0], result[1]
    return None
