"""
Sentiment Analysis Model - reads from trades_history data
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class SentimentAnalyzer:
    def __init__(self):
        self.model = None

    def extract_features(self, data, trade):
        """Extract sentiment features from trade data"""
        # ✅ الربط مع أعمدة البوت الحقيقية (sentiment_score, panic_score)
        # التحويل من مقياس البوت (-10 to 10) إلى مقياس الموديل (0 to 100)
        raw_sent = float(trade.get('sentiment_score', 0) or 0)
        fear_greed = (raw_sent * 5) + 50  # Convert -10/+10 to 0-100
        panic_score = float(trade.get('panic_score', 0) or 0)
        
        sentiment = data.get('sentiment', {})
        positive = sentiment.get('positive_ratio', 0.33)
        negative = sentiment.get('negative_ratio', 0.33)
        neutral = sentiment.get('neutral_ratio', 0.34)
        news_sentiment = sentiment.get('news_sentiment', 0)
        social_volume = sentiment.get('social_volume', 1000)
        trending_score = sentiment.get('trending_score', 0)

        # Feature Engineering
        pos_neg_ratio = positive / (negative + 0.001)
        sentiment_score = positive - negative
        fear_greed_norm = (fear_greed - 50) / 50
        is_fearful = 1 if (fear_greed < 30 or panic_score > 7) else 0
        is_greedy = 1 if fear_greed > 70 else 0
        high_social = 1 if social_volume > 1000 else 0
        strong_positive = 1 if positive > 0.6 else 0
        strong_negative = 1 if negative > 0.6 else 0

        return [
            fear_greed, social_volume, positive, negative, neutral,
            trending_score, news_sentiment,
            pos_neg_ratio, sentiment_score, fear_greed_norm,
            is_fearful, is_greedy, high_social, strong_positive, strong_negative
        ]

    @property
    def feature_names(self):
        return [
            'fear_greed_index', 'social_volume', 'positive_ratio',
            'negative_ratio', 'neutral_ratio', 'trending_score',
            'news_sentiment', 'pos_neg_ratio', 'sentiment_score',
            'fear_greed_norm', 'is_fearful', 'is_greedy',
            'high_social', 'strong_positive', 'strong_negative'
        ]

    def train(self, trades, voting_scores=None, since_timestamp=None):
        """Train sentiment model from trades data"""
        print("\nTraining Sentiment Model (from trades)...")

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
                
                # ✅ لا نتخطى الصفقة أبداً إذا كان هناك عمود sentiment_score (حتى لو 0)
                # هذا يحل مشكلة الـ 0 samples
                if trade.get('sentiment_score') is None and not data.get('sentiment'):
                    skipped += 1
                    continue

                features_list.append(self.extract_features(data, trade))
                profit = float(trade.get('profit_percent', 0))
                labels_list.append(1 if profit > 0.8 else 0)
            except:
                continue
        
        print(f"  Training samples: {len(features_list)} (skipped {skipped} without sentiment)")
        
        if len(features_list) < 50:
            print("Not enough data for Sentiment Model")
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
        print(f"Sentiment Model: Accuracy {accuracy*100:.2f}%")
        return self.model, accuracy

    def predict(self, sentiment_data):
        if self.model is None:
            return 0.5
        X = pd.DataFrame([self.extract_features(sentiment_data)], columns=self.feature_names)
        proba = self.model.predict_proba(X)[0]
        return proba[1] if len(proba) > 1 else 0.5


def train_sentiment_model(trades, voting_scores=None, since_timestamp=None):
    analyzer = SentimentAnalyzer()
    result = analyzer.train(trades, voting_scores, since_timestamp=since_timestamp)
    if result:
        return result[0], result[1]
    return None
