"""
🎭 Sentiment Analysis Model - تحليل مشاعر السوق
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

    def extract_features(self, data):
        """استخراج ميزات المشاعر مع Feature Engineering"""
        positive      = data.get('positive_ratio', 0.33)
        negative      = data.get('negative_ratio', 0.33)
        neutral       = data.get('neutral_ratio', 0.34)
        news_sentiment = data.get('news_sentiment', 0)
        fear_greed    = data.get('fear_greed_index', 50)
        social_volume = data.get('social_volume', 0)

        # Feature Engineering
        pos_neg_ratio   = positive / (negative + 0.001)
        sentiment_score = positive - negative
        fear_greed_norm = (fear_greed - 50) / 50
        is_fearful      = 1 if fear_greed < 30 else 0
        is_greedy       = 1 if fear_greed > 70 else 0
        high_social     = 1 if social_volume > 1000 else 0
        strong_positive = 1 if positive > 0.6 else 0
        strong_negative = 1 if negative > 0.6 else 0

        return [
            fear_greed, social_volume, positive, negative, neutral,
            data.get('trending_score', 0), news_sentiment,
            data.get('reddit_sentiment', 0), data.get('twitter_sentiment', 0),
            data.get('btc_dominance', 50), data.get('market_cap_change_24h', 0),
            data.get('volume_24h_change', 0),
            pos_neg_ratio, sentiment_score, fear_greed_norm,
            is_fearful, is_greedy, high_social, strong_positive, strong_negative
        ]

    @property
    def feature_names(self):
        return [
            'fear_greed_index', 'social_volume', 'positive_ratio',
            'negative_ratio', 'neutral_ratio', 'trending_score',
            'news_sentiment', 'reddit_sentiment', 'twitter_sentiment',
            'btc_dominance', 'market_cap_change', 'volume_24h_change',
            'pos_neg_ratio', 'sentiment_score', 'fear_greed_norm',
            'is_fearful', 'is_greedy', 'high_social', 'strong_positive', 'strong_negative'
        ]

    def train(self, trades, voting_scores=None):
        """تدريب نموذج المشاعر"""
        print("\n🎭 Training Sentiment Analysis Model...")

        features_list, labels_list = [], []
        skipped_no_data = 0
        
        for trade in trades:
            try:
                data = trade.get('data', {})
                if isinstance(data, str):
                    data = json.loads(data)
                sentiment_data = data.get('sentiment', {})
                
                # تخطي الصفقات بدون بيانات مشاعر حقيقية
                if not sentiment_data or sentiment_data.get('fear_greed_index', 50) == 50 and sentiment_data.get('positive_ratio', 0.33) == 0.33:
                    skipped_no_data += 1
                    continue
                
                features_list.append(self.extract_features(sentiment_data))
                profit = float(trade.get('profit_percent', 0))
                labels_list.append(1 if profit > 0.8 else 0)
            except:
                continue
        
        print(f"  📊 Training samples: {len(features_list)} trades (skipped {skipped_no_data} without sentiment data)")

        if len(features_list) < 50:
            print("⚠️ Not enough data for Sentiment Model")
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
        print(f"🎭 Sentiment Model: Accuracy {accuracy*100:.2f}%")
        return self.model, accuracy

    def predict(self, sentiment_data):
        if self.model is None:
            return 0.5
        X = pd.DataFrame([self.extract_features(sentiment_data)], columns=self.feature_names)
        proba = self.model.predict_proba(X)[0]
        return proba[1] if len(proba) > 1 else 0.5

    def get_sentiment_score(self, sentiment_data):
        if not sentiment_data:
            return 0
        score = 0
        fear_greed = sentiment_data.get('fear_greed_index', 50)
        if fear_greed < 30:
            score -= 30
        elif fear_greed > 70:
            score += 30
        else:
            score += (fear_greed - 50) * 0.6
        positive = sentiment_data.get('positive_ratio', 0.33)
        negative = sentiment_data.get('negative_ratio', 0.33)
        score += (positive - negative) * 50
        score += sentiment_data.get('news_sentiment', 0) * 20
        if sentiment_data.get('social_volume', 0) > 1000:
            score += 10
        return max(-100, min(100, score))


def train_sentiment_model(trades, voting_scores=None):
    analyzer = SentimentAnalyzer()
    result = analyzer.train(trades, voting_scores)
    if result:
        return result[0], result[1]
    return None
