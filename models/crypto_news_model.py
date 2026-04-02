"""
📰 Crypto News Analysis Model - تحليل أخبار العملات الرقمية
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
        """استخراج ميزات الأخبار مع Feature Engineering"""
        news_count    = data.get('news_count_24h', 0)
        positive      = data.get('positive_news_count', 0)
        negative      = data.get('negative_news_count', 0)
        neutral       = data.get('neutral_news_count', 0)
        sentiment_avg = data.get('news_sentiment_avg', 0)
        breaking      = data.get('breaking_news_count', 0)
        partnership   = data.get('partnership_news', 0)
        regulation    = data.get('regulation_news', 0)

        # Feature Engineering
        pos_neg_ratio    = positive / (negative + 0.001)
        news_sentiment   = (positive - negative) / (news_count + 0.001)
        breaking_ratio   = breaking / (news_count + 0.001)
        regulation_risk  = 1 if regulation > 0 else 0
        partnership_boost = 1 if partnership > 0 else 0
        high_news_volume = 1 if news_count > 10 else 0
        strong_positive  = 1 if positive > negative * 2 else 0
        strong_negative  = 1 if negative > positive * 2 else 0

        return [
            news_count, positive, negative, neutral, sentiment_avg,
            data.get('news_volume_score', 0), breaking, partnership, regulation,
            data.get('technical_news', 0), data.get('market_news', 0),
            data.get('exchange_news', 0), data.get('news_recency_score', 0),
            data.get('news_source_reliability', 0.5),
            # Feature Engineering
            pos_neg_ratio, news_sentiment, breaking_ratio,
            regulation_risk, partnership_boost, high_news_volume,
            strong_positive, strong_negative
        ]

    @property
    def feature_names(self):
        return [
            'news_count_24h', 'positive_news_count', 'negative_news_count',
            'neutral_news_count', 'news_sentiment_avg', 'news_volume_score',
            'breaking_news_count', 'partnership_news', 'regulation_news',
            'technical_news', 'market_news', 'exchange_news',
            'news_recency_score', 'news_source_reliability',
            'pos_neg_ratio', 'news_sentiment', 'breaking_ratio',
            'regulation_risk', 'partnership_boost', 'high_news_volume',
            'strong_positive', 'strong_negative'
        ]

    def train(self, trades, voting_scores=None):
        """تدريب نموذج الأخبار"""
        print("\n📰 Training Crypto News Model...")

        features_list, labels_list = [], []
        skipped_no_data = 0
        
        for trade in trades:
            try:
                data = trade.get('data', {})
                if isinstance(data, str):
                    data = json.loads(data)
                news_data = data.get('news', {})
                
                # تخطي الصفقات بدون بيانات أخبار حقيقية
                if not news_data or news_data.get('news_count_24h', 0) == 0:
                    skipped_no_data += 1
                    continue
                
                features_list.append(self.extract_features(news_data))
                profit = float(trade.get('profit_percent', 0))
                labels_list.append(1 if profit > 0.8 else 0)
            except:
                continue
        
        print(f"  📊 Training samples: {len(features_list)} trades (skipped {skipped_no_data} without news data)")

        if len(features_list) < 50:
            print("⚠️ Not enough data for Crypto News Model")
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
        print(f"📰 Crypto News Model: Accuracy {accuracy*100:.2f}%")
        return self.model, accuracy

    def predict(self, news_data):
        if self.model is None:
            return 0.5
        X = pd.DataFrame([self.extract_features(news_data)], columns=self.feature_names)
        proba = self.model.predict_proba(X)[0]
        return proba[1] if len(proba) > 1 else 0.5

    def get_news_impact_score(self, news_data):
        if not news_data:
            return 0
        score = 0
        news_count = news_data.get('news_count_24h', 0)
        if news_count > 10:
            score += 20
        elif news_count > 5:
            score += 10
        positive = news_data.get('positive_news_count', 0)
        negative = news_data.get('negative_news_count', 0)
        score += (positive - negative) * 10
        score += news_data.get('breaking_news_count', 0) * 15
        score += news_data.get('partnership_news', 0) * 20
        score -= news_data.get('regulation_news', 0) * 25
        score += news_data.get('news_sentiment_avg', 0) * 30
        return max(-100, min(100, score))


def train_crypto_news_model(trades, voting_scores=None):
    analyzer = CryptoNewsAnalyzer()
    result = analyzer.train(trades, voting_scores)
    if result:
        return result[0], result[1]
    return None
