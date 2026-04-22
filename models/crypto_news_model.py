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
        if not isinstance(data, dict):
            data = {}

        news = data.get('news', {}) or data.get('news_analysis', {})
        if not isinstance(news, dict):
            news = {}

        def _f(d, key, default=0):
            try:
                return float(d.get(key, default) or default)
            except (ValueError, TypeError):
                return float(default)

        news_count    = _f(news, 'news_count_24h', _f(news, 'total', 0))
        positive      = _f(news, 'positive_news_count', _f(news, 'positive', 0))
        negative      = _f(news, 'negative_news_count', _f(news, 'negative', 0))
        neutral       = _f(news, 'neutral_news_count',  _f(news, 'neutral',  0))
        sentiment_avg = _f(news, 'news_sentiment_avg',  _f(news, 'news_score', 0))

        # Feature Engineering
        pos_neg_ratio    = positive / (negative + 0.001)
        news_sentiment   = (positive - negative) / (news_count + 0.001)
        high_news_volume = 1 if news_count > 5 else 0
        strong_positive  = 1 if positive > negative * 2 else 0
        strong_negative  = 1 if negative > positive * 2 else 0

        return [
            news_count, positive, negative, neutral, sentiment_avg,
            pos_neg_ratio, news_sentiment,
            high_news_volume, strong_positive, strong_negative,
        ]

    @property
    def feature_names(self):
        return [
            'news_count_24h', 'positive_news_count', 'negative_news_count',
            'neutral_news_count', 'news_sentiment_avg',
            'pos_neg_ratio', 'news_sentiment',
            'high_news_volume', 'strong_positive', 'strong_negative',
        ]

    def train(self, trades, voting_scores=None, since_timestamp=None):
        """Train crypto news model from trades data"""
        print("\n📰 Training Crypto News Model (from trades)...")

        features_list, labels_list = [], []
        skipped = 0

        for trade in trades:
            try:
                data = trade.get('data', {})
                if isinstance(data, str):
                    data = json.loads(data)
                if not isinstance(data, dict):
                    data = {}

                # ✅ معالجة البيانات المتداخلة
                if 'data' in data and isinstance(data.get('data'), dict):
                    data = data['data']

                # ✅ تخطي إذا لم تكن هناك بيانات أخبار
                news = data.get('news', {}) or data.get('news_analysis', {})
                if not news:
                    skipped += 1
                    continue

                features_list.append(self.extract_features(data))
                profit = float(trade.get('profit_percent', 0) or 0)
                labels_list.append(1 if profit > 0.8 else 0)

            except Exception as e:
                print(f"  ⚠️ Skipping trade: {e}")
                continue

        print(f"  Training samples: {len(features_list)} (skipped {skipped} without news)")

        if len(features_list) < 30:
            print("  ⚠️ Not enough data for Crypto News Model")
            return None

        X = pd.DataFrame(features_list, columns=self.feature_names)
        y = pd.Series(labels_list, name='target')

        stratify_param = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )

        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, self.model.predict(X_test))
        print(f"  ✅ Crypto News Model: Accuracy {accuracy * 100:.2f}%")
        return self.model, accuracy

    def predict(self, data):
        """Predict news sentiment probability (0.0 - 1.0)."""
        if self.model is None:
            return 0.5
        if not isinstance(data, dict):
            return 0.5
        try:
            X     = pd.DataFrame([self.extract_features(data)], columns=self.feature_names)
            proba = self.model.predict_proba(X)[0]
            return float(proba[1]) if len(proba) > 1 else 0.5
        except Exception as e:
            print(f"  ⚠️ News predict error: {e}")
            return 0.5


def train_crypto_news_model(trades, voting_scores=None, since_timestamp=None):
    analyzer = CryptoNewsAnalyzer()
    result   = analyzer.train(trades, voting_scores, since_timestamp=since_timestamp)
    if result:
        return result[0], result[1]
    return None