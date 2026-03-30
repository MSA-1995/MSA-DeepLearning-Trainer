"""
🎭 Sentiment Analysis Model - تحليل مشاعر السوق
يحلل مشاعر السوق من مصادر متعددة (Twitter, Reddit, News)
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
from datetime import datetime, timedelta


class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'fear_greed_index', 'social_volume', 'positive_ratio',
            'negative_ratio', 'neutral_ratio', 'trending_score',
            'news_sentiment', 'reddit_sentiment', 'twitter_sentiment',
            'btc_dominance', 'market_cap_change', 'volume_24h_change'
        ]
    
    def extract_features(self, data):
        """استخراج ميزات المشاعر من البيانات"""
        features = []
        
        # 1. Fear & Greed Index (0-100)
        fear_greed = data.get('fear_greed_index', 50)
        features.append(fear_greed)
        
        # 2. Social Volume (حجم النقاش الاجتماعي)
        social_volume = data.get('social_volume', 0)
        features.append(social_volume)
        
        # 3. Positive/Negative/Neutral Ratios
        positive = data.get('positive_ratio', 0.33)
        negative = data.get('negative_ratio', 0.33)
        neutral = data.get('neutral_ratio', 0.34)
        features.extend([positive, negative, neutral])
        
        # 4. Trending Score (مدى تداول العملة)
        trending = data.get('trending_score', 0)
        features.append(trending)
        
        # 5. News Sentiment
        news_sentiment = data.get('news_sentiment', 0)
        features.append(news_sentiment)
        
        # 6. Reddit Sentiment
        reddit_sentiment = data.get('reddit_sentiment', 0)
        features.append(reddit_sentiment)
        
        # 7. Twitter Sentiment
        twitter_sentiment = data.get('twitter_sentiment', 0)
        features.append(twitter_sentiment)
        
        # 8. BTC Dominance
        btc_dominance = data.get('btc_dominance', 50)
        features.append(btc_dominance)
        
        # 9. Market Cap Change
        market_cap_change = data.get('market_cap_change_24h', 0)
        features.append(market_cap_change)
        
        # 10. Volume 24h Change
        volume_change = data.get('volume_24h_change', 0)
        features.append(volume_change)
        
        return features
    
    def train(self, trades, voting_scores=None):
        """تدريب نموذج المشاعر"""
        print("\n🎭 Training Sentiment Analysis Model...")
        
        features_list = []
        labels_list = []
        
        for trade in trades:
            try:
                data = trade.get('data', {})
                if isinstance(data, str):
                    data = json.loads(data)
                
                # استخراج ميزات المشاعر
                sentiment_data = data.get('sentiment', {})
                features = self.extract_features(sentiment_data)
                features_list.append(features)
                
                # التسمية: 1 إذا ربح > 0.8%، 0 إذا خسر
                profit = float(trade.get('profit_percent', 0))
                labels_list.append(1 if profit > 0.8 else 0)
                
            except Exception as e:
                continue
        
        if len(features_list) < 50:
            print("⚠️ Not enough data for Sentiment Model")
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
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            random_state=42,
            verbose=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # تقييم الأداء
        accuracy = accuracy_score(y_test, self.model.predict(X_test))
        print(f"🎭 Sentiment Model: Accuracy {accuracy*100:.2f}%")
        
        return self.model, accuracy
    
    def predict(self, sentiment_data):
        """التنبؤ بناءً على مشاعر السوق"""
        if self.model is None:
            return 0.5
        
        features = self.extract_features(sentiment_data)
        X = pd.DataFrame([features], columns=self.feature_names)
        
        # احتمالية الشراء
        proba = self.model.predict_proba(X)[0]
        return proba[1] if len(proba) > 1 else 0.5
    
    def get_sentiment_score(self, sentiment_data):
        """حساب درجة المشاعر (-100 إلى +100)"""
        if not sentiment_data:
            return 0
        
        score = 0
        
        # Fear & Greed Index
        fear_greed = sentiment_data.get('fear_greed_index', 50)
        if fear_greed < 30:  # خوف شديد
            score -= 30
        elif fear_greed > 70:  # طمع شديد
            score += 30
        else:
            score += (fear_greed - 50) * 0.6
        
        # Positive/Negative Ratios
        positive = sentiment_data.get('positive_ratio', 0.33)
        negative = sentiment_data.get('negative_ratio', 0.33)
        score += (positive - negative) * 50
        
        # News Sentiment
        news = sentiment_data.get('news_sentiment', 0)
        score += news * 20
        
        # Social Volume
        social_volume = sentiment_data.get('social_volume', 0)
        if social_volume > 1000:  # نقاش عالي
            score += 10
        
        # تطبيع النتيجة
        score = max(-100, min(100, score))
        
        return score


def train_sentiment_model(trades, voting_scores=None):
    """دالة تدريب نموذج المشاعر"""
    analyzer = SentimentAnalyzer()
    result = analyzer.train(trades, voting_scores)
    
    if result:
        model, accuracy = result
        return model, accuracy
    return None
