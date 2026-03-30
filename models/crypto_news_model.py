"""
📰 Crypto News Analysis Model - تحليل أخبار العملات الرقمية
يحلل تأثير الأخبار على أسعار العملات الرقمية
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
from datetime import datetime, timedelta


class CryptoNewsAnalyzer:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'news_count_24h', 'positive_news_count', 'negative_news_count',
            'neutral_news_count', 'news_sentiment_avg', 'news_volume_score',
            'breaking_news_count', 'partnership_news', 'regulation_news',
            'technical_news', 'market_news', 'exchange_news',
            'news_recency_score', 'news_source_reliability'
        ]
        
        # كلمات مفتاحية للتصنيف
        self.positive_keywords = [
            'partnership', 'adoption', 'upgrade', 'launch', 'integration',
            'bullish', 'growth', 'success', 'milestone', 'record',
            'institutional', 'investment', 'approval', 'support'
        ]
        
        self.negative_keywords = [
            'hack', 'scam', 'ban', 'regulation', 'crash', 'bearish',
            'decline', 'loss', 'failure', 'delay', 'concern', 'risk',
            'warning', 'investigation', 'lawsuit', 'fraud'
        ]
        
        self.regulation_keywords = [
            'sec', 'cftc', 'regulation', 'compliance', 'legal',
            'law', 'policy', 'government', 'central bank', 'ban'
        ]
        
        self.partnership_keywords = [
            'partnership', 'collaboration', 'integration', 'alliance',
            'joint', 'agreement', 'deal', 'contract'
        ]
    
    def analyze_news_text(self, text):
        """تحليل نص الخبر واستخراج الميزات"""
        if not text:
            return {
                'sentiment': 0,
                'is_breaking': False,
                'is_partnership': False,
                'is_regulation': False,
                'is_technical': False,
                'is_market': False,
                'is_exchange': False
            }
        
        text_lower = text.lower()
        
        # تحليل المشاعر
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = 1
        elif negative_count > positive_count:
            sentiment = -1
        else:
            sentiment = 0
        
        # تصنيف نوع الخبر
        is_breaking = any(word in text_lower for word in ['breaking', 'urgent', 'alert', 'just in'])
        is_partnership = any(word in text_lower for word in self.partnership_keywords)
        is_regulation = any(word in text_lower for word in self.regulation_keywords)
        is_technical = any(word in text_lower for word in ['upgrade', 'update', 'fork', 'mainnet', 'testnet'])
        is_market = any(word in text_lower for word in ['price', 'market', 'trading', 'volume', 'rally'])
        is_exchange = any(word in text_lower for word in ['exchange', 'listing', 'delisting', 'binance', 'coinbase'])
        
        return {
            'sentiment': sentiment,
            'is_breaking': is_breaking,
            'is_partnership': is_partnership,
            'is_regulation': is_regulation,
            'is_technical': is_technical,
            'is_market': is_market,
            'is_exchange': is_exchange
        }
    
    def extract_features(self, news_data):
        """استخراج ميزات الأخبار"""
        features = []
        
        # 1. عدد الأخبار في 24 ساعة
        news_count = news_data.get('news_count_24h', 0)
        features.append(news_count)
        
        # 2. عدد الأخبار الإيجابية
        positive_count = news_data.get('positive_news_count', 0)
        features.append(positive_count)
        
        # 3. عدد الأخبار السلبية
        negative_count = news_data.get('negative_news_count', 0)
        features.append(negative_count)
        
        # 4. عدد الأخبار المحايدة
        neutral_count = news_data.get('neutral_news_count', 0)
        features.append(neutral_count)
        
        # 5. متوسط مشاعر الأخبار
        sentiment_avg = news_data.get('news_sentiment_avg', 0)
        features.append(sentiment_avg)
        
        # 6. درجة حجم الأخبار
        volume_score = news_data.get('news_volume_score', 0)
        features.append(volume_score)
        
        # 7. عدد الأخبار العاجلة
        breaking_count = news_data.get('breaking_news_count', 0)
        features.append(breaking_count)
        
        # 8. عدد أخبار الشراكات
        partnership_count = news_data.get('partnership_news', 0)
        features.append(partnership_count)
        
        # 9. عدد أخبار التنظيم
        regulation_count = news_data.get('regulation_news', 0)
        features.append(regulation_count)
        
        # 10. عدد أخبار تقنية
        technical_count = news_data.get('technical_news', 0)
        features.append(technical_count)
        
        # 11. عدد أخبار السوق
        market_count = news_data.get('market_news', 0)
        features.append(market_count)
        
        # 12. عدد أخبار البورصات
        exchange_count = news_data.get('exchange_news', 0)
        features.append(exchange_count)
        
        # 13. درجة حداثة الأخبار
        recency_score = news_data.get('news_recency_score', 0)
        features.append(recency_score)
        
        # 14. درجة موثوقية المصدر
        reliability = news_data.get('news_source_reliability', 0.5)
        features.append(reliability)
        
        return features
    
    def train(self, trades, voting_scores=None):
        """تدريب نموذج الأخبار"""
        print("\n📰 Training Crypto News Model...")
        
        features_list = []
        labels_list = []
        
        for trade in trades:
            try:
                data = trade.get('data', {})
                if isinstance(data, str):
                    data = json.loads(data)
                
                # استخراج ميزات الأخبار
                news_data = data.get('news', {})
                features = self.extract_features(news_data)
                features_list.append(features)
                
                # التسمية: 1 إذا ربح > 0.8%، 0 إذا خسر
                profit = float(trade.get('profit_percent', 0))
                labels_list.append(1 if profit > 0.8 else 0)
                
            except Exception as e:
                continue
        
        if len(features_list) < 50:
            print("⚠️ Not enough data for Crypto News Model")
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
        print(f"📰 Crypto News Model: Accuracy {accuracy*100:.2f}%")
        
        return self.model, accuracy
    
    def predict(self, news_data):
        """التنبؤ بناءً على الأخبار"""
        if self.model is None:
            return 0.5
        
        features = self.extract_features(news_data)
        X = pd.DataFrame([features], columns=self.feature_names)
        
        # احتمالية الشراء
        proba = self.model.predict_proba(X)[0]
        return proba[1] if len(proba) > 1 else 0.5
    
    def get_news_impact_score(self, news_data):
        """حساب درجة تأثير الأخبار (-100 إلى +100)"""
        if not news_data:
            return 0
        
        score = 0
        
        # عدد الأخبار
        news_count = news_data.get('news_count_24h', 0)
        if news_count > 10:
            score += 20
        elif news_count > 5:
            score += 10
        
        # الأخبار الإيجابية والسلبية
        positive = news_data.get('positive_news_count', 0)
        negative = news_data.get('negative_news_count', 0)
        score += (positive - negative) * 10
        
        # الأخبار العاجلة
        breaking = news_data.get('breaking_news_count', 0)
        score += breaking * 15
        
        # أخبار الشراكات
        partnership = news_data.get('partnership_news', 0)
        score += partnership * 20
        
        # أخبار التنظيم
        regulation = news_data.get('regulation_news', 0)
        score -= regulation * 25  # التنظيم عادة سلبي
        
        # متوسط المشاعر
        sentiment_avg = news_data.get('news_sentiment_avg', 0)
        score += sentiment_avg * 30
        
        # تطبيع النتيجة
        score = max(-100, min(100, score))
        
        return score


def train_crypto_news_model(trades, voting_scores=None):
    """دالة تدريب نموذج الأخبار"""
    analyzer = CryptoNewsAnalyzer()
    result = analyzer.train(trades, voting_scores)
    
    if result:
        model, accuracy = result
        return model, accuracy
    return None
