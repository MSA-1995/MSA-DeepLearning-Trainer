"""
Crypto News Analysis Model - reads directly from news_sentiment table
"""

import os
import json
import numpy as np
import pandas as pd
import psycopg2
from urllib.parse import unquote
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False


def get_news_from_db(symbol, hours=24):
    """Read news data directly from news_sentiment table"""
    try:
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            return None
        
        from urllib.parse import urlparse
        parsed = urlparse(database_url)
        conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port,
            database=parsed.path[1:],
            user=parsed.username,
            password=unquote(parsed.password)
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sentiment, score, headline
            FROM news_sentiment 
            WHERE symbol = %s 
            AND timestamp > NOW() - INTERVAL '%s hours'
        """, (symbol, hours))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not rows:
            return None
        
        positive = sum(1 for r in rows if r[0] == 'POSITIVE')
        negative = sum(1 for r in rows if r[0] == 'NEGATIVE')
        neutral = sum(1 for r in rows if r[0] == 'NEUTRAL')
        total = len(rows)
        avg_score = sum(r[1] for r in rows) / total if total > 0 else 0
        
        return {
            'news_count_24h': total,
            'positive_news_count': positive,
            'negative_news_count': negative,
            'neutral_news_count': neutral,
            'news_sentiment_avg': avg_score,
            'news_score': avg_score,
            'total': total,
            'positive': positive,
            'negative': negative,
            'neutral': neutral
        }
    except:
        return None


class CryptoNewsAnalyzer:
    def __init__(self):
        self.model = None

    def extract_features(self, data):
        """Extract news features"""
        news_count = data.get('news_count_24h', data.get('total', 0))
        positive = data.get('positive_news_count', data.get('positive', 0))
        negative = data.get('negative_news_count', data.get('negative', 0))
        neutral = data.get('neutral_news_count', data.get('neutral', 0))
        sentiment_avg = data.get('news_sentiment_avg', data.get('news_score', 0))

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

    def train(self, trades, voting_scores=None):
        """Train crypto news model - reads from news_sentiment table directly"""
        print("\nTraining Crypto News Model (from news_sentiment table)...")

        if not LIGHTGBM_AVAILABLE:
            print("⚠️ LightGBM not available")
            return None

        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            print("⚠️ No DATABASE_URL - cannot read from news_sentiment")
            return None

        try:
            from urllib.parse import urlparse
            parsed = urlparse(database_url)
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port,
                database=parsed.path[1:],
                user=parsed.username,
                password=unquote(parsed.password)
            )
            cursor = conn.cursor()
            
            # Get news data by symbol
            cursor.execute("""
                SELECT symbol, sentiment, score, headline
                FROM news_sentiment
                WHERE timestamp > NOW() - INTERVAL '7 days'
                ORDER BY timestamp DESC
            """)
            news_rows = cursor.fetchall()
            
            # Get trades for matching (no time limit)
            cursor.execute("""
                SELECT symbol, profit_percent
                FROM trades_history
                WHERE action = 'SELL'
            """)
            trades_data = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            if not news_rows:
                print("⚠️ No news data in news_sentiment table")
                return None
            
            # Normalize symbol function (BTC/USDT -> BTCUSDT)
            def normalize_symbol(s):
                return s.replace('/', '') if '/' in s else s
            
            # Build news features by symbol (normalized)
            symbol_news = {}
            for row in news_rows:
                symbol = normalize_symbol(row[0])
                sentiment = row[1]
                score = row[2]
                
                if symbol not in symbol_news:
                    symbol_news[symbol] = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0, 'score_sum': 0}
                
                symbol_news[symbol]['total'] += 1
                symbol_news[symbol]['score_sum'] += score
                if sentiment == 'POSITIVE':
                    symbol_news[symbol]['positive'] += 1
                elif sentiment == 'NEGATIVE':
                    symbol_news[symbol]['negative'] += 1
                else:
                    symbol_news[symbol]['neutral'] += 1
            
            # Build training data
            features_list, labels_list = [], []
            for trade in trades_data:
                symbol = normalize_symbol(trade[0])  # Normalize trade symbol too
                profit = trade[1]
                
                if symbol not in symbol_news:
                    continue
                
                news = symbol_news[symbol]
                total = news['total']
                
                news_data = {
                    'news_count_24h': total,
                    'positive_news_count': news['positive'],
                    'negative_news_count': news['negative'],
                    'neutral_news_count': news['neutral'],
                    'news_sentiment_avg': news['score_sum'] / total if total > 0 else 0,
                    'news_score': news['score_sum'] / total if total > 0 else 0,
                    'total': total,
                    'positive': news['positive'],
                    'negative': news['negative'],
                    'neutral': news['neutral']
                }
                
                features_list.append(self.extract_features(news_data))
                labels_list.append(1 if profit > 0.8 else 0)
            
            print(f"  Training samples: {len(features_list)} (from news_sentiment table)")
            
            if len(features_list) < 50:
                print("⚠️ Not enough data for Crypto News Model")
                return None

        except Exception as e:
            print(f"⚠️ Error reading from news_sentiment: {e}")
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


def train_crypto_news_model(trades, voting_scores=None):
    analyzer = CryptoNewsAnalyzer()
    result = analyzer.train(trades, voting_scores)
    if result:
        return result[0], result[1]
    return None
