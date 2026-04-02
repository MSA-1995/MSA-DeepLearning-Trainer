"""
Sentiment Analysis Model - reads directly from news_sentiment table
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


def get_sentiment_from_db(symbol, hours=24):
    """Read sentiment data directly from news_sentiment table"""
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
            SELECT sentiment, score 
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
        
        return {
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'total': total,
            'positive_ratio': positive / total if total > 0 else 0.33,
            'negative_ratio': negative / total if total > 0 else 0.33,
            'neutral_ratio': neutral / total if total > 0 else 0.34,
            'news_score': sum(r[1] for r in rows) / total if total > 0 else 0
        }
    except:
        return None


class SentimentAnalyzer:
    def __init__(self):
        self.model = None

    def extract_features(self, data):
        """Extract sentiment features"""
        positive = data.get('positive_ratio', 0.33)
        negative = data.get('negative_ratio', 0.33)
        neutral = data.get('neutral_ratio', 0.34)
        news_sentiment = data.get('news_score', 0)
        fear_greed = 50
        social_volume = 1000

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

    def train(self, trades, voting_scores=None, since_timestamp=None):
        """Train sentiment model - reads from news_sentiment table, only NEW data since last training"""
        print("\nTraining Sentiment Model (from news_sentiment table)...")

        if not LIGHTGBM_AVAILABLE:
            print("⚠️ LightGBM not available")
            return None

        # Read sentiment data from news_sentiment table
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
            
            # Get ONLY NEW sentiment data since last training (like other advisors)
            if since_timestamp:
                cursor.execute("""
                    SELECT symbol, sentiment, score, timestamp
                    FROM news_sentiment
                    WHERE timestamp > %s
                    ORDER BY timestamp DESC
                """, (since_timestamp,))
            else:
                # First training - get all
                cursor.execute("""
                    SELECT symbol, sentiment, score, timestamp
                    FROM news_sentiment
                    ORDER BY timestamp DESC
                """)
            sentiment_rows = cursor.fetchall()
            
            # Get ONLY NEW trades since last training
            if since_timestamp:
                cursor.execute("""
                    SELECT symbol, profit_percent
                    FROM trades_history
                    WHERE action = 'SELL'
                      AND timestamp > %s
                """, (since_timestamp,))
            else:
                cursor.execute("""
                    SELECT symbol, profit_percent
                    FROM trades_history
                    WHERE action = 'SELL'
                """)
            trades_data = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            if not sentiment_rows:
                print("⚠️ No sentiment data in news_sentiment table")
                return None
            
            # Normalize symbol function (BTC/USDT -> BTCUSDT)
            def normalize_symbol(s):
                return s.replace('/', '') if '/' in s else s
            
            # Build sentiment features by symbol (normalized)
            symbol_sentiment = {}
            for row in sentiment_rows:
                symbol = normalize_symbol(row[0])
                sentiment = row[1]
                score = row[2]
                
                if symbol not in symbol_sentiment:
                    symbol_sentiment[symbol] = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0, 'score_sum': 0}
                
                symbol_sentiment[symbol]['total'] += 1
                symbol_sentiment[symbol]['score_sum'] += score
                if sentiment == 'POSITIVE':
                    symbol_sentiment[symbol]['positive'] += 1
                elif sentiment == 'NEGATIVE':
                    symbol_sentiment[symbol]['negative'] += 1
                else:
                    symbol_sentiment[symbol]['neutral'] += 1
            
            # Build training data
            features_list, labels_list = [], []
            for trade in trades_data:
                symbol = normalize_symbol(trade[0])  # Normalize trade symbol too
                profit = trade[1]
                
                if symbol not in symbol_sentiment:
                    continue
                
                sent = symbol_sentiment[symbol]
                total = sent['total']
                
                sentiment_data = {
                    'positive_ratio': sent['positive'] / total if total > 0 else 0.33,
                    'negative_ratio': sent['negative'] / total if total > 0 else 0.33,
                    'neutral_ratio': sent['neutral'] / total if total > 0 else 0.34,
                    'news_score': sent['score_sum'] / total if total > 0 else 0
                }
                
                features_list.append(self.extract_features(sentiment_data))
                labels_list.append(1 if profit > 0.8 else 0)
            
            print(f"  Training samples: {len(features_list)} (from news_sentiment table)")
            
            if len(features_list) < 50:
                print("⚠️ Not enough data for Sentiment Model")
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
