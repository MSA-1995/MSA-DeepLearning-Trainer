"""
🧠 Deep Learning Trainer - Advanced AI for Trading Bot
Trains advisors and provides predictions using Deep Neural Networks
"""

# ========== LOAD ENV FILE ==========
import os
for _env_file in [
    '/home/container/DeepLearningTrainer/.env',
    '/home/container/.env',
]:
    try:
        with open(_env_file) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith('#') and '=' in _line:
                    _k, _v = _line.split('=', 1)
                    os.environ.setdefault(_k.strip(), _v.strip())
        break
    except:
        pass

import sys
import time
import json
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse, unquote

try:
    import numpy as np
    import pandas as pd
    import os
    os.environ['KERAS_BACKEND'] = 'jax'
    import keras
    from keras import layers
    DL_AVAILABLE = True
    print(f"✅ Keras {keras.__version__} with JAX backend loaded")
except ImportError:
    print("❌ Keras not installed. Run: pip install keras jax jaxlib")
    DL_AVAILABLE = False
    sys.exit(1)

class DeepLearningTrainer:
    def __init__(self, database_url):
        self.database_url = database_url
        self.conn = self._connect_db()
        self.model = None
        self.feature_names = [
            'rsi', 'macd', 'volume_ratio', 'price_momentum', 'confidence',
            'mtf_score', 'risk_score', 'anomaly_score', 
            'exit_score', 'pattern_score', 'ranking_score'
        ]
        self.min_trades_for_training = 50
        
        print("🧠 Deep Learning Trainer initialized")
    
    def _connect_db(self):
        """Connect to PostgreSQL"""
        try:
            parsed = urlparse(self.database_url)
            self._db_params = {
                'host': parsed.hostname,
                'port': parsed.port,
                'database': parsed.path[1:],
                'user': parsed.username,
                'password': unquote(parsed.password)
            }
            conn = psycopg2.connect(**self._db_params)
            print("✅ Database connected")
            return conn
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return None

    def _get_conn(self):
        """Get valid connection - reconnect if closed"""
        try:
            if self.conn.closed:
                raise Exception("closed")
            self.conn.cursor().execute("SELECT 1")
        except Exception:
            try:
                self.conn = psycopg2.connect(**self._db_params)
            except Exception as e:
                print(f"❌ DB reconnect error: {e}")
        return self.conn
    
    def load_training_data(self):
        """Load historical trades for training"""
        if not self.conn:
            return None, None
        
        try:
            cursor = self._get_conn().cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    symbol,
                    profit_percent,
                    action,
                    timestamp,
                    data
                FROM trades_history
                WHERE action = 'SELL'
                AND data IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 1000
            """)
            
            trades = cursor.fetchall()
            cursor.close()
            
            if len(trades) < self.min_trades_for_training:
                print(f"⚠️ Not enough trades. Need {self.min_trades_for_training}, have {len(trades)}")
                return None, None
            
            X = []
            y = []
            
            for trade in trades:
                try:
                    data = trade.get('data', {})
                    if isinstance(data, str):
                        data = json.loads(data)
                    
                    features = [
                        data.get('rsi', 50),
                        data.get('macd', 0),
                        data.get('volume_ratio', 1),
                        data.get('price_momentum', 0),
                        data.get('confidence', 60),
                        data.get('mtf_score', 0),
                        data.get('risk_score', 0),
                        data.get('anomaly_score', 0),
                        data.get('exit_score', 0),
                        data.get('pattern_score', 0),
                        data.get('ranking_score', 0)
                    ]
                    
                    profit = float(trade.get('profit_percent', 0))
                    label = 1 if profit > 0 else 0
                    
                    X.append(features)
                    y.append(label)
                
                except Exception:
                    continue
            
            if len(X) < self.min_trades_for_training:
                print(f"⚠️ Not enough valid trades. Need {self.min_trades_for_training}, have {len(X)}")
                return None, None
            
            print(f"📊 Loaded {len(X)} trades")
            print(f"   Profitable: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
            print(f"   Losses: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
            
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None
    
    def build_model(self, input_dim):
        """Build Deep Neural Network"""
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self):
        """Train Deep Learning model"""
        print("\n" + "="*60)
        print("🎓 Starting Deep Learning Training...")
        print("="*60)
        
        X, y = self.load_training_data()
        if X is None or y is None:
            return False
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\n📚 Training set: {len(X_train)} trades")
        print(f"🧪 Test set: {len(X_test)} trades")
        
        # Build model
        print("\n🧠 Building Deep Neural Network...")
        self.model = self.build_model(X.shape[1])
        
        # Train
        print("\n🎓 Training...")
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n✅ Training complete!")
        print(f"🎯 Accuracy: {accuracy*100:.2f}%")
        print(f"📉 Loss: {loss:.4f}")
        
        # Save model
        self.save_model()
        
        # Save predictions
        self.save_predictions_to_db(accuracy)
        
        # Train advisors
        self.train_advisors()
        
        return True
    
    def save_model(self):
        """Save model to file"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'deep_model.keras')
            self.model.save(model_path)
            print(f"\n💾 Model saved to {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self):
        """Load model from file"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'deep_model.keras')
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                print("✅ Model loaded")
                return True
            else:
                print("⚠️ No saved model found")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, features):
        """Predict trade success probability"""
        if self.model is None:
            return None
        
        try:
            features_array = np.array([features], dtype=np.float32)
            proba = self.model.predict(features_array, verbose=0)[0][0]
            
            return {
                'success_probability': float(proba),
                'confidence_boost': int((proba - 0.5) * 20)
            }
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def save_predictions_to_db(self, accuracy):
        """Save model info to database"""
        if not self.conn or self.model is None:
            return False
        
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dl_predictions (
                    id SERIAL PRIMARY KEY,
                    predictions JSONB NOT NULL,
                    model_accuracy FLOAT,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """)
            
            cursor.execute("DELETE FROM dl_predictions")
            
            predictions_data = {
                'model_type': 'DeepNeuralNetwork',
                'features': self.feature_names,
                'trained_at': datetime.now().isoformat(),
                'status': 'active',
                'layers': [64, 32, 16, 1],
                'accuracy': float(accuracy)
            }
            
            cursor.execute("""
                INSERT INTO dl_predictions (predictions, model_accuracy)
                VALUES (%s, %s)
            """, (json.dumps(predictions_data), float(accuracy)))
            
            conn.commit()
            cursor.close()
            
            print("💾 Predictions saved to database")
            return True
        
        except Exception as e:
            print(f"❌ Error saving predictions: {e}")
            self._get_conn().rollback()
            return False
    
    def train_advisors(self):
        """Train advisors from database trades"""
        print("\n" + "="*60)
        print("🎓 Training Advisors...")
        print("="*60)
        
        try:
            cursor = self._get_conn().cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM trades_history
                WHERE action = 'SELL'
                ORDER BY timestamp DESC
                LIMIT 500
            """)
            trades = cursor.fetchall()
            cursor.close()
            
            if len(trades) < 10:
                print("⚠️ Not enough trades for advisor training")
                return False
            
            print(f"📊 Training from {len(trades)} trades")
            
            # Prepare advisor knowledge
            advisor_knowledge = self._extract_advisor_knowledge(trades)
            
            # Save to database
            self.save_advisors_knowledge(advisor_knowledge)
            
            print("\n✅ All advisors trained successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Advisor training error: {e}")
            return False
    
    def _extract_advisor_knowledge(self, trades):
        """Extract knowledge for all advisors"""
        knowledge = {
            'ai_brain': {'patterns': [], 'traps': []},
            'exit_strategy': {'exits': []},
            'pattern_recognition': {'success': [], 'trap': []},
            'coin_ranking': {}
        }
        
        for trade in trades:
            try:
                data = trade.get('data', {})
                if isinstance(data, str):
                    data = json.loads(data)
                
                profit = float(trade.get('profit_percent', 0))
                symbol = trade.get('symbol')
                
                # AI Brain patterns
                pattern = {
                    'rsi': data.get('rsi', 50),
                    'macd': data.get('macd', 0),
                    'volume_ratio': data.get('volume_ratio', 1),
                    'confidence': data.get('confidence', 60),
                    'profit': profit,
                    'timestamp': str(trade.get('timestamp'))
                }
                
                if profit > 0:
                    knowledge['ai_brain']['patterns'].append(pattern)
                    knowledge['pattern_recognition']['success'].append(pattern)
                else:
                    knowledge['ai_brain']['traps'].append(pattern)
                    knowledge['pattern_recognition']['trap'].append(pattern)
                
                # Exit strategy
                exit_data = {
                    'profit': profit,
                    'reason': data.get('sell_reason', 'UNKNOWN'),
                    'hours_held': data.get('hours_held', 24)
                }
                knowledge['exit_strategy']['exits'].append(exit_data)
                
                # Coin ranking
                if symbol not in knowledge['coin_ranking']:
                    knowledge['coin_ranking'][symbol] = {'trades': 0, 'total_profit': 0}
                knowledge['coin_ranking'][symbol]['trades'] += 1
                knowledge['coin_ranking'][symbol]['total_profit'] += profit
                
            except Exception:
                continue
        
        return knowledge
    
    def save_advisors_knowledge(self, knowledge):
        """Save advisors knowledge to database"""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dl_advisors_knowledge (
                    id SERIAL PRIMARY KEY,
                    advisor_name VARCHAR(100) NOT NULL,
                    knowledge JSONB NOT NULL,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """)
            
            cursor.execute("DELETE FROM dl_advisors_knowledge")
            
            for advisor_name, advisor_data in knowledge.items():
                cursor.execute("""
                    INSERT INTO dl_advisors_knowledge (advisor_name, knowledge)
                    VALUES (%s, %s)
                """, (advisor_name, json.dumps(advisor_data)))
                print(f"💾 {advisor_name} knowledge saved")
            
            conn.commit()
            cursor.close()
            
            print("\n💾 All advisors knowledge saved!")
            return True
            
        except Exception as e:
            print(f"❌ Error saving knowledge: {e}")
            self._get_conn().rollback()
            return False
    
    def run_continuous(self, interval_hours=12):
        """Run training continuously"""
        print(f"\n🚀 Deep Learning Trainer started!")
        print(f"⏰ Training interval: {interval_hours} hours")
        print("="*60)
        
        while True:
            try:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"\n{'='*60}")
                print(f"⏰ {current_time}")
                print(f"{'='*60}")
                
                success = self.train_model()
                
                if success:
                    print(f"\n✅ Training successful")
                else:
                    print(f"\n⚠️ Training skipped - not enough data")
                
                next_time = (datetime.now() + timedelta(hours=interval_hours)).strftime("%H:%M:%S")
                print(f"\n⏰ Next training at: {next_time}")
                time.sleep(interval_hours * 3600)
            
            except KeyboardInterrupt:
                print("\n🛑 Trainer stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print(f"⏰ Retrying in 30 minutes...")
                time.sleep(1800)

def main():
    if not DL_AVAILABLE:
        print("❌ Please install TensorFlow:")
        print("   pip install tensorflow-cpu")
        return
    
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not found!")
        return
    
    trainer = DeepLearningTrainer(database_url)
    trainer.run_continuous(interval_hours=12)

if __name__ == "__main__":
    main()
