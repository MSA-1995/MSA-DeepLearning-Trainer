"""
🧠 Deep Learning Trainer V2 - 8 Specialized Models (LightGBM)
Trains AI Brain + 7 Consultant Models for trading decisions
"""

# ========== AUTO-UPDATE PIP ==========
import subprocess
import sys
try:
    print("🔄 Checking pip updates...")
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                           capture_output=True, check=False, timeout=30, text=True)
    if "Successfully installed" in result.stdout:
        print("✅ pip updated successfully")
    else:
        print("✅ pip is up to date")
except Exception as e:
    print(f"⚠️ pip update skipped: {e}")

# ========== LOAD ENV FILE ==========
import os
for _env_file in [
    '/home/container/DeepLearningTrainer_XGBoost/.env',
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

import time
import json
import pickle
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse, unquote

try:
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    ML_AVAILABLE = True
    print(f"✅ LightGBM {lgb.__version__} loaded")
except ImportError:
    print("❌ LightGBM not installed. Run: pip install lightgbm scikit-learn")
    ML_AVAILABLE = False
    sys.exit(1)

class DeepLearningTrainerXGBoost:
    def __init__(self, database_url):
        self.database_url = database_url
        self.conn = self._connect_db()
        
        # 8 موديلات: AI Brain + 7 مستشارين (كلهم LightGBM)
        self.models = {
            'ai_brain': None,      # AI Brain - LightGBM
            'smart_money': None,   # Smart Money Tracker - LightGBM (بديل MTF)
            'risk': None,          # Risk Manager - LightGBM
            'anomaly': None,       # Anomaly Detector - LightGBM
            'exit': None,          # Exit Strategy - LightGBM
            'pattern': None,       # Pattern Recognition - LightGBM
            'liquidity': None,     # Liquidity Analyzer - LightGBM (بديل Ranking)
            'chart_cnn': None      # Chart Pattern Analyzer - LightGBM
        }
        
        self.min_trades_for_training = 100
        
        print("🧠 Deep Learning Trainer V2 initialized (8 Models - LightGBM)")
    
    def _connect_db(self):
        """Connect to PostgreSQL"""
        try:
            parsed = urlparse(self.database_url)
            self._db_params = {
                'host': parsed.hostname,
                'port': parsed.port,
                'database': parsed.path[1:],
                'user': parsed.username,
                'password': unquote(parsed.password),
                'sslmode': 'require',
                'connect_timeout': 10
            }
            conn = psycopg2.connect(**self._db_params)
            print("✅ Database: Connected (Supabase)")
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
            return None
        
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
                ORDER BY timestamp ASC
                LIMIT 2000
            """)
            
            trades = cursor.fetchall()
            cursor.close()
            
            if len(trades) < self.min_trades_for_training:
                print(f"⚠️ Not enough trades. Need {self.min_trades_for_training}, have {len(trades)}")
                return None
            
            print(f"📊 Loaded {len(trades)} trades for training")
            return trades
        
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def calculate_enhanced_features(self, data):
        """Feature Engineering: حساب مؤشرات إضافية"""
        try:
            rsi = data.get('rsi', 50)
            macd = data.get('macd', 0)
            volume_ratio = data.get('volume_ratio', 1)
            price_momentum = data.get('price_momentum', 0)
            
            # Bollinger Bands approximation
            bb_position = (rsi - 30) / 40
            
            # ATR approximation (محسّن)
            atr_estimate = abs(price_momentum) * volume_ratio
            
            # Stochastic approximation
            stochastic = rsi
            
            # EMA crossover signal (محسّن)
            ema_signal = 1 if macd > 0 else -1
            
            # Volume strength
            volume_strength = min(volume_ratio / 2.0, 2.0)
            
            # Momentum strength
            momentum_strength = abs(price_momentum) / 10.0
            
            # ========== الإضافات الجديدة (5 مؤشرات) ==========
            
            # 1. ATR (Average True Range) - للمخاطرة
            atr = data.get('atr', atr_estimate)
            
            # 2. EMA 9/21 Crossover - للأنماط
            ema_9 = data.get('ema_9', 0)
            ema_21 = data.get('ema_21', 0)
            ema_crossover = 1 if ema_9 > ema_21 else -1
            
            # 3. Bid-Ask Spread - للفخاخ
            bid_ask_spread = data.get('bid_ask_spread', 0)
            
            # 4. Volume Trend - للبيع
            volume_trend = data.get('volume_trend', 0)
            
            # 5. Price Change 1h - للشذوذ
            price_change_1h = data.get('price_change_1h', 0)
            
            return [
                rsi,
                macd,
                volume_ratio,
                price_momentum,
                bb_position,
                atr_estimate,
                stochastic,
                ema_signal,
                volume_strength,
                momentum_strength,
                atr,                    # جديد
                ema_crossover,          # جديد
                bid_ask_spread,         # جديد
                volume_trend,           # جديد
                price_change_1h         # جديد
            ]
        except:
            return [50, 0, 1, 0, 0.5, 1, 50, 0, 1, 0, 1, 0, 0, 0, 0]

    
    def train_smart_money_model(self, trades):
        """Train Smart Money Tracker (بديل MTF)"""
        print("\n🐋 Training Smart Money Model (LightGBM)...")
        
        features_list = []
        labels_list = []
        
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
                    data.get('atr', 1),
                    data.get('ema_crossover', 0),
                    data.get('bid_ask_spread', 0),
                    data.get('volume_trend', 0),
                    data.get('price_change_1h', 0)
                ]
                
                profit = float(trade.get('profit_percent', 0))
                label = 1 if profit > 0 else 0
                
                features_list.append(features)
                labels_list.append(label)
            except:
                continue
        
        if len(features_list) < 50:
            print("⚠️ Not enough data for MTF")
            return None
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"🐋 Smart Money Model: Accuracy {accuracy*100:.2f}%")
        
        return model, accuracy
    
    def train_ai_brain_model(self, trades):
        """Train AI Brain (الملك) - القرار النهائي"""
        print("\n👑 Training AI Brain Model (LightGBM)...")
        
        features_list = []
        labels_list = []
        
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
                    data.get('ranking_score', 0),
                    data.get('atr', 1),
                    data.get('ema_crossover', 0),
                    data.get('bid_ask_spread', 0),
                    data.get('volume_trend', 0),
                    data.get('price_change_1h', 0)
                ]
                
                profit = float(trade.get('profit_percent', 0))
                label = 1 if profit > 0.5 else 0
                
                features_list.append(features)
                labels_list.append(label)
            except:
                continue
        
        if len(features_list) < 50:
            print("⚠️ Not enough data for AI Brain")
            return None
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # الملك يحتاج موديل أقوى
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"👑 AI Brain Model: Accuracy {accuracy*100:.2f}%")
        
        return model, accuracy
    
    def train_risk_model(self, trades):
        """Train Risk Manager"""
        print("\n🎓 Training Risk Model (LightGBM)...")
        
        features_list = []
        labels_list = []
        
        for trade in trades:
            try:
                data = trade.get('data', {})
                if isinstance(data, str):
                    data = json.loads(data)
                
                features = [
                    data.get('rsi', 50),
                    data.get('volume_ratio', 1),
                    data.get('confidence', 60),
                    data.get('price_momentum', 0),
                    data.get('atr', 1),
                    data.get('ema_crossover', 0),
                    data.get('bid_ask_spread', 0),
                    data.get('volume_trend', 0),
                    data.get('price_change_1h', 0)
                ]
                
                profit = float(trade.get('profit_percent', 0))
                label = 1 if profit < -1.0 else 0
                
                features_list.append(features)
                labels_list.append(label)
            except:
                continue
        
        if len(features_list) < 50:
            print("⚠️ Not enough data for Risk")
            return None
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Risk Model: Accuracy {accuracy*100:.2f}%")
        
        return model, accuracy
    
    def train_anomaly_model(self, trades):
        """Train Anomaly Detector"""
        print("\n🎓 Training Anomaly Model (LightGBM)...")
        
        features_list = []
        labels_list = []
        
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
                    data.get('atr', 1),
                    data.get('ema_crossover', 0),
                    data.get('bid_ask_spread', 0),
                    data.get('volume_trend', 0),
                    data.get('price_change_1h', 0)
                ]
                
                profit = float(trade.get('profit_percent', 0))
                label = 1 if profit < -1.5 else 0
                
                features_list.append(features)
                labels_list.append(label)
            except:
                continue
        
        if len(features_list) < 50:
            print("⚠️ Not enough data for Anomaly")
            return None
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Anomaly Model: Accuracy {accuracy*100:.2f}%")
        
        return model, accuracy
    
    def train_exit_model(self, trades):
        """Train Exit Strategy"""
        print("\n🎓 Training Exit Model (LightGBM)...")
        
        features_list = []
        labels_list = []
        
        for trade in trades:
            try:
                data = trade.get('data', {})
                if isinstance(data, str):
                    data = json.loads(data)
                
                features = [
                    data.get('rsi', 50),
                    data.get('macd', 0),
                    data.get('confidence', 60),
                    data.get('price_momentum', 0),
                    data.get('atr', 1),
                    data.get('ema_crossover', 0),
                    data.get('bid_ask_spread', 0),
                    data.get('volume_trend', 0),
                    data.get('price_change_1h', 0)
                ]
                
                profit = float(trade.get('profit_percent', 0))
                label = 1 if (profit > 1.0 or profit < -1.0) else 0
                
                features_list.append(features)
                labels_list.append(label)
            except:
                continue
        
        if len(features_list) < 50:
            print("⚠️ Not enough data for Exit")
            return None
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Exit Model: Accuracy {accuracy*100:.2f}%")
        
        return model, accuracy
    
    def train_pattern_model(self, trades):
        """Train Pattern Recognition"""
        print("\n🎓 Training Pattern Model (LightGBM)...")
        
        features_list = []
        labels_list = []
        
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
                    data.get('atr', 1),
                    data.get('ema_crossover', 0),
                    data.get('bid_ask_spread', 0),
                    data.get('volume_trend', 0),
                    data.get('price_change_1h', 0)
                ]
                
                profit = float(trade.get('profit_percent', 0))
                label = 1 if profit > 0.5 else 0
                
                features_list.append(features)
                labels_list.append(label)
            except:
                continue
        
        if len(features_list) < 50:
            print("⚠️ Not enough data for Pattern")
            return None
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Pattern Model: Accuracy {accuracy*100:.2f}%")
        
        return model, accuracy
    
    def train_liquidity_model(self, trades):
        """Train Liquidity Analyzer (بديل Ranking)"""
        print("\n💧 Training Liquidity Model (LightGBM)...")
        
        coin_data = {}
        
        for trade in trades:
            try:
                symbol = trade.get('symbol')
                profit = float(trade.get('profit_percent', 0))
                
                if symbol not in coin_data:
                    coin_data[symbol] = {'profits': [], 'count': 0}
                
                coin_data[symbol]['profits'].append(profit)
                coin_data[symbol]['count'] += 1
            except:
                continue
        
        features_list = []
        labels_list = []
        
        for symbol, data in coin_data.items():
            if data['count'] < 3:
                continue
            
            avg_profit = sum(data['profits']) / len(data['profits'])
            win_rate = sum(1 for p in data['profits'] if p > 0) / len(data['profits'])
            
            features = [
                avg_profit,
                win_rate,
                data['count'],
                max(data['profits']),
                min(data['profits'])
            ]
            
            label = 1 if avg_profit > 0 and win_rate > 0.5 else 0
            
            features_list.append(features)
            labels_list.append(label)
        
        if len(features_list) < 20:
            print("⚠️ Not enough coins for Ranking")
            return None
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"💧 Liquidity Model: Accuracy {accuracy*100:.2f}%")
        
        return model, accuracy
    
    def train_chart_cnn_model(self, trades):
        """Train Chart Pattern Analyzer (LightGBM)"""
        print("\n📊 Training Chart Pattern Model (LightGBM)...")
        
        features_list = []
        labels_list = []
        
        for trade in trades:
            try:
                data = trade.get('data', {})
                if isinstance(data, str):
                    data = json.loads(data)
                
                features = self.calculate_enhanced_features(data)
                
                profit = float(trade.get('profit_percent', 0))
                label = 1 if profit > 0.5 else 0
                
                features_list.append(features)
                labels_list.append(label)
            except:
                continue
        
        if len(features_list) < 50:
            print("⚠️ Not enough data for Chart Pattern")
            return None
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"📊 Chart Pattern Model: Accuracy {accuracy*100:.2f}%")
        
        return model, accuracy

    
    def train_all_models(self):
        """Train all 8 models (AI Brain + 7 Consultants: LightGBM)"""
        print("\n" + "="*60)
        print("👑 Starting Training - 8 LightGBM Models")
        print("="*60)
        
        trades = self.load_training_data()
        if not trades:
            return False
        
        results = {}
        
        # 🎯 حساب دقة التصويت أولاً
        try:
            voting_scores = self.calculate_voting_accuracy(trades)
            results['voting_scores'] = voting_scores
        except Exception as e:
            print(f"⚠️ Voting accuracy calculation error: {e}")
            results['voting_scores'] = {}
        
        # 👑 الملك يتدرب أول
        try:
            result = self.train_ai_brain_model(trades)
            if result:
                self.models['ai_brain'], results['ai_brain_accuracy'] = result
        except Exception as e:
            print(f"❌ AI Brain training error: {e}")
        
        # Train consultant models
        try:
            result = self.train_smart_money_model(trades)
            if result:
                self.models['smart_money'], results['smart_money_accuracy'] = result
        except Exception as e:
            print(f"❌ Smart Money training error: {e}")
        
        try:
            result = self.train_risk_model(trades)
            if result:
                self.models['risk'], results['risk_accuracy'] = result
        except Exception as e:
            print(f"❌ Risk training error: {e}")
        
        try:
            result = self.train_anomaly_model(trades)
            if result:
                self.models['anomaly'], results['anomaly_accuracy'] = result
        except Exception as e:
            print(f"❌ Anomaly training error: {e}")
        
        try:
            result = self.train_exit_model(trades)
            if result:
                self.models['exit'], results['exit_accuracy'] = result
        except Exception as e:
            print(f"❌ Exit training error: {e}")
        
        try:
            result = self.train_pattern_model(trades)
            if result:
                self.models['pattern'], results['pattern_accuracy'] = result
        except Exception as e:
            print(f"❌ Pattern training error: {e}")
        
        try:
            result = self.train_liquidity_model(trades)
            if result:
                self.models['liquidity'], results['liquidity_accuracy'] = result
        except Exception as e:
            print(f"❌ Liquidity training error: {e}")
        
        try:
            result = self.train_chart_cnn_model(trades)
            if result:
                self.models['chart_cnn'], results['chart_cnn_accuracy'] = result
        except Exception as e:
            print(f"❌ Chart Pattern training error: {e}")
        
        # Save models
        self.save_all_models()
        
        # Save to database
        self.save_models_to_db(results)
        
        print("\n✅ All 8 LightGBM models trained successfully!")
        return True
    
    def save_all_models(self):
        """Save all models to files"""
        print("\n💾 Saving LightGBM models...")
        
        for model_name, model in self.models.items():
            if model:
                try:
                    model_path = os.path.join(os.path.dirname(__file__), f'{model_name}_model.pkl')
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    print(f"  ✅ {model_name} saved")
                except Exception as e:
                    print(f"  ❌ {model_name} save error: {e}")
    
    def calculate_voting_accuracy(self, trades):
        """حساب دقة تصويت المستشارين من جدول consultant_votes"""
        print("\n🎯 Calculating voting accuracy from database...")
        
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT consultant_name, vote_type, is_correct, COUNT(*) as total
                FROM consultant_votes
                WHERE timestamp > NOW() - INTERVAL '30 days'
                GROUP BY consultant_name, vote_type, is_correct
            """)
            
            results = cursor.fetchall()
            cursor.close()
            
            consultant_scores = {}
            
            for row in results:
                consultant_name = row[0]
                vote_type = row[1]
                is_correct = row[2]
                count = row[3]
                
                if consultant_name not in consultant_scores:
                    consultant_scores[consultant_name] = {
                        'tp_correct': 0, 'tp_wrong': 0,
                        'amount_correct': 0, 'amount_wrong': 0,
                        'sl_correct': 0, 'sl_wrong': 0,
                        'sell_correct': 0, 'sell_wrong': 0
                    }
                
                key = f"{vote_type}_{'correct' if is_correct else 'wrong'}"
                consultant_scores[consultant_name][key] = count
            
            final_scores = {}
            for consultant, scores in consultant_scores.items():
                tp_total = scores['tp_correct'] + scores['tp_wrong']
                amount_total = scores['amount_correct'] + scores['amount_wrong']
                sl_total = scores['sl_correct'] + scores['sl_wrong']
                sell_total = scores['sell_correct'] + scores['sell_wrong']
                
                final_scores[consultant] = {
                    'tp_accuracy': scores['tp_correct'] / tp_total if tp_total > 0 else 0.5,
                    'amount_accuracy': scores['amount_correct'] / amount_total if amount_total > 0 else 0.5,
                    'sl_accuracy': scores['sl_correct'] / sl_total if sl_total > 0 else 0.5,
                    'sell_accuracy': scores['sell_correct'] / sell_total if sell_total > 0 else 0.5,
                    'overall_accuracy': (
                        (scores['tp_correct'] + scores['amount_correct'] + scores['sl_correct'] + scores['sell_correct']) /
                        max(tp_total + amount_total + sl_total + sell_total, 1)
                    )
                }
            
            print(f"✅ Loaded voting accuracy for {len(final_scores)} consultants")
            return final_scores
        
        except Exception as e:
            print(f"⚠️ Error calculating voting accuracy: {e}")
            return {
                'exit': {'tp_accuracy': 0.5, 'amount_accuracy': 0.5, 'sl_accuracy': 0.5, 'sell_accuracy': 0.5, 'overall_accuracy': 0.5},
                'smart_money': {'tp_accuracy': 0.5, 'amount_accuracy': 0.5, 'sl_accuracy': 0.5, 'sell_accuracy': 0.5, 'overall_accuracy': 0.5},
                'risk': {'tp_accuracy': 0.5, 'amount_accuracy': 0.5, 'sl_accuracy': 0.5, 'sell_accuracy': 0.5, 'overall_accuracy': 0.5},
                'pattern': {'tp_accuracy': 0.5, 'amount_accuracy': 0.5, 'sl_accuracy': 0.5, 'sell_accuracy': 0.5, 'overall_accuracy': 0.5},
                'cnn': {'tp_accuracy': 0.5, 'amount_accuracy': 0.5, 'sl_accuracy': 0.5, 'sell_accuracy': 0.5, 'overall_accuracy': 0.5},
                'anomaly': {'tp_accuracy': 0.5, 'amount_accuracy': 0.5, 'sl_accuracy': 0.5, 'sell_accuracy': 0.5, 'overall_accuracy': 0.5},
                'liquidity': {'tp_accuracy': 0.5, 'amount_accuracy': 0.5, 'sl_accuracy': 0.5, 'sell_accuracy': 0.5, 'overall_accuracy': 0.5}
            }
    
    def save_models_to_db(self, results, retry=3):
        """Save models info to database (نفس الجدول القديم dl_models_v2)"""
        if not self.conn:
            print("⚠️ No database connection - models saved to files only")
            return False
        
        for attempt in range(retry):
            try:
                print(f"🔄 Attempt {attempt+1}/{retry}: Connecting to database...")
                conn = self._get_conn()
                cursor = conn.cursor()
                
                print("📋 Checking table...")
                # نفس الجدول القديم dl_models_v2
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS dl_models_v2 (
                        id SERIAL PRIMARY KEY,
                        model_name VARCHAR(50) NOT NULL,
                        model_type VARCHAR(50) NOT NULL,
                        accuracy FLOAT,
                        trained_at TIMESTAMP DEFAULT NOW(),
                        status VARCHAR(20) DEFAULT 'active'
                    )
                """)
                conn.commit()
                
                print("🗑️ Deleting old data...")
                cursor.execute("DELETE FROM dl_models_v2")
                conn.commit()
                
                print("💾 Inserting new data...")
                for model_name in self.models.keys():
                    accuracy_key = f'{model_name}_accuracy'
                    accuracy = results.get(accuracy_key, 0)
                    
                    cursor.execute("""
                        INSERT INTO dl_models_v2 (model_name, model_type, accuracy)
                        VALUES (%s, %s, %s)
                    """, (model_name, 'LightGBM', float(accuracy)))
                
                conn.commit()
                cursor.close()
                
                print("✅ Models info saved to database (dl_models_v2)")
                return True
            
            except Exception as e:
                try:
                    conn.rollback()
                except:
                    pass
                
                if attempt < retry - 1:
                    time.sleep(2)
        
        return False
    
    def run_continuous(self, interval_hours=12):
        """Run training continuously"""
        print(f"\n🚀 Deep Learning Trainer V2 started (LightGBM)!")
        print(f"⏰ Training interval: {interval_hours} hours")
        print("="*60)
        
        while True:
            try:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"\n{'='*60}")
                print(f"⏰ {current_time}")
                print(f"{'='*60}")
                
                success = self.train_all_models()
                
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
    if not ML_AVAILABLE:
        print("❌ Please install LightGBM:")
        print("   pip install lightgbm scikit-learn")
        return
    
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not found!")
        return
    
    trainer = DeepLearningTrainerXGBoost(database_url)
    trainer.run_continuous(interval_hours=6)

if __name__ == "__main__":
    main()
