"""
🤖 Deep Learning Trainer - Main Class
Orchestrates training of all 12 LightGBM models and runs continuous loop.
"""

import os
import pickle
import time
from datetime import datetime, timedelta

from database import get_db_connection, close_db_connection
from db_manager import DatabaseManager
from alerts import send_critical_alert
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consultants.models import (
    train_smart_money_model,
    train_risk_model,
    train_anomaly_model,
    train_exit_model,
    train_pattern_model,
    train_liquidity_model,
    train_chart_cnn_model,
    train_meta_learner_model,
)
from models.sentiment_model import train_sentiment_model
from models.crypto_news_model import train_crypto_news_model
from models.volume_prediction_model import train_volume_prediction_model

TRAIN_PIPELINE = [
    ('smart_money', train_smart_money_model),
    ('risk',        train_risk_model),
    ('anomaly',     train_anomaly_model),
    ('exit',        train_exit_model),
    ('pattern',     train_pattern_model),
    ('liquidity',   train_liquidity_model),
    ('chart_cnn',   train_chart_cnn_model),
    ('sentiment',   train_sentiment_model),
    ('crypto_news', train_crypto_news_model),
    ('volume_pred', train_volume_prediction_model),
    ('meta_learner', train_meta_learner_model),
]


class DeepLearningTrainerLightGBM:
    def __init__(self):
        self.db     = DatabaseManager()
        self.models = {name: None for name, _ in TRAIN_PIPELINE}
        self._load_models_from_db()

    def _load_models_from_db(self):
        """Load existing models from database on startup."""
        try:
            conn = self.db._get_conn()
            if not conn:
                return
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM dl_models_v2")
                count = cursor.fetchone()[0]
            close_db_connection(conn)

            if count == 0:
                print("ℹ️ No saved models found. Will train from scratch.")
                return

            print(f"✅ Found {count} saved models in database. Loading...")
            base_dir = os.path.dirname(os.path.dirname(__file__))
            trained_models_dir = os.path.join(base_dir, 'trained_models')

            loaded = 0
            for model_name in self.models.keys():
                path = os.path.join(trained_models_dir, f'{model_name}_model.pkl')
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    loaded += 1

            print(f"✅ Loaded {loaded}/{len(self.models)} models from disk.")

        except Exception as e:
            print(f"⚠️ Could not load models from DB: {e}")

    # ========== Training ==========

    def train_all_models(self):
        """Train models - missing models train on ALL trades, existing models train on NEW trades only."""
        print("\n" + "=" * 60)
        print("👑 Starting Training - 11 LightGBM Models")
        print("=" * 60)

        new_trades_count, since_timestamp = self.db.get_new_trades_count()
        
        if new_trades_count == 0:
            print("⏭️ No new trades or news found. Skipping training cycle.")
            return False

        # Handle per-model training for missing models
        if new_trades_count == -1 and isinstance(since_timestamp, list):
            missing_models = since_timestamp
            print(f"ℹ️ Missing models: {missing_models}")
            print(f"📊 Missing models will train on ALL trades")
            
            # Separate sentiment/crypto_news from other missing models
            missing_trade_models = [m for m in missing_models if m not in ['sentiment', 'crypto_news']]
            missing_news_models = [m for m in missing_models if m in ['sentiment', 'crypto_news']]
            
            if missing_trade_models:
                trades_all = self.db.load_training_data(since_timestamp=None)
            else:
                trades_all = None
            
            trades_new = self.db.load_training_data(since_timestamp=self.db.get_existing_model_timestamp())
            if trades_all is None:
                trades_all = trades_new
        elif since_timestamp is None:
            # First training → load ALL trades
            print(f"📊 First training → loading ALL trades")
            trades_all = self.db.load_training_data(since_timestamp=None)
            trades_new = trades_all
            missing_models = [
                'smart_money', 'risk', 'anomaly', 'exit', 'pattern',
                'liquidity', 'chart_cnn', 'sentiment', 'crypto_news',
                'volume_pred', 'meta_learner'
            ]
            if not trades_all:
                return False
        else:
            print(f"📊 Loading new data since {since_timestamp}")
            trades_all = None
            missing_models = []
            
            # Check if only news (no new trades)
            if since_timestamp == "NEWS_ONLY":
                print("📰 Only new news found - sentiment and crypto_news will train on NEW only")
                # Get oldest training time to use as real timestamp
                oldest_training = self.db.get_existing_model_timestamp()
                since_timestamp = oldest_training  # Replace "NEWS_ONLY" with real timestamp
                trades_new = self.db.load_training_data(since_timestamp=oldest_training) if oldest_training else []
                if not trades_new:
                    trades_new = []
            else:
                trades_new = self.db.load_training_data(since_timestamp=since_timestamp)
                if not trades_new:
                    trades_new = []
        
        results = {}
        trained_consultants = {}

        try:
            voting_scores = self.db.calculate_voting_accuracy(trades_new)
        except Exception as e:
            print(f"⚠️ Voting accuracy error: {e}")
            voting_scores = {}

        for model_name, train_fn in TRAIN_PIPELINE:
            if model_name == 'meta_learner':
                continue
            
            # sentiment and crypto_news always train from news_sentiment table
            if model_name in ['sentiment', 'crypto_news']:
                if model_name in missing_models:
                    print(f"  📊 {model_name}: training from news_sentiment (missing model)")
                elif new_trades_count > 0:
                    print(f"  📊 {model_name}: training from news_sentiment (new data)")
                else:
                    print(f"  ⏭️ {model_name}: skipping (no new news)")
                    continue
                
                try:
                    # Pass since_timestamp to train only new data
                    ts_for_news = None if model_name in missing_models else since_timestamp
                    result = train_fn(trades_new, voting_scores, since_timestamp=ts_for_news)
                    if result:
                        model, accuracy = result
                        self.models[model_name] = model
                        trained_consultants[model_name] = model
                        results[f'{model_name}_accuracy'] = accuracy
                except Exception as e:
                    print(f"❌ {model_name} training error: {e}")
                continue
            
            # Skip other models if no new trades
            if model_name not in missing_models:
                if not trades_new or len(trades_new) == 0:
                    print(f"  ⏭️ {model_name}: skipping (no new trades)")
                    continue
                trades = trades_new
                print(f"  📊 {model_name}: training on NEW trades only")
            else:
                # Missing model - train on all trades
                if not trades_all or len(trades_all) == 0:
                    print(f"  ⏭️ {model_name}: skipping (no data)")
                    continue
                trades = trades_all
                print(f"  📊 {model_name}: training on ALL trades")
            
            try:
                result = train_fn(trades, voting_scores)
                if result:
                    model, accuracy = result
                    self.models[model_name] = model
                    trained_consultants[model_name] = model
                    results[f'{model_name}_accuracy'] = accuracy
            except Exception as e:
                print(f"❌ {model_name} training error: {e}")

        try:
            # Meta-learner trains ONLY when there are NEW trades (never when models are missing)
            if new_trades_count > 0 and isinstance(since_timestamp, str) and since_timestamp != "NEWS_ONLY":
                print(f"\n👑🧠 Meta-Learner: training with new trades since {since_timestamp}...")
                meta_result = train_meta_learner_model(self.db, trained_consultants, voting_scores, since_timestamp=since_timestamp)
                if meta_result:
                    self.models['meta_learner'], results['meta_learner_accuracy'] = meta_result
            else:
                print("\n👑🧠 Meta-Learner: skipping (no new trades)")
        except Exception as e:
            print(f"❌ meta_learner training error: {e}")
            send_critical_alert("Model Training Error", "Meta-Learner failed to train", str(e))

        # Load old accuracies for models that didn't train this session
        try:
            conn = self.db._get_conn()
            if conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT model_name, accuracy FROM dl_models_v2")
                    for row in cursor.fetchall():
                        name, acc = row
                        key = f"{name}_accuracy"
                        if key not in results and acc is not None:
                            results[key] = float(acc)
                from database import close_db_connection
                close_db_connection(conn)
        except Exception as e:
            print(f"Could not load old accuracies: {e}")

        self.save_all_models()
        self.db.save_models_to_db(self.models, results)

        print("\n✅ All LightGBM models trained successfully!")
        return True

    def save_all_models(self):
        """Save all trained models to .pkl files"""
        print("\n💾 Saving LightGBM models...")
        base_dir = os.path.dirname(os.path.dirname(__file__))
        trained_models_dir = os.path.join(base_dir, 'trained_models')

        if not os.path.exists(trained_models_dir):
            os.makedirs(trained_models_dir)

        for model_name, model in self.models.items():
            if model:
                try:
                    path = os.path.join(trained_models_dir, f'{model_name}_model.pkl')
                    with open(path, 'wb') as f:
                        pickle.dump(model, f)
                    print(f"  ✅ {model_name} saved to trained_models/")
                except Exception as e:
                    print(f"  ❌ {model_name} save error: {e}")

    # ========== Continuous Loop ==========

    def run_continuous(self, interval_hours=6):
        """Run training loop — trains immediately, then every N hours."""
        print(f"\n🚀 Deep Learning Trainer V2 started (LightGBM)!")
        print(f"⏰ Training triggers: Immediately, then every {interval_hours} hours.")
        print("=" * 60)

        while True:
            try:
                print(f"\n{'='*60}")
                print(f"🎯 Training triggered: Initial run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                success = self.train_all_models()

                if success:
                    print("\n✅ Training cycle completed successfully.")

                next_training_time = datetime.now() + timedelta(hours=interval_hours)
                print(f"\n{'-'*60}")
                print(f"⏰ Next training cycle scheduled for: {next_training_time.strftime('%Y-%m-%d %H:%M:%S')}")

                while datetime.now() < next_training_time:
                    time.sleep(60)

            except KeyboardInterrupt:
                print("\n🛑 Trainer stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                send_critical_alert("Training Loop Error", "Training loop encountered an error", str(e))
                time.sleep(300)
