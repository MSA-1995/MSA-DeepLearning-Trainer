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
)
from models.sentiment_model import train_sentiment_model
from models.crypto_news_model import train_crypto_news_model
from models.volume_prediction_model import train_volume_prediction_model
from models.meta_trading import train_meta_trading


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
    ('meta_trading',  train_meta_trading),
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

        # Recalculate symbol_memory first (needed for meta_trading training)
        print("📊 Recalculating symbol_memory from trades...")
        self.db.recalculate_symbol_memory()

        # Get missing models list
        missing_models = self.db.get_missing_models()
        
        # Determine what data to load
        if missing_models:
            print(f"📊 Missing models will train on ALL trades")
            print(f"📊 Existing models will train on NEW trades")
            
            # Load ALL trades for missing models
            trades_all = self.db.load_training_data(since_timestamp=None)
            if not trades_all:
                print("❌ No trades found in database")
                return False
            
            # Load NEW trades for existing models (from oldest model timestamp)
            oldest_timestamp = self.db.get_oldest_model_timestamp()
            trades_new = self.db.load_training_data(since_timestamp=oldest_timestamp)
        else:
            # All models exist - check for new trades
            oldest_timestamp = self.db.get_oldest_model_timestamp()
            if not oldest_timestamp:
                print("⚠️ No models found, starting first training")
                trades_all = self.db.load_training_data(since_timestamp=None)
                trades_new = trades_all
                missing_models = [
                    'smart_money', 'risk', 'anomaly', 'exit', 'pattern',
                    'liquidity', 'chart_cnn', 'sentiment', 'crypto_news',
                    'volume_pred', 'meta_trading'
                ]
                if not trades_all:
                    return False
            else:
                trades_new = self.db.load_training_data(since_timestamp=oldest_timestamp)
                if not trades_new or len(trades_new) == 0:
                    print("⏭️ No new trades found. Skipping training cycle.")
                    return False
                # Minimum 50 trades required for quality training
                if len(trades_new) < 50:
                    print(f"⏭️ Only {len(trades_new)} new trades found. Waiting for at least 50 trades before training.")
                    return False
                trades_all = None
        
        results = {}
        trained_consultants = {}

        try:
            voting_scores = self.db.calculate_voting_accuracy(trades_new if trades_new else [])
        except Exception as e:
            print(f"⚠️ Voting accuracy error: {e}")
            voting_scores = {}

        # Train all models except meta_learner
        for model_name, train_fn in TRAIN_PIPELINE:
            if model_name == 'meta_learner':
                continue
            
            # Determine which trades to use
            if model_name in missing_models:
                # Missing model - train on ALL trades
                if not trades_all or len(trades_all) == 0:
                    print(f"  ⏭️ {model_name}: skipping (no data)")
                    continue
                trades_to_use = trades_all
                print(f"  📊 {model_name}: training on ALL trades ({len(trades_to_use)})")
            else:
                # Existing model - train on NEW trades only
                if not trades_new or len(trades_new) == 0:
                    print(f"  ⏭️ {model_name}: skipping (no new trades)")
                    continue
                trades_to_use = trades_new
                print(f"  📊 {model_name}: training on NEW trades ({len(trades_to_use)})")
            
            try:
                result = train_fn(trades_to_use, voting_scores)
                if result:
                    model, accuracy = result
                    self.models[model_name] = model
                    trained_consultants[model_name] = model
                    results[f'{model_name}_accuracy'] = accuracy
                    print(f"  ✅ {model_name}: Accuracy {accuracy*100:.2f}%")
            except Exception as e:
                print(f"  ❌ {model_name} training error: {e}")

        # Train meta_trading with db_manager
        try:
            if 'meta_trading' in missing_models:
                # Missing - train on ALL trades
                if trades_all and len(trades_all) > 0:
                     print(f"\n👑 Meta-Model: training on ALL trades ({len(trades_all)})...")
                     meta_result = train_meta_trading(trades_all, voting_scores, since_timestamp=None, db_manager=self.db)
                     if meta_result:
                         self.models['meta_trading'], results['meta_trading_accuracy'] = meta_result
                         print(f"  ✅ Meta-Model: Accuracy {results['meta_trading_accuracy']*100:.2f}%")
                else:
                    print(f"\n👑 Meta-Model: skipping (no data)")
            else:
                # Existing - train on NEW trades
                if trades_new and len(trades_new) > 0:
                     print(f"\n👑 Meta-Model: training on NEW trades ({len(trades_new)})...")
                     oldest_timestamp = self.db.get_oldest_model_timestamp()
                     meta_result = train_meta_trading(trades_new, voting_scores, since_timestamp=oldest_timestamp, db_manager=self.db)
                     if meta_result:
                         self.models['meta_trading'], results['meta_trading_accuracy'] = meta_result
                         print(f"  ✅ Meta-Model: Accuracy {results['meta_trading_accuracy']*100:.2f}%")
                else:
                    print(f"\n👑 Meta-Model: skipping (no new trades)")
        except Exception as e:
            print(f"  ❌ meta_trading training error: {e}")

        # Load old accuracies for models that didn't train this session
        self._load_existing_accuracies(results)

        # Save models to disk and database
        self.save_all_models()
        success = self.db.save_models_to_db(self.models, results)
        
        if success:
            print("\n✅ All models trained and saved successfully!")
        else:
            print("\n⚠️ Training completed but some models failed to save to database")
        
        return True

    def _load_existing_accuracies(self, results):
        """Load accuracies from database for models that didn't train this session."""
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
                close_db_connection(conn)
        except Exception as e:
            print(f"⚠️ Could not load existing accuracies: {e}")

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
