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
        """Train all models - only on new trades since last training."""
        print("\n" + "=" * 60)
        print("👑 Starting Training - 12 LightGBM Models")
        print("=" * 60)

        new_trades_count = self.db.get_new_trades_count()
        print(f"📊 New trades since last training: {new_trades_count}")

        if new_trades_count == 0:
            print("⏭️ No new trades found. Skipping training cycle.")
            return False

        print(f"✅ Found {new_trades_count} new trades. Starting training...")
        trades = self.db.load_training_data()
        if not trades:
            return False

        results = {}
        trained_consultants = {}

        try:
            voting_scores = self.db.calculate_voting_accuracy(trades)
        except Exception as e:
            print(f"⚠️ Voting accuracy error: {e}")
            voting_scores = {}

        for model_name, train_fn in TRAIN_PIPELINE:
            if model_name == 'meta_learner':
                continue
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
            meta_result = train_meta_learner_model(self.db, trained_consultants, voting_scores)
            if meta_result:
                self.models['meta_learner'], results['meta_learner_accuracy'] = meta_result
        except Exception as e:
            print(f"❌ meta_learner training error: {e}")
            send_critical_alert("Model Training Error", "Meta-Learner failed to train", str(e))

        self.save_all_models()
        self.db.save_models_to_db(self.models, results)

        print("\n✅ All 12 LightGBM models trained successfully!")
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
