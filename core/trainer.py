"""
🤖 Deep Learning Trainer - Main Class
Orchestrates training of all 12 LightGBM models and runs continuous loop.
"""

import os
import sys
import pickle
import time
from datetime import datetime, timedelta

# ── Add models folder to path ─────────────────────────────────
_core_dir    = os.path.dirname(os.path.abspath(__file__))
_root_dir    = os.path.dirname(_core_dir)
_models_dir  = os.path.join(_root_dir, 'models')

if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)

from database import get_db_connection, close_db_connection
from db_manager import DatabaseManager
from alerts import send_critical_alert

from chart_cnn_model import train_chart_cnn_model
from sentiment_model import train_sentiment_model
from crypto_news_model import train_crypto_news_model
from volume_prediction_model import train_volume_prediction_model
from candle_expert_model import train_candle_expert_model
from meta_trading import train_meta_trading
from smart_money_model import train_smart_money_model
from risk_manager_model import train_risk_model
from anomaly_detector_model import train_anomaly_model
from exit_strategy_model import train_exit_strategy_model
from pattern_recognition_model import train_pattern_recognition_model
from liquidity_analyzer_model import train_liquidity_model


TRAIN_PIPELINE = [
    ('smart_money',   train_smart_money_model),
    ('risk',          train_risk_model),
    ('anomaly',       train_anomaly_model),
    ('exit',          train_exit_strategy_model),
    ('pattern',       train_pattern_recognition_model),
    ('liquidity',     train_liquidity_model),
    ('chart_cnn',     train_chart_cnn_model),
    ('candle_expert', train_candle_expert_model),
    ('sentiment',     train_sentiment_model),
    ('crypto_news',   train_crypto_news_model),
    ('volume_pred',   train_volume_prediction_model),
    ('meta_trading',  train_meta_trading),
]


class DeepLearningTrainerLightGBM:
    def __init__(self):
        self.db                  = DatabaseManager()
        self.models              = {name: None for name, _ in TRAIN_PIPELINE}
        self.trained_consultants = {}
        self._load_models_from_db()

    def _load_models_from_db(self):
        """Load existing models from database on startup."""
        conn = None
        try:
            conn = self.db._get_conn()
            if not conn:
                return

            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM dl_models_v2")
                count = cursor.fetchone()[0]

            if count == 0:
                print("ℹ️ No saved models found. Will train from scratch.")
                return

            print(f"✅ Found {count} saved model(s) in database. Loading...")

            trained_models_dir = os.path.join(_root_dir, 'trained_models')

            loaded = 0
            for model_name in self.models:
                path = os.path.join(trained_models_dir, f'{model_name}_model.pkl')
                if os.path.exists(path):
                    try:
                        with open(path, 'rb') as f:
                            self.models[model_name] = pickle.load(f)
                        loaded += 1
                    except Exception as e:
                        print(f"⚠️ Could not load {model_name}: {e}")

            print(f"✅ Loaded {loaded}/{len(self.models)} models from disk.")

        except Exception as e:
            print(f"⚠️ Could not load models from DB: {e}")
        finally:
            if conn:
                close_db_connection(conn)

    # ========== Training ==========

    def train_all_models(self):
        """Train models - missing models train on ALL trades, existing models train on NEW trades only."""
        print("\n" + "=" * 60)
        print("👑 Starting Training - 12 LightGBM Models")
        print("=" * 60)

        should_train, since_ts = self.db.check_training_trigger(300)
        missing_models         = self.db.get_missing_models()

        if not should_train and not missing_models:
            print("⏭️ Skipping cycle: No enough new trades (+300) since last attempt.")
            return False

        if missing_models:
            print("📊 Missing models will train on ALL trades")
            trades_all = self.db.load_training_data(since_timestamp=None)
            trades_new = trades_all if since_ts is None else self.db.load_training_data(since_timestamp=since_ts)
        else:
            trades_all = None
            trades_new = self.db.load_training_data(since_timestamp=since_ts)

        if not trades_new and not trades_all:
            print("❌ No data available for training.")
            return False

        print("📊 Recalculating symbol_memory from trades...")
        self.db.recalculate_symbol_memory()

        results       = {}
        voting_scores = self.db.calculate_voting_accuracy(trades_new or trades_all)

        # ── Train all models except meta_trading ──────────────────
        for model_name, train_fn in TRAIN_PIPELINE:
            if model_name == 'meta_trading':
                continue

            if model_name in missing_models:
                if not trades_all:
                    print(f"  ⏭️ {model_name}: skipping (no data)")
                    continue
                trades_to_use = trades_all
                print(f"  📊 {model_name}: training on ALL trades ({len(trades_to_use)})")
            else:
                if not trades_new:
                    print(f"  ⏭️ {model_name}: skipping (no new trades)")
                    continue
                trades_to_use = trades_new
                print(f"  📊 {model_name}: training on NEW trades ({len(trades_to_use)})")

            try:
                result = train_fn(trades_to_use, voting_scores)
                if result:
                    model, accuracy = result
                    self.models[model_name]              = model
                    self.trained_consultants[model_name] = model
                    results[f'{model_name}_accuracy']    = accuracy
                    print(f"  ✅ {model_name}: Accuracy {accuracy * 100:.2f}%")
            except Exception as e:
                print(f"  ❌ {model_name} training error: {e}")

        # ── Train meta_trading ────────────────────────────────────
        try:
            if 'meta_trading' in missing_models:
                if trades_all:
                    print(f"\n👑 Meta-Model: training on ALL trades ({len(trades_all)})...")
                    meta_result = train_meta_trading(
                        trades_all, voting_scores,
                        since_timestamp=None,
                        db_manager=self.db,
                        trained_consultants=self.trained_consultants,
                    )
                    if meta_result:
                        self.models['meta_trading'], results['meta_trading_accuracy'] = meta_result
                        print(f"  ✅ Meta-Model: Accuracy {results['meta_trading_accuracy'] * 100:.2f}%")
                else:
                    print("\n👑 Meta-Model: skipping (no data)")
            else:
                if trades_new:
                    print(f"\n👑 Meta-Model: training on NEW trades ({len(trades_new)})...")
                    oldest_timestamp = self.db.get_oldest_model_timestamp()
                    meta_result = train_meta_trading(
                        trades_new, voting_scores,
                        since_timestamp=oldest_timestamp,
                        db_manager=self.db,
                        trained_consultants=self.trained_consultants,
                    )
                    if meta_result:
                        self.models['meta_trading'], results['meta_trading_accuracy'] = meta_result
                        print(f"  ✅ Meta-Model: Accuracy {results['meta_trading_accuracy'] * 100:.2f}%")
                else:
                    print("\n👑 Meta-Model: skipping (no new trades)")
        except Exception as e:
            print(f"  ❌ meta_trading training error: {e}")

        self._load_existing_accuracies(results)
        self.save_all_models()
        success = self.db.save_models_to_db(self.models, results)

        if success:
            print("\n✅ All models trained and saved successfully!")
        else:
            print("\n⚠️ Training completed but some models failed to save to database")

        return True

    def _load_existing_accuracies(self, results):
        """Load accuracies from database for models that didn't train this session."""
        conn = None
        try:
            conn = self.db._get_conn()
            if not conn:
                return
            with conn.cursor() as cursor:
                cursor.execute("SELECT model_name, accuracy FROM dl_models_v2")
                for name, acc in cursor.fetchall():
                    key = f"{name}_accuracy"
                    if key not in results and acc is not None:
                        results[key] = float(acc)
        except Exception as e:
            print(f"⚠️ Could not load existing accuracies: {e}")
        finally:
            if conn:
                close_db_connection(conn)

    def save_all_models(self):
        """Save all trained models to .pkl files."""
        print("\n💾 Saving LightGBM models...")
        trained_models_dir = os.path.join(_root_dir, 'trained_models')
        os.makedirs(trained_models_dir, exist_ok=True)

        for model_name, model in self.models.items():
            if model is None:
                continue
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
                print(f"\n{'=' * 60}")
                print(f"🎯 Training triggered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                success = self.train_all_models()

                if success:
                    print("\n✅ Training cycle completed successfully.")

                next_run = datetime.now() + timedelta(hours=interval_hours)
                print(f"\n{'-' * 60}")
                print(f"⏰ Next training cycle: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")

                while datetime.now() < next_run:
                    time.sleep(60)

            except KeyboardInterrupt:
                print("\n🛑 Trainer stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                send_critical_alert("Training Loop Error", "Training loop encountered an error", str(e))
                time.sleep(300)