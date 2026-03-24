"""
🤖 Deep Learning Trainer - Main Class
Orchestrates training of all 8 LightGBM models and runs continuous loop.
"""

import os
import pickle
import time
from datetime import datetime, timedelta

from database import get_db_connection, close_db_connection
from db_manager import DatabaseManager
from alerts import send_critical_alert
from models import (
    train_smart_money_model,
    train_risk_model,
    train_anomaly_model,
    train_exit_model,
    train_pattern_model,
    train_liquidity_model,
    train_chart_cnn_model,
    train_rescue_model,
    train_meta_learner_model, # 👑🧠 The New King!
)

# Ordered list of consultants for the king to learn from
TRAIN_PIPELINE = [
    ('smart_money', train_smart_money_model),
    ('risk',        train_risk_model),
    ('anomaly',     train_anomaly_model),
    ('exit',        train_exit_model),
    ('pattern',     train_pattern_model),
    ('liquidity',   train_liquidity_model),
    ('chart_cnn',   train_chart_cnn_model),
    ('rescue',      train_rescue_model), # 🤪 Added The Jester
    ('meta_learner', train_meta_learner_model), # 👑🧠 The New King!
]


class DeepLearningTrainerXGBoost:
    def __init__(self):
        self.db     = DatabaseManager()
        self.models = {name: None for name, _ in TRAIN_PIPELINE}
        print("🧠 Deep Learning Trainer V2 initialized (LightGBM)")

    # ========== Training ==========

    def train_all_models(self):
        """Train all models sequentially, with Meta-Learner at the end."""
        print("\n" + "=" * 60)
        print("👑 Starting Training - 9 LightGBM Models")
        print("=" * 60)

        trades = self.db.load_training_data()
        if not trades:
            return False

        results = {}
        trained_consultants = {}

        # Calculate consultant voting accuracy first
        try:
            voting_scores = self.db.calculate_voting_accuracy(trades)
        except Exception as e:
            print(f"⚠️ Voting accuracy error: {e}")
            voting_scores = {}

        # Train each model (except Meta-Learner)
        for model_name, train_fn in TRAIN_PIPELINE:
            if model_name == 'meta_learner':
                continue # Skip for now
            try:
                result = train_fn(trades, voting_scores)
                if result:
                    model, accuracy = result
                    self.models[model_name] = model
                    trained_consultants[model_name] = model # Save for the king
                    results[f'{model_name}_accuracy'] = accuracy
            except Exception as e:
                print(f"❌ {model_name} training error: {e}")
                if model_name == 'ai_brain':
                    send_critical_alert("Model Training Error", "AI Brain failed to train", str(e))

        # Now, train the Meta-Learner using the trained consultants and the db manager
        try:
            meta_result = train_meta_learner_model(self.db, trained_consultants)
            if meta_result:
                self.models['meta_learner'], results['meta_learner_accuracy'] = meta_result
        except Exception as e:
            print(f"❌ meta_learner training error: {e}")
            send_critical_alert("Model Training Error", "Meta-Learner failed to train", str(e))


        self.save_all_models()
        self.db.save_models_to_db(self.models, results)

        print("\n✅ All 9 LightGBM models trained successfully!")
        return True

    def save_all_models(self):
        """Save all trained models to .pkl files"""
        print("\n💾 Saving LightGBM models...")
        base_dir = os.path.dirname(__file__)
        for model_name, model in self.models.items():
            if model:
                try:
                    path = os.path.join(base_dir, f'{model_name}_model.pkl')
                    with open(path, 'wb') as f:
                        pickle.dump(model, f)
                    print(f"  ✅ {model_name} saved")
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
                # --- تدريب فوري --- 
                print(f"\n{'='*60}")
                print(f"🎯 Training triggered: Initial run at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
                print(f"{'='*60}")

                success = self.train_all_models()

                if success:
                    print("\n✅ Training cycle completed successfully.")
                else:
                    print("\n⚠️ Training skipped - not enough data.")

                # --- جدولة الدورة القادمة --- 
                next_training_time = datetime.now() + timedelta(hours=interval_hours)
                print(f"\n{'-'*60}")
                print(f"⏰ Next training cycle scheduled for: {next_training_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'-'*60}")

                # --- الانتظار --- 
                while datetime.now() < next_training_time:
                    # طباعة رسالة انتظار كل 30 دقيقة
                    remaining_time = next_training_time - datetime.now()
                    # تحويل الوقت المتبقي إلى ساعات ودقائق
                    hours, remainder = divmod(remaining_time.total_seconds(), 3600)
                    minutes, _ = divmod(remainder, 60)
                    print(f"⏳ Waiting for next cycle... Time remaining: {int(hours)}h {int(minutes)}m", end="\r")
                    time.sleep(60) # تحقق كل دقيقة
                
                print("\n") # سطر جديد قبل بدء الدورة التالية

            except KeyboardInterrupt:
                print("\n🛑 Trainer stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                send_critical_alert("Training Loop Error", "Training loop encountered an error", str(e))
                print(f"⏰ Retrying in 5 minutes...")
                time.sleep(300)
