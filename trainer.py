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
    train_ai_brain_model,
    train_smart_money_model,
    train_risk_model,
    train_anomaly_model,
    train_exit_model,
    train_pattern_model,
    train_liquidity_model,
    train_chart_cnn_model,
    train_rescue_model,
)

# Ordered list: AI Brain first, then consultants
TRAIN_PIPELINE = [
    ('ai_brain',    train_ai_brain_model),
    ('smart_money', train_smart_money_model),
    ('risk',        train_risk_model),
    ('anomaly',     train_anomaly_model),
    ('exit',        train_exit_model),
    ('pattern',     train_pattern_model),
    ('liquidity',   train_liquidity_model),
    ('chart_cnn',   train_chart_cnn_model),
    ('rescue',      train_rescue_model), # 🤪 Added The Jester
]


class DeepLearningTrainerXGBoost:
    def __init__(self):
        self.db     = DatabaseManager()
        self.models = {name: None for name, _ in TRAIN_PIPELINE}
        print("🧠 Deep Learning Trainer V2 initialized (9 Models - LightGBM)")

    # ========== Training ==========

    def train_all_models(self):
        """Train all 8 models sequentially"""
        print("\n" + "=" * 60)
        print("👑 Starting Training - 9 LightGBM Models")
        print("=" * 60)

        trades = self.db.load_training_data()
        if not trades:
            return False

        results = {}

        # Calculate consultant voting accuracy first
        try:
            voting_scores = self.db.calculate_voting_accuracy(trades)
        except Exception as e:
            print(f"⚠️ Voting accuracy error: {e}")
            voting_scores = {}

        # Train each model
        for model_name, train_fn in TRAIN_PIPELINE:
            try:
                result = train_fn(trades, voting_scores)
                if result:
                    self.models[model_name], results[f'{model_name}_accuracy'] = result
            except Exception as e:
                print(f"❌ {model_name} training error: {e}")
                if model_name == 'ai_brain':
                    send_critical_alert("Model Training Error", "AI Brain failed to train", str(e))

        self.save_all_models()
        self.db.save_models_to_db(list(self.models.keys()), results)

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

    def run_continuous(self, interval_hours=6, trades_trigger=100):
        """Run training loop — triggers every N hours OR every N new trades"""
        print(f"\n🚀 Deep Learning Trainer V2 started (LightGBM)!")
        print(f"⏰ Training triggers:")
        print(f"   • Every {interval_hours} hours")
        print(f"   • Every {trades_trigger} new trades")
        print("=" * 60)

        last_training_time = datetime.now()
        last_trade_count   = 0

        while True:
            try:
                current_time        = datetime.now().strftime("%H:%M:%S")
                hours_since         = (datetime.now() - last_training_time).total_seconds() / 3600
                new_trades          = self.db.get_new_trades_count()
                new_trades_since    = new_trades - last_trade_count

                should_train   = False
                trigger_reason = ""

                if hours_since >= interval_hours:
                    should_train   = True
                    trigger_reason = f"{interval_hours} hours passed"
                elif new_trades_since >= trades_trigger:
                    should_train   = True
                    trigger_reason = f"{new_trades_since} new trades"

                if should_train:
                    print(f"\n{'='*60}")
                    print(f"⏰ {current_time}")
                    print(f"🎯 Training triggered: {trigger_reason}")
                    print(f"{'='*60}")

                    success = self.train_all_models()

                    if success:
                        print("\n✅ Training successful")
                        last_training_time = datetime.now()
                        last_trade_count   = new_trades
                    else:
                        print("\n⚠️ Training skipped - not enough data")

                    next_time = (datetime.now() + timedelta(hours=interval_hours)).strftime("%H:%M:%S")
                    print(f"\n⏰ Next check:")
                    print(f"   • Time-based:  {next_time} ({interval_hours}h)")
                    print(f"   • Trade-based: {trades_trigger - new_trades_since} trades remaining")

                else:
                    # Status update every 30 minutes
                    if int(hours_since * 60) % 30 == 0:
                        print(f"⏳ {current_time} | Waiting... "
                              f"({new_trades_since}/{trades_trigger} trades, "
                              f"{hours_since:.1f}/{interval_hours}h)")

                time.sleep(60)

            except KeyboardInterrupt:
                print("\n🛑 Trainer stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                send_critical_alert("Training Loop Error", "Training loop encountered an error", str(e))
                print("⏰ Retrying in 5 minutes...")
                time.sleep(300)
