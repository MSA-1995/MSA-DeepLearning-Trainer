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
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consultants.models import (
    train_smart_money_model,
    train_risk_model,
    train_anomaly_model,
    train_exit_model,
    train_pattern_model,
    train_liquidity_model,
    train_chart_cnn_model,
    train_meta_learner_model, # 👑🧠 The New King!
)
from models.sentiment_model import train_sentiment_model
from models.crypto_news_model import train_crypto_news_model
from models.volume_prediction_model import train_volume_prediction_model

# Ordered list of consultants for the king to learn from
TRAIN_PIPELINE = [
    ('smart_money', train_smart_money_model),
    ('risk',        train_risk_model),
    ('anomaly',     train_anomaly_model),
    ('exit',        train_exit_model),
    ('pattern',     train_pattern_model),
    ('liquidity',   train_liquidity_model),
    ('chart_cnn',   train_chart_cnn_model),
    ('sentiment',   train_sentiment_model),  # 🎭 New: Sentiment Analysis
    ('crypto_news', train_crypto_news_model),  # 📰 New: Crypto News
    ('volume_pred', train_volume_prediction_model),  # 📊 New: Volume Prediction
    ('meta_learner', train_meta_learner_model), # 👑🧠 The New King!
]


class DeepLearningTrainerXGBoost:
    def __init__(self):
        self.db     = DatabaseManager()
        self.models = {name: None for name, _ in TRAIN_PIPELINE}


    # ========== Training ==========

    def train_all_models(self):
        """Train all models sequentially, with Meta-Learner at the end."""
        print("\n" + "=" * 60)
        print("👑 Starting Training - 12 LightGBM Models")
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

        # Now, train the Meta-Learner using the trained consultants and the db manager
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
        base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to MSA-DeepLearning-Trainer
        trained_models_dir = os.path.join(base_dir, 'trained_models')
        
        # Create trained_models directory if it doesn't exist
        if not os.path.exists(trained_models_dir):
            os.makedirs(trained_models_dir)
            print(f"  📁 Created directory: {trained_models_dir}")
        
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
                # --- تدريب فوري --- 
                print(f"\n{'='*60}")
                print(f"🎯 Training triggered: Initial run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                success = self.train_all_models()

                if success:
                    print("\n✅ Training cycle completed successfully.")

                # --- جدولة الدورة القادمة --- 
                next_training_time = datetime.now() + timedelta(hours=interval_hours)
                print(f"\n{'-'*60}")
                print(f"⏰ Next training cycle scheduled for: {next_training_time.strftime('%Y-%m-%d %H:%M:%S')}")

                # --- الانتظار --- 
                while datetime.now() < next_training_time:
                    # طباعة رسالة انتظار كل 30 دقيقة
                    remaining_time = next_training_time - datetime.now()
                    # تحويل الوقت المتبقي إلى ساعات ودقائق
                    hours, remainder = divmod(remaining_time.total_seconds(), 3600)
                    minutes, _ = divmod(remainder, 60)
                    time.sleep(60) # تحقق كل دقيقة
                


            except KeyboardInterrupt:
                print("\n🛑 Trainer stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                send_critical_alert("Training Loop Error", "Training loop encountered an error", str(e))
                time.sleep(300)