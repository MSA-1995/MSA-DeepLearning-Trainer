"""
🗄️ Database Manager
Handles all PostgreSQL operations: connect, load, save.
"""

import json
import pickle
from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import RealDictCursor

from alerts import send_critical_alert
from database import get_db_connection, close_db_connection


class DatabaseManager:
    def __init__(self):
        self.min_trades_for_training = 100

    # ========== Connection ==========

    def _get_conn(self):
        """Return valid connection"""
        return get_db_connection()

    # ========== Load ==========

    def load_training_data(self):
        """Load historical SELL trades for training"""
        conn = self._get_conn()
        if not conn:
            return None
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT symbol, profit_percent, action, timestamp, data
                FROM trades_history
                WHERE action = 'SELL'
                  AND data IS NOT NULL
                ORDER BY timestamp ASC
                LIMIT 2000
            """)
            trades = cursor.fetchall()
            cursor.close()
            close_db_connection(conn)

            if len(trades) < self.min_trades_for_training:
                print(f"⚠️ Not enough trades. Need {self.min_trades_for_training}, have {len(trades)}")
                return None

            print(f"📊 Loaded {len(trades)} trades for training")
            return trades
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            close_db_connection(conn)
            return None

    def calculate_voting_accuracy(self, trades):
        """حساب دقة تصويت المستشارين من جدول consultant_votes"""
        print("\n🎯 Calculating voting accuracy from database...")
        try:
            conn   = self._get_conn()
            if not conn:
                return {}
            cursor = conn.cursor()
            cursor.execute("""
                SELECT consultant_name, vote_type, is_correct, COUNT(*) as total
                FROM consultant_votes
                WHERE timestamp > NOW() - INTERVAL '30 days'
                GROUP BY consultant_name, vote_type, is_correct
            """)
            rows = cursor.fetchall()
            cursor.close()

            consultant_scores = {}
            for row in rows:
                name, vote_type, is_correct, count = row
                if name not in consultant_scores:
                    consultant_scores[name] = {
                        'tp_correct': 0,     'tp_wrong': 0,
                        'amount_correct': 0, 'amount_wrong': 0,
                        'sl_correct': 0,     'sl_wrong': 0,
                        'sell_correct': 0,   'sell_wrong': 0,
                        'buy_correct': 0,    'buy_wrong': 0
                    }
                key = f"{vote_type}_{'correct' if is_correct else 'wrong'}"
                consultant_scores[name][key] = count

            final_scores = {}
            for consultant, scores in consultant_scores.items():
                tp_total     = scores['tp_correct']     + scores['tp_wrong']
                amount_total = scores['amount_correct'] + scores['amount_wrong']
                sl_total     = scores['sl_correct']     + scores['sl_wrong']
                sell_total   = scores['sell_correct']   + scores['sell_wrong']
                buy_total    = scores['buy_correct']    + scores['buy_wrong']

                final_scores[consultant] = {
                    'tp_accuracy':     scores['tp_correct']     / tp_total     if tp_total     > 0 else 0.5,
                    'amount_accuracy': scores['amount_correct'] / amount_total if amount_total > 0 else 0.5,
                    'sl_accuracy':     scores['sl_correct']     / sl_total     if sl_total     > 0 else 0.5,
                    'sell_accuracy':   scores['sell_correct']   / sell_total   if sell_total   > 0 else 0.5,
                    'buy_accuracy':    scores['buy_correct']    / buy_total    if buy_total    > 0 else 0.5,
                    'overall_accuracy': (
                        (scores['tp_correct'] + scores['amount_correct'] + scores['sl_correct'] +
                         scores['sell_correct'] + scores['buy_correct']) /
                        max(tp_total + amount_total + sl_total + sell_total + buy_total, 1)
                    )
                }

            if final_scores:
                print(f"✅ Loaded voting accuracy for {len(final_scores)} consultants:")
                for name, s in final_scores.items():
                    print(f"   • {name}: Overall {s['overall_accuracy']*100:.1f}% | "
                          f"Buy {s['buy_accuracy']*100:.1f}% | Sell {s['sell_accuracy']*100:.1f}%")
            else:
                print("⚠️ No voting data found yet (table is new)")

            return final_scores

        except Exception as e:
            print(f"⚠️ Error calculating voting accuracy: {e}")
            close_db_connection(conn)
            return {}

    # ========== Save ==========

    def save_models_to_db(self, models, results, retry=3):
        """Save model accuracy info to dl_models_v2 table using a safe, non-destructive approach."""
        conn = self._get_conn()
        if not conn:
            print("⚠️ No database connection - models saved to files only")
            return False

        for attempt in range(retry):
            try:
                print(f"🔄 Attempt {attempt+1}/{retry}: Saving models to database...")
                with conn.cursor() as cursor:
                    # Step 1: Create the table if it doesn't already exist.
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS dl_models_v2 (
                            id         SERIAL PRIMARY KEY,
                            model_name VARCHAR(50)  NOT NULL,
                            model_type VARCHAR(50)  NOT NULL,
                            accuracy   FLOAT,
                            trained_at TIMESTAMP    DEFAULT NOW(),
                            status     VARCHAR(20)  DEFAULT 'active'
                        );
                    """)
                    # Step 2: Add the 'model_data' column if it doesn't exist. This is a safe way to migrate.
                    cursor.execute("ALTER TABLE dl_models_v2 ADD COLUMN IF NOT EXISTS model_data BYTEA;")

                    # Step 3: Clear only the old models, leaving the table structure intact.
                    cursor.execute("DELETE FROM dl_models_v2;")

                    # Step 4: Insert the newly trained models.
                    for model_name, model in models.items():
                        if model is None:
                            continue
                        accuracy = results.get(f'{model_name}_accuracy', 0)
                        model_bytes = pickle.dumps(model)
                        cursor.execute("""
                            INSERT INTO dl_models_v2 (model_name, model_type, model_data, accuracy)
                            VALUES (%s, %s, %s, %s);
                        """, (model_name, 'LightGBM', psycopg2.Binary(model_bytes), float(accuracy)))

                # Commit all changes (schema and data) in a single transaction.
                conn.commit()
                print("✅ Models info saved to database (dl_models_v2)")
                close_db_connection(conn)
                return True

            except Exception as e:
                print(f"❌ Error on attempt {attempt+1}: {e}")
                if conn:
                    try:
                        conn.rollback()
                        print("  - Transaction has been rolled back.")
                    except Exception as rb_e:
                        print(f"  - Additionally, failed to rollback transaction: {rb_e}")

                if attempt < retry - 1:
                    import time
                    print("  - Retrying in 3 seconds...")
                    time.sleep(3)
                    # Re-establish connection for the next attempt
                    close_db_connection(conn)
                    conn = self._get_conn()
                    if not conn:
                        print("⚠️ Could not re-establish DB connection for retry. Aborting.")
                        return False
                else:
                    print("❌ Max retries reached. Failed to save models to database.")
                    break # Exit loop
        
        close_db_connection(conn)
        return False

    def get_new_trades_count(self):
        """عدد الصفقات الجديدة منذ آخر تدريب"""
        try:
            conn   = self._get_conn()
            if not conn:
                return 0
            cursor = conn.cursor()

            cursor.execute("SELECT MAX(trained_at) FROM dl_models_v2")
            result        = cursor.fetchone()
            last_training = result[0] if result and result[0] else datetime.now() - timedelta(days=30)

            cursor.execute("""
                SELECT COUNT(*)
                FROM trades_history
                WHERE action = 'SELL'
                  AND timestamp > %s
            """, (last_training,))
            new_trades = cursor.fetchone()[0]
            cursor.close()
            close_db_connection(conn)
            return new_trades

        except Exception as e:
            print(f"⚠️ Error counting new trades: {e}")
            close_db_connection(conn)
            return 0
