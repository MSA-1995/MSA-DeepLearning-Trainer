"""
🗄️ Database Manager
Handles all PostgreSQL operations: connect, load, save.
"""

import json
import pickle
from datetime import datetime, timedelta
import time

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

    def load_training_data(self, limit=None, offset=None):
        """Load historical SELL trades for training, with support for batching."""
        conn = self._get_conn()
        if not conn:
            return None
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = """
                    SELECT symbol, profit_percent, action, timestamp, data
                    FROM trades_history
                    WHERE action = 'SELL'
                      AND data IS NOT NULL
                    ORDER BY timestamp DESC
                """
                params = []
                if limit is not None:
                    query += " LIMIT %s"
                    params.append(limit)
                if offset is not None:
                    query += " OFFSET %s"
                    params.append(offset)

                cursor.execute(query, tuple(params))
                trades = cursor.fetchall()

                if offset is None or offset == 0:
                    if len(trades) < self.min_trades_for_training:
                        print(f"⚠️ Not enough trades. Need {self.min_trades_for_training}, have {len(trades)}")
                        return None
                    print(f"📊 Loaded {len(trades)} trades for training")
            
            return trades
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
        finally:
            close_db_connection(conn)

    def get_total_trades_count(self):
        """Get the total number of SELL trades available for training."""
        conn = self._get_conn()
        if not conn:
            return 0
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM trades_history
                    WHERE action = 'SELL'
                      AND data IS NOT NULL
                """)
                count = cursor.fetchone()[0]
            return count
        except Exception as e:
            print(f"❌ Error getting total trades count: {e}")
            return 0
        finally:
            close_db_connection(conn)

    def load_ai_decisions(self, limit=1000):
        """Load historical decisions from ai_decisions table."""
        conn = self._get_conn()
        if not conn:
            return []
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT symbol, decision, confidence, timestamp
                    FROM ai_decisions
                    WHERE decision IN ('BUY', 'SELL')
                    ORDER BY id DESC
                    LIMIT %s
                """, (limit,))
                decisions = cursor.fetchall()
            print(f"🧠 Loaded {len(decisions)} historical AI decisions.")
            return decisions
        except Exception as e:
            print(f"❌ Error loading AI decisions: {e}")
            return []
        finally:
            close_db_connection(conn)

    def load_traps(self, limit=500):
        """Load data from the trap_memory table."""
        conn = self._get_conn()
        if not conn:
            return []
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT symbol, data, timestamp
                    FROM trap_memory
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (limit,))
                traps = cursor.fetchall()
            print(f"🪤 Loaded {len(traps)} records from trap memory.")
            return traps
        except Exception as e:
            print(f"❌ Error loading trap memory: {e}")
            return []
        finally:
            close_db_connection(conn)

    def load_trades(self, limit=2000):
        """Load all types of trades from trades_history table."""
        conn = self._get_conn()
        if not conn:
            return []
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT symbol, profit_percent, action, timestamp, data
                    FROM trades_history
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (limit,))
                trades = cursor.fetchall()
            print(f"📜 Loaded {len(trades)} records from trades history.")
            return trades
        except Exception as e:
            print(f"❌ Error loading trades history: {e}")
            return []
        finally:
            close_db_connection(conn)

    def calculate_voting_accuracy(self, trades):
        """حساب دقة تصويت المستشارين من جدول consultant_votes"""
        print("\n🎯 Calculating voting accuracy from database...")
        conn = self._get_conn()
        if not conn:
            return {}
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT consultant_name, vote_type, is_correct, COUNT(*) as total
                    FROM consultant_votes
                    WHERE timestamp > NOW() - INTERVAL '30 days'
                    GROUP BY consultant_name, vote_type, is_correct
                """)
                rows = cursor.fetchall()

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
            return {}
        finally:
            close_db_connection(conn)

    # ========== Save ==========

    def save_models_to_db(self, models, results, retry=3):
        """Save model accuracy info to dl_models_v2 table using a safe, non-destructive approach."""
        for attempt in range(retry):
            conn = None  # Initialize conn to None for each attempt
            try:
                conn = self._get_conn()
                if not conn:
                    print(f"⚠️ Attempt {attempt+1}/{retry}: No database connection available. Retrying...")
                    time.sleep(5) # Wait longer if the pool is empty
                    continue

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
                            status     VARCHAR(20)  DEFAULT 'active',
                            model_data BYTEA,       -- Store pickled model
                            CONSTRAINT uq_model_name_type UNIQUE (model_name, model_type)
                        );
                    """)

                    # Step 2: Use INSERT ... ON CONFLICT to update or insert.
                    for model_name, model_obj in models.items():
                        if model_obj is None:
                            continue
                        accuracy = results.get(f'{model_name}_accuracy', 0)
                        
                        # Pickle the model object to store it
                        pickled_model = pickle.dumps(model_obj)

                        cursor.execute("""
                            INSERT INTO dl_models_v2 (model_name, model_type, accuracy, trained_at, model_data)
                            VALUES (%s, %s, %s, NOW(), %s)
                            ON CONFLICT (model_name, model_type) 
                            DO UPDATE SET
                                accuracy = EXCLUDED.accuracy,
                                trained_at = EXCLUDED.trained_at,
                                model_data = EXCLUDED.model_data;
                        """, (model_name, 'LightGBM', float(accuracy), psycopg2.Binary(pickled_model)))

                conn.commit()
                print("✅ Models info saved to database (dl_models_v2)")
                return True

            except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
                print(f"❌ Attempt {attempt+1}/{retry} failed with connection error: {e}")
                if conn:
                    try:
                        # For connection errors, the connection is likely dead. Close it.
                        close_db_connection(conn, force_close=True)
                    except Exception as close_e:
                        print(f"- Error while closing failed connection: {close_e}")
                if attempt < retry - 1:
                    time.sleep(5) # Wait before retrying
            except Exception as e:
                print(f"❌ Attempt {attempt+1}/{retry} failed with general error: {e}")
                if conn:
                    try:
                        conn.rollback()
                    except Exception as rb_e:
                        print(f"- Error during rollback: {rb_e}")
                if attempt < retry - 1:
                    time.sleep(2)
            finally:
                if conn:
                    close_db_connection(conn)
        
        send_critical_alert(
            "DB Save Failure", 
            "Failed to save models to the database after multiple retries.",
            f"Check database connectivity and logs."
        )
        return False

    def get_new_trades_count(self):
        """عدد الصفقات الجديدة منذ آخر تدريب - يفحص كل نموذج بشكل مستقل"""
        conn = self._get_conn()
        if not conn:
            return 0
        try:
            with conn.cursor() as cursor:
                # جلب آخر trained_at لكل نموذج بشكل مستقل
                cursor.execute("""
                    SELECT model_name, trained_at 
                    FROM dl_models_v2 
                    ORDER BY trained_at ASC
                """)
                rows = cursor.fetchall()

                if not rows:
                    # لا يوجد أي نموذج → تدريب من الصفر
                    print("ℹ️ No models found → will train from scratch")
                    return 999999

                # أقدم trained_at بين كل النماذج
                oldest_training = rows[0][1]
                missing_models  = 11 - len(rows)  # 11 نموذج مطلوب

                if missing_models > 0:
                    # فيه نماذج ناقصة → تدريب من الصفر لها
                    print(f"ℹ️ {missing_models} models missing → will retrain all")
                    return 999999

                # كل النماذج موجودة → فحص الصفقات الجديدة منذ أقدم تدريب
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM trades_history
                    WHERE action = 'SELL'
                      AND timestamp > %s
                """, (oldest_training,))
                new_trades = cursor.fetchone()[0]

            return new_trades
        except Exception as e:
            print(f"⚠️ Error counting new trades: {e}")
            return 0
        finally:
            close_db_connection(conn)
