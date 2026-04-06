"""
🗄️ Database Manager
Handles all PostgreSQL operations: connect, load, save.
"""

import json
import pickle
import gzip
from datetime import datetime, timedelta
import time

import psycopg2
from psycopg2.extras import RealDictCursor

from alerts import send_critical_alert
from database import get_db_connection, close_db_connection


class DatabaseManager:
    def __init__(self):
        self.min_trades_for_training = 1

    # ========== Connection ==========

    def _get_conn(self):
        """Return valid connection"""
        return get_db_connection()

    # ========== Load ==========

    def load_training_data(self, since_timestamp=None):
        """Load SELL trades for training."""
        conn = self._get_conn()
        if not conn:
            return None
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SET statement_timeout = '120s'")
                
                if since_timestamp is None:
                    # Load ALL trades
                    cursor.execute("""
                        SELECT *
                        FROM trades_history
                        WHERE action = 'SELL' AND data IS NOT NULL
                    """)
                else:
                    # Load NEW trades
                    cursor.execute("""
                        SELECT *
                        FROM trades_history
                        WHERE action = 'SELL' AND data IS NOT NULL AND timestamp > %s
                    """, (since_timestamp,))
                
                trades = cursor.fetchall()
                print(f"📊 Loaded {len(trades)} trades")
            
            return trades
        except Exception as e:
            print(f"❌ Error: {e}")
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
                cursor.execute("SET statement_timeout = '30s'")
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_consultant_votes_timestamp 
                    ON consultant_votes (timestamp);
                """)
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

    # ========== Model Management ==========

    def get_missing_models(self):
        """Get list of missing models that need to be trained."""
        conn = self._get_conn()
        if not conn:
            return ['smart_money', 'risk', 'anomaly', 'exit', 'pattern',
                    'liquidity', 'chart_cnn', 'sentiment', 'crypto_news',
                    'volume_pred', 'meta_trading']
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT model_name FROM dl_models_v2")
                existing = {row[0] for row in cursor.fetchall()}
            
            required = {'smart_money', 'risk', 'anomaly', 'exit', 'pattern',
                       'liquidity', 'chart_cnn', 'sentiment', 'crypto_news',
                       'volume_pred', 'meta_trading'}
            
            missing = list(required - existing)
            return missing
        except Exception as e:
            print(f"⚠️ Error checking missing models: {e}")
            return []
        finally:
            close_db_connection(conn)
    
    def get_oldest_model_timestamp(self):
        """Get the oldest training timestamp among all models."""
        conn = self._get_conn()
        if not conn:
            return None
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT MIN(trained_at) FROM dl_models_v2")
                result = cursor.fetchone()
                return result[0] if result and result[0] else None
        except Exception as e:
            print(f"⚠️ Error getting oldest timestamp: {e}")
            return None
        finally:
            close_db_connection(conn)

    # ========== Save ==========

    def save_models_to_db(self, models, results, retry=3):
        """Save models to database with their accuracies."""
        for attempt in range(retry):
            conn = None
            try:
                conn = self._get_conn()
                if not conn:
                    time.sleep(5)
                    continue

                with conn.cursor() as cursor:
                    cursor.execute("SET statement_timeout = '10s'")
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS dl_models_v2 (
                            id         SERIAL PRIMARY KEY,
                            model_name VARCHAR(50),
                            model_type VARCHAR(50),
                            accuracy   FLOAT,
                            trained_at TIMESTAMP DEFAULT NOW(),
                            model_data BYTEA
                        );
                    """)
                    cursor.execute("DELETE FROM dl_models_v2;")
                conn.commit()

                # Save each model separately with commit after each
                saved_count = 0
                for model_name, model_obj in models.items():
                    if model_obj is None:
                        continue
                    accuracy = results.get(f'{model_name}_accuracy', 0)
                    if accuracy < 0.01:
                        continue
                    
                    try:
                        # Compress model before saving
                        pickled_model = pickle.dumps(model_obj)
                        compressed_model = gzip.compress(pickled_model)
                        original_size = len(pickled_model) / (1024 * 1024)
                        compressed_size = len(compressed_model) / (1024 * 1024)
                        
                        with conn.cursor() as cursor:
                            cursor.execute("SET statement_timeout = '60s'")
                            cursor.execute("""
                                INSERT INTO dl_models_v2 (model_name, model_type, accuracy, model_data)
                                VALUES (%s, %s, %s, %s);
                            """, (model_name, 'LightGBM', float(accuracy), compressed_model))
                        conn.commit()
                        saved_count += 1
                        print(f"  ✅ {model_name}: {accuracy*100:.1f}% ({original_size:.2f}MB → {compressed_size:.2f}MB)")
                    except Exception as e:
                        print(f"  ❌ {model_name}: {str(e)[:50]}")
                        try:
                            conn.rollback()
                        except:
                            pass
                
                print(f"✅ {saved_count}/11 models saved")
                return saved_count > 0

            except Exception as e:
                print(f"❌ Attempt {attempt+1}/{retry} failed: {str(e)[:100]}")
                if conn:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                if attempt < retry - 1:
                    time.sleep(2)
            finally:
                if conn:
                    close_db_connection(conn)
        
        send_critical_alert(
            "DB Save Failure", 
            "Failed to save models to database after multiple retries.",
            f"Check database connectivity and logs."
        )
        return False

    def load_symbol_memory(self):
        """Load symbol memory for all symbols."""
        conn = self._get_conn()
        if not conn:
            return {}
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM symbol_memory")
                rows = cursor.fetchall()
            memory = {row['symbol']: dict(row) for row in rows}
            print(f"🧠 Loaded symbol memory for {len(memory)} symbols.")
            return memory
        except Exception as e:
            print(f"⚠️ Error loading symbol memory: {e}")
            return {}
        finally:
            close_db_connection(conn)

    def load_causal_data(self, limit=10000):
        """Load causal data for training."""
        conn = self._get_conn()
        if not conn:
            return []
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM causal_data
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (limit,))
                rows = cursor.fetchall()
            print(f"🔗 Loaded {len(rows)} causal data records.")
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"⚠️ Error loading causal data: {e}")
            return []
        finally:
            close_db_connection(conn)

    def recalculate_symbol_memory(self):
        """
        حساب وتحديث جدول symbol_memory بـ جميع الأعمدة:
        - الـ 7 أعمدة (sentiment_avg, whale_confidence_avg, إلخ)
        - الـ 4 أعمدة الناقصة (courage_boost, time_memory_modifier, إلخ)
        """
        print("\n🧠 Recalculating Symbol Memory...")
        conn = self._get_conn()
        if not conn:
            return False
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # جلب جميع الصفقات
                cursor.execute("""
                    SELECT symbol, profit_percent, data, timestamp, trade_quality, hours_held
                    FROM trades_history
                    WHERE action = 'SELL' AND data IS NOT NULL
                    ORDER BY symbol, timestamp
                """)
                trades = cursor.fetchall()
            
            if not trades:
                print("⚠️ No trades found for recalculation")
                return False
            
            # تجميع الصفقات حسب الرمز
            symbol_trades = {}
            for trade in trades:
                symbol = trade['symbol']
                if symbol not in symbol_trades:
                    symbol_trades[symbol] = []
                symbol_trades[symbol].append(trade)
            
            # حساب الإحصائيات لكل رمز
            updates = []
            for symbol, sym_trades in symbol_trades.items():
                try:
                    stats = self._calculate_symbol_stats(symbol, sym_trades)
                    updates.append((symbol, stats))
                except Exception as e:
                    print(f"⚠️ Error calculating stats for {symbol}: {e}")
            
            # تحديث الـ database
            with conn.cursor() as cursor:
                for symbol, stats in updates:
                    cursor.execute("""
                        INSERT INTO symbol_memory 
                        (symbol, win_count, total_trades, avg_profit, trap_count,
                         sentiment_avg, whale_confidence_avg, profit_loss_ratio, volume_trend,
                         panic_score_avg, optimism_penalty_avg, psychological_summary,
                         courage_boost, time_memory_modifier, pattern_score, win_rate_boost)
                        VALUES 
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol) DO UPDATE SET
                        win_count = EXCLUDED.win_count,
                        total_trades = EXCLUDED.total_trades,
                        avg_profit = EXCLUDED.avg_profit,
                        trap_count = EXCLUDED.trap_count,
                        sentiment_avg = EXCLUDED.sentiment_avg,
                        whale_confidence_avg = EXCLUDED.whale_confidence_avg,
                        profit_loss_ratio = EXCLUDED.profit_loss_ratio,
                        volume_trend = EXCLUDED.volume_trend,
                        panic_score_avg = EXCLUDED.panic_score_avg,
                        optimism_penalty_avg = EXCLUDED.optimism_penalty_avg,
                        psychological_summary = EXCLUDED.psychological_summary,
                        courage_boost = EXCLUDED.courage_boost,
                        time_memory_modifier = EXCLUDED.time_memory_modifier,
                        pattern_score = EXCLUDED.pattern_score,
                        win_rate_boost = EXCLUDED.win_rate_boost
                    """, (
                        symbol,
                        stats['win_count'],
                        stats['total_trades'],
                        stats['avg_profit'],
                        stats['trap_count'],
                        stats['sentiment_avg'],
                        stats['whale_confidence_avg'],
                        stats['profit_loss_ratio'],
                        stats['volume_trend'],
                        stats['panic_score_avg'],
                        stats['optimism_penalty_avg'],
                        stats['psychological_summary'],
                        stats['courage_boost'],
                        stats['time_memory_modifier'],
                        stats['pattern_score'],
                        stats['win_rate_boost']
                    ))
                conn.commit()
            
            print(f"✅ Updated {len(updates)} symbols in symbol_memory")
            return True
        
        except Exception as e:
            print(f"❌ Error recalculating symbol memory: {e}")
            return False
        finally:
            close_db_connection(conn)

    def _calculate_symbol_stats(self, symbol, trades):
        """حساب جميع إحصائيات الرمز من الصفقات"""
        import numpy as np
        
        profits = []
        wins = 0
        losses = 0
        trap_count = 0
        
        sentiment_scores = []
        whale_scores = []
        volume_scores = []
        panic_scores = []
        optimism_penalties = []
        
        for trade in trades:
            try:
                profit = float(trade.get('profit_percent', 0))
                profits.append(profit)
                
                if profit > 0:
                    wins += 1
                else:
                    losses += 1
                
                # جلب البيانات المخزنة
                data = trade.get('data', {})
                if isinstance(data, str):
                    data = json.loads(data)
                elif not isinstance(data, dict):
                    data = {}
                
                trade_quality = trade.get('trade_quality', 'OK')
                if trade_quality in ['TRAP', 'RISKY']:
                    trap_count += 1
                
                # Sentiment
                sentiment = data.get('sentiment', {})
                if sentiment:
                    sentiment_scores.append(float(sentiment.get('news_sentiment', 0)))
                
                # Whale Confidence
                whale = data.get('whale_confidence', 0)
                whale_scores.append(float(whale))
                
                # Volume Trend
                volume_ratio = data.get('volume_ratio', 1.0)
                volume_scores.append(float(volume_ratio))
                
                # Panic Score
                panic = data.get('panic_score', 0)
                panic_scores.append(float(panic))
                
                # Optimism Penalty
                optimism = data.get('optimism_penalty', 0)
                optimism_penalties.append(float(optimism))
            
            except Exception as e:
                continue
        
        total = len(trades)
        avg_profit = float(np.mean(profits)) if profits else 0
        win_rate = wins / max(total, 1)
        
        # profit_loss_ratio
        avg_win = float(np.mean([p for p in profits if p > 0])) if any(p > 0 for p in profits) else 0.1
        avg_loss = abs(float(np.mean([p for p in profits if p <= 0])) if any(p <= 0 for p in profits) else 1.0)
        profit_loss_ratio = avg_win / max(avg_loss, 0.01)
        
        # Averages
        sentiment_avg = float(np.mean(sentiment_scores)) if sentiment_scores else 0
        whale_avg = float(np.mean(whale_scores)) if whale_scores else 0
        volume_trend = float(np.mean(volume_scores)) if volume_scores else 1.0
        panic_avg = float(np.mean(panic_scores)) if panic_scores else 0
        optimism_avg = float(np.mean(optimism_penalties)) if optimism_penalties else 0
        
        # Psychological summary
        psychological_summary = "Neutral"
        if sentiment_avg > 30 and panic_avg < 20:
            psychological_summary = "Bullish"
        elif sentiment_avg < -30 and panic_avg > 60:
            psychological_summary = "Bearish"
        elif panic_avg > 70:
            psychological_summary = "Panic"
        elif sentiment_avg > 50:
            psychological_summary = "Optimistic"
        
        # Courage Boost (بناءً على نسبة الفوز)
        courage_boost = min(win_rate * 20, 15) if win_rate >= 0.6 else 0
        
        # Time Memory Modifier (بناءً على consistency)
        time_mod = 0
        if win_rate >= 0.75:
            time_mod = 10
        elif win_rate >= 0.65:
            time_mod = 5
        elif win_rate < 0.35:
            time_mod = -10
        
        # Pattern Score (بناءً على عدد الصفقات والأرباح)
        pattern_score = min((total / 10) * (win_rate * 2), 20) if total > 5 else 0
        
        # Win Rate Boost (bonus مباشر)
        win_rate_boost = 0
        if win_rate >= 0.80:
            win_rate_boost = 15
        elif win_rate >= 0.70:
            win_rate_boost = 10
        elif win_rate >= 0.60:
            win_rate_boost = 5
        
        return {
            'win_count': wins,
            'total_trades': total,
            'avg_profit': avg_profit,
            'trap_count': trap_count,
            'sentiment_avg': sentiment_avg,
            'whale_confidence_avg': whale_avg,
            'profit_loss_ratio': profit_loss_ratio,
            'volume_trend': volume_trend,
            'panic_score_avg': panic_avg,
            'optimism_penalty_avg': optimism_avg,
            'psychological_summary': psychological_summary,
            'courage_boost': courage_boost,
            'time_memory_modifier': time_mod,
            'pattern_score': pattern_score,
            'win_rate_boost': win_rate_boost
        }
