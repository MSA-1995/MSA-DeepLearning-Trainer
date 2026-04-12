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
                        WHERE lower(action) = 'sell' AND data IS NOT NULL
                    """)
                else:
                    # Load NEW trades
                    cursor.execute("""
                        SELECT *
                        FROM trades_history
                        WHERE lower(action) = 'sell' AND data IS NOT NULL AND timestamp > %s
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

    def calculate_voting_accuracy(self, trades):
        """حساب دقة تصويت كل مستشار بناءً على نتائج الصفقات الحقيقية"""
        advisors = ['smart_money', 'risk', 'anomaly', 'exit', 'pattern', 'liquidity', 'chart_cnn', 'sentiment', 'crypto_news', 'volume_pred']
        results = {adv: {'tp_accuracy': 0.5, 'sell_accuracy': 0.5} for adv in advisors}
        
        if not trades:
            return results

        advisor_stats = {adv: {'hits': 0, 'total': 0} for adv in advisors}

        for t in trades:
            try:
                raw_data = t.get('data', {})
                if isinstance(raw_data, str):
                    data = json.loads(raw_data)
                else:
                    data = raw_data
                
                buy_votes = data.get('buy_votes', {})
                profit = float(t.get('profit_percent', 0))

                for adv, vote in buy_votes.items():
                    if adv in advisor_stats and vote == 1:
                        advisor_stats[adv]['total'] += 1
                        if profit > 0.5:  # اعتبار التصويت ناجحاً إذا تحقق ربح
                            advisor_stats[adv]['hits'] += 1
            except:
                continue

        for adv in advisors:
            if advisor_stats[adv]['total'] > 0:
                results[adv]['tp_accuracy'] = advisor_stats[adv]['hits'] / advisor_stats[adv]['total']
        
        return results

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
                            model_data BYTEA,
                            status     VARCHAR(20) DEFAULT 'active'
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
                    # Normalize accuracy: handle both fraction (0.78) and percentage (78.0)
                    if accuracy > 1.0:
                        accuracy = accuracy / 100.0
                    if accuracy < 0.01:
                        print(f"  ⏭️ {model_name}: skipping (accuracy too low: {accuracy:.3f})")
                        continue
                    
                    try:
                        # Compress model before saving
                        pickled_model = pickle.dumps(model_obj)
                        compressed_model = gzip.compress(pickled_model)
                        original_size = len(pickled_model) / (1024 * 1024)
                        compressed_size = len(compressed_model) / (1024 * 1024)
                        
                        with conn.cursor() as cursor:
                            cursor.execute("SET statement_timeout = '60s'")
                            # Always store accuracy as fraction (0.0-1.0)
                            acc_to_save = accuracy / 100.0 if accuracy > 1.0 else float(accuracy)
                            cursor.execute("""
                                INSERT INTO dl_models_v2 (model_name, model_type, accuracy, model_data, status)
                                VALUES (%s, %s, %s, %s, %s);
                            """, (model_name, 'LightGBM', acc_to_save, compressed_model, 'active'))
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
                    SELECT symbol, profit_percent, data, timestamp, trade_quality, hours_held,
                           sentiment_score, whale_confidence, panic_score, optimism_penalty
                    FROM trades_history
                    WHERE lower(action) = 'sell' AND data IS NOT NULL
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
                    # Ensure values fit field lengths
                    symbol_val = symbol[:20] if len(symbol) > 20 else symbol
                    volume_trend_val = str(stats['volume_trend'])[:10]
                    psychological_summary_val = stats['psychological_summary'][:10]

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
                        symbol_val,
                        stats['win_count'],
                        stats['total_trades'],
                        stats['avg_profit'],
                        stats['trap_count'],
                        stats['sentiment_avg'],
                        stats['whale_confidence_avg'],
                        stats['profit_loss_ratio'],
                        volume_trend_val,
                        stats['panic_score_avg'],
                        stats['optimism_penalty_avg'],
                        psychological_summary_val,
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
                
                trade_quality = trade.get('trade_quality') or 'OK'
                if trade_quality in ['TRAP', 'RISKY']:
                    trap_count += 1
                
                # Sentiment
                sentiment_scores.append(float(trade.get('sentiment_score', 0)))

                # Whale Confidence
                whale_scores.append(float(trade.get('whale_confidence', 0)))

                # Volume Trend (from data JSON)
                volume_ratio = data.get('volume_ratio', 1.0)
                volume_scores.append(float(volume_ratio))

                # Panic Score
                panic_scores.append(float(trade.get('panic_score', 0)))

                # Optimism Penalty
                optimism_penalties.append(float(trade.get('optimism_penalty', 0)))
            
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
