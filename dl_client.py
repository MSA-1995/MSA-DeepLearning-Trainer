"""
🔌 Deep Learning Client
Used by main trading bot to get predictions from Deep Learning model
"""

import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse, unquote

class DLClient:
    def __init__(self, database_url):
        self.database_url = database_url
        self.conn = None
        self._connect_db()
    
    def _connect_db(self):
        """Connect to database"""
        try:
            parsed = urlparse(self.database_url)
            self._db_params = {
                'host': parsed.hostname,
                'port': parsed.port,
                'database': parsed.path[1:],
                'user': parsed.username,
                'password': unquote(parsed.password)
            }
            self.conn = psycopg2.connect(**self._db_params)
        except Exception as e:
            print(f"❌ DL Client DB error: {e}")
            self.conn = None
    
    def _get_conn(self):
        """Get valid connection"""
        try:
            if self.conn.closed:
                raise Exception("closed")
            self.conn.cursor().execute("SELECT 1")
        except Exception:
            try:
                self.conn = psycopg2.connect(**self._db_params)
            except:
                pass
        return self.conn
    
    def is_model_available(self):
        """Check if Deep Learning model is trained and available"""
        if not self.conn:
            return False
        
        try:
            cursor = self._get_conn().cursor()
            cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = 'dl_predictions'
                )
            """)
            exists = cursor.fetchone()[0]
            cursor.close()
            
            if not exists:
                return False
            
            cursor = self._get_conn().cursor()
            cursor.execute("SELECT COUNT(*) FROM dl_predictions")
            count = cursor.fetchone()[0]
            cursor.close()
            
            return count > 0
        except Exception:
            return False
    
    def get_confidence_adjustment(self, rsi, macd, volume_ratio, price_momentum, confidence):
        """Get confidence adjustment from Deep Learning model"""
        if not self.is_model_available():
            return 0
        
        try:
            # For now, return simple adjustment based on stored predictions
            # In production, this would call the actual model
            cursor = self._get_conn().cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT model_accuracy FROM dl_predictions
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                accuracy = result['model_accuracy']
                # Simple adjustment based on model accuracy
                if accuracy > 0.7:
                    return 3
                elif accuracy > 0.6:
                    return 2
                else:
                    return 1
            
            return 0
        except Exception:
            return 0
    
    def get_advisor_knowledge(self, advisor_name):
        """Get knowledge for specific advisor"""
        if not self.conn:
            return None
        
        try:
            cursor = self._get_conn().cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT knowledge FROM dl_advisors_knowledge
                WHERE advisor_name = %s
                ORDER BY timestamp DESC
                LIMIT 1
            """, (advisor_name,))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return result['knowledge']
            return None
        except Exception:
            return None
