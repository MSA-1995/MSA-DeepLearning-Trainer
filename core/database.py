import os
import psycopg2
from psycopg2 import pool

# Global connection pool
_db_pool = None

def get_db_connection():
    """
    Establishes a database connection using a connection pool.
    """
    global _db_pool
    if _db_pool is None:
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                raise ValueError("DATABASE_URL environment variable not set.")
            
            _db_pool = psycopg2.pool.SimpleConnectionPool(
                1, 5, dsn=database_url,
                connect_timeout=5
            )
            print("[OK] Database connection pool created.")
        except Exception as e:
            print(f"[ERROR] Failed to create database connection pool: {e}")
            return None

    try:
        conn = _db_pool.getconn()
        return conn
    except Exception as e:
        print(f"[ERROR] Failed to get database connection from pool: {e}")
        return None

def close_db_connection(conn):
    """
    Returns a connection to the pool.
    """
    global _db_pool
    if _db_pool and conn:
        _db_pool.putconn(conn)

def close_all_connections():
    """
    Closes all connections in the pool.
    """
    global _db_pool
    if _db_pool:
        _db_pool.closeall()
        print("[INFO] All database connections closed.")
