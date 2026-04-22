import os
import psycopg2
from psycopg2 import pool

# Global connection pool
_db_pool = None


def get_db_connection():
    """
    Establishes a database connection using a connection pool.
    Returns a connection object or None on failure.
    """
    global _db_pool

    if _db_pool is None:
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                raise ValueError("DATABASE_URL environment variable not set.")

            _db_pool = psycopg2.pool.SimpleConnectionPool(
                1, 20,
                dsn=database_url,
                connect_timeout=5,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=3,
            )
            print("[OK] Database connection pool created.")
        except Exception as e:
            print(f"[ERROR] Failed to create database connection pool: {e}")
            _db_pool = None
            return None

    try:
        conn = _db_pool.getconn()
        if conn:
            conn.autocommit = False  # explicit transaction control
        return conn
    except pool.PoolError as e:
        print(f"[ERROR] Connection pool exhausted: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to get database connection from pool: {e}")
        return None


def close_db_connection(conn):
    """
    Returns a connection to the pool.
    Handles rollback if transaction is in progress.
    """
    global _db_pool

    if not _db_pool or not conn:
        return

    try:
        # Rollback any pending transaction
        if conn and not conn.closed:
            try:
                conn.rollback()
            except Exception:
                pass
        _db_pool.putconn(conn)
    except pool.PoolError as e:
        print(f"[WARNING] Could not return connection to pool: {e}")
        try:
            if conn and not conn.closed:
                conn.close()
        except Exception:
            pass
    except Exception as e:
        print(f"[WARNING] Error closing connection: {e}")
        try:
            if conn and not conn.closed:
                conn.close()
        except Exception:
            pass


def close_all_connections():
    """
    Closes all connections in the pool.
    """
    global _db_pool

    if _db_pool:
        try:
            _db_pool.closeall()
            print("[INFO] All database connections closed.")
            _db_pool = None
        except Exception as e:
            print(f"[ERROR] Error closing all connections: {e}")
            _db_pool = None


def test_db_connection():
    """
    Test database connectivity.
    Returns True if successful, False otherwise.
    """
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return False
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
        return result is not None
    except Exception as e:
        print(f"[ERROR] Database test failed: {e}")
        return False
    finally:
        if conn:
            close_db_connection(conn)