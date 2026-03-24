"""
🚨 Critical Alerts - Discord Webhook + Encryption
"""

import os
from config import STATUS_STORAGE_METHOD
import base64
from datetime import datetime

# Optional imports (auto-installed by entry point)
try:
    import requests
except ImportError:
    requests = None

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
except ImportError:
    Fernet = None
    hashes = None
    PBKDF2HMAC = None
    default_backend = None

ENCRYPTED_CRITICAL_WEBHOOK = "gAAAAABpwvw0pPVH-QpC6XaqbfA2_Tpelbc5ZSnXpbTvxd-IjRHfF9BxlJFlk5ppgFE5uVzm2oEvtiRl9xZREKREuNzz5ZH3bzcZ6VmzufOu_F_VSrkO8-PgVZrbiFsUjZRihOXW2MT4WUmewtNjyrabini37aBJgLQrTgYY7QKRQWeHlMFmGP2fD4inb2rrQU4V2KmnbSOwpcBu85xQK5vqsd9OQpDs4Gjvt0-3S8TxyeOO2_OMTxM="

# Master key for password encryption (must match trading bot)
_MASTER_KEY = base64.urlsafe_b64encode(b'MSA_TRADING_BOT_2026_SECRET_KEY!')
_MASTER_CIPHER = Fernet(_MASTER_KEY)

def get_encryption_key():
    """Get encryption key - supports Fernet encrypted password (Mirrors trading bot logic)"""
    # Try Environment Variable first
    encrypted_key = os.getenv('ENCRYPTION_KEY')
    
    if encrypted_key:
        try:
            # Try to decrypt with Fernet (strong encryption, for server)
            decrypted = _MASTER_CIPHER.decrypt(encrypted_key.encode()).decode()
            print("✅ Encryption key loaded from environment variable (Fernet decrypted)")
            return decrypted
        except:
            # Fallback: try base64 (old method)
            try:
                decoded = base64.b64decode(encrypted_key).decode()
                print("✅ Encryption key loaded from environment variable (base64 decoded)")
                return decoded
            except:
                # Use as-is (plain text, for local .env)
                print("✅ Encryption key loaded from environment variable (plain text)")
                return encrypted_key
    
    # Fallback to flash drive (for local development)
    key_file = r"D:\bot_keys.txt"
    try:
        if os.path.exists(key_file):
            with open(key_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    key = lines[2].strip()
                    if key:
                        print("✅ Encryption key loaded from flash drive")
                        return key
            print("⚠️ Flash drive key file is invalid or key is empty.")
            return None
        else:
            # This is not an error, just info.
            print("ℹ️ Flash drive not found. This is normal on a server.")
            return None
    except Exception as e:
        print(f"❌ Error reading encryption key from flash drive: {e}")
        return None


def get_critical_webhook():
    """Decrypt critical webhook URL"""
    if not Fernet:
        return None
    try:
        _KEY = get_encryption_key()
        if not _KEY:
            return None
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'binance_bot_salt_2026',
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(_KEY.encode()))
        fernet = Fernet(key)
        webhook = fernet.decrypt(ENCRYPTED_CRITICAL_WEBHOOK.encode()).decode()
        return webhook
    except:
        return None


CRITICAL_WEBHOOK = get_critical_webhook()

# --- Status Message Management ---
STATUS_MESSAGE_ID = None
# Use os.path.dirname to ensure the path is relative to the current file's location
STATUS_MESSAGE_ID_FILE = os.path.join(os.path.dirname(__file__), 'trainer_status_message_id.txt')
STARTUP_TIME = None

def load_status_message_id():
    """Load the status message ID from the configured storage (db or file)."""
    global STATUS_MESSAGE_ID
    if STATUS_STORAGE_METHOD == 'database':
        from database import get_db_connection, close_db_connection
        conn = get_db_connection()
        if not conn:
            STATUS_MESSAGE_ID = None
            return
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM bot_settings WHERE key = 'trainer_status_message_id'")
            row = cursor.fetchone()
            if row and row[0]:
                STATUS_MESSAGE_ID = row[0]
                print(f"✅ Loaded Trainer status message ID from database: {STATUS_MESSAGE_ID}")
            else:
                print("🤔 No Trainer status message ID found in database. A new one will be created.")
                STATUS_MESSAGE_ID = None
            cursor.close()
        except Exception as e:
            print(f"❌ Error loading Trainer status message ID from database: {e}")
            STATUS_MESSAGE_ID = None
        finally:
            close_db_connection(conn)
    else: # file method
        if not os.path.exists(STATUS_MESSAGE_ID_FILE):
            print("🤔 No Trainer status message ID file found. A new one will be created.")
            STATUS_MESSAGE_ID = None
            return
        try:
            with open(STATUS_MESSAGE_ID_FILE, 'r') as f:
                STATUS_MESSAGE_ID = f.read().strip()
                if STATUS_MESSAGE_ID:
                    print(f"✅ Loaded Trainer status message ID from file: {STATUS_MESSAGE_ID}")
                else:
                    print("🤔 Status message ID file is empty. A new one will be created.")
                    STATUS_MESSAGE_ID = None
        except Exception as e:
            print(f"❌ Error loading Trainer status message ID from file: {e}")
            STATUS_MESSAGE_ID = None

def save_status_message_id(message_id):
    """Save the status message ID to the configured storage (db or file)."""
    global STATUS_MESSAGE_ID
    if STATUS_STORAGE_METHOD == 'database':
        from database import get_db_connection, close_db_connection
        conn = get_db_connection()
        if not conn:
            return
        try:
            cursor = conn.cursor()
            # Ensure the table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_settings (
                    key VARCHAR(255) PRIMARY KEY,
                    value TEXT
                );
            """)
            cursor.execute("INSERT INTO bot_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", ('trainer_status_message_id', str(message_id)))
            conn.commit()
            STATUS_MESSAGE_ID = str(message_id)
            print(f"💾 Saved Trainer status message ID to database: {message_id}")
            cursor.close()
        except Exception as e:
            print(f"❌ Error saving Trainer status message ID to database: {e}")
        finally:
            close_db_connection(conn)
    else: # file method
        try:
            os.makedirs(os.path.dirname(STATUS_MESSAGE_ID_FILE), exist_ok=True)
            with open(STATUS_MESSAGE_ID_FILE, 'w') as f:
                f.write(str(message_id) if message_id else '')
            STATUS_MESSAGE_ID = str(message_id) if message_id else None
            if message_id:
                print(f"💾 Saved Trainer status message ID to file: {message_id}")
            else:
                print("📝 Cleared Trainer status message ID file.")
        except Exception as e:
            print(f"❌ Error saving Trainer status message ID to file: {e}")

def send_status_embed(title, fields, color='blue', message_id=None):
    """Send or edit a status embed message on Discord. This is a specialized version.
    """
    if not CRITICAL_WEBHOOK or not requests:
        print("⚠️ Discord webhook URL or requests library is not available.")
        return None

    colors = {'blue': 0x0000ff}

    embed = {
        "title": title,
        "color": colors.get(color, 0x0000ff),
        "fields": fields,
        "footer": {"text": "MSA Trading Bot • AI Powered"}, # Exact footer requested
        "timestamp": datetime.utcnow().isoformat()
    }

    data = {"embeds": [embed]}
    
    if message_id:
        url = f"{CRITICAL_WEBHOOK}/messages/{message_id}"
        method = 'patch'
    else:
        url = f"{CRITICAL_WEBHOOK}?wait=true" # wait=true is needed to get the message ID back
        method = 'post'

    try:
        response = requests.request(method, url, json=data, timeout=10)
        response.raise_for_status()
        if not response.text:
            return None
        return response.json()
    except requests.exceptions.RequestException as e:
        if e.response and e.response.status_code == 404:
            print(f"ℹ️ Discord message {message_id} not found. It was likely deleted.")
        else:
            print(f"❌ Discord API Error: {e}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred while sending to Discord: {e}")
        return None

def send_startup_notification():
    """Send or update the trainer status message at script startup."""
    global STARTUP_TIME, STATUS_MESSAGE_ID
    load_status_message_id()

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    STARTUP_TIME = now_str
    title = "BOT RESTARTED" # Exact title requested
    fields = [
        {"name": "Trainer Script", "value": "ACTIVE", "inline": False}, # Changed from AI Brain
        {"name": "Last update", "value": datetime.now().strftime('%H:%M:%S'), "inline": False},
        {"name": "Last Restart", "value": now_str, "inline": False}
    ]

    response_data = None
    if STATUS_MESSAGE_ID:
        response_data = send_status_embed(title, fields, 'blue', message_id=STATUS_MESSAGE_ID)

    if not response_data:
        if STATUS_MESSAGE_ID:
             print("ℹ️ Failed to edit status message. Creating a new one.")
        response_data = send_status_embed(title, fields, 'blue')

    if response_data and 'id' in response_data:
        save_status_message_id(response_data['id'])

def send_heartbeat_notification():
    """Periodically update the status message to show the script is alive."""
    global STARTUP_TIME, STATUS_MESSAGE_ID
    
    if not STATUS_MESSAGE_ID:
        print("⚠️ No status message ID for heartbeat. Re-creating notification.")
        send_startup_notification()
        return

    restart_time = STARTUP_TIME if STARTUP_TIME else "Unknown"
    title = "BOT RESTARTED" # Exact title requested
    fields = [
        {"name": "Trainer Script", "value": "ACTIVE", "inline": False}, # Changed from AI Brain
        {"name": "Last update", "value": datetime.now().strftime('%H:%M:%S'), "inline": False},
        {"name": "Last Restart", "value": restart_time, "inline": False}
    ]
    
    response_data = send_status_embed(title, fields, 'blue', message_id=STATUS_MESSAGE_ID)
    
    if not response_data:
        print("⚠️ Failed to send heartbeat. Re-creating notification.")
        STATUS_MESSAGE_ID = None
        save_status_message_id(None) # Clear the invalid ID from file
        send_startup_notification()


def send_critical_alert(error_type, message, details=None):
    """Send critical error alert to Discord"""
    if not CRITICAL_WEBHOOK or not requests:
        return

    fields = [
        {"name": "Bot",        "value": "Training Bot",                              "inline": True},
        {"name": "Error Type", "value": error_type,                                  "inline": True},
        {"name": "Timestamp",  "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "inline": True},
        {"name": "Message",    "value": message,                                      "inline": False},
    ]

    if details:
        fields.append({"name": "Details", "value": str(details)[:1000], "inline": False})

    embed = {
        "title": "🚨 CRITICAL ALERT",
        "color": 0xff0000,
        "fields": fields,
        "footer": {"text": "MSA Training Bot • System Alerts"},
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        requests.post(CRITICAL_WEBHOOK, json={"embeds": [embed]}, timeout=5)
    except:
        pass
