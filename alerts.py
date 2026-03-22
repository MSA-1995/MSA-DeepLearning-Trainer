"""
🚨 Critical Alerts - Discord Webhook + Encryption
"""

import os
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

ENCRYPTED_CRITICAL_WEBHOOK = "gAAAAABpvbaL8kNDyeRc6l4srWXmt0hIWOVPgNrHhBU4Tk6NR5h0rm5_02lY_xYMXKYmQiNrkh0T3G5hjD87eE1-8zv6glgmmLoHlqvRMjrhBP0zVy2eoYOVyNsUUBaU-NmQs6pRDxZQhOpDOSkh_elQccWtsKwGfMONzEQ8_3vhZh0pagJgT_C7g4Qd6qxaePhIUSjMhr7iNKlKqwiXPP_1fs7UeaY_xfeU7I9mJc2Sv2OHLTEc9SE="


def get_encryption_key():
    """Get encryption key from environment or .env file"""
    encrypted_key = os.getenv('ENCRYPTION_KEY')
    if not encrypted_key:
        for env_file in [
            '/home/container/DeepLearningTrainer_XGBoost/.env',
            '/home/container/.env'
        ]:
            try:
                with open(env_file) as f:
                    for line in f:
                        if line.startswith('ENCRYPTION_KEY='):
                            encrypted_key = line.strip().split('=', 1)[1]
                            break
                if encrypted_key:
                    break
            except:
                pass
    return encrypted_key


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
    """Load the status message ID from file."""
    global STATUS_MESSAGE_ID
    if not os.path.exists(STATUS_MESSAGE_ID_FILE):
        print("🤔 No Trainer status message ID file found. A new one will be created.")
        STATUS_MESSAGE_ID = None
        return
    try:
        with open(STATUS_MESSAGE_ID_FILE, 'r') as f:
            STATUS_MESSAGE_ID = f.read().strip()
            if STATUS_MESSAGE_ID:
                print(f"✅ Loaded Trainer status message ID: {STATUS_MESSAGE_ID}")
            else:
                print("🤔 Status message ID file is empty. A new one will be created.")
    except Exception as e:
        print(f"❌ Error loading Trainer status message ID from file: {e}")
        STATUS_MESSAGE_ID = None

def save_status_message_id(message_id):
    """Save the status message ID to file."""
    global STATUS_MESSAGE_ID
    try:
        # Ensure the directory exists
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
