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

ENCRYPTED_CRITICAL_WEBHOOK = "gAAAAABpuay0FYK_AXFBy_trEWffy5Ho8xzGr4-zSrASVWnVqipfKR3_k6C9VsucFp1qPEzcHaXDb8txhiVUkFrXFKTD9XIguwTnCZcpj6FqnGTKi7-jaCDb3eHEdeNiZcmKpax4ma_WNrlRHLJDTVDSuWvtff41bmMLyohJ3_ezK3Ox0-8iHeVDnutL1oyU7sMHwWfWY4f12xvc--03MTYqu42u_0IfNbEvyCt2LGvDNlVIJcCkQeg="


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
