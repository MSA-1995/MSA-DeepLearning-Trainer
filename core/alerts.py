"""
🚨 Critical Alerts
"""

from datetime import datetime, timezone


TRAINING_WEBHOOK = None
CRITICAL_WEBHOOK = None


def send_training_notification(title, fields, color_hex='2ecc71'):
    """No-op: Discord notifications disabled."""
    return None


def send_critical_alert(error_type, message, details=None):
    """Log critical alerts to console only."""
    print(f"🚨 CRITICAL ALERT [{error_type}] - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Message: {message}")
    if details:
        details_str = str(details)
        if len(details_str) > 1000:
            details_str = details_str[:1000] + "..."
        print(f"   Details: {details_str}")
    return None


def send_discord_embed(title, fields, color_hex, message_id=None, webhook_url=None):
    """No-op: Discord notifications disabled."""
    return None