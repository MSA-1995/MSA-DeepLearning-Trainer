"""
🚨 Critical Alerts - Discord Webhook
"""

import requests
from datetime import datetime, timezone
import os

# --- Webhook URLs ---
# يقرأ مباشرة من متغيرات البيئة التي تم تحميلها في deep_trainer_v2.py
TRAINING_WEBHOOK = os.getenv('TRAINING_WEBHOOK')
CRITICAL_WEBHOOK = os.getenv('CRITICAL_WEBHOOK')


def send_training_notification(title, fields, color_hex='2ecc71'):
    """Send a rich notification about the training process."""
    send_discord_embed(title, fields, color_hex, message_id=None, webhook_url=TRAINING_WEBHOOK)

def send_critical_alert(error_type, message, details=None):
    """Send a critical error alert to Discord."""
    fields = [
        {"name": "Trainer", "value": "Deep Learning Trainer", "inline": True},
        {"name": "Error Type", "value": error_type, "inline": True},
        {"name": "Timestamp", "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "inline": True},
        {"name": "Message", "value": str(message), "inline": False}
    ]
    if details:
        # Ensure details are string and truncate if too long
        details_str = str(details)
        if len(details_str) > 1000:
            details_str = details_str[:1000] + "..."
        fields.append({"name": "Details", "value": f"```\n{details_str}\n```", "inline": False})

    send_discord_embed("🚨 CRITICAL TRAINER ALERT", fields, 'e74c3c', message_id=None, webhook_url=CRITICAL_WEBHOOK)


def send_discord_embed(title, fields, color_hex, message_id=None, webhook_url=None):
    """
    Sends or edits a rich embed message to a Discord webhook.
    Returns the response JSON from Discord if successful, otherwise None.
    """
    if not webhook_url:
        # If the specific webhook is not available, do not send.
        return None

    # Convert color from hex string to integer
    try:
        color_int = int(color_hex.lstrip('#'), 16)
    except (ValueError, TypeError):
        color_int = 0 # Default to black if conversion fails

    embed = {
        "title": title,
        "color": color_int,
        "fields": fields,
        "footer": {
            "text": "MSA Deep Learning Trainer"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Determine the URL and method (POST for new, PATCH for edit)
    url = f"{webhook_url}/messages/{message_id}" if message_id else webhook_url
    method = 'PATCH' if message_id else 'POST'
    
    try:
        response = requests.request(
            method,
            url,
            json={"embeds": [embed]},
            timeout=10
        )
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as e:
        # If editing failed (e.g., message deleted), response.status_code will be 404
        # We return None to indicate failure.
        print(f"⚠️ Discord API Error: {e.response.status_code} - {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send Discord notification: {e}")
        return None
