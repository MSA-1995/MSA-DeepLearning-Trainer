"""
🧠 Deep Learning Trainer V2 - Entry Point
8 Specialized LightGBM Models: AI Brain + 7 Consultants
"""

# ========== Add src to path ==========
import sys
import os
# المسار إلى المجلد الرئيسي للمشروع (TradingBota_AI)
# يفترض أن هذا السكربت موجود في TradingBota_AI/scripts/MSA-DeepLearning-Trainer
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src_path = os.path.join(project_root, 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)
    print(f"✅ Added '{src_path}' to Python path")

# =====================================

# ========== AUTO-UPDATE PIP ==========
import subprocess
import sys

try:
    print("🔄 Checking pip updates...")
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
        capture_output=True, check=False, timeout=30, text=True
    )
    if "Successfully installed" in result.stdout:
        print("✅ pip updated successfully")
    else:
        print("✅ pip is up to date")
except Exception as e:
    print(f"⚠️ pip update skipped: {e}")

# ========== AUTO-INSTALL ==========
_packages = ['requests', 'cryptography', 'lightgbm', 'scikit-learn', 'numpy', 'pandas', 'psycopg2-binary']
for _pkg in _packages:
    try:
        __import__(_pkg.replace('-', '_').split('.')[0])
    except ImportError:
        print(f"📦 Installing {_pkg}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', _pkg],
                       capture_output=True, check=False, timeout=120)

# ========== LOAD ENV FILE & KEYS ==========
import os

# --- 1. Load .env file ---
# Look for .env in the script's directory first, then in common server paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_env_loaded = False
for _env_file in [
    os.path.join(_script_dir, '.env'), # Local .env file
    '/home/container/DeepLearningTrainer_XGBoost/.env',
    '/home/container/.env',
]:
    try:
        if os.path.exists(_env_file):
            with open(_env_file, 'r', encoding='utf-8') as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line and not _line.startswith('#') and '=' in _line:
                        _k, _v = _line.split('=', 1)
                        os.environ.setdefault(_k.strip(), _v.strip())
            print(f"✅ Environment variables loaded from {_env_file}")
            _env_loaded = True
            break # Stop after finding the first .env file
    except Exception as e:
        print(f"⚠️ Could not load environment file {_env_file}: {e}")

# --- 2. Load Encryption Key from Flash Drive (if not already in environment) ---
if not os.getenv('ENCRYPTION_KEY'):
    print("🔑 ENCRYPTION_KEY not found in environment, checking flash drive...")
    _key_file = r"D:\bot_keys.txt"
    try:
        if os.path.exists(_key_file):
            with open(_key_file, 'r', encoding='utf-8') as f:
                _lines = f.readlines()
                if len(_lines) >= 3:
                    _key = _lines[2].strip() # Key is on the third line
                    if _key:
                        os.environ['ENCRYPTION_KEY'] = _key
                        print("✅ Encryption key loaded from flash drive (D:\\bot_keys.txt)")
                    else:
                        print("⚠️ Key on third line of flash drive file is empty.")
                else:
                    print("⚠️ Flash drive key file is invalid (fewer than 3 lines).")
        else:
            # This is not an error, just info. The user might be on the server.
            print("ℹ️ Flash drive key file not found. This is normal on a server.")
    except Exception as e:
        print(f"❌ Error reading encryption key from flash drive: {e}")

# ========== MAIN ==========
import threading
import time
from trainer import DeepLearningTrainerXGBoost

def main():
    trainer = DeepLearningTrainerXGBoost()
    # Train immediately, then every 6 hours.
    trainer.run_continuous(interval_hours=6)


if __name__ == "__main__":
    # --- Run Main Application ---
    main()
