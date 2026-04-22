"""
🧠 Deep Learning Trainer V2 - Entry Point
8 Specialized LightGBM Models: AI Brain + 7 Consultants
"""

# ========== SETUP SYS.PATH ==========
import sys
import os
import time

try:
    _script_dir   = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))
    _src_path     = os.path.join(_project_root, 'src')
    if _src_path not in sys.path:
        sys.path.insert(0, _src_path)
    print(f"✅ System path configured to include: {_src_path}")
except Exception as e:
    print(f"❌ CRITICAL: Failed to configure system path. Error: {e}")
    sys.exit(1)

# ========== AUTO-UPDATE PIP ==========
import subprocess

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
_packages = [
    'requests', 'cryptography', 'lightgbm',
    'scikit-learn', 'numpy', 'pandas', 'psycopg2-binary'
]

for _pkg in _packages:
    try:
        __import__(_pkg.replace('-', '_').split('.')[0])
    except ImportError:
        print(f"📦 Installing {_pkg}...")
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', _pkg],
            capture_output=True, check=False, timeout=120
        )

# ========== LOAD ENV FILE ==========
_env_loaded = False
_env_paths  = [
    os.path.join(_script_dir, '.env'),
    os.path.join(os.path.dirname(_script_dir), '.env'),
    os.path.join(_project_root, '.env'),
    '/home/container/DeepLearningTrainer_XGBoost/.env',
    '/home/container/.env',
]

for _env_file in _env_paths:
    try:
        if os.path.exists(_env_file):
            with open(_env_file, 'r', encoding='utf-8') as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line and not _line.startswith('#') and '=' in _line:
                        _k, _v = _line.split('=', 1)
                        os.environ.setdefault(_k.strip(), _v.strip())
            print(f"✅ Environment variables loaded from: {_env_file}")
            _env_loaded = True
            break
    except Exception as e:
        print(f"⚠️ Could not load environment file {_env_file}: {e}")

if not _env_loaded:
    print("⚠️ No .env file found in any of the expected paths.")

# ========== LOAD ENCRYPTION KEY ==========
if not os.getenv('ENCRYPTION_KEY'):
    print("🔑 ENCRYPTION_KEY not found in environment, checking key file...")
    _key_file = os.getenv('KEY_FILE_PATH', r"D:\bot_keys.txt")
    try:
        if os.path.exists(_key_file):
            with open(_key_file, 'r', encoding='utf-8') as _f:
                _lines = _f.readlines()
            if len(_lines) >= 3:
                _key = _lines[2].strip()
                if _key:
                    os.environ['ENCRYPTION_KEY'] = _key
                    print(f"✅ Encryption key loaded from: {_key_file}")
                else:
                    print("⚠️ Key on third line of key file is empty.")
            else:
                print("⚠️ Key file is invalid (fewer than 3 lines).")
        else:
            print("ℹ️ Key file not found. This is normal on a server.")
    except Exception as e:
        print(f"❌ Error reading encryption key from file: {e}")

# ========== MAIN ==========
from trainer import DeepLearningTrainerLightGBM


def main():
    try:
        trainer = DeepLearningTrainerLightGBM()
        trainer.run_continuous(interval_hours=6)
    except Exception as e:
        print(f"❌ An unexpected error occurred in the main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()