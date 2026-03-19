"""
🧠 Deep Learning Trainer V2 - Entry Point
8 Specialized LightGBM Models: AI Brain + 7 Consultants
"""

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

# ========== LOAD ENV FILE ==========
import os

for _env_file in [
    '/home/container/DeepLearningTrainer_XGBoost/.env',
    '/home/container/.env',
]:
    try:
        with open(_env_file) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith('#') and '=' in _line:
                    _k, _v = _line.split('=', 1)
                    os.environ.setdefault(_k.strip(), _v.strip())
        break
    except:
        pass

# ========== MAIN ==========
from trainer import DeepLearningTrainerXGBoost


def main():
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not found!")
        return

    trainer = DeepLearningTrainerXGBoost(database_url)
    # Train every 6 hours OR every 100 new trades (whichever comes first)
    trainer.run_continuous(interval_hours=6, trades_trigger=100)


if __name__ == "__main__":
    main()
