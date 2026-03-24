"""
🧠 Deep Learning Trainer V2 - Entry Point
8 Specialized LightGBM Models: AI Brain + 7 Consultants
"""

# ========== SETUP SYS.PATH ==========
# This is the most critical part. It tells the script where to find the 'src' folder.
import sys
import os
try:
    # Get the directory of the current script (MSA-DeepLearning-Trainer)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to the project root (TradingBot-AI)
    _project_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))
    # Construct the path to the 'src' directory
    _src_path = os.path.join(_project_root, 'src')
    # Add the 'src' path to the top of Python's search paths
    if _src_path not in sys.path:
        sys.path.insert(0, _src_path)
    print(f"✅ System path configured to include: {_src_path}")
except Exception as e:
    print(f"❌ CRITICAL: Failed to configure system path. Cannot continue. Error: {e}")
    sys.exit(1) # Exit if we can't find the src directory

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
_packages = ['requests', 'cryptography', 'lightgbm', 'scikit-learn', 'numpy', 'pandas', 'psycopg2-binary']
for _pkg in _packages:
    try:
        __import__(_pkg.replace('-', '_').split('.')[0])
    except ImportError:
        print(f"📦 Installing {_pkg}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', _pkg],
                       capture_output=True, check=False, timeout=120)

# ========== LOAD ENV FILE & KEYS ==========
# --- 1. Load .env file ---
# Look for .env in the script's directory first, then in common server paths
_env_loaded = False
for _env_file in [
    os.path.join(_script_dir, '.env'), # Local .env file
    os.path.join(_project_root, '.env'), # Project root .env file
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

# ========== MAIN ==========#
from trainer import DeepLearningTrainerXGBoost
from db_manager import DatabaseManager # استيراد مدير قاعدة البيانات

def apply_db_schema_fix():
    """
    يقوم بتطبيق إصلاح لمرة واحدة على جدول 'dl_models_v2'
    عن طريق إضافة قيد UNIQUE. تم تصميم هذا ليعمل تلقائيًا وأمان،
    حتى لو تم تطبيق الإصلاح بالفعل.
    """
    print("🔧 محاولة تطبيق إصلاح مخطط قاعدة البيانات لـ 'dl_models_v2'...")
    db_manager = None
    conn = None
    try:
        db_manager = DatabaseManager()
        conn = db_manager._get_conn()
        if not conn:
            print("⚠️ تعذر الحصول على اتصال بقاعدة البيانات لتطبيق إصلاح المخطط. ستتم إعادة المحاولة في التشغيل التالي.")
            return

        with conn.cursor() as cursor:
            # هذا الأمر سيفشل إذا كان القيد موجودًا بالفعل، وهو ما نريده.
            alter_command = "ALTER TABLE dl_models_v2 ADD CONSTRAINT uq_model_name_type UNIQUE (model_name, model_type);"
            cursor.execute(alter_command)
            conn.commit()
            print("✅ تم تطبيق قيد UNIQUE بنجاح على جدول 'dl_models_v2'.")

    except Exception as e:
        # رمز الخطأ '42710' في PostgreSQL هو لـ 'duplicate_object'.
        # يمكننا تجاهله بأمان، لأنه يعني أن القيد موجود بالفعل.
        if '42710' in str(e) or "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            print("ℹ️ مخطط قاعدة البيانات محدث بالفعل. لا حاجة لإجراء تغييرات.")
        else:
            # لأي خطأ آخر، نقوم بطباعته والتراجع.
            print(f"❌ حدث خطأ غير متوقع أثناء تطبيق إصلاح المخطط: {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception as rb_e:
                    print(f"❌ فشل التراجع عن المعاملة: {rb_e}")
    finally:
        if conn and db_manager:
            db_manager._close_conn(conn)

def main():
    trainer = DeepLearningTrainerXGBoost()
    # Train immediately, then every 6 hours.
    trainer.run_continuous(interval_hours=6)


if __name__ == "__main__":
    # --- تطبيق إصلاح المخطط قبل التشغيل ---
    apply_db_schema_fix()

    # --- تشغيل التطبيق الرئيسي ---
    main()
