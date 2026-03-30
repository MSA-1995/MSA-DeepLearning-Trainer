@echo off
echo =================================================
echo  Starting MSA Deep Learning Trainer v2 (Local)
echo =================================================
echo.

REM --- تحديد مسار السكربت ---
set SCRIPT_PATH=%~dp0core\deep_trainer_v2.py

echo Running script: %SCRIPT_PATH%
echo.

REM --- تشغيل السكربت باستخدام بايثون ---
python "%SCRIPT_PATH%"

echo.
echo =================================================
echo  Training script has finished.
echo =================================================
pause