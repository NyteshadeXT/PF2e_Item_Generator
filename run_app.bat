@echo off
cd /d "C:\Users\kkroe\Desktop\shopgen_web" || (echo [ERROR] Can't cd & pause & exit /b 1)
set "PORT=5000"

where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY (echo [ERROR] Python not found & pause & exit /b 1)

if not exist ".venv\Scripts\activate.bat" (
  echo [Setup] Creating venv...
  %PY% -m venv .venv || (echo venv creation failed & pause & exit /b 1)
)
call ".venv\Scripts\activate.bat" || (echo venv activation failed & pause & exit /b 1)

if not exist requirements.txt (
  >requirements.txt echo flask
  >>requirements.txt echo pandas
)
python -m pip install --upgrade pip >nul
pip install -r requirements.txt || (echo pip install failed & pause & exit /b 1)

echo [Info] starting server on http://127.0.0.1:%PORT% ...
start "Shop Inventory Server" cmd /k "call .venv\Scripts\activate.bat && python app.py"
REM open a tab immediately (the app will also open one itself)
start "" "http://127.0.0.1:%PORT%"
