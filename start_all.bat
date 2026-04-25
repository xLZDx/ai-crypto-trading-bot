@echo off
cd /d "%~dp0"
echo Starting AI Trading Assistance...

echo Terminating old processes to prevent freezing...
taskkill /F /IM python.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul

echo Training All Machine Learning Models...
cmd /c "if exist venv\Scripts\activate.bat (call venv\Scripts\activate.bat) & python src\engine\train_all_models.py"

echo Starting Bot...
start "AI Trading Bot" cmd /k "if exist venv\Scripts\activate.bat (call venv\Scripts\activate.bat) & pip install websockets vaderSentiment ccxt python-dotenv flask pandas scikit-learn joblib mcp google-generativeai youtube-transcript-api beautifulsoup4 requests debugpy & python -m debugpy --listen 0.0.0.0:5678 src\main.py"

echo Starting Dashboard...
start "AI Trading Dashboard" cmd /k "if exist venv\Scripts\activate.bat (call venv\Scripts\activate.bat) & pip install flask & python src\dashboard\app.py"

echo Starting MCP Server...
start "AI Trading MCP Server" cmd /k "if exist venv\Scripts\activate.bat (call venv\Scripts\activate.bat) & python src\mcp_server\server.py"

echo Applications have been started in separate windows.
echo To view them, look for the "AI Trading Bot", "Dashboard", and "MCP Server" console windows.
pause
