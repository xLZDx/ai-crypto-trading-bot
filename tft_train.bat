@echo off
set PYTHONUNBUFFERED=1
cd /d "D:\test 2\AI trading assistance"
"D:\test 2\AI trading assistance\venv\Scripts\python.exe" -u -m src.engine.train_tft_model --timeframe 1h --epochs 10 >> "D:\test 2\AI trading assistance\logs\tft_standalone.log" 2>&1
