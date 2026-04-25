# AI Trading Assistance Bot

## Overview
The AI Trading Assistance Bot is a comprehensive, autonomous cryptocurrency trading system and dashboard. It combines traditional algorithmic trading techniques, machine learning predictive models, and cutting-edge Agentic Large Language Models (LLMs) to make intelligent, risk-adjusted trading decisions.

## Why We Built This
This project was built to create a robust, end-to-end trading system that doesn't just rely on static technical indicators, but rather leverages an ensemble of approaches:
1. **Technical Analysis**: Using math and historical patterns (e.g., Elliott Waves).
2. **Machine Learning**: Predicting future short-term price action using advanced Gradient Boosting classifiers trained on years of historical data.
3. **Agentic AI Overlay**: Utilizing LLMs to read real-time news and macroeconomics to act as a definitive "Risk Manager" that can veto catastrophic trades (e.g., stopping the bot from buying during a sudden hack or market crash).
4. **Transparency**: A Flask-based web dashboard to interact with the bot, view trade history, retrain models, and ask the AI assistant for market insights.

## Tech Stack
- **Backend Core**: Python 3
- **Exchange API Integration**: `ccxt` (Binance Spot & Futures, supporting Testnet/Mainnet)
- **Web Dashboard**: Flask (Python), HTML, CSS, JavaScript
- **Machine Learning**: `scikit-learn` (HistGradientBoostingClassifier), `pandas`, `joblib`
- **AI/LLM Integration**: `google-generativeai` (Gemini 2.5 Flash)
- **Data Storage**: Local JSON (`data/state.json`, `data/trades.json`), GZipped CSVs for historical market data.

## System Architecture

### 1. Data Ingestion (`src/data_ingestion/`, `src/tools/`)
- Downloads and updates OHLCV data from Binance.
- Uses `binance_archive_downloader.py` to bulk download historical archives from `data.binance.vision` (1h and 1m intervals) for massive ML training datasets.
- Smart updating system to only append new candles and avoid memory overhead using GZip and streaming.

### 2. Analysis & Machine Learning (`src/analysis/`, `src/engine/train_model.py`)
- **Elliott Wave Analyzer**: Pure Python ZigZag algorithm to find peaks and troughs without heavy external libraries.
- **Feature Engineering**: Generates rolling SMAs, RSI, MACD, Bollinger Bands, Stochastic oscillators, and volatility metrics.
- **ML Predictive Model**: Uses `HistGradientBoostingClassifier` to predict if the price will be higher in the near future based on complex feature combinations. The model handles class balancing and uses early stopping to prevent overfitting.

### 3. Execution & Risk Management (`src/engine/`)
- **Order Manager**: Interfaces directly with Binance via `ccxt`. Handles Spot and Futures (Long/Short) execution, precision formatting, and balance checking.
- **Agentic LLM (`agentic_llm.py`)**: Before a trade is executed, this module feeds the technical justification and recent news headlines to Gemini 2.5 Flash. The LLM responds in strict JSON to either `APPROVE` or `REJECT` (VETO) the trade based on fundamental context.

### 4. Dashboard (`src/dashboard/app.py`)
- Real-time monitoring web interface.
- Displays current portfolio state, recent trades, active signals, and live server logs.
- Includes an interactive Chat interface hooked up to Gemini, allowing users to ask for market analysis, summarize YouTube/Article links via scraper tools, or manually trigger background model retraining.

---

## Project State for AI Assistants (Context Handoff)

**Attention AI Coding Assistants:** If you are picking up this project, here is exactly where we left off and what you need to know.

### Current State
- The system is fully translated into English.
- The dashboard is using a stable, single-column stacked card layout (fixed from a previous broken CSS grid layout).
- ML prediction crashes caused by missing 1-minute feature data have been resolved.
- A strict rollback tracking rule is in effect. Always refer to `notes_for_ai.md` and `rollback_plan.md` for historical context.
- Process restart scripts (`stop_all.bat`, `restart_all.ps1`) have been updated to aggressively kill hanging `python.exe` background processes.

### Immediate Pending Task
**Goal:** Integrate a Telegram Bot for live trade notifications.
1. The Telegram bot needs to alert the user when a new trade is opened, closed, or vetoed by the Agentic LLM.
2. It should tie into the existing trade tracking loop without blocking the main engine (avoid parallelism issues; stick to synchronous or non-blocking asynchronous requests).

### Core Constraints & Rules
- **Execution Style**: Work on tasks strictly one by one. No parallelism or concurrent execution to avoid freezing.
- **Stop Guessing**: If debugging, request the user to attach their IDE to port `5678` or read from `logs/trading.log`. Do not guess fixes.
- **Logging**: Ensure all new modules use standard `logging` to output to the console and the global trading log.
- **Rollback Tracking**: Document all major changes in `notes_for_ai.md` at the end of the session.

---

## How to Run
1. Ensure your `.env` file is configured with your `API_KEY`, `API_SECRET`, and `GEMINI_API_KEY`.
2. Run `stop_all.bat` to clear any hanging background processes.
3. Run `restart_all.bat` (or your chosen startup script) to spin up the data modules, execution engine, and the Flask dashboard.
4. Access the dashboard at `http://localhost:5000`.