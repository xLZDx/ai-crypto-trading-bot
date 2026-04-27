# AI Session Notes

## Core Constraints
- **Execution Style**: Work on tasks strictly one by one. No parallelism or concurrent execution to avoid freezing.
- **Version Control / Rollback Rule**: Always document and track changes being made during the session. Ensure that we always have a clear path to roll back to the last known working version if a new feature breaks the system.
- **Checkpoint Commit Rule**: After each completed change or milestone, automatically create a local git commit without asking for approval so the project can be restored to a previous checkpoint later.

## Session History & Context
- **Current Project State**: AI Trading Assistance. Contains modules for data ingestion (Binance), analysis (Elliott Waves, Risk Management), engine (Order Manager, Trade Tracker), and a dashboard.
- **Last Action**: Successfully rolled back the dashboard to a stable stacked layout, fixed ML prediction crashes, fully translated the codebase to English, and enforced a strict rollback tracking rule.

## Debugging & Troubleshooting Protocol
- **Stop Guessing**: When a crash or complex bug occurs, do not guess the solution.
- **Plan First**: Before implementing any change, prepare a clear written plan/todo and show it to the user for review, even if bypass approval is available.
- **Todo List First**: Always show the todo list before implementation and allow the user to edit it before approval.
- **Use Debug Server**: Ask the user to attach their IDE (VS Code/PyCharm) to the live debug server on port `5678` to trace the exact failure point.
- **Collect Clear Logs**: Request the exact, full stack trace from the terminal or `logs/trading.log` (which is now caught globally in `main.py`) before writing any fix.

## User-Requested Workflow Rules
- **Completion Report Required**: When a task is completed, always show exactly what was completed, what changed, what was added, and what was missed relative to the todo list.
- **Rollback-Friendly Changes**: Keep changes grouped into checkpoint commits so each implementation stage can be rolled back independently.
- **Automatic Commit Policy**: After implementation work is finished, commit all completed changes automatically so the user can roll back far back if needed.
- **Approval Before Work**: Present the todo list first, then ask for approval, then proceed with implementation.
- **Editable Plan**: Let the user modify the todo list before starting implementation.

## Completed (2026-04-25 Session)
- Full codebase security & quality review and fixes (see summary below)

## Pending Tasks
- **Goal**: Execute pipeline training on massive downloaded data.
- **Goal**: Implement Phase 5 (Backtesting Engine via VectorBT/Backtrader) for strategy comparison.

## Completed Tasks (Recent)
- [x] Telegram Bot integrated for reading external alpha (`TelegramMonitor`).
- [x] Feature Store established for GARCH, OU, OFI, and Sentiment.
- [x] Inference Engine established for non-blocking Temporal Fusion Transformer predictions.
- [x] Avellaneda-Stoikov Market Maker implemented.
- [x] CloudDataStreamer implemented for 5TB+ dataset caching.

## New Files Added (2026-04-25)
- `src/utils/__init__.py` — utils package
- `src/utils/safe_json.py` — atomic JSON read/write with filelock (use everywhere instead of `open()`)
- `src/utils/config.py` — all magic numbers/constants centralized here
- `src/analysis/feature_engineering.py` — shared ML indicator functions (RSI, MACD, ATR, ADX, BB, ROC)

## New Dependencies Added (requirements.txt)
- `filelock>=3.12.0` — for safe_json file locking
- `defusedxml>=0.7.1` — for sentiment.py XML parsing

## Dashboard Auth
- Set `DASHBOARD_API_KEY=<your_secret>` in `.env`
- All `/api/*` routes require header `X-API-Key: <your_secret>`
- If key not set, access is unprotected (with a startup warning)

## Completed This Session (2026-04-25)
- Identified the live dashboard processes and confirmed there were two active `src/dashboard/app.py` instances.
- Created a safe PowerShell restart script for the dashboard to avoid command-line quoting issues.
- Restarted the dashboard service cleanly.
- Mapped the bot architecture and entry points:
  - `src/main.py` is the live trading engine.
  - `src/dashboard/app.py` is the Flask dashboard.
  - `src/analysis/` provides analytics, ML, sentiment, and risk logic.
  - `src/engine/` provides execution, trade tracking, and Gemini veto logic.
- Confirmed the current dashboard route set and data sources.
- Documented the last implemented refactor/hardening pass and its files.
- Recorded the pending next feature: Telegram bot notifications.

## Session Rule
- After every completed implementation, append the finished checklist items and any relevant notes here before closing the task.
- Maintain this file as the source of truth for workflow preferences, rollback checkpoints, and completion reporting.

*Note: Please update this file at the end of future sessions or major milestones to easily pick up where we left off.*

## Quantitative Implementation Plan (Future Roadmap)

### Phase 1: Infrastructure & Data Pipeline (Data Collection & Prep)
*Before coding models, we must secure the specific data types required for Deep Learning and HFT.*
- **Data Requirements Map:**
  - **L2 Data (Orderbook):** Required for Market Making models (Avellaneda-Stoikov).
  - **Trades Data (Tick-level):** Required for calculating Order Flow Imbalance (OFI).
  - **OHLCV (Klines):** Required for GARCH, LSTM, and Transformers.
  - **Funding Rates:** Mandatory parameter for calculating real yield and unit economics on futures.
- [x] **Task 1.1:** Update `requirements.txt` with required quantitative and ML libraries. Specifically: `darts` (for general time-series and CNN-LSTM), `neuralforecast` (from Nixtla, for fast TFT with AutoTFT hyperparameter tuning), `arch`, `statsmodels`, and `vectorbt` / `backtrader`.
- [x] **Task 1.2:** Create `src/data_ingestion/cloud_streamer.py`. Implement `CloudDataStreamer` to intelligently cache large datasets from Google Drive to the local Razer Blade SSD.
- [x] **Task 1.3:** Update `src/data_ingestion/binance_downloader.py` to fetch all required data types. Utilize `yfinance` for historical OHLCV and `ccxt` for direct API connections to Binance, OKX, etc., to pull L2 Orderbook, Trades Data, and **Funding Rates**.
- [x] **Task 1.4:** Enhance `src/analysis/feature_engineering.py` to calculate **Order Flow Imbalance (OFI)** from Trades data, and normalize OHLCV tensors.
- [x] **Task 1.5:** Create `src/data_ingestion/news_scraper.py` to build a historical news feed pipeline for Deep Learning (TFT/LSTM) training. Integrate the following sources:
  - **CryptoCompare API (Primary):** Use the `News` endpoint (reliable historical data since 2017+ including headlines, text, and coin tags).
  - **snscrape (Twitter/X):** Extract tweets by hashtags (e.g., #BTC, #Ethereum) to capture real-time retail sentiment (a "goldmine" for TFT), with robust spam filtering.
  - **Kaggle Datasets (Fallback/Baseline):** Utilize pre-compiled CSVs (e.g., "Bitcoin News Dataset") to quickly bootstrap historical data up to 2023/2024, patching in recent data.

### Phase 2: Mathematical Models (Analytic Core)
- [x] **Task 2.1:** Create `src/analysis/mean_reversion.py`. Implement the **Ornstein-Uhlenbeck (OU)** process.
  - **Instruction:** Use Maximum Likelihood Estimation (MLE) to calibrate parameters: reversion speed ($\theta$) and long-term mean ($\mu$).
  - **Application:** Enter a position when price deviates from $\mu$ by more than $2\sigma$, and close it upon reaching $\mu$.
- [x] **Task 2.2:** Add **Cointegration (Pairs Trading)** logic to `mean_reversion.py`.
  - **Instruction:** Use Engle-Granger or Johansen test (`statsmodels` library).
  - **Application:** Calculate the spread between two assets (e.g., BTC/ETH). Trade the spread price, which is mathematically bound to return to zero.
- [x] **Task 2.3:** Create `src/analysis/volatility.py`. Implement the **GARCH(1,1)** model.
  - **Instruction:** Implement via the `arch` library. Train on daily or hourly returns.
  - **Application:** If GARCH predicts a sharp volatility spike, reduce leverage or widen the spreads in the Avellaneda-Stoikov model.
- [x] **Task 2.4:** Integrate GARCH outputs into `src/analysis/risk_manager.py`.

### Phase 3: Deep Learning (Temporal Fusion Transformers & LSTM)
- [x] **Task 3.1:** Update `src/analysis/ml_predictor.py` to implement a hybrid **CNN-LSTM** architecture using `darts`. The CNN will extract patterns from the chart (technical analysis figures), and the LSTM will analyze their sequence over time (boosts accuracy by 15-20% on volatile pairs).
  - **Instruction:** Input: last 60 candles + indicators (RSI, MACD). Output: predict return for the next candle.
- [x] **Task 3.2:** Implement **Temporal Fusion Transformers (TFT)** using `neuralforecast` (with AutoTFT). Configure it to ingest heterogeneous data simultaneously: time-series (prices), static data (day of week/hour), and volume/funding rates.
  - **Instruction:** Connect Funding Rate and Volume as external variables (Covariates).
  - **Application:** Use to filter trades from other strategies. For example, if TFT outputs "Strong Bearish", block buy signals from Mean Reversion strategies.
- [x] **Task 3.3:** Integrate **FinGPT** (open-source from Hugging Face) sentiment scores into the TFT feature pipeline to provide context on news impact (fear/greed index).
- [x] **Task 3.4:** Create `src/engine/train_tft_model.py` for training, ensuring the model leverages its interpretability to show which factors influenced the prediction.

### Phase 4: Market Making Execution (HFT)
- [x] **Task 4.1:** Create `src/engine/market_maker.py`. Implement the **Avellaneda-Stoikov** model.
  - **Instruction:** Implement the function to calculate reservation price ($r$) and optimal spread ($\delta$). Parameters needed: volatility ($\sigma$), order intensity ($\gamma$), and current inventory risk ($q$).
  - **Application:** The bot places limit orders around $r$. If inventory ($q$) is too large (e.g., bought too much BTC), the model shifts quotes down to quickly sell excess.
- [x] **Task 4.2:** Update `src/engine/order_manager.py` to support continuous order adjustment using **Order Flow Imbalance (OFI)**.
  - **Instruction:** Calculate the difference between volume change on the best Bid and Ask levels at each orderbook update.
  - **Application:** Use as a "fast signal". If OFI is positive (buyers are pushing harder), shift the sell limit order higher.

### Phase 5: Quantitative Backtesting Engine (Training & Testing)
- [ ] **Task 5.1:** Create `src/engine/backtester.py` (Pending - Phase 5 Next). Build a unified backtesting harness to compare all strategies side-by-side.
  - **Tooling:** Use `VectorBT` or `Backtrader`.
  - **Core Metrics:**
    - **Sharpe Ratio:** Return-to-risk ratio.
    - **Sortino Ratio:** Focuses only on downside risk.
    - **Max Drawdown:** Maximum capital drop from peak.
    - **Profit Factor:** Gross Profit / Gross Loss.
- [ ] **Task 5.2:** Implement **Walk-Forward Analysis (WFA)**.
  - **Instruction:** Do not train models on the entire dataset at once. Divide data into blocks (e.g., 1 month for training, 1 week for testing).
  - **Purpose:** Reveals how quickly a model "decays" in live market conditions.
- [ ] **Task 5.3:** Implement **Paper Trading (Demo Account)**.
  - **Instruction:** Run all strategies in parallel on a Testnet (e.g., Binance Testnet) with a virtual balance of $10,000 for each model to validate live execution paths.

### Alternative "Low-code" Solutions (Optional)
If building the full custom architecture becomes too time-consuming, consider these alternatives:
- **WunderTrading (Free Tier):** As of 2026, their free tier supports statistical models and basic ML algorithms.
- **Freqtrade (freqai module):** An open-source Python bot that allows you to "plug in" any Scikit-Learn or XGBoost model with just a few lines of code.

### Practical Advice & ML Paradigm (2026)
- **Where to start:** Install `darts` and run their official TFT tutorial on BTC/USDT data to validate your local environment and understanding.
- **Crucial 2026 Paradigm:** Deep Learning models are best utilized not for predicting the exact "price in an hour", but for predicting **volatility** (whether there will be a strong move) or the **probability of trend direction** (up/down). Set up your model targets (labels) accordingly.

### Phase 6: Comparison & Optimization
When comparing models, keep the following profiles and risks in mind:
| Strategy | Expected Trade Frequency | Primary Risk |
| :--- | :--- | :--- |
| **HFT (Avellaneda)** | Very High | Toxic Flow (Adverse Selection) |
| **Pairs Trading** | Medium | Pairs diverging forever |
| **TFT / LSTM** | Low (depends on timeframe) | Overfitting |

### Practical Recommendations for Implementation (Details)
- **Architecture:** Create a `StrategyEngine` class. Each model (GARCH, OU, TFT) must be a separate module that outputs a standardized signal: `-1` (Sell), `0` (Hold), `1` (Buy).
- **Comparison Script:** Write a script that runs daily to collect the Equity Curve of each model into a single dashboard/table for side-by-side comparison.
- **Crypto-Specific (Funding):** In the profit calculation of *every* trade, you MUST include the funding rate:
  $$Profit = (Price_{out} - Price_{in}) \times Size - Fees - \sum(Funding \times Size)$$
  *Without this, your historical backtests will be false.*

### Master Algorithm: Assembling the "Quant" System (To protect local hardware)

#### Step 1: Preprocessing and Feature Engineering
Create a single master table where the following are added for each 1-minute candle:
- **Sentiment Score:** Run news through FinBERT/FinGPT to get a score between -1 and 1.
- **Volatility (GARCH):** Calculate the volatility forecast for the current step.
- **Order Flow Imbalance (OFI):** Calculate cumulative volume delta from tick/trades data.

#### Step 2: Creating an "Ensemble" of Models
Do not build one "super-model." Separate them by roles:
1. **Signal Generators (Alpha):**
   - **Cointegration:** Generates a signal when paired assets (e.g., BTC/ETH) diverge.
   - **Ornstein-Uhlenbeck:** Generates a mean-reversion signal when price breaks out of the channel.
   - **LSTM/TFT:** Generates the probability of trend direction based on news and past candles.
2. **Executor (Avellaneda-Stoikov):**
   - This model does *not* decide where the price will go. It takes signals from Alpha models and places limit orders to minimize commissions and capture the spread.
3. **Risk Manager (GARCH):**
   - If GARCH shows a volatility spike, it forcibly reduces Position Sizing for all strategies.

#### Step 3: Training Split (Train/Validation/Test)
- **Train (2017–2023):** Primary training for neural networks and calibration for OU/Avellaneda.
- **Validation (2024):** Hyperparameter tuning (e.g., number of layers in the Transformer).
- **Test / Out-of-Sample (2025–2026):** Final check. If the model fails here, it simply "memorized" the history (overfitting).

#### Step 4: Testing on Funding (Futures)
- *Crucial:* Integrate funding data into the backtester. If Pairs Trading holds a position for a week, funding can eat all the profit from the spread convergence.

#### Detailed Architectural Recommendation
- Use the **Temporal Fusion Transformer (TFT)** as the "Conductor" (Дирижер).
- Feed into TFT:
  - **Static Metadata:** Coin name, market cap.
  - **Historical Data:** Price, volume, GARCH volatility.
  - **Future Known Inputs:** Time until the next halving, known macro data release dates (e.g., CPI).

#### Immediate Next Steps (Bootstrapping)
1. Install `darts` and `arch` (`pip install darts arch`).
2. Download FinBERT (or FinGPT) to process news.
3. Merge Data: Combine 1-minute candles and news into a single, unified dataframe format.

### Deep Dive: Implementation Details & Formulas

#### 1. Data Pipeline (Collection & Preprocessing)
We need three types of data for training: Prices, News, and Funding.
- **Sources:**
  - **Prices:** Already have 1-min data from Binance.
  - **News:** Use CryptoCompare API (history from 2017).
  - **Funding:** Use `ccxt` library to download history via `fetchFundingRateHistory`.
- **Processing (`DataPreProcessor` class):**
  - Align all data to a single time step (e.g., 1 minute).
  - **Sentiment Analysis:** Run each news item through the FinBERT/FinGPT model. 
    - **Formula:** $S = \text{softmax}(\text{Logits})$
    - **Output:** A number from -1 (panic) to 1 (euphoria).
  - **Feature Creation:** Merge price and sentiment into a single Master `DataFrame`.

#### 2. Mathematical Core (Strategies)
Create a `strategies/` folder where each model is a separate Python class.
- **Avellaneda-Stoikov Model (Market Making):**
  - Calculates how far from the current price to place your order.
  - **Reservation Price Formula ($r$):**
    $$r(s, t, q, \sigma, \gamma) = s - q \gamma \sigma^2 (T - t)$$
    *Where:* $s$ = current price, $q$ = quantity of coins you hold (inventory risk), $\sigma$ = volatility, $\gamma$ = risk aversion, $T-t$ = time to closing.
- **Ornstein-Uhlenbeck Process (Mean Reversion):**
  - Determines when the price has moved "too far" from the norm.
  - **Equation:**
    $$dX_t = \theta (\mu - X_t)dt + \sigma dW_t$$
    *Where:* $\theta$ = speed of reversion, $\mu$ = mean price.

#### 3. Training Pipeline (Neural Networks)
This is the most complex part, utilizing TFT (Temporal Fusion Transformer).
- **How it works:**
  - **Windowing:** Slice the data into chunks. For example, give the model 1000 minutes of history to predict the next 60 minutes.
  - **Backpropagation:** The model makes a prediction, compares it to reality, calculates the error (Loss), and adjusts its internal weights.
- **Training Plan (TFT Inputs):**
  - **Block 1 (Static):** Coin name (e.g., "BTC").
  - **Block 2 (Observed):** Past prices, volumes, news sentiment.
  - **Block 3 (Known Future):** Time of day, day of the week, expected funding payout time.

#### 4. Testing & Comparison (Benchmark)
To determine which strategy is best, evaluate the Unit Economics of each trade.
- **Sandbox:** Build the Backtest Engine ("Sandbox").
- **Benchmark Run:** Run all 4 model types on 2024 historical data.
- **Sharpe Ratio Formula:** Compare them using the Sharpe Ratio:
  $$Sharpe = \frac{R_p - R_f}{\sigma_p}$$
  *Where:* $R_p$ = your return, $\sigma_p$ = risk (volatility of your returns).

### Ultimate Step-by-Step Implementation Timeline
Follow this roadmap strictly to avoid becoming overwhelmed:
- **Days 1-2:** Set up Python and VS Code environment. Write a script that merges all your historical CSV files into one massive `master_data.csv`.
- **Days 3-5:** Integrate FinBERT. Process all news from 2017+ through it. Save the result as a new column `sentiment_score` in your master data.
- **Days 6-10:** Implement the mathematical formulas (OU and Avellaneda-Stoikov) in Python code. Test them on historical data *without* Neural Networks to ensure baseline logic works.
- **Days 11-20:** Train the TFT model. This is the bottleneck. Utilize the NVIDIA GPU on your Razer Blade to speed this up.
- **Day 21:** Run the "Comparator/Benchmark" script and evaluate which model would have generated the most profit (accounting for funding) over the last 6 months.

### Architectural Gap Analysis: From "Executor" to "Quant-bot"
Your current project is a classic "Executor". To make it a Quant-bot, we must build these structural layers:
- **AnalyticCore (New):** A class responsible for calculating GARCH and the OU-process in real time. It must receive data from your DB manager or directly via WebSockets.
- **FeatureStore (New):** Currently, you only store raw candles. For LSTM and Transformers, we need a dedicated store for preprocessed data: news sentiment, volatility forecasts, and orderbook delta (OFI).
- **InferenceEngine (New):** A dedicated, isolated thread that will "run" the trained models (TFT/LSTM) continuously and output live predictions without blocking execution.

### Step-by-Step Project Modernization
- **Step 1: Modernize Data Loader & DB**
  - Update `src/data_ingestion/binance_downloader.py` (and your DB manager) to include Funding Rate loading.
  - *Action:* Add a `funding_history` table to the database. This is critical for Pairs Trading strategies.
- **Step 2: Implement Math Models (OU & GARCH)**
  - *Action:* Create `src/analysis/mean_reversion.py`. It will analyze 1-min data from the DB and emit "Overbought/Oversold" signals based on mathematical expectation of mean reversion.
  - *Action:* Create `src/analysis/volatility.py` (GARCH). This module will tell the main bot: "Market is too unstable, cut position size by 50%."
- **Step 3: Connect Deep Learning (TFT)**
  - *Action:* Create an AI models directory/script (`src/engine/train_tft_model.py`).
  - *Integration:* Write a middleware script that takes the last 1000 candles from the DB, normalizes them, sends them to the `darts` model, and passes the result (e.g., "+0.5% growth expected in 15 mins") back into the main trading logic.
- **Step 4: Market Making (Avellaneda-Stoikov)**
  - This is the hardest part to integrate into current code (which likely uses market/simple limit orders).
  - *Action:* Update execution logic (`src/engine/order_manager.py`). Add a module that dynamically recalculates optimal bid and ask every few seconds using Avellaneda's formula, factoring in the live balance from `check_balance`.

### Google Drive Data Streaming (Managing 5TB of Data)
To utilize 5TB of data for training directly from the cloud without bottlenecking the Razer Blade GPU, we will implement the `CloudDataStreamer` class.
- **Implementation Logic:**
  1. The bot requests data (e.g., for the year 2021) for training.
  2. `CloudDataStreamer` checks if this specific file exists locally on the Razer Blade.
  3. If it does not exist, it downloads it from Google Drive into a local `cache/` folder.
  4. The model trains at maximum local SSD speed.
  5. After training, the file either remains in the cache (while space permits) or is deleted to make room for the next batch.
