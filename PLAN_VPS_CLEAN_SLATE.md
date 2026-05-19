# VPS Clean Slate + Data Pipeline Redesign — FINAL PLAN v11

**Created:** 2026-05-19  
**Status:** FINAL — approved by operator, all agent review rounds incorporated  
**VPS:** 5.104.81.27 (Tokyo, Contabo, 400 GB SSD, 24 GB RAM, Ubuntu 24.04)  
**Branch:** `dev/vps-clean-slate` — all work here, merge to main only after full testing

---

## Phase 0 — Housekeeping + Git Branch

```
git checkout -b dev/vps-clean-slate
git push -u origin dev/vps-clean-slate
```

- Run `tests/test_dashboard.py` — record baseline (0 failures required)
- Remove dead code / debug prints via `refactor-cleaner` agent
- Remove unused imports; update `CLAUDE.md`
- Verify `.gitignore` excludes: `data/parquet/`, `data/raw/`, `models/`, `logs/`, `.env`
- PR → main only after: all phases + smoke-test + 0 test failures + code-reviewer approval

**VPS hardening (run once on fresh VPS, before Phase 1):**
```bash
ufw default deny incoming
ufw allow 22/tcp
ufw allow from <operator_IP> to any port 5000   # Flask dashboard — your IP only
ufw enable
# /etc/ssh/sshd_config:
PasswordAuthentication no
PermitRootLogin prohibit-password               # key-only root login; correct for solo operator
# auto security patches:
apt install unattended-upgrades
dpkg-reconfigure --priority=low unattended-upgrades
# brute-force protection:
apt install fail2ban
```
- **Separate env profiles** — do NOT reuse the same `config.yaml` across modes. Create distinct profiles:
  - `config/training.yaml` — high memory limits, no order output, verbose logging
  - `config/backtest.yaml` — historical slippage model, no live API calls
  - `config/paper.yaml` — live data, paper order sink, full signal logging
  - `config/live.yaml` — live data, real order placement, conservative limits
  - Each profile loaded via `APP_ENV=training|backtest|paper|live` env var at startup

---

## Phase 1 — Bug Fixes (VPS)

**Fix A — ZMQ_BUS_KEY**
```
python -c "import secrets; print(secrets.token_hex(32))"
```
Append as `ZMQ_BUS_KEY=<value>` to `/root/trading-bot/.env`. Grep all ZMQ subscribers before restarting — not just the two known services.

**Fix A2 — Verify all secrets present in .env**

VPS is a clean slate — confirm every key is transferred before any phase that calls an external API:

| Key | Required by |
|-----|-------------|
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | Phase 1C, Phase 8, live trading |
| `BINGX_API_KEY` / `BINGX_API_SECRET` | Live trading (if active) |
| `HETZNER_API_TOKEN` | **Phase 8 CPU training** — orchestrator will crash without this |
| `VASTAI_API_KEY` | **Phase 8 GPU training** |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Zombie-server alerts (Phase 8) |

Command to verify (run on VPS): `grep -E "BINANCE|BINGX|HETZNER|VASTAI|TELEGRAM" /root/trading-bot/.env | cut -d= -f1`

**Binance API key scope (MANDATORY — create restricted key):**

The API key must be created with the minimum required scopes only. Never create an unrestricted key for the VPS:
- ✅ Enable: `Read` (account info, market data)
- ✅ Enable: `Spot & Margin Trading` (for spot orders)
- ✅ Enable: `Futures` (for USDT-M futures — only if futures strategy active)
- ❌ Disable: `Withdrawals` — if the key is compromised, attacker cannot drain funds
- ❌ Disable: `Universal Transfer`
- Restrict by IP: whitelist VPS IP `5.104.81.27` only

**DASHBOARD_API_KEY — hard abort on missing key:**

The dashboard must fail to start (not silently skip auth) if `DASHBOARD_API_KEY` is missing or empty from `.env`. A silent fallback to unauthenticated mode leaves the Flask dashboard exposed on port 5000 to the internet. Check at app startup:
```python
DASHBOARD_API_KEY = os.environ.get("DASHBOARD_API_KEY", "")
if not DASHBOARD_API_KEY:
    raise SystemExit("FATAL: DASHBOARD_API_KEY not set in .env — refusing to start without auth")
```

**Fix B — Agent heartbeat timestamps**
Write fresh `data/agent_status.json`: all `status: inactive`, `last_heartbeat: null`.

**Fix C — WebSocket timeouts**
`src/main.py:1437` `ping_timeout=20` → `ping_timeout=60`  
`src/main.py:1438` `close_timeout=10` → `close_timeout=15`  
`systemctl restart trading-bot`

---

## Phase 2 — Transfer 49 GB data/parquet to VPS

Stop `trading-realtime`, `trading-bot`, then:

> **WARNING — prevent Windows sleep before starting rsync.** If the laptop suspends on minute 15, the SSH pipe breaks and leaves partially-written Parquet files on VPS that are silently corrupt. Run before rsync:
> ```powershell
> powercfg /change standby-timeout-ac 0   # disable sleep on AC power
> ```
> Re-enable after transfer: `powercfg /change standby-timeout-ac 30`

```
rsync -av --progress -e "ssh -i ~/.ssh/trading_bot" \
  "D:/test 2/AI trading assistance/data/parquet/" \
  root@5.104.81.27:/root/trading-bot/data/parquet/
```
~28 min at 30 MB/s. Verify file count after transfer.

> **Note:** `-z` (compress) is intentionally omitted. Parquet files are already Snappy-compressed — adding `-z` burns CPU with zero transfer savings.

**Immediately after transfer — one-time parquet backup to Google Drive:**
```
rclone copy /root/trading-bot/data/parquet/ gdrive:trading-bot-backup/parquet-archive/
```
This runs once manually. The daily cron (Phase 5) permanently excludes `data/parquet/`. Without this step, VPS is a single point of failure for 49 GB.

---

## Phase 3 — Migrate CSV.gz → Parquet on VPS, Archive CSV.gz

**Schema validation BEFORE migration:**
```python
import pyarrow.parquet as pq
existing = pq.read_schema("data/parquet/BTCUSDT/1h/yyyymm=2025-01/data_0.parquet")
migrated = pq.read_schema("<output_path>")
assert existing == migrated
```
Force timestamp to **`datetime64[us]`** — this IS the PyArrow 13+ default. Do NOT force `ns`: PyArrow 13+ writes `us` by default, and forcing `ns` causes spurious schema diff failures on every new file. Binance timestamps are millisecond-precision — `us` retains full fidelity, no precision loss.

**Migration of existing 49 GB (mixed ns/us) — Option A (safest):**
```
1. Write all batches to data/parquet_us/  (original data/parquet/ untouched)
2. Verify fully: file count + schema assert us + fingerprint check
3. STOP BOT
4. rename data/parquet/    → data/parquet_backup/
5. rename data/parquet_us/ → data/parquet/
6. START BOT
7. Delete data/parquet_backup/ after first successful training run confirms correctness
```
Process in ~5 GB batches to allow per-batch rollback. `rename()` is atomic on Linux (single syscall) — safe with bot stopped.

- Run `python scripts/migrate_csv_to_parquet.py` after schema check passes
- Move all `data/raw/*.csv.gz` → `data/raw_archive/`
- **Touch every moved file to reset mtime to arrival time** (rsync preserves source mtime; without touch, cron deletes files immediately if source mtime > 7 days):
  ```
  find /root/trading-bot/data/raw_archive/ -name "*.csv.gz" -exec touch {} +
  ```
- Cron (explicit UTC):
  ```
  TZ=UTC
  0 4 * * * find /root/trading-bot/data/raw_archive/ -name "*.csv.gz" -mtime +7 -delete
  ```

**Same touch rule for all future CSV.gz deliveries via rsync** — add `touch` to the delivery script.

---

## Phase 4 — Code Change: CSV.gz as Temp-Only

New flow: download CSV.gz → convert to parquet → `mv` to `data/raw_archive/` → auto-deleted after 7 days.

`src/data_ingestion/ohlcv_parquet_loader.py`:
- Replace silent `return pd.DataFrame()` with:
  ```python
  raise FileNotFoundError(f"No parquet data for {symbol}/{timeframe}. Run backfill first.")
  ```
- Remove CSV.gz fallback from `load_funding` (~lines 83–89) separately
- Update all callers — explicit `FileNotFoundError` handling, no silent empty-DataFrame propagation
- Orchestrator must call `pyarrow.parquet.read_schema()` on load and compare against stored schema standard before feeding data to any model
- CSV migration script must default to `--skip` (not overwrite) when a Parquet file already exists for the same partition — silent overwrite of valid data is destructive

**Failed validation — quarantine, do not delete:**

If new parquet fails schema diff, fingerprint mismatch, or drift sanity check, move to `data/quarantine/` with timestamped name:
```
data/quarantine/
  20260519_143022_BTCUSDT_1h_schema_fail.parquet
  20260519_150401_ETHUSDT_4h_fingerprint_mismatch.parquet
```
Never auto-delete quarantined files — manual audit first. The data may be recoverable, or the failure may indicate a bug in the validator. Add to rclone exclusions (Phase 5): `--exclude "data/quarantine/**"`

---

## Phase 5 — rclone + Google Drive (Full Golden Copy, Daily)

Full golden copy: `/root/trading-bot/` → `gdrive:trading-bot-backup/`, once per day.

**Mandatory exclusions:**
```
--exclude "data/parquet/**"      ← 49 GB — backed up once in Phase 2, not daily
--exclude "data/raw_archive/**"
--exclude "logs/**"
--exclude "*.pyc"
--exclude "__pycache__/**"
--exclude "*.lock"
--exclude "**/*.tmp"
--exclude "data/cache/**"
--exclude "data/quarantine/**"
```

Setup:
1. `rclone config` on VPS — create "gdrive" remote (browser OAuth URL)
2. Cron:
   ```
   TZ=UTC
   0 3 * * * rclone sync /root/trading-bot/ gdrive:trading-bot-backup/ \
     --exclude "data/parquet/**" --exclude "data/raw_archive/**" \
     --exclude "logs/**" --exclude "*.pyc" --exclude "__pycache__/**" \
     --exclude "*.lock" --exclude "**/*.tmp" --exclude "data/cache/**" \
     --exclude "data/quarantine/**" \
     --log-file=/root/trading-bot/logs/rclone.log
   ```
3. Remove Windows Task Scheduler sync from local machine

---

## Phase 6 — Smoke-Test on Synthetic Data

Run BEFORE archiving real training state.

**Synthetic data requirements:**
- CPU models (`data/parquet_test/base_test/`, ~100 MB): AR(1)/GBM prices, volume correlated with volatility, ≥50k rows, exact `OHLCV_COLS` schema, `datetime64[ns]`, partition `yyyymm=YYYY-MM`
- GPU models (`data/parquet_test/tft_test/`, ~200 MB): same + sequence length ≥ lookback window (realism matters for TFT/OFT). **Minimum 200k rows** — TFT with lookback=168 needs ~200k/168 ≈ 1190 unique sequences for stable training; 50k gives only ~297 sequences which is insufficient
- Assert `df.dtypes` before saving

Exercise `training_rules.json` dispatch: ≥2 symbols × ≥2 timeframes.

**CPU test (Hetzner CCX33):** create → SSH → train → verify artifacts on VPS via rsync pull → verify server **DELETED** via Hetzner API.

**GPU test (Vast.ai RTX 4090):** bid → SSH → train → verify artifacts → verify instance **DESTROYED** via Vast.ai API.

Only proceed to Phase 7 after both pass.

---

## Phase 7 — Archive Current Training State

**Stop the bot.** Confirm no `running` entries in `dashboard_jobs.json` before touching `models/`.

Archive to `data/training_archive/YYYY-MM-DD/`:

| Source | Destination |
|--------|-------------|
| `models/` | `training_archive/YYYY-MM-DD/models/` |
| `data/training_runs_history.json` | `training_archive/YYYY-MM-DD/` |
| `data/training_status_report.json` | `training_archive/YYYY-MM-DD/` |
| `data/agent_status.json` | `training_archive/YYYY-MM-DD/` |
| `data/dashboard_jobs.json` | `training_archive/YYYY-MM-DD/` |
| `data/bake_off_cut_list.json` | `training_archive/YYYY-MM-DD/` |

After copying: clear originals. Archive never deleted.  
Never touch: `data/training_rules.json`, `data/parquet/`, `.env`.

---

## Phase 8 — Retrain All Models from Scratch

### Retrain Philosophy — Baseline Research Reset

This is NOT "retrain everything the same way as before." The historical trade log (2026-04-25 → 2026-05-17) showed that scalping destroyed 98% of P&L (−$1,090 of −$1,112), and zero ML-driven strategies generated a single live trade. The retrain is a clean research reset with a new deployment strategy.

**Rule 1 — Scalping stays in paper/experimental.**
Scalping_ML is trained (data already proves it can identify short-term moves), but it MUST NOT receive real capital until it demonstrates positive EV after fees/slippage in a dedicated paper canary of ≥500 trades and ≥30 days. Default after retrain: `Scalping_ML live: false`.

**Rule 2 — Optimize for after-fee expectancy, not accuracy.**
Training loss and WF accuracy are secondary metrics. The primary optimization target for every model is:
```
profit_factor = gross_profit / gross_loss  (target > 1.5)
after_fee_sharpe                           (target > 0.5 live)
max_drawdown                               (hard limit)
```
A model with WF Acc 52% and Sharpe 0.8 is better than WF Acc 55% and Sharpe 0.1.

**Rule 3 — Validate per regime, not just overall.**
For each model, run walk-forward separately for `bull / bear / chop / high_vol` regimes. A model that works only in bull markets is a bull-only strategy — deploy it only when regime = bull, not as an always-on signal.

**Rule 4 — Core retrain focus: 3 validated combos first.**
Do not try to activate all strategies simultaneously. The first live deployment uses only:
```
Combo A: Trend RF (1h/4h) + Meta-Labeler filter + Regime Router (TRENDING only) + GARCH sizing
Combo B: Base RF (1h)     + Meta-Labeler filter + GARCH sizing
Combo C: Volatility Breakout (rule) + Regime Router + Meta filter
```
Everything else (futures short, TFT, OFT, scalping, rule zoo) goes to paper/canary only until the interaction matrix (see Phase 9) produces Sharpe > 0 data.

**Rule 5 — Disable the rule strategy zoo on first live run.**
After retrain, disable in `strategy_config.json`: `ElliottWave_ML`, `Ichimoku_Cloud`, `MACD_Divergence`, `Keltner_Breakout`, `Supertrend`, `VWAP_Reversion`, `Donchian_Breakout`, `OU_Entry`, all Ensemble variants. Keep as baselines but `live: false`. Fewer active strategies = cleaner attribution data.

**Rule 6 — Meta-Labeler telemetry is mandatory from day 1.**
In the previous run, `model_confidence = NULL` on all 1,350 trades — Meta-Labeler was supposedly live but left no trace in any trade record. After retrain, `execution_audit.jsonl` must log on every order:
```json
{
  "model_confidence": 0.73,
  "meta_passed":      true,
  "predicted_prob":   0.73,
  "threshold":        0.54,
  "strategy":         "Trend_ML",
  "model":            "trend_model_4h",
  "timeframe":        "4h",
  "regime":           "trending",
  "expected_ev":      0.0042
}
```
Without this, the interaction matrix is impossible and the Master Allocator has no signal.

**Rule 7 — Test combos in sequence, not simultaneously.**
The interaction matrix is built by deploying one component at a time and measuring Sharpe at each stage. Do NOT activate all filters at once on live capital — you lose the ability to attribute P&L to individual components. Sequence:
```
Stage 1: Trend RF (1h/4h) alone               — first live
Stage 2: Trend + Meta-Labeler                 — after Stage 1 Sharpe > 0 (≥50 trades)
Stage 3: Trend + Regime Router                — after Stage 2 Sharpe > 0 (≥50 trades)
Stage 4: Trend + Meta + Regime                — after Stage 3 Sharpe > 0 (≥50 trades)
Stage 5: Trend + Meta + Regime + GARCH sizing — after Stage 4 Sharpe > 0 (≥50 trades)
```
Each stage transition requires a new canary period (≥50 trades, see Phase 9 canary thresholds). `execution_audit.jsonl` fields `meta_passed`, `regime_used`, `garch_used` make each stage attributable in the Interaction Matrix dashboard card.

**Rule 8 — First live deployment set (hard constraint).**
After retrain completes, only the following may receive real capital on day 1:
```
✅  Trend Momentum RF — 1h and 4h timeframes
✅  Meta-Labeler filter (only after telemetry verified in paper — meta_passed not NULL)
✅  Regime Router
✅  GARCH position sizing
❌  Scalping_ML — paper/canary only (Rule 1)
❌  Futures Short — paper/canary only until Trend combo proves positive live Sharpe
❌  TFT / OFT — paper/canary only
❌  ElliottWave, rule zoo — disabled (Rule 5)
```
If `meta_passed` is still NULL in the first 5 paper trades, halt and fix instrumentation before any live trade.

**Deployment priority after retrain:**
```
1. Telemetry instrumentation (verify BEFORE any live trade)
2. Meta-Labeler (highest AUC: 0.641 — most important filter)
3. Regime-aware Trend RF (structurally compatible with crypto)
4. Volatility Breakout (rule-based, fast to validate)
5. Portfolio Allocator (after interaction matrix proves ≥3 combos)
6. Futures Short, TFT, OFT, Scalping — only after proven in canary
```

### Disk Space Pre-Check (hard gate)

Before starting any training run, the orchestrator must verify free disk space:
```python
import shutil
free_gb = shutil.disk_usage('/root/trading-bot').free / (1024 ** 3)
if free_gb < 20:
    raise SystemError(f"Low disk space: {free_gb:.1f} GB free (need ≥20 GB). Aborting.")
```

DuckDB spills temp files to `data/cache/duckdb_temp` under memory pressure. If disk fills mid-run, the `.db` file can corrupt silently — partial transaction, wrong query results on the next read. Growth sources over time: `training_archive/`, `logs/`, `data/oos_signals/`, `data/cache/duckdb_temp/`. With 400 GB SSD and 49 GB parquet the 20 GB threshold is safe but concrete.

**DuckDB connection must set RAM limit and temp directory (MANDATORY):**
```python
con = duckdb.connect(database_path)
con.execute("SET memory_limit='18GB'")        # NOT PRAGMA — DuckDB uses SET, not PRAGMA
con.execute("SET temp_directory='/root/trading-bot/data/cache/duckdb_temp/'")
```
`PRAGMA` is SQLite syntax — DuckDB silently ignores unknown PRAGMAs, meaning the limit was never applied in earlier versions. Use `SET`.

**Singleton connection pattern (MANDATORY):** Do NOT open multiple DuckDB connections to the same file. With 3 concurrent consumers each claiming 18 GB on a 24 GB VPS, aggregate demand = 54 GB → OOM. Enforce one process-wide connection in `src/data/duckdb_pool.py` (lazy singleton), shared by all readers. Trainers on Hetzner/Vast use their own connections on different hosts — that is fine.

### Pre-Flight Checklist (run before real retrain)

```bash
python scripts/preflight_train.py
```

Script must verify and report PASS/FAIL for each:

| Check | What it verifies |
|-------|-----------------|
| Disk free ≥ 20 GB | Same as pre-check above |
| Parquet file count | Matches expected count from last verified sync |
| Schema valid | `pq.read_schema()` on sample files per symbol/TF |
| No running jobs | `dashboard_jobs.json` has no `status: running` entries |
| API keys present | All 5 keys in `.env` (Binance, BingX, Hetzner, Vast.ai, Telegram) |
| GDrive backup exists | `rclone lsd gdrive:trading-bot-backup/` returns entries |
| `training_rules.json` valid | JSON parses, all required fields present |
| OOS directory writable | `touch data/oos_signals/.write_test` succeeds |
| Hetzner credentials | `GET /v1/servers` returns 200 |
| Vast.ai credentials | `GET /api/v0/instances` returns 200 |

Any FAIL → abort with non-zero exit code. Orchestrator must check exit code before proceeding.

### Training Order

```
regime → base → trend → futures → scalping → meta → tft → oft
```

Regime (GMM, unsupervised) trains first — no dependencies. Its state features are **optional/injectable** in downstream models (not a hard dependency — avoids tightly coupled pipeline).

### OOS Signals for Meta (mandatory)

After each of base / trend / futures completes, save OOS predictions **with run_id**:
```
data/oos_signals/<run_id>/
  base.parquet
  trend.parquet
  futures.parquet
```
`run_id` = UTC timestamp of training run start (e.g. `2026-05-20T14:00:00`). Each OOS file carries `run_id` in its path AND as a column.

Before meta training begins: **hard check** — all three files must exist under the **same `run_id`**. If any missing OR if `run_id` values differ (stale file from previous partial run) → stop, do not proceed.

**Why run_id matters:** if `base` completes, `trend` crashes, and retrain restarts from `trend`, meta would silently consume a fresh `base` OOS + stale `trend` OOS from the prior run. run_id isolation prevents this.

### Checkpoint Protocol (Pull-based rsync)

After each model completes on Hetzner/Vast.ai:
1. Model artifacts saved to training server `models/`
2. **VPS pulls from training server** (not push):
   ```
   rsync -avz -e "ssh -i ~/.ssh/trading_bot" \
     root@<hetzner_ip>:/root/models/ /root/trading-bot/models/
   ```
   VPS initiates the pull — no private keys placed on the temporary Hetzner server.
3. Append entry to `training_runs_history.json` on VPS
4. Verify file present on VPS before starting next model

### Env Manifest

Capture at training start, save as `env_manifest.json` alongside model artifacts:
```python
import importlib.metadata, torch, platform
manifest = {
    "python": platform.python_version(),
    "lightgbm": importlib.metadata.version("lightgbm"),
    "scikit-learn": importlib.metadata.version("scikit-learn"),  # NOT "sklearn"
    "torch": torch.__version__,
    "cuda": torch.version.cuda,   # NOT nvcc — may not be in PATH on Vast.ai
    "pyarrow": importlib.metadata.version("pyarrow"),
    "numpy": importlib.metadata.version("numpy"),
}
```
Add `capture_env_manifest()` to `src/utils/env_manifest.py`, call from training orchestrator pre-run.

### Infrastructure

**CPU** (regime, base, trend, futures, scalping, meta) — Hetzner CCX33: ~8.5h, ~€0.85. Server **DELETED** at end AND in exception handler.

**GPU** (tft, oft) — Vast.ai RTX 4090: ~4-6h, ~$1.50-2.20. Instance **DESTROYED** at end AND in exception handler.

**Zombie-server protection (MANDATORY):**

Deletion/destruction must use an exponential-backoff retry loop. If the Hetzner or Vast.ai API returns 500 / timeout, the server stays running and burns budget silently.

```python
import time
for attempt in range(3):
    try:
        api.delete_server(server_id)   # or api.destroy_instance(instance_id)
        break
    except APIError:
        time.sleep(30 * (2 ** attempt))  # 30s → 60s → 120s
else:
    send_telegram_alert(
        f"CRITICAL: Failed to destroy {server_id} after 3 attempts. "
        f"Manual action required — check billing dashboard immediately."
    )
```

Apply this wrapper to **both** the normal teardown AND the exception handler.

---

## Phase 9 — Champion/Challenger Baseline System

### Storage Structure

```
data/baselines/
  v1_2026-05-19/
    metadata.json          (date, git hash, dataset_hash, notes)
    metrics.json           (all metrics: model × TF × symbol)
    data_manifest.json     (per-symbol: train_start, train_end, n_bars)
    feature_schema.json    (columns, dtypes, frac-diff d, feature_pipeline_version)
    env_manifest.json      (python/lightgbm/cuda/torch versions)
    data_fingerprint_cache.json  (mtime+size fast-path cache)
    model_snapshots/
  current.json             → { "active_baseline": "v1_2026-05-19" }
```

### Dataset Fingerprint

Hash **logical data**, not file binary (binary sha256 changes on PyArrow re-encoding of identical data):
```python
fingerprint = {
    "schema_hash":   sha256(column_names + dtypes),
    "row_count":     total_rows,
    "timestamp_min": earliest_bar.isoformat(),
    "timestamp_max": latest_bar.isoformat(),
    "per_column":    {col: sha256(col_values) for col in df.columns},
    "file_count":    n_parquet_files,
    "symbol_tf_coverage": {sym: [tfs] for sym in symbols},
}
```

**Streaming hash (MANDATORY — 48 GB does NOT fit in memory):**
```python
import pyarrow.parquet as pq, hashlib
h = hashlib.sha256()
pf = pq.ParquetFile(path)
for batch in pf.iter_batches(batch_size=50_000):
    for col_name in batch.schema.names:
        buf = batch.column(col_name).buffers()[1]  # zero-copy Arrow buffer
        if buf:
            h.update(buf)
per_column_hash = h.hexdigest()
```
Never load the full DataFrame — use `iter_batches()` with incremental SHA256.

Fast-path: compare `{path: (mtime, size)}` first. Full logical hash only when any file shows mtime or size change.

**Cache writes must be atomic:**
```python
import os, json, tempfile
tmp = path + ".tmp"
with open(tmp, 'w') as f:
    json.dump(cache, f)
os.replace(tmp, path)   # atomic on Linux — prevents corruption on crash
```
Cache in `data_fingerprint_cache.json`.

Comparison agent rejects baseline vs challenger with mismatched `dataset_hash`.

### Feature Pipeline Version

`feature_schema.json` must include `feature_pipeline_version` (e.g. `"v2.1"`). Increment on any algorithm change, not just column renames. RSI_v1 and RSI_v2 may share a column name but differ in computation.

### Metrics (per model × TF × symbol)

**Tier 1 — Financial (blocking):** `Sharpe`, `EV`, `Calmar`, `max_drawdown`, per-symbol Sharpe floor (no symbol drops > -10%)

**Tier 2 — ML (informational):** `PR-AUC`, `Precision(TP)`, `Recall(TP)`, `OOS log-loss` (meta: log-loss only)

**ML Integrity:** `avg_uniqueness` (detects label overlap / leakage)

### Rebaseline Decision Logic

```
TIER 1 — must not regress:
  ✅ Sharpe AND EV both not worse (both fall → REJECT, no override)
  ✅ Calmar not worse; max_drawdown not worse
  ✅ No single symbol Sharpe drops > -10%

TIER 2 — statistically significant improvement required:
  ✅ At least one Tier 1 metric improves with statistical significance
     Method: Stationary Block Bootstrap (arch.bootstrap.StationaryBootstrap)
             Politis-Romano automatic block length
             N=1000 resamples; CI lower bound > 0
  ✅ Improvement must be on a financial metric (Sharpe, EV, or Calmar)
     win_rate alone does NOT qualify

TIER 3 — display only, no gate:
  PR-AUC, Precision, Recall — shown in table, cannot approve or block
```

### Trading-Cost Stress Test (required before canary)

Run backtest with: `fees × 1.5`, `slippage × 2`, `latency spikes`. P&L negative → fragile → do not promote.

### Canary Deployment (model-type-specific thresholds)

5% capital allocation alongside current champion. Gate criteria:

| Model type | Min calendar days | Min trades | Notes |
|------------|-------------------|------------|-------|
| Scalping | 14 | 500 | Either condition met: `min(14 days, 500 trades)`. 500 gives ~2.2σ; but 14 days of data is sufficient if strategy trades slower than expected |
| Base / Meta | 14 | 100 | Mid-frequency |
| Trend | 30 | 50 | **OR logic:** promote when `(50 trades OR 30 days) AND actual_trades >= 10`. ~1.4 trades/day at $500 bankroll; 50 trades in 14 days impossible by design |
| Futures | 30 | 50 | **OR logic:** same as Trend. `(50 trades OR 30 days) AND actual_trades >= 10` |
| Regime (GMM) | 21 | N/A | Gate on ≥3 full regime-state cycle changes |
| TFT / OFT | 14 | 30 | Medium-to-low frequency |

Promotion criteria (all must pass):
- `|live_sharpe - backtest_sharpe| / backtest_sharpe < 0.30`
- No single-day drawdown exceeding stress-tested max
- **Challenger live Sharpe ≥ Champion live Sharpe** (not just vs backtest)

**MIN_NOTIONAL protection:** if 5% capital < exchange minimum order size ($5–$10 depending on pair), canary must trade the minimum allowed lot instead of 5%. At $500 bankroll this is not yet binding ($25 > $10), but if balance drops to $100, `5% = $5` and some pairs reject the order. Check `MIN_NOTIONAL` from exchange filter at order time; fall back to minimum lot silently (do not skip the canary).

### Correlation-Aware Portfolio Gate

Before executing signals:
- Pairwise Pearson correlation > 0.7 (30-day rolling) defines a correlated cluster
- `exposure_cap`: max 20% total capital in any one correlated cluster
- `correlation_penalty`: reduce position size when correlation > threshold
- Roadmap: migrate to rolling correlation + beta-to-BTC weighting

**Component:** `src/risk/correlation_gate.py` (new file — does not exist yet)

### Rollback Playbook

If challenger causes live degradation, rollback to last known good baseline in under 2 minutes:
```bash
python -m src.governance.baseline_manager rollback --to v1_2026-05-19
systemctl restart trading-bot
```
`rollback`: restores `current.json` pointer, swaps model artifact symlinks. Verify correct version loaded: `curl localhost:5000/api/strategy/full | jq '.baseline_version'`

### Order Execution Audit Log

For every filled order, append a record to `data/execution_audit.jsonl`:
```json
{
  "ts":                    "2026-05-19T14:30:22Z",
  "signal_id":             "uuid",
  "model_version":         "v1_2026-05-19",
  "feature_snapshot_hash": "sha256-of-features-at-decision-time",
  "strategy":              "Trend_ML",
  "model":                 "trend_model_4h",
  "timeframe":             "4h",
  "mode":                  "live",
  "predicted_prob":        0.73,
  "model_confidence":      0.73,
  "threshold":             0.60,
  "expected_ev":           0.0042,
  "meta_passed":           true,
  "regime_used":           "trending",
  "garch_used":            true,
  "actual_fill_price":     65432.10,
  "slippage":              0.00018,
  "latency_ms":            47,
  "pnl_usdt":              null
}
```
`pnl_usdt` is `null` at entry; write a second record with `"event": "close"` and the same `signal_id` when the position closes. Without `feature_snapshot_hash` + `model_version` on every trade it is impossible to distinguish model error from execution error in post-mortem analysis. Without `meta_passed` + `regime_used` + `garch_used` the Interaction Matrix card cannot attribute P&L by combo.

**Write discipline for `execution_audit.jsonl` (WAL pattern):**
- Open in **append-only** mode (`open(path, 'a')`) — never overwrite
- Call `f.flush(); os.fsync(f.fileno())` after each write — guarantees durability on VPS crash
- Rotate when file exceeds 100 MB: rename to `execution_audit_YYYYMMDD.jsonl`, open fresh file
- A partial last line (VPS crash mid-write) is detectable and skippable at read time — JSONL ensures all prior records are intact

### Interaction Matrix Dashboard Card

Purpose: attribute live P&L to each filter combo without manual log grep. Appears in the Analytics tab once `execution_audit.jsonl` accumulates records.

**Backend endpoint — `GET /api/analytics/interaction_matrix`** (new, `src/dashboard/app.py`):
- Reads `data/execution_audit.jsonl` line-by-line; skips malformed last line (crash-safe)
- Returns `{"ok": true, "has_data": false, "rows": []}` when file missing or zero records
- Accepts `?mode=live|paper|all` query param (default: `all`)
- Groups **close** records by `(meta_passed, regime_used != null, garch_used)` booleans
- Per group computes: `n_trades`, `win_rate`, `avg_pnl_usdt`, `trade_sharpe = mean(pnl_usdt) / std(pnl_usdt)` (null when n < 2)

**Five canonical combos:**

| # | Label | meta_passed | regime_used | garch_used |
|---|-------|-------------|-------------|------------|
| 1 | Trend only | false | false | false |
| 2 | Trend + Meta | true | false | false |
| 3 | Trend + Regime | false | true | false |
| 4 | Trend + Meta + Regime | true | true | false |
| 5 | Trend + Meta + Regime + GARCH | true | true | true |

**Frontend card** (`src/dashboard/templates/index.html`, Analytics tab):
- Card id: `card-analytics-interaction-matrix`
- Auto-loads when Analytics tab clicked (alongside `anLoad()` — add call in tab click handler)
- JS function: `loadInteractionMatrix()`
- Columns: Combo | Trades | Win% | Avg P&L (USDT) | Trade Sharpe
- Color: Sharpe > 0.5 → green, 0–0.5 → yellow, < 0 → red, null → gray `—`
- Mode toggle: Live / Paper / All
- Placeholder state: *"Waiting for first live trades with `meta_passed` / `regime_used` / `garch_used` logged in `execution_audit.jsonl`"*

### Components to Build / Extend

| Component | Path | Note |
|-----------|------|------|
| Baseline manager | `src/governance/baseline_manager.py` | New — single file; `PromotionPolicy` + `rollback()` as class methods |
| Dataset fingerprinter | `src/utils/dataset_fingerprint.py` | New — logical hash + mtime cache |
| Env manifest | `src/utils/env_manifest.py` | New — `capture_env_manifest()` |
| Comparison agent | `src/agents/model_comparison_agent.py` | New — supervised loop |
| Stress tester | `src/utils/trading_cost_stress_test.py` | New |
| Correlation gate | `src/risk/correlation_gate.py` | New |
| Dashboard card | Analytics tab | New card — table + canary status + promote button |
| Promote endpoint | **Reuse** `POST /api/analytics/baseline` (`app.py:6418`) | Do NOT duplicate |
| Interaction Matrix endpoint | `GET /api/analytics/interaction_matrix` (`app.py`) | New — reads `execution_audit.jsonl`, groups by combo |
| Interaction Matrix card | Analytics tab, `card-analytics-interaction-matrix` (`index.html`) | New card — auto-loads on tab click, mode toggle |

---

## Phase 10 — Risk Controls: Kill-Switch, Liquidity Filter, Outage Mode

### Kill-Switch (extend existing `src/risk/kill_switch.py`)

Already implemented in `KillSwitchConfig` (lines 46–52):
- `drawdown_pct_threshold: 0.08` (8% peak-to-trough) ← already exists
- `latency_p99_ms_threshold: 500.0` (ms) ← already exists
- `daily_loss_R_multiple` proxy for losses ← already exists

**What to ADD:**
- `slippage_pct_threshold` field to `KillSwitchConfig` — dedicated slippage trigger
- Corresponding check in `_iter_triggers` after line 209
- Caller supplies `slippage_pct` key in metrics dict

### Exchange Outage Mode (extend `src/main.py`)

Already implemented: startup connectivity check with exponential backoff (5s→80s), WebSocket reconnect with backoff (capped 60s).

**What is MISSING — HIGH priority:**
- No runtime `ws_connected` boolean that the order-placement path consults
- During WebSocket reconnect window, trade loop continues uninhibited — can open positions against stale/absent prices

**What to ADD:**
- Boolean flag `ws_connected` set `False` on WebSocket disconnect (~line 1479), `True` on reconnect
- Pre-trade gate reads flag: if `ws_connected == False` → block new-position orders; allow position-closing/risk-reduction orders through
- **State Reconciliation on every reconnect** (before setting `ws_connected = True`):

  During a 60-second disconnect, the exchange may have triggered a TP/SL, filled an order, or price may have gapped. Blind reconnect means the bot continues trading against stale local state.

  ```python
  # On reconnect — BEFORE ws_connected = True:
  futures_positions = client.futures_position_information()   # GET /fapi/v2/positionRisk
  spot_open_orders  = client.get_open_orders()                # GET /api/v3/openOrders
  spot_account      = client.get_account()                    # GET /api/v3/account

  # Reconcile diffs:
  #   Position closed on exchange but open locally  → mark closed, book realized P&L
  #   Order filled on exchange but pending locally  → update order status, adjust inventory
  #   Balance mismatch                              → overwrite local cache with exchange value
  # THEN: ws_connected = True
  ```

### PreTradeGate — Unified Check + Two-Lock Pattern (MANDATORY)

All safety conditions (kill-switch, ws_connected, warmup_complete, SAFE_MODE, NaN/Inf guard) must be checked in a **single unified** `PreTradeGate.check(ctx)` call — NOT scattered across independent if-statements at different call sites (classic missed-check-path bug).

**Two separate locks are required:**

```python
trading_lock = threading.Lock()   # serializes order placement only
flag_lock    = threading.Lock()   # serializes flag mutations only

# Order placement path:
with trading_lock:
    gate = PreTradeGate.check(ctx)   # reads flags under flag_lock internally
    if not gate.allow:
        log(gate.reason); return
    exchange.create_order(...)

# Flag mutation path (WebSocket thread, kill-switch, SAFE_MODE toggle):
with flag_lock:
    ws_connected = False   # or SAFE_MODE = "read_only", etc.
```

**Why two locks (not one):** A single `trading_lock` used for both order placement AND WebSocket flag writes causes deadlock. The WebSocket I/O thread holds the lock while waiting for a network response; the trading loop is blocked waiting for the lock. With separate locks, `trading_lock` only wraps the critical placement section and `flag_lock` only wraps flag writes — no cross-thread wait.

**TOCTOU protection:** `PreTradeGate.check()` reads all flags atomically under `flag_lock` internally. The caller never reads flags separately before calling `check()` — that pattern creates a window where a flag can change between the read and the order.

### Position Sizing Gate (new)

Hard limits enforced at order-generation time, before any signal reaches the exchange:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_risk_per_trade` | 0.25–0.5% of bankroll | At $500: $1.25–$2.50 per trade |
| `max_daily_risk` | 2% of bankroll | At $500: $10/day max loss |
| `max_open_positions` | N (configure per strategy) | Prevents correlated over-exposure |

If a signal would exceed any limit → **size down to the limit, do not skip the trade**. Log the adjustment. If sizing down would result in an order below MIN_NOTIONAL → skip the trade and log reason.

**Component:** add `PositionSizingGate` to `src/risk/position_sizing.py` (new file or extend existing).

### Read-Only Safe Mode

Add `SAFE_MODE` operational flag. When `SAFE_MODE=read_only`:
- Bot receives market data normally
- Computes signals and features normally
- Writes to `execution_audit.jsonl` as paper trades (tagged `"mode": "paper"`)
- **Does NOT send any orders to the exchange**

Trigger conditions (set automatically):
- After any new model deploy (stay in `read_only` for first 30 minutes)
- After reconnect with anomalous state diff (reconciliation found closed positions or balance mismatch)
- After drift alert from Phase 11
- After kill-switch reset (manual operator re-enable → starts in `read_only`, operator explicitly promotes to `live`)

**Component:** add `SAFE_MODE` env var + gate in order placement path in `src/main.py`.

### Model Warmup Requirement

After bot restart or model reload, block order placement until:

| Indicator | Minimum bars required |
|-----------|----------------------|
| RSI (14) | 14 bars |
| EMA (any period N) | 3×N bars (to stabilize) |
| Feature rolling windows | max lookback across all active features |
| Regime GMM state | 1 full prediction cycle |

**Startup data loading sequence (7 steps — mandatory order):**
```
1. Load historical closed bars from Parquet
2. Drop the partial last bar (current incomplete candle)
3. Assert all timestamps are UTC (no naive datetimes)
4. Assert Parquet recency — youngest bar not older than max lookback window
5. Fetch gap bars via REST to bring history up to the current minute
5.5. Assert contiguity — no gap AND no overlap between Parquet tail and REST bars
     (gap = bars missing; overlap = REST bar duplicates a Parquet bar → double-counting)
6. Enable WebSocket for live candle updates
7. Set warmup_complete = True after max_required_bars accumulated
```

Only after step 7 does `PreTradeGate.check()` allow order placement. Steps 1–5.5 run synchronously at startup before the WebSocket connects.

Implementation: maintain `warmup_complete: bool` flag set `False` on start, `True` after `max_required_bars` loaded. Pre-trade gate checks this flag alongside `ws_connected`.

NaN/Inf in features during warmup → order blocked silently (not an error — expected during initialization).

### NaN / Inf Guardrail

Before every model inference call:
```python
assert np.isfinite(features).all(), (
    f"Non-finite values in feature vector: "
    f"{features.columns[~np.isfinite(features).all()].tolist()}"
)
```
LightGBM does NOT raise on NaN input — it silently applies its internal NaN handler which can produce unexpected predictions. A single bad funding rate value, divide-by-zero in a feature, or corrupt parquet partition propagates invisibly to a live trade signal.

Log and skip the signal on assertion failure (do not crash the bot).

### Clock Drift Monitoring

Binance signed requests use server timestamp. If VPS clock drifts vs exchange:
- Requests rejected with `INVALID_TIMESTAMP` (margin: default 5000ms recvWindow)
- Candle alignment shifts → wrong bar attribution
- Funding timestamps misalign → wrong funding P&L accounting

Check at bot startup and every 5 minutes:
```python
exchange_time_ms = client.get_server_time()["serverTime"]
local_time_ms    = int(time.time() * 1000)
drift_ms         = abs(local_time_ms - exchange_time_ms)
if drift_ms > 500:
    alert(f"Clock drift {drift_ms}ms — sync NTP immediately")
```
Fix: `chronyc makestep` (Ubuntu 24.04 — `ntpdate` is deprecated and not installed by default; `chronyc makestep` forces an immediate step correction via the running chronyd daemon without risking a conflicting daemon). Add to bot startup as a pre-flight check.

### Exchange Precision Normalization

Before every order placement, normalize quantity and price to exchange-mandated precision:
```python
# Preferred: CCXT built-in precision (handles edge cases automatically)
qty   = exchange.amount_to_precision(symbol, raw_qty)
price = exchange.price_to_precision(symbol, raw_price)

# Alternative if using raw Binance API — use Decimal to avoid float binary hazard:
from decimal import Decimal, ROUND_DOWN
step  = Decimal(str(step_size))
qty   = float((Decimal(str(raw_qty)) / step).quantize(Decimal('1'), rounding=ROUND_DOWN) * step)
# WARNING: math.floor(0.123 / 0.001) = 122.999... = 122 due to float imprecision. Never use math.floor for this.
```
Without this, the exchange silently rejects orders with precision errors or returns `LOT_SIZE` / `PRICE_FILTER` error codes that are easy to mistake for network failures. Different pairs have different `step_size` and `tick_size` — fetch from `GET /api/v3/exchangeInfo` and cache (24h TTL).

### Funding-Rate Blackout Windows (Futures)

Within 2 minutes before each funding settlement timestamp, futures liquidity deteriorates sharply:

| UTC time | Funding event |
|----------|---------------|
| 07:58–08:00 | 08:00 funding settlement |
| 15:58–16:00 | 16:00 funding settlement |
| 23:58–00:00 | 00:00 funding settlement |

During blackout windows:
- **Do not open new futures positions**
- Widen slippage assumption by 2× for any fills that occur (if position already open)
- Optionally reduce leverage by 50% on existing positions

**Implementation:** check `datetime.now(timezone.utc)` at order time, skip entry if within 2-minute blackout. (`datetime.utcnow()` is deprecated in Python 3.12+ and returns a naive datetime — use `datetime.now(timezone.utc)` for timezone-aware UTC.)

### Minimum Liquidity Filter (new, dynamic check)

Check at order-generation time, 60-second TTL cache using:
- `GET /api/v3/ticker/24hr` for volume
- `GET /api/v3/ticker/bookTicker` for spread and depth

| Metric | Spot minimum | Futures minimum |
|--------|-------------|-----------------|
| 24h volume (USD) | $50M | $100M |
| Bid-ask spread | ≤ 0.05% (5 bps) | ≤ 0.03% |
| Book depth at 0.1% from mid | ≥ $50K | ≥ $50K |

All 20 symbols currently pass the volume threshold. Binding constraint is spread: **SHIB, HBAR, ICP, SUI** occasionally widen to 0.08–0.15% during UTC 02:00–06:00. Dynamic check catches this; static filter at session start would miss it.

Filter is **dynamic** — checked at order-generation time, not once at startup.

---

## Phase 11 — Online Drift Monitoring (extend existing `src/risk/drift_monitor.py`)

**Extend existing file** — do NOT create a new `drift_monitor_agent.py` (would duplicate logic with existing `src/risk/drift_monitor.py`, `drift_psi.py`, `drift_baseline.py`).

| Metric | Threshold | Notes |
|--------|-----------|-------|
| PSI — price/return features | 0.20 | Near-normal distribution |
| PSI — volume/liquidation/funding | 0.25–0.35 | Fat-tailed; standard 0.2 gives false positives |
| KL Divergence | TBD per feature | Daily aggregated report only |
| Feature drift (mean/std) | > 3σ | Per-feature z-score |
| Volatility drift | > 2σ rolling 30d | Market regime change signal |

**Minimum sample size:** 500 bars per window before computing PSI (prevents unstable estimates).

**Two-level check:**
1. **Hourly:** PSI vs rolling 24-hour window (NOT static training distribution) — prevents false positives from intraday seasonality (Asia/EU/US session opens, 8h funding rate cycles)
2. **Daily:** KL divergence vs training distribution — deeper drift vs original training data

Flow: drift exceeds threshold → `data/drift_report.json` updated → dashboard flag "Drift Detected". Operator decides: ignore (seasonal), investigate, or trigger retrain. **Never automatic.**

Reference distributions saved at training time: `data/baselines/vN/train_distributions/`

---

## Execution Gates

```
Phase 0   (Housekeeping + branch + VPS hardening)      — one GO, ~45 min
Phase 1   (Bug fixes A/B/C)                            — ~5 min
Phase 2   (rsync parquet + one-time GDrive backup)     — ~30 min + ~60 min backup
Phase 3   (CSV.gz migration + touch + cron)            — ~20 min
Phase 4   (Code changes: FileNotFoundError, schema)    — ~30 min + tests
Phase 5   (rclone daily sync setup)                    — ~10 min + browser
──────────────────────────────────────────────────────────────────────────
Phase 6   (Smoke-test: synthetic data + CPU + GPU)     — separate GO
──────────────────────────────────────────────────────────────────────────
Phase 7   (Archive training state, bot stopped)        — separate GO after Phase 6
Phase 8   (Retrain: regime first, OOS signals, pull rsync, checkpoints)  — separate GO
Phase 9   (Baseline system + canary + stress test)     — parallel with 8 or after
Phase 10  (Kill-switch slippage + outage mode + liquidity filter)  — parallel
Phase 11  (Extend drift monitor)                       — after Phase 9
──────────────────────────────────────────────────────────────────────────
PR review + merge to main                              — after all phases + 0 test failures
```

---

## Infrastructure Billing Rules (MANDATORY)

- **Hetzner**: always **DELETE** server — never power off. Delete in exception handler. Confirm via API. Log server ID at creation.
- **Vast.ai**: always **DESTROY** instance — never stop. Destroy in exception handler. Confirm via API.
