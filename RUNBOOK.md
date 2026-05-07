# RUNBOOK — AI Trading Assistance

One-page operator runbook. Phase F deliverable. Everything below assumes
you are at `D:\test 2\AI trading assistance` with `venv\` activated and
the dashboard reachable at <http://127.0.0.1:5000>.

---

## 1. Daily go/no-go check

Before letting the bot trade real money, verify these five items:

| # | Check | How |
|---|---|---|
| 1 | **Bot loop is alive** | Dashboard banner shows no `bot died Xs ago` critical |
| 2 | **No data-staleness errors** | Dashboard banner shows no `data feed inconsistency` |
| 3 | **Trade mode is what you expect** | Overview tab → Live Trading card shows the right pill |
| 4 | **Models retrained recently** | ML/Strategy tab → Model Training table → no rows in `STALE` (>7d) |
| 5 | **Circuit breakers respond** | Run the drill: `python -m src.engine.breaker_drill` (4/4 pass) |

If any item fails, **switch trade mode to PAPER** (Overview → Trade Mode → 📄 PAPER) until fixed.

---

## 2. Three trading modes (PR 6)

| Mode | Real orders | Real money | When to use |
|---|---|---|---|
| **PAPER** | No — booked to internal virtual balance only | No | Default. Always-on shadow trading. |
| **TESTNET** | Yes, to Binance testnet | No (fake money) | Verify exchange wiring without risk |
| **MAINNET** | Yes, to Binance mainnet | **YES** | Production only. Requires explicit confirm. |

Switch via the Overview tab. PAPER ↔ TESTNET is one click; MAINNET requires a confirm dialog. Switch back to PAPER any time something looks off.

---

## 3. Common operations

### Start everything from scratch
```powershell
.\restart_all.ps1
```
Brings up monitor (5001), dashboard (5000), bot, FastAPI, orderbook collector, debug supervisor. PIDs saved to `data/process_ids.json`.

### Stop everything
```powershell
.\stop_all.ps1
```

### Manual retrain
```powershell
python -m src.engine.train_all_models
```
Or via dashboard: **ML/Strategy tab → Pipeline Orchestrator → ▶ Run Train + Multi-TF Backtest**.

### Auto-retrain (with regression guard)
```powershell
python -m src.engine.auto_retrain --tolerance 0.05 --rollback
```
Snapshots WF Sharpe, runs full pipeline, restores meta files if WF Sharpe drops > 5%. Schedule daily via Windows Task Scheduler.

### Long-horizon backtest (5y robustness)
```powershell
python -m src.engine.long_horizon_backtest --horizon long
```
Or dashboard: **POST /api/backtest/long_horizon {"horizon": "long"}**. Auto-picks 1h/4h/1d/1w (skips 5m at 5y to avoid 250M-row blowup).

### Force-test circuit breakers
```powershell
python -m src.engine.breaker_drill
```
Confirms max-DD / API-latency / stale-feed triggers fire correctly. Should show `4/4 pass`.

### Audit trade ↔ signal ↔ model trail
```powershell
python -m src.engine.audit_trail --max-trades 100
```
Reports orphan orders (no matching signal), untraced signals (no model), missing artifacts (signal references a deleted model).

### Switch trade mode from CLI (no dashboard)
```powershell
# trade_mode field in data/control.json. Edit directly OR:
$body = '{"mode":"paper"}'
curl.exe -X POST -H "Content-Type: application/json" -d $body http://127.0.0.1:5000/api/control/trade_mode
```

---

## 4. Where things live

| Data | Path | Notes |
|---|---|---|
| 1s OHLCV archives | `data/raw/historical/<sym>_spot_1s.csv.gz` | Source of truth (~48 GB) |
| Resampled multi-TF | `data/raw/<sym>_<tf>.csv.gz` | Auto-built by `resample_ohlcv` |
| Parquet store | `data/db/` | DuckDB-backed; replaces QuestDB |
| News partitions | `data/parquet/_NEWS/news/yyyymm=*/` | GDELT + Reddit + CryptoCompare |
| Model artifacts | `models/<key>_<tf>_*.{joblib,json}` | Per-TF; legacy filenames also written |
| Live state | `data/state.json` | Latest signal per symbol |
| Live trades | `data/trades.json` | Append-only trade log |
| Pipeline status | `data/pipeline_status.json` | Orchestrator's heartbeat |
| Auto-retrain status | `data/auto_retrain_status.json` | Last cycle's verdict |
| TF pinning | `data/strategy_tf_pinning.json` | Per-strategy TF assignments (auto + manual) |
| Control flags | `data/control.json` | running, trade_mode, kill_switch |
| Logs | `logs/*.log` | Per-component; last 100 lines visible in Monitor tab |

---

## 5. Incident response

### Bot is hot-looping with errors
1. **Check the banner** — top of dashboard shows critical issues with counts.
2. **Switch to PAPER** if real-money trading is currently on.
3. **Find the error**:
   ```powershell
   Select-String -Path logs\bot.log -Pattern "ERROR" | Select-Object -Last 20
   ```
4. **Restart bot only**:
   ```powershell
   .\stop_all.ps1
   .\restart_all.ps1
   ```

### Resampled CSV has gap rows again (won't load)
PR 7's `_write_csv_gz` drops NaN-OHLC rows before write — but if a hand-modified file shows the bug, scrub it:
```powershell
python -m src.utils.scrub_resampled_csvs
```
Idempotent. Fast (~1 min for all symbols).

### Dashboard cards stuck on "Loading…"
PR 10 fixed silent failure — if you still see this, the underlying endpoint is genuinely down. The chip should show `offline (HTTP n)` or `unreachable` after a few seconds. Check `logs/dashboard.log` for the failing endpoint.

### Pipeline orchestrator dead but status says "running"
PR 7's `api_pipeline_status` flips to `error` automatically when the subprocess is gone. Re-run from the dashboard **▶ Run** button.

### Mainnet trade fired by mistake
1. **PAPER mode immediately** (Overview → 📄 PAPER).
2. **Manually flatten** any open positions on Binance.
3. Run audit:
   ```powershell
   python -m src.engine.audit_trail
   ```
4. File the incident in `data/incidents/<date>.md`.

---

## 6. Pre-deploy checklist (quarterly)

Run before any major code merge to mainnet:

- [ ] `python tests/test_dashboard.py --offline` → 0 NEW failures (the `launch_bot.ps1` pre-existing fail can stay)
- [ ] `python -m src.engine.breaker_drill` → 4/4 pass
- [ ] `python -m src.engine.audit_trail` → 0 orphan_orders + 0 missing_artifacts
- [ ] `python -m src.engine.auto_retrain --tolerance 0.05 --rollback` → verdict `accepted`
- [ ] 7-day PAPER run with no critical banner issues
- [ ] Open positions = 0 before mode switch from PAPER → MAINNET
- [ ] `data/strategy_tf_pinning.json` reflects latest backtest's `auto` map

---

## 7. Contact / further reading

- `PLAN_2026_05_07.md` — current session plan
- `INSTITUTIONAL_UPGRADE_PLAN.md` — original 18-point architecture
- `updated_architecture_plan_en.md` — L1–L5 reference
- `APP_DOCUMENTATION.md` — code-walk for every module
- `CLAUDE.md` — agent rules (approval gate, restart rule, etc.)
