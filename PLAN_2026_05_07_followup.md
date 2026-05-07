# PLAN — 2026-05-07 follow-up batch (post PR-17)

User reported 8 issues after the first 11 PRs landed. Committed to "Option A"
for the data backfill (full 1s top-up + re-resample, no shortcuts).

---

## P0 — dashboard fundamentally broken

The fact that ALL these issues are happening simultaneously points to a single
root cause:
- Refresh buttons across the dashboard do nothing
- Monitor tab cards stay on "Loading…" forever
- Simulator tab is stuck (state IDLE, none of the buttons advance state)
- Pipeline orchestrator pill says "error / running since 23:42 (yesterday)"
- Trade-mode switch reportedly still not switching

Hypothesis: a JS error in one of the recent template edits is breaking the
global event-handler chain. When that happens, **every** onclick="..." in
the page silently fails because the function is undefined or threw before
it was assigned.

Action:
1. Open dashboard log + browser-console errors via curl checks
2. Find the broken JS — most likely my PR-9 / PR-10 / PR-11 template edits
3. Fix it; verify in a fresh hard-reload

## P0 — training row "Train" buttons don't work

User clicks ▶ Train on a row, gets a confirm dialog, accepts — and nothing
happens. Want:
- Train button transitions to Stop when training is in flight
- Status column reflects RUNNING
- TF picker per row, defaulting to the row's best-TF (from
  data/strategy_tf_pinning.json that PR 12 already maintains)

Need to look at what the click currently does — likely the confirm() succeeds
but the POST silently fails (404 endpoint, or wrong body shape).

## P0 — pipeline orchestrator "error" state

The pipeline status file from yesterday's run is leaking into the pill. The
NoneType crash is fixed (PR 7) but the status file still shows status=error
because that orchestrator process exited cleanly hours ago. Add a "Reset"
button on the orchestrator card, OR auto-reset the file once the orchestrator
isn't alive AND the user hasn't clicked Run.

---

## P1 — UX reworks

### Stability heatmap rework (item 4)
- Style like the Model Training card (table, sortable columns)
- Per-column number colours (green good, red bad, gold great)
- Compact column width
- Show every TF column even when 0 runs (placeholder cells, not hidden)
- Add description column (second from right) — short blurb per strategy

### Model Training table — add description column
Same treatment: a "Description" column with one-line model summary
(market, target, what it predicts).

### Status column visible improvements (item 1)
PR 9 added the status pill but the user wants it more visible — make it
the primary indicator, animate when running, click-to-cancel.

---

## P2 — data top-up (Option A approved)

User approved Option A: full 1s archive top-up via Binance archive, then
re-resample.

Plan:
1. Use existing `binance_archive_downloader.py` with `--start 2025-01-01
   --timeframe 1s` for every symbol.
2. After download, re-run `resample_ohlcv` to refresh all higher TFs.
3. After resample, run `auto_retrain` to retrain all models on the
   refreshed data.
4. After retrain, run multi-TF backtest; tag results with
   `years_back: 5y_full`.

Trade-off: ~17 months × 20 symbols of 1s data. 10-30 GB download depending
on coin volume. 1-3 days at Binance archive bandwidth (free, no key).

Show results per timeframe in dashboard so user can compare.

---

## Sequence

1. **PR-18** — diagnose + fix the broken JS chain (P0). Lights everything up.
2. **PR-19** — training row Train button (P0). Status column transitions,
   TF picker, click-to-cancel.
3. **PR-20** — pipeline orchestrator reset + auto-clear stale status (P0).
4. **PR-21** — heatmap rework (P1) — sortable table style.
5. **PR-22** — model training description column (P1).
6. **PR-23** — kick off Option A 1s archive top-up. Multi-day background.
7. **PR-24** — once Option A finishes, full retrain + multi-TF backtest.

Per CLAUDE.md: tests + restart + commit between each PR.
