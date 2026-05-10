# Data Sources — Reference

This document catalogs every external data feed the bot can ingest. Each
source has a connector at `src/data_governance/connectors/<name>.py`
implementing the `DataSourceConnector` contract.

Three priority tiers:
- **Tier 0 — Core** (no auth, always-on)
- **Tier 1 — Auxiliary** (free, mostly no auth, additional context)
- **Tier 2 — Premium** (free tier of paid services, or auth required)

Configure via `data/data_governance.json` (auto-created on first run with
defaults). Run `python -m src.data_governance.orchestrator --list` to see
what's registered.

---

## Tier 0 — Core (enabled by default, no auth)

| Source | Host | Connector | What it provides | Cost |
|---|---|---|---|---|
| Binance Spot/Futures | `data.binance.vision`, `api.binance.com` | `binance_archive_downloader.py` + `realtime_db_writer.py` | Klines, depth, trades, funding | Free |
| Bybit | `api.bybit.com` | `bybit.py` | Spot + perps klines | Free |
| OKX | `www.okx.com` | `okx.py` | Spot + perps candles | Free |
| Coinbase Exchange | `api.exchange.coinbase.com` | `coinbase.py` | Regulated US venue spot | Free |
| Kraken | `api.kraken.com` | `kraken.py` | EUR + USD pairs spot | Free |
| CoinGecko | `api.coingecko.com` | `coingecko.py` | Total mcap, BTC dominance, top-250 | Free (~30 req/min) |
| Crypto Fear & Greed | `api.alternative.me` | `fear_greed.py` | Daily sentiment composite (0-100) | Free |

**Why all the venues?** Cross-venue divergence (e.g. Bybit-Binance basis,
Coinbase-Binance premium) is a leading indicator of liquidity migration.

---

## Tier 1 — Auxiliary (free, mostly no auth)

| Source | Connector | What it provides | Auth | Cost |
|---|---|---|---|---|
| DeFiLlama | `defillama.py` | DeFi TVL by chain | No | Free |
| FRED (Fed) | `fred.py` | DXY, VIX, US10Y, gold, oil, M2 | `FRED_API_KEY` | Free |
| CryptoCompare News | `cryptocompare_news.py` | ~20-source aggregated news | Optional `CRYPTOCOMPARE_API_KEY` | 100k/mo free |
| Funding rates | already in `funding_rate_downloader.py` | Binance Futures funding | No | Free |

### Setup — FRED (recommended)

1. Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html
2. Add to `.env`:
   ```
   FRED_API_KEY=your_key_here
   ```
3. Restart the bot — orchestrator picks it up automatically.

### Setup — CryptoCompare News

Optional but raises quota from anonymous (60/hr) to 100k/month.
1. Register at https://www.cryptocompare.com/cryptopian/api-keys
2. Add to `.env`:
   ```
   CRYPTOCOMPARE_API_KEY=your_key_here
   ```

---

## Tier 2 — Premium (free tier of paid; auth required)

| Source | Connector | What it provides | Auth | Cost |
|---|---|---|---|---|
| CoinGlass | `coinglass.py` | Aggregated funding across 8 venues | `COINGLASS_API_KEY` | Free + paid |
| Reddit | `reddit.py` | Sentiment from r/cryptocurrency, r/bitcoin | `REDDIT_CLIENT_ID/SECRET/USER_AGENT` | Free |

### Setup — CoinGlass

1. Register at https://www.coinglass.com/
2. Get API key in dashboard
3. Add to `.env`:
   ```
   COINGLASS_API_KEY=your_key_here
   ```

### Setup — Reddit

1. Create an app at https://www.reddit.com/prefs/apps (script type)
2. Note `client_id` (under app name) and `secret`
3. `pip install --no-cache-dir praw`
4. Add to `.env`:
   ```
   REDDIT_CLIENT_ID=...
   REDDIT_CLIENT_SECRET=...
   REDDIT_USER_AGENT=ai-trading/0.1 (by u/your_username)
   ```

---

## Suggested but not yet implemented

These have higher value but involve heavier setup. Implement if/when needed.

| Source | What | Why valuable | Cost |
|---|---|---|---|
| **Glassnode** | On-chain (active addresses, exchange net flows) | Leading indicators on holder behavior | Free tier (lagged) → $39/mo for live |
| **Santiment** | Social volume, dev activity, whale txs | Cross-validates news sentiment | Free tier limited |
| **LunarCrush** | "Galaxy Score" — composite social rank | Dense feature | Free tier limited |
| **CryptoQuant** | Miner outflows, exchange reserves | Cycle-top indicators | Free tier + paid |
| **Etherscan / BSCscan / Solscan** | Wallet labels, large txs | "Smart money" tracking | Free 5/sec |
| **Dune Analytics** | Custom on-chain SQL | Anything you can query | Free + paid |
| **Nansen** | Smart-money wallet labels, intent | Best-in-class | Paid only |
| **The Block** RSS | Curated institutional reports | Quality news | Free RSS |
| **NewsAPI.org** | Broad news, crypto category | Backup news source | 100/day free |
| **YouTube transcripts** | Influencer commentary (deps already installed) | Hype / FUD detection | Free |
| **Telegram channels** | Already wired in `telegram_monitor.py` | Alpha leaks | Free |
| **TradingEconomics** | Macro economic calendar (CPI, FOMC, NFP) | Avoid trades around binary events | Free + paid |
| **CoinMetrics** | Institutional-grade on-chain (free + paid) | Reference data | Free tier |
| **dYdX, GMX, Hyperliquid** | DEX perp funding | Retail/CEX flow divergence | Free |
| **Kaiko** | Institutional tick + L2, all venues | Best clean L3 | Paid ($$$) |

---

## Architecture

```
┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
│ data_governance/   │    │  rate_limiter      │    │   QuestDB (hot)    │
│ ├ config.py        │    │  ├ get_limiter     │◄───┤   ILP port 9009    │
│ ├ registry.py      │    │  └ react_to_resp   │    │                    │
│ ├ base.py          │    └────────────────────┘    │   Tables:          │
│ ├ orchestrator.py  │           ▲                  │   - market_data    │
│ └ connectors/      │           │ wraps every      │   - news_sentiment │
│   ├ bybit          ├──┐        │ HTTP call        │   - model_signals  │
│   ├ okx            │  │        │                  │   - trade_events   │
│   ├ coinbase       │  └────────┘                  └─────────┬──────────┘
│   ├ kraken         │                                        │ nightly
│   ├ coingecko      │                                        ▼ rollover
│   ├ fear_greed     │                              ┌────────────────────┐
│   ├ fred           │                              │  Parquet (cold)    │
│   ├ defillama      │                              │  data/parquet/     │
│   ├ cryptocompare  │                              │  by symbol+tf      │
│   ├ coinglass      │                              └─────────┬──────────┘
│   └ reddit         │                                        │ retention
└────────────────────┘                                        ▼ archive
                                                    ┌────────────────────┐
                                                    │  Google Drive      │
                                                    │  (paid backup)     │
                                                    └────────────────────┘
```

### Data lifecycle

1. **Bootstrap** (one-time): orchestrator runs `pull_history()` for every
   enabled connector. Bulk-loads ~1 year of context into QuestDB.
2. **Realtime** (continuous): each connector's `realtime_loop()` polls on
   its configured interval (`poll_sec` in `data_governance.json`).
3. **Rollover** (nightly): `realtime_db_writer.cold_rollover_loop` snapshots
   QuestDB → Parquet so the cold store is always at most 24 h behind.
4. **Archive** (weekly+): `RetentionManager` identifies fully-trained
   partitions; `GoogleDriveBackup` (when configured) uploads them and
   marks them eligible for local pruning.

---

## Cost summary

| What you do | Monthly cost | Result |
|---|---|---|
| Use only Tier 0 (no keys) | $0 | 7 venues, sentiment, dominance, F&G |
| + FRED key | $0 | Macro context (DXY/VIX/yields) |
| + CryptoCompare key | $0 | Higher news quota |
| + CoinGlass key | $0 (free tier) | Aggregated funding |
| + Reddit auth | $0 | Social sentiment |
| + Glassnode/Santiment paid | ~$40-100/mo | On-chain leading indicators |
| + Kaiko (institutional) | $$$ | Clean L3 tick data |

**Recommendation**: start with Tier 0 + FRED + CryptoCompare. That's $0/mo
and covers 80% of the alpha most strategies need.
