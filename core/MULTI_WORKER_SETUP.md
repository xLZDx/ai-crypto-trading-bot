# Multi-Worker Cluster Setup (Razer + Ivan + future)

**Date:** 2026-05-14
**Audience:** operator setting up distributed training across multiple laptops.
**Scope:** config + network setup only. No code changes required to add a new worker.

---

## Architecture (what's already in place)

The cluster orchestrator at [src/training/distributed/orchestrator.py](../src/training/distributed/orchestrator.py) runs on the **master** machine (Razer) on port 7700. Each **worker** runs [src/training/distributed/worker.py](../src/training/distributed/worker.py) and registers itself via `POST /api/cluster/register` every `HEARTBEAT_SEC=15` seconds.

The registration payload includes: `node_id, name, hostname, ip, port, gpu_name, gpu_vram_gb, cpu_cores, ram_gb, cuda_available, lane, status, current_task, last_seen, cpu_percent, gpu_percent, gpu_mem_used_mb, gpu_mem_total_mb, uptime_s`.

The orchestrator persists state to `data/orchestrator_state.json` so the registry survives restarts.

---

## Master (Razer) setup

1. **`.env`** — set the bind host so Ivan can reach the orchestrator over LAN:

   ```bash
   ORCHESTRATOR_BIND_HOST=0.0.0.0
   WORKER_AUTH_KEY=<shared-secret-48-chars>      # required for cross-machine; optional on a single box
   ```

   Generate the auth key with `python -c "import secrets; print(secrets.token_urlsafe(48))"` and use the same value on every worker.

2. **Firewall** — allow inbound TCP 7700 (Windows: `New-NetFirewallRule -DisplayName "AI-Trader Cluster" -Direction Inbound -LocalPort 7700 -Protocol TCP -Action Allow`).

3. **Find Razer's LAN IP** — `ipconfig | findstr IPv4` or `Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.PrefixOrigin -ne 'WellKnown'} | Select-Object IPAddress`. Note it for Ivan's worker config.

4. **Restart** — `.\restart_all.ps1`. The cluster orchestrator starts at step [4.9/6].

5. **Verify** — `Invoke-RestMethod http://127.0.0.1:7700/api/cluster/status`. Should return `{"workers": [...]}`. Empty list is fine until a worker registers.

---

## Worker (Ivan) setup

On the Ivan laptop (or any new worker):

1. **Clone / sync the repo** to a fresh checkout. The worker only needs `src/training/distributed/worker.py` and its imports (utils, ML pipeline). Bot code and trainers are required because the worker executes trainer subprocesses end-to-end.

2. **`.env`** on Ivan:

   ```bash
   GEMINI_API_KEY=<not strictly needed for worker but matches master config>
   WORKER_BIND_HOST=0.0.0.0
   WORKER_AUTH_KEY=<same secret as master>
   ```

3. **Launch** — Ivan has a canonical launcher at `C:\ai-worker\restart_workers.ps1` (hardened with `taskkill /F /T` + WMI verify + port fallback per the operator's memory notes). Otherwise:

   ```powershell
   D:\test 2\AI trading assistance\venv\Scripts\python.exe -m src.training.distributed.worker `
       --master http://<RAZER-LAN-IP>:7700 `
       --name Ivan `
       --lane gpu `
       --host 0.0.0.0
   ```

4. **Verify** — on Razer:

   ```powershell
   Invoke-RestMethod http://127.0.0.1:7700/api/cluster/status | Select-Object -Expand workers
   ```

   Should now include an `Ivan` row with `lane=gpu`, `status=idle`, `gpu_name` populated.

---

## Lane semantics

| Lane    | Accepts                                                  |
|---------|----------------------------------------------------------|
| `cpu`   | CPU-only training (HistGBT, regime RF, meta-labeler)     |
| `gpu`   | TFT, OFT, anything in `_RESOURCE_KIND` with kind=`neural` |
| `any`   | Either (worker doesn't strictly enforce — takes whatever) |

The dispatcher in [src/training/distributed/orchestrator.py:732](../src/training/distributed/orchestrator.py#L732) routes by `resource_kind`:
- `resource_kind=cpu` -> lane in `{cpu, any}`
- `resource_kind=neural` -> lane in `{gpu, any}`

**For TFT specifically:** declare your worker as `--lane gpu` to ensure TFT jobs land there, not on a CPU-only worker that would fall back to PyTorch CPU (which is what triggered the operator's "TFT was running on CPU" complaint).

---

## Roster discipline (recommendation)

The orchestrator's worker list is currently **live-discovery only** — any machine that successfully POSTs to `/register` with the right auth key becomes a worker. There's no persisted "expected roster".

To detect a worker that should be online but isn't (Ivan unplugged, network issue), maintain a roster at [data/workers.json](../data/workers.json):

```json
{
  "workers": [
    {"name": "Razer-Local",      "lane": "cpu",     "expected": true,  "owner": "operator", "notes": "always present as master + cpu lane"},
    {"name": "Razer-GPU",        "lane": "gpu",     "expected": true,  "owner": "operator", "notes": "RTX 3080 Ti Laptop 16GB — TFT/OFT primary"},
    {"name": "Ivan",             "lane": "gpu",     "expected": true,  "owner": "operator", "notes": "secondary GPU laptop on LAN"}
  ]
}
```

Then a roster-vs-live diff endpoint surfaces stragglers. **Not implemented yet** — flagged as Phase 5b follow-up. The roster file itself is harmless to add today (no code reads it).

---

## Auto-restart on worker death

The master doesn't auto-respawn workers — they manage themselves. If Ivan crashes, the operator (or Ivan's local watchdog) restarts the worker.daemon.

Within Razer, the dashboard's `dashboard_watchdog` script keeps the dashboard + cluster orchestrator alive. **No auto-restart for the worker process itself** — that's intentional because the worker may legitimately be down (e.g., Ivan packed up the laptop for travel).

---

## Common gotchas

| Symptom                                          | Cause                                  | Fix                                                          |
|--------------------------------------------------|----------------------------------------|--------------------------------------------------------------|
| Worker stays "idle" forever                      | wrong --master URL                     | check Razer's LAN IP; firewall allows 7700                   |
| Worker registers then disappears every 15s       | auth key mismatch                      | match `WORKER_AUTH_KEY` exactly on both sides                |
| Cluster shows `TEST_*` workers                   | stale registry from prior testing      | call `/api/cluster/registry/reap` (zombie sweep, 60 s cycle) |
| TFT job runs on CPU even though GPU is available | no `lane=gpu` worker registered        | start Ivan worker with `--lane gpu`, OR add a Razer-GPU       |
| Worker's GPU not visible in registration         | torch CUDA install missing on worker   | run worker's `install_cuda_torch.ps1`                        |

---

## What this doc does NOT cover

- **Optuna study coordination across workers** — separate setup (CIO Agent / `data/optuna_orchestrator.db`).
- **DEX data ingestion on worker** — not required; the worker only consumes the master's data feeds.
- **Cloud / VM workers** — same pattern as Ivan, just point `--master` at the public Razer IP and open the firewall accordingly. Production deployments should add TLS termination in front of the orchestrator HTTP server.
