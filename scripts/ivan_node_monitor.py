"""Ivan node monitor — runs ON Ivan via SSH, prints JSON to stdout.

Razer's health monitor SSHes to Ivan and runs:
    python scripts/ivan_node_monitor.py

Output: single JSON line with GPU, CPU, memory, training processes.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time


def _nvidia_smi() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,utilization.memory,"
             "memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            timeout=6, text=True
        ).strip()
        rows = []
        for line in out.splitlines():
            p = [x.strip() for x in line.split(",")]
            rows.append({
                "name":        p[0],
                "gpu_pct":     float(p[1]) if p[1] not in ("[N/A]","") else None,
                "mem_pct":     float(p[2]) if p[2] not in ("[N/A]","") else None,
                "mem_used_mb": float(p[3]) if p[3] not in ("[N/A]","") else None,
                "mem_total_mb":float(p[4]) if p[4] not in ("[N/A]","") else None,
                "temp_c":      float(p[5]) if p[5] not in ("[N/A]","") else None,
                "power_w":     float(p[6]) if p[6] not in ("[N/A]","") else None,
            })
        return {"gpus": rows, "error": None}
    except Exception as e:
        return {"gpus": [], "error": str(e)}


def _cpu_mem() -> dict:
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=1)
        vm  = psutil.virtual_memory()
        return {
            "cpu_pct":    cpu,
            "ram_used_gb": round(vm.used / 1e9, 2),
            "ram_total_gb":round(vm.total / 1e9, 2),
            "ram_pct":    vm.percent,
        }
    except Exception as e:
        return {"cpu_pct": None, "error": str(e)}


def _training_procs() -> list[dict]:
    """List Python processes that look like workers or trainers."""
    procs = []
    try:
        import psutil
        for p in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent", "memory_info"]):
            try:
                cmd = " ".join(p.info["cmdline"] or [])
                if "python" not in p.info["name"].lower() and "python" not in cmd.lower():
                    continue
                if not any(kw in cmd for kw in ("worker", "train", "orchestrator")):
                    continue
                procs.append({
                    "pid":     p.info["pid"],
                    "cpu_pct": round(p.cpu_percent(interval=0.1), 1),
                    "mem_mb":  round((p.info["memory_info"].rss if p.info["memory_info"] else 0) / 1e6, 1),
                    "cmd":     cmd[-120:],
                })
            except Exception:
                pass
    except Exception as e:
        procs.append({"error": str(e)})
    return procs


def main():
    gpu  = _nvidia_smi()
    cpu  = _cpu_mem()
    proc = _training_procs()

    report = {
        "ts":        time.strftime("%H:%M:%S"),
        "hostname":  os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "Ivan",
        "gpu":       gpu,
        "cpu_mem":   cpu,
        "procs":     proc,
    }
    print(json.dumps(report))


if __name__ == "__main__":
    main()
