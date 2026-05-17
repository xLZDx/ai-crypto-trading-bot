"""Kill duplicate role-processes — keep the OLDEST of each role."""
from __future__ import annotations
import psutil

ROLE_TOKENS = {
    'BOT':         ('main.py',                       'AI trading'),
    'DASH':        ('dashboard',                     'app.py'),
    'MONITOR':     ('monitor',                       'server.py'),
    'WATCHLIST':   ('src.data_ingestion.watchlist_downloader', ''),
    'REALTIME':    ('src.data_ingestion.realtime_db_writer',  ''),
    'ORDERBOOK_W': ('src.data_ingestion.orderbook_parquet_writer', ''),
    'ORDERBOOK_C': ('src.data_ingestion.orderbook_collector',  ''),
    'DGOV':        ('src.data_governance.orchestrator',        ''),
    'DEBUG':       ('scripts.debug_supervisor',                ''),
    'DASHWATCH':   ('scripts.dashboard_watchdog',              ''),
    'SWEEPWATCH':  ('scripts.training_sweep_watchdog',         ''),
    'ORCH':        ('src.training.distributed.orchestrator',   ''),
    'WORKER':      ('src.training.distributed.worker',         ''),
}


def classify(cmd: str) -> str | None:
    cmd_l = cmd
    for role, (tok_a, tok_b) in ROLE_TOKENS.items():
        if tok_a in cmd_l and (not tok_b or tok_b in cmd_l):
            return role
    return None


def main():
    by_role: dict[str, list] = {role: [] for role in ROLE_TOKENS}
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            if p.info['name'] != 'python.exe':
                continue
            cmd = ' '.join(p.info['cmdline'] or [])
            role = classify(cmd)
            if role:
                by_role[role].append((p.info['pid'], p.info['create_time']))
        except Exception:
            continue
    killed = []
    kept = []
    for role, procs in by_role.items():
        procs.sort(key=lambda x: x[1])
        if not procs:
            continue
        keep_pid = procs[0][0]
        kept.append((role, keep_pid))
        print(f"  {role:<12} {len(procs)} alive; keep oldest pid={keep_pid}")
        for pid, _ in procs[1:]:
            try:
                psutil.Process(pid).kill()
                killed.append((role, pid))
            except Exception as e:
                print(f"    kill {pid} err: {e}")
    print()
    print(f"killed {len(killed)} duplicates, kept {len(kept)} anchors")
    for r, p in killed:
        print(f"  killed {r:<12} pid={p}")


if __name__ == '__main__':
    main()
