"""Diagnose why the TFT smoke is stuck."""
import psutil
import time

print('Python processes (cpu>2% OR mem>200MB):')
for p in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
    try:
        if p.info['name'] != 'python.exe':
            continue
        cpu = p.cpu_percent(interval=0.5)
        rss_mb = p.memory_info().rss / 1e6
        if cpu < 2 and rss_mb < 200:
            continue
        cmd = ' '.join(p.info['cmdline'] or [])
        age_min = (time.time() - p.info['create_time']) / 60
        kind = '?'
        if 'distributed.worker' in cmd:
            kind = 'WORKER'
        elif 'distributed.orchestrator' in cmd:
            kind = 'ORCH'
        elif 'src.dashboard' in cmd:
            kind = 'DASH'
        elif 'main.py' in cmd:
            kind = 'BOT'
        elif 'tft' in cmd.lower():
            kind = 'TFT-TRAIN'
        elif 'data_ingestion' in cmd:
            kind = 'INGEST'
        elif 'loky' in cmd or 'joblib' in cmd:
            kind = 'LOKY'
        print(f'  pid={p.info["pid"]:>6} [{kind:<10}] cpu={cpu:5.1f}% mem={rss_mb:6.0f}MB age={age_min:5.1f}min')
        if kind in ('TFT-TRAIN', 'WORKER', 'LOKY', '?'):
            print(f'           cmd: {cmd[:240]}')
    except Exception:
        continue
