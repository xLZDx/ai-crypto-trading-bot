"""Kill + respawn dashboard process to pick up code changes."""
import psutil, subprocess, time, json, urllib.request, os
ROOT = r'D:\test 2\AI trading assistance'
VENV = ROOT + r'\venv\Scripts\python.exe'
DETACHED = 0x00000008 | 0x00000200

killed = []
for p in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        if p.info['name'] != 'python.exe':
            continue
        cmd = ' '.join(p.info['cmdline'] or [])
        normalised = cmd.replace('\\', '/')
        if 'src.dashboard.app' in cmd or 'dashboard/app.py' in normalised:
            psutil.Process(p.info['pid']).kill()
            killed.append(p.info['pid'])
    except Exception:
        pass
print(f'killed dashboard pids: {killed}')
time.sleep(3)

p = subprocess.Popen(
    [VENV, '-m', 'src.dashboard.app'], cwd=ROOT, creationflags=DETACHED,
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    close_fds=True)
print(f'dashboard new pid={p.pid}')
time.sleep(7)

api_key = 'AC54481C0A1662C01AF95EF07E50038BAF239AC35899250B0CE3E605AFA220B9'
req = urllib.request.Request('http://127.0.0.1:5000/api/training/progress')
req.add_header('X-API-Key', api_key)
try:
    r = urllib.request.urlopen(req, timeout=8)
    body = json.loads(r.read())
    print(f"/api/training/progress -> HTTP {r.status} ok={body.get('ok')} count={body.get('count')}")
except Exception as e:
    print(f"/api/training/progress -> FAILED: {e}")
