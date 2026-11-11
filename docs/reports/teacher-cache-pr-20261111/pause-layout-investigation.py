from pathlib import Path
import os,signal,json
r=Path('/root/experiments/lingbot-teacher-cache-pr-20261111');pid=int((r/'driver.pid').read_text())
assert Path(f'/proc/{pid}/cwd').resolve()==r
assert b'driver.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
s=json.loads((r/'status.json').read_text());print('Pausing own queue after launched child finishes:',s.get('run'),pid)
os.kill(pid,signal.SIGTERM)
