from pathlib import Path
import json,hashlib,subprocess
R=Path('/root/experiments/lingbot-teacher-cache-pr-20261111')
p=R/'harness/bench.sh';s=p.read_text().replace('"$PY" "$HERE/summarize.py" "$RUN"','if [ "${CACHE_BUILD_ONLY:-0}" != "1" ]; then\n  "$PY" "$HERE/summarize.py" "$RUN"\nfi');p.write_text(s)
p=R/'driver.py';s=p.read_text().replace("CACHE_BUILD_SUBSET=str(int(small)),", "CACHE_BUILD_SUBSET=str(int(small)),CACHE_BUILD_ONLY=str(int(kind=='build')),");p.write_text(s)
files=[R/'candidate/lingbotvla/data/vla_data/teacher_cache.py',R/'candidate/lingbotvla/data/vla_data/multi_vla_dataset.py',R/'candidate/tasks/vla/train_lingbotvla.py',R/'candidate/tests/test_teacher_cache.py']
(R/'source_freeze.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
subprocess.run(['python3',str(R/'summarize.py')],check=True)
with (R/'driver.log').open('x') as f:p=subprocess.Popen(['python3','driver.py'],cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(R/'driver.pid').write_text(str(p.pid));print('started',p.pid)
