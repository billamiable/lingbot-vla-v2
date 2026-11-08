import json,statistics
from pathlib import Path
r=Path('/root/experiments/lingbot-profile-20261108')
s=json.loads((r/'status.json').read_text())
print(s.get('state'),s.get('run'),s.get('error',''))
for p in sorted((r/'runs').glob('*')):
    files=[p/f'rank{i}.jsonl' for i in range(8)]
    rows=[list(map(json.loads,f.read_text().splitlines())) if f.exists() else [] for f in files]
    n=min(map(len,rows))
    times=[max(rr[i]['elapsed_s'] for rr in rows) for i in range(n)]
    pre=times[10:29];post=times[36:60];traces=list(p.glob('trace_rank*.json.gz'))
    print(p.name,'step',n,'pre',round(statistics.mean(pre),4) if pre else None,
          'post',round(statistics.mean(post),4) if post else None,'traces',len(traces))
