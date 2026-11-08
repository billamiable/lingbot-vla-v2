import csv
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path('/root/experiments/lingbot-ac-mbs-20261108')

def pct(xs, p):
    xs = sorted(xs)
    return xs[max(0, math.ceil(len(xs)*p)-1)]

def analyze(path):
    rank = []
    for i in range(8):
        p = path / f'rank{i}.jsonl'
        rank.append({r['step']: r for r in map(json.loads, p.read_text().splitlines())} if p.exists() else {})
    common = sorted(set.intersection(*(set(x) for x in rank)))
    if not common:
        return dict(run=path.name, records=[len(x) for x in rank])
    result = dict(run=path.name, records=[len(x) for x in rank], last_step=max(common), windows={})
    for first, last in [(11,30), (31,80), (31,200)]:
        ids = [s for s in common if first <= s <= last]
        if not ids:
            continue
        times = [max(r[s]['elapsed_s'] for r in rank) for s in ids]
        med = st.median(times)
        waits = [max(r[s].get('data_wait_s', 0) for r in rank) for s in ids]
        values = dict(n=len(ids), mean_s=st.mean(times), median_s=med, p95_s=pct(times,.95),
                      samples_per_s=256*len(ids)/sum(times), max_data_wait_s=max(waits),
                      mean_data_wait_s=st.mean(waits), slow_steps=[s for s,t in zip(ids,times) if t>1.3*med],
                      loss_mean=st.mean(rank[0][s]['loss'] for s in ids),
                      vla_loss_mean=st.mean(rank[0][s]['vla_loss'] for s in ids),
                      grad_norm_mean=st.mean(rank[0][s]['grad_norm'] for s in ids))
        result['windows'][f'{first}-{last}'] = values
    allrows = [row for r in rank for row in r.values()]
    result['finite'] = all(math.isfinite(row[k]) for row in allrows for k in
                           ['loss','vla_loss','depth_loss','future_depth_loss','future_video_loss','grad_norm','elapsed_s'])
    result['max_grad_norm'] = max((r['grad_norm'] for r in allrows),default=0)
    result['clipped_updates_rank0'] = sum(r['grad_norm']>1 for r in rank[0].values())
    result['peak_allocated_gib'] = max((r.get('peak_allocated_bytes',0) for r in allrows),default=0)/2**30
    result['peak_reserved_gib'] = max((r.get('peak_reserved_bytes',0) for r in allrows),default=0)/2**30
    smi=path/'smi.csv'
    if smi.exists():
        rows=[r for r in csv.reader(smi.read_text().splitlines()) if len(r)>=9]
        result['smi_peak_mib']=max((int(r[2]) for r in rows),default=0)
        result['max_temperature_c']=max((int(r[4]) for r in rows),default=0)
        result['thermal_slowdown_samples']=sum(any('Active' == v.strip() for v in r[7:9]) for r in rows)
    return result

results=[analyze(p) for p in sorted((ROOT/'runs').iterdir()) if p.is_dir()]
(ROOT/'summary_all.json').write_text(json.dumps(results,indent=2))
for r in results:
    w=r.get('windows',{}).get('31-200') or r.get('windows',{}).get('31-80') or {}
    print(r['run'],r.get('last_step',0),'mean',round(w.get('mean_s',0),4),'median',round(w.get('median_s',0),4),
          'samples/s',round(w.get('samples_per_s',0),2),'peakMiB',r.get('smi_peak_mib',0),'finite',r.get('finite'))
