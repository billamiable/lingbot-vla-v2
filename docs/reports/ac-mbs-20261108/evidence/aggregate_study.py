import json
from pathlib import Path
import statistics as st

ROOT=Path('/root/experiments/lingbot-ac-mbs-20261108')
rows=json.loads((ROOT/'summary_all.json').read_text())
cells={}
for cell in ['main_ac32','main_noac16','opt_ac32','opt_noac16']:
    members=[r for r in rows if r['run'].startswith('alloc_'+cell+'_r') and r.get('records')==[200]*8]
    if not members:
        continue
    windows={}
    for window in ['11-30','31-80','31-200']:
        ws=[r['windows'][window] for r in members]
        mean=st.mean(w['mean_s'] for w in ws)
        windows[window]=dict(mean_s=mean,mean_run_median_s=st.mean(w['median_s'] for w in ws),
                             samples_per_s=256/mean,per_run_mean_s=[w['mean_s'] for w in ws],
                             per_run_median_s=[w['median_s'] for w in ws],
                             slow_steps={r['run']:r['windows'][window]['slow_steps'] for r in members},
                             mean_data_wait_s=st.mean(w['mean_data_wait_s'] for w in ws))
        windows[window]['loss_mean'] = st.mean(w['loss_mean'] for w in ws)
        windows[window]['vla_loss_mean'] = st.mean(w['vla_loss_mean'] for w in ws)
    cells[cell]=dict(runs=[r['run'] for r in members],n_runs=len(members),windows=windows,
                     peak_mib=max(r.get('smi_peak_mib',0) for r in members),
                     peak_allocated_gib=max(r.get('peak_allocated_gib',0) for r in members),
                     finite=all(r['finite'] for r in members),
                     max_grad_norm=max(r['max_grad_norm'] for r in members),
                     clipped_updates=sum(r['clipped_updates_rank0'] for r in members),
                     thermal_slowdown_samples=sum(r.get('thermal_slowdown_samples',0) for r in members))
comparisons={}
for code in ['main','opt']:
    if code+'_ac32' in cells and code+'_noac16' in cells:
        a=cells[code+'_ac32']['windows']['31-200']['mean_s']
        b=cells[code+'_noac16']['windows']['31-200']['mean_s']
        comparisons[code]=dict(ac32_s=a,noac16_s=b,ac32_throughput_gain_pct=100*(b/a-1),ac32_time_reduction_pct=100*(1-a/b))
for recipe in ['ac32','noac16']:
    if 'main_'+recipe in cells and 'opt_'+recipe in cells:
        a=cells['main_'+recipe]['windows']['31-200']['mean_s']
        b=cells['opt_'+recipe]['windows']['31-200']['mean_s']
        comparisons[recipe]=dict(main_s=a,opt_s=b,optimized_speedup=a/b)
ready=len(cells)==4 and all(c['n_runs']==2 for c in cells.values())
out=dict(ready=ready,cells=cells,comparisons=comparisons)
(ROOT/'aggregate.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
