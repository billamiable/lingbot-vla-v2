from pathlib import Path
import csv,json,math,statistics as st
R=Path(__file__).resolve().parent
runs={}
for p in sorted((R/'runs').glob('*')):
 rank_files=[p/f'rank{i}.jsonl' for i in range(8)]
 if not all(f.exists() for f in rank_files):continue
 rows=[[json.loads(x) for x in f.read_text().splitlines()] for f in rank_files]
 n=min(map(len,rows));r=dict(records=list(map(len,rows)),finite=all(math.isfinite(v[k]) for rr in rows for v in rr for k in ['loss','grad_norm','elapsed_s']))
 if n:
  r['first']={k:rows[0][0][k] for k in ['loss','vla_loss','depth_loss','future_depth_loss','future_video_loss','grad_norm']}
 if n>=30:
  t=[max(rr[i]['elapsed_s'] for rr in rows) for i in range(10,30)]
  wait=[max(rr[i]['data_wait_s'] for rr in rows) for i in range(10,30)]
  r.update(mean_s=st.mean(t),median_s=st.median(t),data_wait_s=st.mean(wait),samples_per_s=256/st.mean(t))
 smi=p/'smi.csv'
 samples=list(csv.reader(smi.read_text().splitlines())) if smi.exists() else []
 r['peak_mib']=max((int(x[2]) for x in samples if len(x)>=9),default=0)
 runs[p.name]=r
numeric={}
for name in ['num_base_1','num_off_fixed','num_verify_fixed','num_read_fixed','num_base_2']:
 d={}
 for kind in ['grad','parameter']:
  files=list((R/'runs'/name).glob(kind+'_rank*.json'))
  if len(files)!=8:continue
  data=[json.loads(f.read_text()) for f in files];logical={}
  for x in data:
   for row in x['rows']:logical[row['name']]=logical.get(row['name'],True) and row.get('bit_equal',True)
  d[kind]=dict(relative_l2=(sum(x['sum_diff2'] for x in data)/max(sum(x['sum_ref2'] for x in data),1e-300))**.5,
   max_abs=max(row.get('max_abs',0) for x in data for row in x['rows']),equal=sum(logical.values()),parameters=len(logical))
 if d:numeric[name]=d
perf={}
for arm in ['base','cache']:
 vals=[runs.get('final_'+arm+'_'+str(i),{}) for i in [1,2]]
 if all(x.get('records')==[31]*8 for x in vals):
  mean=st.mean(x['mean_s'] for x in vals);perf[arm]=dict(mean_s=mean,samples_per_s=256/mean,peak_mib=max(x['peak_mib'] for x in vals),runs=vals)
ready=len(perf)==2 and (R/'verify_full_fixed.receipt.json').exists() and json.loads((R/'verify_full_fixed.receipt.json').read_text())['passed']
result=dict(complete=ready,performance=perf,numerical=numeric,runs=runs)
if len(perf)==2:result['throughput_gain_percent']=100*(perf['base']['mean_s']/perf['cache']['mean_s']-1)
(R/'results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(complete=ready,performance={k:v['mean_s'] for k,v in perf.items()},numeric=list(numeric))))
