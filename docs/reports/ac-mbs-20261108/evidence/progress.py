import json
from pathlib import Path

r=Path('/root/experiments/lingbot-ac-mbs-20261108')
s=json.loads((r/'status.json').read_text())
rows=json.loads((r/'summary_all.json').read_text())
done=[x['run'] for x in rows if x['run'].startswith('alloc_') and x.get('records')==[200]*8]
current=next((x for x in rows if x['run']==s.get('run')),None)
print('STATE',s.get('state'),'RUN',s.get('run'),'COMPLETED',len(done),'/8')
if current:
    w=current.get('windows',{}).get('31-200') or current.get('windows',{}).get('11-30') or {}
    print('STEP',current.get('last_step',0),'MEAN',round(w.get('mean_s',0),4),
          'MEDIAN',round(w.get('median_s',0),4),'SAMPLES/S',round(w.get('samples_per_s',0),2),
          'PEAK_MIB',current.get('smi_peak_mib',0),'FINITE',current.get('finite'))
if s.get('state')=='error':
    print('ERROR',s.get('error'))
