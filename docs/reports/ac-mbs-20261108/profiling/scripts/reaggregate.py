#!/usr/bin/env python3
"""Reaggregate per-rank profiler results, optionally reparsing full Kineto traces."""
import argparse
import importlib.util
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parents[1]/'evidence')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--from-raw',action='store_true',help='Require and reparse all 48 full compressed traces')
args=parser.parse_args()
if args.output.exists():parser.error('--output must be a new file')
source=Path(__file__).with_name('analyze.py')
spec=importlib.util.spec_from_file_location('profiler_analysis',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result=[]
for run in sorted((args.evidence/'runs').glob('*')):
    if not (run/'rank0.jsonl').exists():continue
    ranks=[]
    if not run.name.startswith('validate_'):
        for rank in range(8):
            raw=run/f'trace_rank{rank}.json.gz'
            cached=run/f'trace_rank{rank}.analysis.json'
            ranks.append(module.analyze_trace(raw) if args.from_raw else json.loads(cached.read_text()))
    result.append(dict(run=run.name,timing=module.timing(run),ranks=ranks))
expected=json.loads((args.evidence/'analysis.json').read_text())
assert result==expected,'Reaggregated results differ from archived analysis'
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2))
print('PASS: '+('raw traces reparsed' if args.from_raw else 'cached per-rank event statistics reaggregated')+'; all results exactly match.')
