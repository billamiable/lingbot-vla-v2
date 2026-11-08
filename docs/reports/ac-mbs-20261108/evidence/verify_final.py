"""Read-only verification of the completed authorized study."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('/root/experiments/lingbot-ac-mbs-20261108')
aggregate = json.loads((root/'aggregate.json').read_text())
assert aggregate['ready']
checks = {}
for cell, result in aggregate['cells'].items():
    for run in result['runs']:
        receipt = json.loads((root/f'{run}.receipt.json').read_text())
        assert receipt['passed'] and receipt['rank_records'] == [200]*8
        assert receipt['allocator'] == 'expandable_segments:True'
        config = json.loads((root/'runs'/run/'effective_args.json').read_text())['train']
        ac = '_ac32' in cell
        assert config['micro_batch_size'] == (32 if ac else 16)
        assert config['gradient_accumulation_steps'] == (1 if ac else 2)
        assert config['enable_gradient_checkpointing'] == ac
        assert config['global_batch_size'] == 256
        assert config['seed'] == 42 and config['lr'] == 1e-4
        assert config['max_steps'] == 50000 and config['save_steps'] == 0
        assert config['enable_profiling'] is False
        log = (root/f'{run}.launcher.log').read_text()
        assert 'trainer exited with' not in log.lower(), run
        checks[run] = {'passed': True, 'steps_per_rank': 200, 'allocator': receipt['allocator']}
audit = json.loads((root/'consistency_audit.json').read_text())
pairs = [p for p in audit['input_pairs'] if p['a'].startswith('alloc_')]
assert len(pairs) == 4
assert all(p['compared_updates'] == 24 and p['compared_tensor_fields'] >= 312 and not p['differences'] for p in pairs)
assert all(c['finite'] for c in aggregate['cells'].values())
manifest = json.loads((root/'manifest.json').read_text())
sources = {}
for code in ['main','opt']:
    head = subprocess.check_output(['git','-C',str(root/code),'rev-parse','HEAD'],text=True).strip()
    tracked = subprocess.check_output(['git','-C',str(root/code),'status','--porcelain','--untracked-files=no'],text=True)
    assert head == manifest[code] and not tracked
    sources[code] = head
inventory = json.loads((root/'model_inventory.json').read_text())
for item in inventory:
    stat = Path(item['path']).stat()
    assert stat.st_size == item['size'] and stat.st_mtime_ns == item['mtime_ns'], item['path']
data_hash = hashlib.sha256(Path(manifest['data_list']).read_bytes()).hexdigest()
assert data_hash == '5e81a32ec171260d782f381ffdb5020b6c80ecd2f623f33687d83b88a572ebdc'
result = dict(passed=True, runs=checks, input_pairs=pairs, sources=sources,
              model_files_size_mtime_unchanged=len(inventory), data_list_sha256=data_hash,
              note='Input fingerprints cover first three updates only; inventory compares size/mtime, not full weight hashes.')
(root/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'passed':True,'runs':len(checks),'input_pairs':len(pairs),'model_files':len(inventory)}))
