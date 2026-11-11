from pathlib import Path
import json
r=Path('/root/experiments/lingbot-teacher-cache-pr-20261111')
p=r/'harness/validation.py';s=p.read_text();s+='''

def install_layout_probe(module):
    if os.environ.get('CAPTURE_TARGET_LAYOUT')!='1':return
    original=module.log_teacher_cache_diff
    def wrapped(micro_batch,online,step):
        if step==1:
            def info(x):return dict(shape=list(x.shape),stride=list(x.stride()),offset=x.storage_offset(),dtype=str(x.dtype))
            data={k:dict(online=info(v),cached=info(micro_batch[k])) for k,v in online.items()}
            Path(os.environ['BENCH_OUT'],f'target_layout_rank{os.environ["RANK"]}.json').write_text(json.dumps(data,indent=2))
        return original(micro_batch,online,step)
    module.log_teacher_cache_diff=wrapped
''';p.write_text(s)
p=r/'harness/step_timer.py';s=p.read_text().replace('        install_subset(sys.modules["__main__"])','        install_subset(sys.modules["__main__"])\n        from validation import install_layout_probe\n        install_layout_probe(sys.modules["__main__"])');p.write_text(s)
