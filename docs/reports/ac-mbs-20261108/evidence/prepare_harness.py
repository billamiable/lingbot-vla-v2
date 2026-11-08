"""Build an instrumented copy of the original benchmark, without changing training math."""
from pathlib import Path
import shutil
import hashlib

root = Path('/root/experiments/lingbot-ac-mbs-20261108')
dst = root / 'study_harness'
shutil.copytree(root / 'harness', dst)
p = dst / 'step_timer.py'
s = p.read_text().replace('import ast\n', 'import ast\nimport hashlib\n')
s = s.replace('self.started = None', 'self.started = None\n        self.data_wait_s = None')
s = s.replace('        row = dict(\n', '        row = dict(\n            data_wait_s=self.data_wait_s,\n            peak_allocated_bytes=torch.cuda.max_memory_allocated(),\n            peak_reserved_bytes=torch.cuda.max_memory_reserved(),\n')
extra = '''
    def fetched(self, step, micro_batches):
        import torch
        self.data_wait_s = time.perf_counter() - self.started
        if step > 3:
            return
        tensors = {}
        for key in sorted(set.intersection(*(set(b) for b in micro_batches))):
            values = [b[key] for b in micro_batches]
            if all(isinstance(v, torch.Tensor) and v.device.type == "cpu" and v.ndim > 0 for v in values):
                try:
                    combined = torch.cat(values, dim=0).contiguous()
                    raw = combined.view(torch.uint8).numpy().tobytes()
                    tensors[key] = dict(shape=list(combined.shape), dtype=str(combined.dtype), sha256=hashlib.sha256(raw).hexdigest())
                except (RuntimeError, TypeError):
                    pass
        path = Path(os.environ["BENCH_OUT"]) / f"inputs_rank{self.rank}.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(dict(step=step, micro_batches=len(micro_batches), tensors=tensors)) + "\\n")

'''
s = s.replace('\ndef rewrite(source):', '\n' + extra + '\ndef rewrite(source):')
needle = '    before.setdefault(fetch.lineno, []).append(indent(fetch.lineno) + "_bench.begin(global_step)\\n")\n'
assert needle in s
s = s.replace(needle, needle + '    before.setdefault(fetch.end_lineno + 1, []).append(indent(fetch.lineno) + "_bench.fetched(global_step, micro_batches)\\n")\n')
p.write_text(s)
compile(s, str(p), 'exec')
(root / 'study_harness_sha256.txt').write_text('\n'.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}' for f in dst.iterdir() if f.is_file())+'\n')
print('Study harness ready:', dst)
