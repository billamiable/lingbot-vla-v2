"""Observe the original benchmark using CPU/CUDA traces and phase ranges."""
import ast
from contextlib import nullcontext
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
from types import ModuleType

import base_timer


class Timer(base_timer.Timer):
    def __init__(self, out, steps):
        super().__init__(out, steps)
        self.step = 0
        self.profiler = None
        self.update_range = None
        self.out = Path(out)

    def range(self, label):
        import torch
        if 31 <= self.step <= 33:
            return torch.profiler.record_function(label)
        return nullcontext()

    def begin(self, step):
        import torch
        self.step = step
        if step == 30:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
                record_shapes=False, profile_memory=False, with_stack=False,
            )
            self.profiler.start()
        super().begin(step)
        self.update_range = self.range(f'update/{step}')
        self.update_range.__enter__()

    def finish(self, step, *args, **kwargs):
        super().finish(step, *args, **kwargs)
        self.update_range.__exit__(None, None, None)
        if self.profiler is not None:
            self.profiler.step()
            if step == 33:
                self.profiler.stop()
                path = self.out / f'trace_rank{self.rank}.json'
                self.profiler.export_chrome_trace(str(path))
                with path.open('rb') as inp, gzip.open(str(path)+'.gz','wb') as out:
                    shutil.copyfileobj(inp,out)
                path.unlink()
                self.profiler = None


def wrapped(label, body):
    return ast.With(items=[ast.withitem(context_expr=ast.Call(
        func=ast.Attribute(value=ast.Name(id='_bench',ctx=ast.Load()),attr='range',ctx=ast.Load()),
        args=[ast.Constant(label)],keywords=[]))],body=body)


class Markers(ast.NodeTransformer):
    def __init__(self):
        self.counts = {}

    def mark(self,label,body):
        self.counts[label] = self.counts.get(label,0)+1
        return wrapped(label,body)

    def visit_For(self,node):
        if ast.unparse(node.target)=='micro_batch' and ast.unparse(node.iter)=='micro_batches':
            original=node.body
            forward=next(i for i,n in enumerate(original) if isinstance(n,ast.With) and any(ast.unparse(x.context_expr)=='model_fwd_context' for x in n.items))
            backward=next(i for i,n in enumerate(original) if isinstance(n,ast.With) and any(ast.unparse(x.context_expr)=='model_bwd_context' for x in n.items))
            body=[self.mark('phase/teacher_and_prepare',original[:forward]),
                  self.mark('phase/forward',original[forward:backward]),
                  self.mark('phase/backward',original[backward:backward+1])]
            tail=original[backward+1:]
            if tail and isinstance(tail[0],ast.If) and 'layer_comm' in ast.unparse(tail[0].test):
                body.append(self.mark('phase/after_backward_reshard',tail[:1]));tail=tail[1:]
            body.append(self.mark('phase/microbatch_finalize',tail))
            node.body=[self.mark('microbatch',body)]
            return node
        return self.generic_visit(node)

    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call):
            name=ast.unparse(node.value.func)
            if name in ['optimizer.step','optimizer.zero_grad','lr_scheduler.step']:
                return self.mark('phase/'+name,[node])
        return self.generic_visit(node)


def main():
    root=Path.cwd()
    text=base_timer.rewrite((root/base_timer.TRAINER).read_text())
    transformer=Markers()
    tree=transformer.visit(ast.parse(text))
    ast.fix_missing_locations(tree)
    assert transformer.counts['microbatch']==1 and transformer.counts['phase/forward']==1
    source=ast.unparse(tree)
    out=Path(os.environ['BENCH_OUT'])
    if os.environ['RANK']=='0':
        (out/'trainer.executed.py').write_text(source)
        (out/'phase_markers.json').write_text(json.dumps(transformer.counts,indent=2))
    sys.path.insert(0,str(root))
    sys.argv=[str(root/base_timer.TRAINER),*sys.argv[1:]]
    module=ModuleType('__main__')
    module.__dict__.update(_bench=Timer(out,int(os.environ['BENCH_STEPS'])),__file__=sys.argv[0])
    sys.modules['__main__']=module
    exec(compile(tree,sys.argv[0],'exec'),module.__dict__)


if __name__=='__main__':
    main()
