"""Run tasks/vla/train_lingbotvla.py of a checkout for a fixed number of steps and time each step.

    torchrun ... step_timer.py CONFIG [--key value ...]      (cwd = the checkout; see bench.sh)

The trainer source is rewritten in memory at four landmarks and otherwise executed as is:
  - each step starts timing just before the batch is fetched (CUDA synchronized), and stops
    after the trainer has logged the step (CUDA synchronized), so data wait, the teachers,
    forward/backward, the optimizer and logging are all inside;
  - the loop stops after $BENCH_STEPS steps; the LR schedule keeps the configured max_steps;
  - the four checkpoint guards are disabled, so nothing is saved.
Each rank writes $BENCH_OUT/rank<R>.jsonl: one row per step with elapsed_s, the teacher time the
trainer measured (depth_forward_time), grad_norm and the five losses the trainer logs
(mean over ranks, so identical on every rank). summarize.py takes the max over ranks.
"""

import ast
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType

TRAINER = "tasks/vla/train_lingbotvla.py"
CHECKPOINT_GUARDS = (
    "args.train.save_steps and global_step % args.train.save_steps == 0",
    "not already_saved",
    "args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0",
    "save_checkpoint_path is not None",
)


class Timer:
    def __init__(self, out, steps):
        self.rank = int(os.environ["RANK"])
        self.stop_step = steps
        self.handle = open(Path(out) / f"rank{self.rank}.jsonl", "x", buffering=1)
        self.started = None

    def begin(self, step):
        import torch

        torch.cuda.synchronize()
        self.started = time.perf_counter()

    def finish(self, step, depth_forward_time, grad_norm, **losses):
        import torch

        torch.cuda.synchronize()
        row = dict(
            step=step,
            elapsed_s=time.perf_counter() - self.started,
            teacher_s=float(depth_forward_time),
            grad_norm=float(grad_norm),
            **{k: float(v) for k, v in losses.items()},
        )
        self.handle.write(json.dumps(row) + "\n")


def rewrite(source):
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    nodes = list(ast.walk(main))
    lines = source.splitlines(keepends=True)
    before, replace = {}, {}

    def one(pred, label):
        found = [n for n in nodes if pred(n)]
        if len(found) != 1:
            raise ValueError(f"expected one {label}, found {len(found)}")
        return found[0]

    def indent(lineno):
        line = lines[lineno - 1]
        return line[: len(line) - len(line.lstrip())]

    fetch = one(
        lambda n: isinstance(n, ast.Try)
        and any(isinstance(c, ast.Call) and ast.unparse(c.func) == "next" and ast.unparse(c.args[0]) == "data_iterator"
                for c in ast.walk(n)),
        "batch fetch",
    )
    before.setdefault(fetch.lineno, []).append(indent(fetch.lineno) + "_bench.begin(global_step)\n")
    for expr in CHECKPOINT_GUARDS:
        guard = one(lambda n: isinstance(n, ast.If) and ast.unparse(n.test) == expr, f"guard `{expr}`")
        if expr == CHECKPOINT_GUARDS[0]:
            pad = indent(guard.lineno)
            before.setdefault(guard.lineno, []).extend(
                pad + s + "\n"
                for s in (
                    "_bench.finish(global_step, depth_forward_time, grad_norm, loss=total_loss, vla_loss=total_vla_loss, "
                    "depth_loss=total_depth_loss, future_depth_loss=total_future_depth_loss, "
                    "future_video_loss=total_future_video_loss)",
                    "if global_step >= _bench.stop_step:",
                    "    reached_max_steps = True",
                    "    break",
                )
            )
        replace[guard.lineno] = indent(guard.lineno) + f"if False and ({expr}):  # bench: no checkpoint\n"
    out = "".join("".join(before.get(i, [])) + replace.get(i, line) for i, line in enumerate(lines, 1))
    compile(out, TRAINER, "exec", dont_inherit=True)
    return out


def main():
    root = Path.cwd()
    source = rewrite((root / TRAINER).read_text())
    out = Path(os.environ["BENCH_OUT"])
    if os.environ.get("RANK") == "0":
        (out / "trainer.executed.py").write_text(source)
    sys.path.insert(0, str(root))
    sys.argv = [str(root / TRAINER), *sys.argv[1:]]
    module = ModuleType("__main__")
    module.__dict__.update(_bench=Timer(out, int(os.environ["BENCH_STEPS"])), __file__=sys.argv[0])
    sys.modules["__main__"] = module
    exec(compile(source, sys.argv[0], "exec", dont_inherit=True), module.__dict__)


if __name__ == "__main__":
    main()
