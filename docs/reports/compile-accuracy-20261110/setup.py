from pathlib import Path
import subprocess,shutil
ROOT=Path('/root/experiments/lingbot-compile-accuracy-20261110')
OLD=Path('/root/experiments/lingbot-regional-20261109')
assert not (ROOT/'src').exists()
subprocess.run(['git','clone','--shared',str(OLD/'src'),str(ROOT/'src')],check=True)
subprocess.run(['git','checkout','--detach','71a41bc'],cwd=ROOT/'src',check=True)
subprocess.run(['git','apply',str(OLD/'study.patch')],cwd=ROOT/'src',check=True)
shutil.copytree(OLD/'harness',ROOT/'harness',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(ROOT/'numeric.py',ROOT/'harness/regional_numeric.py')
s=(ROOT/'harness/step_timer.py').read_text()
s=s.replace('        self.data_wait_s = None','        self.data_wait_s = None\n        self.profiler = None')
s=s.replace('        torch.cuda.synchronize()\n        self.started = time.perf_counter()', '''        from regional_numeric import set_step
        set_step(step)
        if os.environ.get('STUDY_PROFILE')=='1' and self.rank==0 and step==11:
            self.profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False,profile_memory=False)
            self.profiler.start()
        torch.cuda.synchronize()
        self.started = time.perf_counter()''')
s=s.replace('        self.handle.write(json.dumps(row) + "\\n")', '''        self.handle.write(json.dumps(row) + "\\n")
        if self.profiler is not None and step==13:
            self.profiler.stop()
            out=Path(os.environ['BENCH_OUT'])
            rows=[]
            for event in self.profiler.key_averages():
                rows.append(dict(name=event.key,count=event.count,self_cpu_us=event.self_cpu_time_total,self_gpu_us=getattr(event,'self_device_time_total',0),gpu_us=getattr(event,'device_time_total',0)))
            (out/'operator_profile.json').write_text(json.dumps(rows,indent=2))
            trace=out/'trace.json';self.profiler.export_chrome_trace(str(trace))
            import gzip,shutil
            with trace.open('rb') as source,gzip.open(str(trace)+'.gz','wb') as dest:shutil.copyfileobj(source,dest)
            trace.unlink();self.profiler=None''')
s=s.replace('        if step > 3:\n            return','        if step > 3 and os.environ.get("STUDY_FULL_INPUT_HASH")!="1":\n            return')
s=s.replace('            optimizer.step()\\n", "            from regional_numeric import audit_gradients\\n            audit_gradients(model, global_step)\\n            optimizer.step()\\n"','            optimizer.step()\\n", "            from regional_numeric import audit_gradients, audit_parameters\\n            audit_gradients(model, global_step)\\n            optimizer.step()\\n            audit_parameters(model, global_step)\\n"')
assert 'audit_parameters(model, global_step)' in s
(ROOT/'harness/step_timer.py').write_text(s)
# Propagate torchrun failures instead of masking them behind a shell echo.
p=ROOT/'harness/bench.sh';s=p.read_text();s=s.replace(' || echo "trainer exited with $?" >&2','');p.write_text(s)
# Save region graphs, generated code, and selected boundary activations without tracing IO.
p=ROOT/'src/regional_runtime.py';s=p.read_text()
s=s.replace('    # A fullgraph error', '''    if os.environ.get('STUDY_CODEGEN')=='1' and os.environ.get('RANK')=='0':
        import torch._inductor.config as inductor_config
        inductor_config.trace.enabled=True
        inductor_config.trace.debug_dir=str(Path(output_dir).parent/'codegen')
    # A fullgraph error''')
s=s.replace('    def backend(gm, example_inputs):','''    def backend(gm, example_inputs):
        if os.environ.get('STUDY_CODEGEN')=='1' and os.environ.get('RANK')=='0':
            root=Path(output_dir).parent/'fx';root.mkdir(exist_ok=True)
            index=len(list(root.glob('*.txt')))
            (root/f'{index:03d}.txt').write_text(gm.print_readable(print_output=False))''')
s=s.replace('    def wrap(forward):','    def wrap(forward,label):')
s=s.replace('            return entry(*args,**kwargs)','''            from regional_numeric import boundary
            phase='qkv' if kwargs.get('compute_kqv',False) else 'omlp'
            boundary(label+'.'+phase+'.input',args)
            if os.environ.get('STUDY_PROFILE')=='1':
                with torch.profiler.record_function('decoder.'+label+'.'+phase):
                    result=entry(*args,**kwargs)
            else:
                result=entry(*args,**kwargs)
            boundary(label+'.'+phase+'.output',result,register_grad=True)
            return result''')
s=s.replace('    for layer in list(vlm)+list(expert):','    for label,layer in [(f"vlm.{i:02d}",x) for i,x in enumerate(vlm)]+[(f"expert.{i:02d}",x) for i,x in enumerate(expert)]:')
s=s.replace('layer.forward=wrap(original)','layer.forward=wrap(original,label)')
p.write_text(s)
(ROOT/'references').mkdir(exist_ok=True)
print('Prepared isolated source and harness',ROOT)
