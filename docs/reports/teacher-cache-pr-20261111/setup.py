from pathlib import Path
import shutil,json,hashlib
R=Path('/root/experiments/lingbot-teacher-cache-pr-20261111')
shutil.copytree('/root/experiments/lingbot-ac-mbs-20261108/study_harness',R/'harness',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(R/'validation.py',R/'harness/validation.py')
p=R/'harness/step_timer.py';s=p.read_text()
s=s.replace('        self.data_wait_s = time.perf_counter() - self.started','        self.data_wait_s = time.perf_counter() - self.started\n        from validation import flow_fixture\n        flow_fixture(micro_batches,step,self.rank)')
s=s.replace('        for key in sorted(set.intersection(*(set(b) for b in micro_batches))):','        for key in sorted(set.intersection(*(set(b) for b in micro_batches))):\n            if key.startswith("teacher_") or key in ["pil_images","future_pil_images"]:\n                continue')
s=s.replace('        from dataclasses import asdict','        from validation import install_subset\n        install_subset(sys.modules["__main__"])\n        from dataclasses import asdict')
s=s.replace('    compile(out, TRAINER, "exec", dont_inherit=True)','    out = out.replace("            optimizer.step()\\n", "            from validation import audit_gradients, audit_parameters\\n            audit_gradients(model, global_step)\\n            optimizer.step()\\n            audit_parameters(model, global_step)\\n")\n    compile(out, TRAINER, "exec", dont_inherit=True)')
p.write_text(s)
p=R/'harness/bench.sh';s=p.read_text().replace(' || echo "trainer exited with $?" >&2','');p.write_text(s)
files=[R/'candidate/lingbotvla/data/vla_data/teacher_cache.py',R/'candidate/lingbotvla/data/vla_data/multi_vla_dataset.py',R/'candidate/tasks/vla/train_lingbotvla.py',R/'candidate/tests/test_teacher_cache.py']
(R/'source_freeze.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print('Validation harness prepared')
