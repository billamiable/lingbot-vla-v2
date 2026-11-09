from pathlib import Path
r=Path('/root/experiments/lingbot-regional-20261109')
p=r/'harness/step_timer.py';s=p.read_text()
old='        if step > 3:\n            return'
assert s.count(old)==1
s=s.replace(old,'        from regional_numeric import flow_fixture\n        flow_fixture(micro_batches, step, self.rank)\n'+old)
# The numerical gradient audit runs only in smoke mode, before the optimizer update.
old='    out = out.replace("    args = parse_args(Arguments)\\n",'
assert s.count(old)==1
insert='''    out = out.replace("            optimizer.step()\\n", "            from regional_numeric import audit_gradients\\n            audit_gradients(model, global_step)\\n            optimizer.step()\\n")
'''
s=s.replace(old,insert+old)
p.write_text(s)
compile(s,str(p),'exec')
print(p)
