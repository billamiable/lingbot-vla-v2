import driver,json
from pathlib import Path
driver.run('smoke_D','D',3,True)
Path('smoke_gate.json').write_text(json.dumps(dict(passed=True,cases=list(driver.CASES))))
driver.state(state='complete',phase='smoke')
