"""Exercise every distinct task/model branch; outputs excluded from formal results."""
import json,traceback
from pathlib import Path
from grid import ROOT,tasks
from run import run
seen=set();reports=[]
for task in tasks():
    key=(task['family'],task['model'],task['dataset'] if task['family'] in ('matrix','inr','sciml') else 'default',task.get('placement',''))
    if key in seen:continue
    seen.add(key)
    c=dict(task,method='quotientopt',lr=.001,beta=.9,eps=1e-6,balance_interval=1,seed=42,protocol='smoke_only')
    try:
        result=run(c,ROOT/'smoke_runs',device='cuda:0',smoke_epochs=2)
        reports.append(dict(key=key,status='pass',run_id=result['run_id']))
    except Exception as e:
        reports.append(dict(key=key,status='fail',error=repr(e)));traceback.print_exc()
    (ROOT/'smoke_report.json').write_text(json.dumps(reports,indent=2))
    print(reports[-1],flush=True)
