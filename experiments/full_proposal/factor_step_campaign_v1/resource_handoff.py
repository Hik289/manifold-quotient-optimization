"""Move freed CPU slots to the already-authorized fixed extension queue."""
from pathlib import Path
import sqlite3,time,subprocess,os,json,sys
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'new_task_campaign_v1'
marker=ROOT/'resource_handoff_complete.json'
if marker.exists():
    raise SystemExit('Resource handoff already completed')
for attempt in range(2880):
    db=sqlite3.connect(BASE/'queue.sqlite',timeout=120)
    states=dict(db.execute('select status,count(*) from jobs group by status'))
    db.close()
    if states.get('pending',0)==0 and states.get('running',0)<=4:
        break
    time.sleep(10)
else:
    raise RuntimeError('Handoff timed out; existing workers remain active')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QUOTIENT_DATA=str(ROOT.parent/'data'),
         OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
pids=json.loads((ROOT/'worker_pids.json').read_text())
new=[]
for slot in range(4,12):
    worker=subprocess.Popen(['nice','-n','10',sys.executable,'-u','campaign.py','--slot',str(slot)],
                            cwd=ROOT,env=env,stdout=(ROOT/f'worker_{slot}.log').open('a'),
                            stderr=subprocess.STDOUT,start_new_session=True)
    new.append(worker.pid)
pids.extend(new)
(ROOT/'worker_pids.json').write_text(json.dumps(pids)+'\n')
marker.write_text(json.dumps(dict(new_pids=new,time_unix=time.time()))+'\n')
print('Resource handoff completed:',new,flush=True)
