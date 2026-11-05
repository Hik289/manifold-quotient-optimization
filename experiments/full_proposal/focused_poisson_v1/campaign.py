"""Fixed four-family expansion; test-blind selection, durable queue, no deletion."""
import argparse, gc, hashlib, json, os, sqlite3, time, traceback
from pathlib import Path
from grid import ROOT, run_id, encode


def initialize():
    plan=json.loads((ROOT/'plan.json').read_text())
    db=sqlite3.connect(ROOT/'queue.sqlite',timeout=120)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, ordinal INTEGER, status TEXT, config TEXT, message TEXT, updated REAL)')
    # Execution order only: every registered LR and arm remains in the plan.
    priority=plan['learning_rates']
    methods=['adamw','sgdm','balance_only','quotientopt']
    ordered=sorted(plan['configs'],key=lambda c:(priority.index(c['lr']),methods.index(c['method']),[t['name'] for t in plan['tasks']].index(c['task_name'])))
    with db:
        db.executemany('INSERT OR IGNORE INTO jobs VALUES (?,?,?,?,?,?)',
                       [(run_id(c),i,'pending',encode(c),'',time.time()) for i,c in enumerate(ordered)])
    return db


def work(slot,task_filter=None):
    import fcntl
    lock=(ROOT/f'worker_{slot}.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    from run import run,Task,seed
    torch.set_num_threads(1)
    db=initialize()
    device=os.environ.get('FOCUS_DEVICE','cpu')
    owner=f'focused {device} worker {slot}'
    with db:
        db.execute("UPDATE jobs SET status='pending' WHERE status='running' AND message=?",(owner,))
    while True:
        db.execute('BEGIN IMMEDIATE')
        if task_filter:
            row=db.execute("SELECT id,config FROM jobs WHERE status='pending' AND json_extract(config,'$.task_name')=? ORDER BY ordinal LIMIT 1",(task_filter,)).fetchone()
        else:
            row=db.execute("SELECT id,config FROM jobs WHERE status='pending' ORDER BY ordinal LIMIT 1").fetchone()
        if not row:
            db.commit();break
        rid,raw=row
        db.execute("UPDATE jobs SET status='running',message=?,updated=? WHERE id=?",(owner,time.time(),rid));db.commit()
        c=json.loads(raw)
        print('START',rid,c,flush=True)
        try:
            result=run(c,device=device)
            status=result['status'];message=str(result['final_validation'])
        except Exception as error:
            status='failed';message=repr(error)
            traceback.print_exc()
            folder=ROOT/'runs'/rid;folder.mkdir(parents=True,exist_ok=True)
            record=dict(status='failed',config=c,error=message,traceback=traceback.format_exc())
            # Retain a conservative sensitivity control: the best finite validation
            # checkpoint of a later-stopped baseline must not disappear.
            if (folder/'best_validation_state.pt').exists():
                try:
                    seed();task=Task(c,'cpu')
                    state=torch.load(folder/'best_validation_state.pt',map_location='cpu',weights_only=False)
                    task.net.load_state_dict(state['model'])
                    record['best_finite_validation']=json.loads((folder/'best_validation.json').read_text())
                    record['selected_test']=task.evaluate('test')
                except Exception as secondary:
                    record['checkpoint_test_error']=repr(secondary)
            (folder/'failure.json').write_text(json.dumps(record,indent=2)+'\n')
        with db:
            db.execute('UPDATE jobs SET status=?,message=?,updated=? WHERE id=?',(status,message,time.time(),rid))
        print('END',rid,status,message,flush=True)
        gc.collect()
        if 'disk_full' in message:
            break


def profile():
    import torch
    from run import Task,seed
    torch.set_num_threads(1)
    plan=json.loads((ROOT/'plan.json').read_text())
    results=[]
    for task in plan['tasks']:
        c=next(c for c in plan['configs'] if c['task_name']==task['name'] and c['method']=='quotientopt' and c['lr']==.001)
        seed();start=time.time();t=Task(c,'cpu')
        initialization=time.time()-start
        durations=[]
        for _ in range(2):
            start=time.time();t.train_epoch();t.evaluate('val');t.diagnostics();durations.append(time.time()-start)
        results.append(dict(task=task['name'],initialization_seconds=initialization,
                            seconds_per_epoch=sum(durations)/len(durations),smoke_only=True))
        print(json.dumps(results[-1]),flush=True)
    (ROOT/'resource_profile.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--slot',type=int,default=0);p.add_argument('--profile',action='store_true');p.add_argument('--task')
    a=p.parse_args()
    if a.profile:profile()
    else:work(a.slot,a.task)
