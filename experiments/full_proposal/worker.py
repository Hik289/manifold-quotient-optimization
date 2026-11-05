"""Durable exhaustive worker; all successes, failures and aliases remain auditable."""
import os,time,json,sqlite3,traceback,argparse,gc,signal
from pathlib import Path
from grid import ROOT,all_configs,run_id,encode

def effective(c):
    d=dict(c)
    if c['method'] in ('adamw','sgdm'):
        d['eps']=1e-6;d['balance_interval']=1
    return d

def work(worker,total,device):
    import fcntl
    lock=(ROOT/f'worker_{worker}.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    from run import run
    db=sqlite3.connect(ROOT/'queue.sqlite',timeout=120)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, ordinal INTEGER, status TEXT, config TEXT, message TEXT, updated REAL)')
    db.execute('CREATE TABLE IF NOT EXISTS aliases (id TEXT PRIMARY KEY, target TEXT)');db.commit()
    with db:db.execute("UPDATE jobs SET status='resume' WHERE status='running' AND message LIKE ?",(f'worker {worker},%',))
    for ordinal,c in enumerate(all_configs()):
        if ordinal%total!=worker:continue
        ec=effective(c);rid=run_id(ec);original=run_id(c)
        if rid!=original:
            db.execute('INSERT OR IGNORE INTO aliases VALUES (?,?)',(original,rid));db.commit()
        try:
            with db:
                state=db.execute('SELECT status FROM jobs WHERE id=?',(rid,)).fetchone()
                if state and state[0] not in ('resume','retry'):continue
                db.execute('INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?)',(rid,ordinal,'running',encode(ec),f'worker {worker}, device {device}',time.time()))
            print('START',ordinal,rid,ec,flush=True)
            result=run(ec,device=device)
            status=result['status'];message=str(result['final_validation'])
        except Exception as e:
            status='infeasible_rank' if 'infeasible_rank' in str(e) else 'failed'
            message=repr(e);traceback.print_exc()
            folder=ROOT/'runs'/rid;folder.mkdir(parents=True,exist_ok=True)
            (folder/'failure.json').write_text(json.dumps(dict(status=status,error=message,traceback=traceback.format_exc(),config=ec),indent=2))
        with db:db.execute('UPDATE jobs SET status=?,message=?,updated=? WHERE id=?',(status,message,time.time(),rid))
        print('END',rid,status,message,flush=True)
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        try:
            from report import collect
            collect()
        except Exception:traceback.print_exc()
        if 'disk_full' in message:
            print('STOP: checkpoint storage capacity reached; existing data preserved',flush=True);break

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--total',type=int,default=3);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    work(a.worker,a.total,a.device)
