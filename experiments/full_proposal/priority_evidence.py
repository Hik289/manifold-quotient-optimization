"""Prioritize existing fair comparison blocks; never select by test outcome.

graph_lr: every method x all seven proposal learning rates, beta=.9.
gauge_matrix: every method x all twelve proposal gauges, same sensing data.
The full queue and its long runs remain active. Config IDs and the shared
transactional queue prevent duplicated work.
"""
import argparse,json,sqlite3,time,traceback,os,gc
from pathlib import Path
from grid import ROOT,tasks,METHODS,LRS,extras,run_id,encode

def planned(group):
    if group=='graph_lr':
        task=next(t for t in tasks() if t['family']=='graph' and t['dataset']=='cora'
                  and t['model']=='gcn' and t['placement']=='input' and t['rank']==4)
        for lr in LRS:
            for method in METHODS:
                yield dict(task,method=method,lr=lr,beta=.9,eps=1e-6,
                           balance_interval=1,seed=42,protocol='full_proposal_v1')
    else:
        yield from (c for c in extras() if c['family']=='matrix' and 'gauge_kind' in c)

def work(group,slot,total,device):
    import fcntl
    lock=(ROOT/f'priority_{group}_{slot}.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Set CUDA_VISIBLE_DEVICES externally before this process imports torch.
    from run import run
    import torch
    if device=='cpu':torch.set_num_threads(1)
    db=sqlite3.connect(ROOT/'queue.sqlite',timeout=120)
    owner=f'priority {group} slot {slot}, device {device}'
    with db:
        db.execute("UPDATE jobs SET status='resume' WHERE status='running' AND message=?",(owner,))
    for ordinal,c in enumerate(planned(group)):
        if ordinal%total!=slot:continue
        rid=run_id(c)
        db.execute('BEGIN IMMEDIATE')
        state=db.execute('SELECT status FROM jobs WHERE id=?',(rid,)).fetchone()
        if state and state[0] not in ('resume','retry'):
            db.commit();continue
        db.execute('INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?)',
                   (rid,-1000000-ordinal,'running',encode(c),owner,time.time()));db.commit()
        print('START',rid,c,flush=True)
        try:
            result=run(c,device=device);status=result['status'];message=str(result['final_validation'])
        except Exception as error:
            status='infeasible_rank' if 'infeasible_rank' in str(error) else 'failed'
            message=repr(error);traceback.print_exc()
            folder=ROOT/'runs'/rid;folder.mkdir(parents=True,exist_ok=True)
            (folder/'failure.json').write_text(json.dumps(dict(status=status,error=message,
                     config=c,traceback=traceback.format_exc()),indent=2))
        with db:db.execute('UPDATE jobs SET status=?,message=?,updated=? WHERE id=?',
                          (status,message,time.time(),rid))
        print('END',rid,status,message,flush=True)
        from report import collect
        collect()
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        if 'disk_full' in message:break

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--group',choices=['graph_lr','gauge_matrix'],required=True)
    p.add_argument('--slot',type=int,default=0);p.add_argument('--total',type=int,default=1)
    p.add_argument('--device',default='cpu');a=p.parse_args()
    work(a.group,a.slot,a.total,a.device)
