"""Execute existing central vision configurations while long LM jobs run.

Uses the same IDs and transactionally shared queue, never creates extra
hyperparameter trials. This prevents the first 1000-pass LM jobs from
delaying the entire vision family for days.
"""
import json,sqlite3,time,traceback,gc,os
from grid import ROOT,configs,run_id,encode
from run import run
import torch
db=sqlite3.connect(ROOT/'queue.sqlite',timeout=120)
for ordinal,c in enumerate(configs()):
    if c['family']!='vision':continue
    # The central setting is first and fully covers all dataset/model/rank
    # combinations. The ordinary workers retain the exhaustive remainder.
    if (c['lr'],c['beta'],c['eps'],c['balance_interval'])!=(.001,.9,1e-6,1):continue
    rid=run_id(c)
    db.execute('BEGIN IMMEDIATE')
    if db.execute('SELECT status FROM jobs WHERE id=?',(rid,)).fetchone():db.commit();continue
    db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?)',(rid,ordinal,'running',encode(c),'priority vision worker, device cuda:0',time.time()));db.commit()
    try:
        result=run(c,device='cuda:0');status=result['status'];message=str(result['final_validation'])
    except Exception as e:
        status='failed';message=repr(e);traceback.print_exc()
        folder=ROOT/'runs'/rid;folder.mkdir(parents=True,exist_ok=True)
        (folder/'failure.json').write_text(json.dumps(dict(status=status,error=message,config=c,traceback=traceback.format_exc()),indent=2))
    with db:db.execute('UPDATE jobs SET status=?,message=?,updated=? WHERE id=?',(status,message,time.time(),rid))
    print(rid,status,message,flush=True);gc.collect();torch.cuda.empty_cache()
    if 'disk_full' in message:break
