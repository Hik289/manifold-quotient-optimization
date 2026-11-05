"""Verify full horizons, untouched test logs, frozen source hashes, and selection."""
import collections,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'new_task_combined/progress.json').read_text())
checks=[]
for record in data['records']:
    folder=ROOT/record['artifact_directory']/'runs'/record['run_id']
    if record['status']=='complete':
        result=json.loads((folder/'result.json').read_text())
        assert result['config']==record['config']
        assert result['completed_epochs']==record['config']['epochs']
        rows=[json.loads(x) for x in (folder/'trajectory.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in rows]==list(range(1,record['config']['epochs']+1))
        assert all(r['test'] is None for r in rows)
        key=result['selection_key']
        chosen=min(rows,key=lambda row:(row['validation'][key],row['epoch']))
        assert chosen['epoch']==result['best_epoch']
        assert chosen['validation'][key]==result['best_validation']
        source=ROOT/record['artifact_directory']
        assert all(hashlib.sha256((source/name).read_bytes()).hexdigest()==value for name,value in result['metadata']['source_sha256'].items())
        checks.append(dict(run_id=record['run_id'],horizon=result['completed_epochs'],status='complete_checked'))
    elif record['status']=='failed':
        failure=json.loads((folder/'failure.json').read_text())
        assert failure['config']==record['config'] and failure['traceback']
        checks.append(dict(run_id=record['run_id'],status='failure_retained',error=failure['error']))
    else:
        raise AssertionError('Primary expansion is still active')
assert len(checks)==208
for name,block in data['tasks'].items():
    assert block['all_attempts_finished'] and block['planned']==52
    for method,chosen in block['selected'].items():
        candidates=[r for r in data['records'] if r['status']=='complete' and r['config']['task_name']==name and r['config']['method']==method]
        best=min(candidates,key=lambda r:(r['validation_score'],r['config']['lr'],r['config'].get('other_lr',r['config']['lr']),r['best_epoch']))
        assert best['run_id']==chosen['run_id']
result=dict(status='passed',checked=208,counts=dict(collections.Counter(r['status'] for r in checks)),failure_types=dict(collections.Counter(r['error'] for r in checks if 'error' in r)),checks=checks)
(ROOT/'new_task_combined/audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
