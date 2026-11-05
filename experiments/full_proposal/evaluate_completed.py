"""Add post-training train/test gaps to early completed records, without retraining."""
import json,torch
from grid import ROOT
from run import Task,seed,SOURCE_HASHES
for file in (ROOT/'runs').glob('*/result.json'):
    r=json.loads(file.read_text())
    if r.get('status')!='complete' or 'final_training_metrics' in r:continue
    c=r['config'];checkpoint=file.parent/f"checkpoint_{c['epochs']}.pt"
    if not checkpoint.exists():continue
    seed();task=Task(c,'cuda:0');state=torch.load(checkpoint,map_location='cuda:0',weights_only=False)
    task.net.load_state_dict(state['model']);train=task.evaluate('train');test=task.evaluate('test')
    # Checkpoint reconstruction must reproduce the already reported test metric.
    difference=max(abs(test[k]-r['final_test'][k]) for k in test if k!='tokens')
    if difference>1e-5:raise AssertionError((r['run_id'],'checkpoint metric mismatch',difference))
    r['final_training_metrics']=train;r['test_minus_train']={k:test[k]-train[k] for k in test if k in train and k!='tokens'}
    r['posthoc_evaluation_source']=SOURCE_HASHES;r['checkpoint_test_reproduction_error']=difference
    file.write_text(json.dumps(r,indent=2)+'\n');print(r['run_id'],train,r['test_minus_train'],flush=True)
    del task;torch.cuda.empty_cache()
