"""Check fixed data/order and paired, distinct model initializations."""
import hashlib,json,torch
from grid import ROOT,run_id
from run import Task,seed
torch.set_num_threads(1)
plan=json.loads((ROOT/'plan.json').read_text())
def digest(tensors):
    h=hashlib.sha256()
    for name,tensor in tensors:
        h.update(name.encode());h.update(str((tensor.dtype,tuple(tensor.shape))).encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
checks=[]
for domain in ['cora_gcn','sdf_siren']:
    entries=[]
    for config in [c for c in plan['configs'] if c['replication_task']==domain]:
        seed();task=Task(config,'cpu')
        model_hash=digest(task.net.state_dict().items())
        if config['family']=='graph':
            d=task.graph;data_hash=digest([(k,getattr(d,k)) for k in ['x','y','edge_index','train_mask','val_mask','test_mask']])
        else:
            ds=task.tr.dataset.dataset
            items=[(str(i),t) for i,t in enumerate(ds.tensors)]
            items += [(name,torch.tensor(getattr(task,name).dataset.indices)) for name in ['tr','va','te']]
            data_hash=digest(items)
        checkpoint=ROOT/'runs'/run_id(config)/'checkpoint_0.pt'
        if checkpoint.exists():
            actual=torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
            assert digest(actual.items())==model_hash
        entries.append(dict(initialization=config['initialization_seed'],method=config['method'],model=model_hash,data=data_hash,rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))
    assert len({e['data'] for e in entries})==1
    assert len({e['rng'] for e in entries})==1
    by_seed={s:{e['model'] for e in entries if e['initialization']==s} for s in [42,43,44]}
    assert all(len(v)==1 for v in by_seed.values())
    assert len(set().union(*by_seed.values()))==3
    checks.append(dict(task=domain,fixed_data_and_splits=True,fixed_shuffle_rng=True,paired_initialization_across_methods=True,distinct_model_initializations=3,entries=entries))
(ROOT/'protocol_checks.json').write_text(json.dumps(dict(status='passed',checks=checks),indent=2)+'\n')
print('Initialization protocol passed:',[c['task'] for c in checks])
