"""Common validation-only initial scale, checked against saved initial models."""
import hashlib,json,torch
from grid import ROOT,run_id
from run import Task,seed
torch.set_num_threads(1)
plan=json.loads((ROOT/'plan.json').read_text())
result={}
for task in plan['tasks']:
    config=next(c for c in plan['configs'] if c['task_name']==task['name'] and c['method']=='adamw' and c['lr']==.001)
    folder=ROOT/'runs'/run_id(config)
    checkpoint=folder/'checkpoint_0.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
    seed();model=Task(config,'cpu');model.net.load_state_dict(state)
    validation=model.evaluate('val')
    checked=0
    for phase in [ROOT,ROOT.parent/'factor_step_campaign_v1']:
        p=json.loads((phase/'plan.json').read_text())
        for c in p['configs']:
            if c['task_name']!=task['name']:continue
            other=phase/'runs'/run_id(c)/'checkpoint_0.pt'
            if other.exists():
                values=torch.load(other,map_location='cpu',weights_only=False)['model']
                assert state.keys()==values.keys() and all(torch.equal(state[k],values[k]) for k in state)
                checked+=1
    result[task['name']]=dict(validation=validation,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             checked_initial_models=checked,initial_models_bit_identical=True)
(ROOT/'initial_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
