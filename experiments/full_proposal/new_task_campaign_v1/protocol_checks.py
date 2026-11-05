"""Audit data separation and test-access timing, excluded from formal results."""
import json,torch
from grid import ROOT
from run import Task,seed,run

torch.set_num_threads(1)
plan=json.loads((ROOT/'plan.json').read_text())
c=next(c for c in plan['configs'] if c['task_name']=='completion128_r8' and c['method']=='quotientopt' and c['lr']==.001)
seed();task=Task(c,'cpu')
train,val,test=[set(x.tolist()) for x in [task.order,task.val_order,task.test_order]]
assert not train&val and not train&test and not val&test
assert len(train|val|test)==128*128
with torch.no_grad():
    product=task.net.B.double()@task.net.A.double()
    expected=float((product-task.truth.double()).flatten()[task.val_order].square().mean())
assert task.evaluate('val')['mse']==expected

counter={'epochs':0,'test_calls_at_epoch':[]}
old_train,old_evaluate=Task.train_epoch,Task.evaluate
def train_epoch(self):
    output=old_train(self);counter['epochs']+=1;return output
def evaluate(self,split):
    if split=='test':
        counter['test_calls_at_epoch'].append(counter['epochs'])
        assert counter['epochs']==5,'Test accessed before the smoke horizon completed'
    return old_evaluate(self,split)
Task.train_epoch,Task.evaluate=train_epoch,evaluate
output=ROOT/'protocol_checks'
result=run(c,output=output,device='cpu',smoke_epochs=5)
assert result['status']=='smoke_only' and result['completed_epochs']==5
assert counter['test_calls_at_epoch']==[5,5]
rows=[json.loads(x) for x in (output/result['run_id']/'trajectory.jsonl').read_text().splitlines()]
assert all(r['test'] is None for r in rows)
best=min(rows,key=lambda r:(r['validation']['mse'],r['epoch']))
assert result['best_epoch']==best['epoch'] and result['best_validation']==best['validation']['mse']
report=dict(status='passed',smoke_only=True,disjoint_entry_counts=[len(train),len(val),len(test)],
            validation_mse_matches_heldout_mask=True,test_calls_at_epoch=counter['test_calls_at_epoch'],
            no_per_epoch_test_evaluation=True,selected_checkpoint_matches_validation_minimum=True)
(ROOT/'protocol_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
