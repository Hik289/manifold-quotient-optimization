"""Compare uninterrupted and checkpoint-resumed training, including data order."""
import json,torch
from pathlib import Path
from grid import ROOT,run_id
from run import run
c=dict(family='inr',dataset='coordinate',model='mlp',rank=8,epochs=4,method='quotientopt',lr=.001,beta=.9,eps=1e-6,balance_interval=1,seed=42,protocol='resume_regression_check_only')
rid=run_id(c);base=ROOT/'resume_checks'
uninterrupted=run(c,base/'uninterrupted',device='cuda:0',smoke_epochs=4)
run(c,base/'interrupted',device='cuda:0',smoke_epochs=2)
(base/'interrupted'/rid/'result.json').unlink()
resumed=run(c,base/'interrupted',device='cuda:0',smoke_epochs=4)
a=torch.load(base/'uninterrupted'/rid/'checkpoint_4.pt',map_location='cpu',weights_only=False)['model']
b=torch.load(base/'interrupted'/rid/'checkpoint_4.pt',map_location='cpu',weights_only=False)['model']
maximum=max(float((a[k]-b[k]).abs().max()) for k in a)
report=dict(max_parameter_error=maximum,final_loss_error=abs(uninterrupted['final_train_loss']-resumed['final_train_loss']),all_pass=maximum<1e-6,formal_result=False)
(ROOT/'resume_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
assert report['all_pass']
