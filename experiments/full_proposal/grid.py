"""Literal proposal grid, with explicit interpretations for underspecified items.

No result is inferred from a queued configuration. Seed is always 42.
FNO is an architecture, so SciML crosses the three named PDEs with all
three named models. Graph module placement is a separate experimental axis.
"""
import hashlib,itertools,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
METHODS=['adamw','sgdm','balance_only','quotientopt']
LRS=[1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2]
BETAS=[0.,.8,.9,.95,.99]
EPS=[1e-8,1e-7,1e-6,1e-5,1e-4]
INTERVALS=[1,5,10,25]

def tasks():
    for depth,rank,condition,noise in itertools.product([2,4,8,16,32],[8,16,32],[1,10,100,1000],[0,.01,.05]):
        yield dict(family='deep_linear',dataset='synthetic',model='chain',depth=depth,rank=rank,condition=condition,noise=noise,epochs=2000)
    for dataset,size,rank in itertools.product(['sensing','completion','regression'],[128,256,512],[4,8,16,32]):
        yield dict(family='matrix',dataset=dataset,model='factorization',size=size,rank=rank,epochs=2000)
    for dataset,model,rank in itertools.product(['cifar10','cifar100','tinyimagenet','imagenet100'],['mlp','resnet18','vit_small'],[4,8,16,32]):
        yield dict(family='vision',dataset=dataset,model=model,rank=rank,epochs=1000)
    for dataset,model,placement,rank in itertools.product(['cora','citeseer','pubmed','ogbn-arxiv'],['gcn','gat','sage','gcnii'],['input','hidden','classifier'],[4,8,16]):
        yield dict(family='graph',dataset=dataset,model=model,placement=placement,rank=rank,epochs=1000)
    for dataset,model,rank in itertools.product(['image','sdf','coordinate'],['mlp','siren','fourier'],[8,16,32]):
        yield dict(family='inr',dataset=dataset,model=model,rank=rank,epochs=2000)
    for dataset,model,rank in itertools.product(['poisson','burgers','darcy'],['pinn','deeponet','fno'],[8,16,32]):
        yield dict(family='sciml',dataset=dataset,model=model,rank=rank,epochs=2000)
    for dataset,model,rank in itertools.product(['wikitext2','wikitext103_subset','tinystories_subset'],['10m','25m','50m'],[4,8,16]):
        yield dict(family='lm',dataset=dataset,model=model,rank=rank,epochs=1000,evaluation_unit='one complete fixed training shard pass')

def encode(c): return json.dumps(c,sort_keys=True,separators=(',',':'))
def run_id(c): return hashlib.sha256(encode(c).encode()).hexdigest()[:20]

def interleaved_tasks():
    from collections import defaultdict
    buckets=defaultdict(list)
    for task in tasks():buckets[task['family']].append(task)
    order=['deep_linear','matrix','graph','inr','sciml','lm','vision']
    for i in range(max(map(len,buckets.values()))):
        for family in order:
            if i<len(buckets[family]):yield buckets[family][i]

def configs():
    # One central setting for every task/method comes first; no task selection.
    for task in interleaved_tasks():
        for method in METHODS:
            yield dict(task,method=method,lr=.001,beta=.9,eps=1e-6,balance_interval=1,seed=42,protocol='full_proposal_v1')
    for task in interleaved_tasks():
        for method,lr,beta,eps,interval in itertools.product(METHODS,LRS,BETAS,EPS,INTERVALS):
            if (lr,beta,eps,interval)==(.001,.9,1e-6,1): continue
            yield dict(task,method=method,lr=lr,beta=beta,eps=eps,balance_interval=interval,seed=42,protocol='full_proposal_v1')

def extras():
    # All A1--A14, not just the archived subset. Applied to the single-interface
    # reference across both conditions and the complete hyperparameter grid.
    for condition,ablation,rank in itertools.product([100,1000],range(1,15),[4,8,16,32]):
        for lr,beta,eps,interval in itertools.product(LRS,BETAS,EPS,INTERVALS):
            yield dict(family='matrix',dataset='factorization',model='factorization',size=128,rank=rank,condition=condition,epochs=2000,method='quotientopt',lr=lr,beta=beta,eps=eps,balance_interval=interval,seed=42,ablation=ablation,protocol='full_proposal_v1')
    for family in ['matrix','deep_linear','vision']:
        for method,kind,value in itertools.product(METHODS,['scalar','dense'],range(7)):
            if kind=='dense' and value>4: continue
            base=dict(family=family,dataset={'matrix':'sensing','deep_linear':'synthetic','vision':'cifar10'}[family],model={'matrix':'factorization','deep_linear':'chain','vision':'resnet18'}[family],size=128,depth=8,rank=16,condition=100,noise=0.,epochs=1000,method=method,lr=.001,beta=.9,eps=1e-6,balance_interval=1,seed=42,protocol='full_proposal_v1')
            yield dict(base,gauge_kind=kind,gauge_value=10.**(value-3 if kind=='scalar' else value))
    for method in METHODS:
        yield dict(family='vision',dataset='cifar100',model='resnet18',rank=16,epochs=1000,method=method,lr=.001,beta=.9,eps=1e-6,balance_interval=1,seed=42,replace_example=True,protocol='full_proposal_v1')

def all_configs(): yield from configs(); yield from extras()

def summary():
    from collections import Counter
    counts=Counter(t['family'] for t in tasks())
    n=sum(counts.values())*4*7*5*5*4
    return dict(task_rank_settings=dict(counts),main_configs=n,ablation_configs=2*14*4*7*5*5*4,gauge_configs=3*4*(7+5),stability_neighbor_configs=4,seed=42,
        note='Each method has the same logical tuning budget. Numerically irrelevant baseline eps/interval values are recorded as aliases of the same executable configuration, not independent replications.',
        lm_protocol='GPT-2 byte-pair vocabulary, 1M training tokens for each named subset; all WikiText-2 training text; 1000 complete passes and validation evaluations. Validation/test caps 65536 tokens each. Actual tokens are recorded.',
        data_policy='ImageNet-100 uses the public clane9/imagenet-100 release, with recorded fingerprint and class map. It is never replaced by Tiny-ImageNet.',
        rank_policy='Rank exceeding either matrix dimension is mathematically infeasible for the full-rank quotient and is recorded as infeasible_rank, never clipped.',
        checkpoint_policy='Save complete resumable state at 0,1,2,5,10,20,50,100,200,500,1000 and each subsequent 100 epochs; stop with disk_full rather than delete checkpoints.')

if __name__=='__main__':
    (ROOT/'grid_summary.json').write_text(json.dumps(summary(),indent=2)+'\n')
    print(json.dumps(summary(),indent=2))
