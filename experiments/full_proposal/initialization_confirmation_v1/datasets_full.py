"""Auditable data locations and deterministic train/validation/test splits."""
import json,os,hashlib,urllib.request,zipfile,tarfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset,Subset

HERE=Path(__file__).resolve().parent
DATA=Path(os.environ.get('QUOTIENT_DATA',str(HERE/'data')))
DATA.mkdir(parents=True,exist_ok=True)

def record(name,**info):
    (DATA/(name+'_provenance.json')).write_text(json.dumps(info,indent=2,default=str)+'\n')

def vision(name,replace_example=False):
    import torchvision.datasets as ds
    import torchvision.transforms as T
    size=32 if name.startswith('cifar') else 64 if name=='tinyimagenet' else 224
    transform=T.Compose([T.Resize((size,size)),T.ToTensor(),T.Normalize((.5,)*3,(.5,)*3)])
    if name.startswith('cifar'):
        root=Path(os.environ.get('QUOTIENT_VISION_DATA', str(DATA / 'vision')))
        cls=ds.CIFAR10 if name=='cifar10' else ds.CIFAR100
        if not root.exists():root=DATA/'vision'
        full=cls(str(root),train=True,download=True,transform=transform)
        test=cls(str(root),train=False,download=True,transform=transform)
        classes=10 if name=='cifar10' else 100
    elif name=='tinyimagenet':
        root=DATA/'tiny-imagenet-200'
        if not root.exists():
            url='https://cs231n.stanford.edu/tiny-imagenet-200.zip'
            dest=DATA/'tiny-imagenet-200.zip'
            if not dest.exists():urllib.request.urlretrieve(url,dest)
            with zipfile.ZipFile(dest) as z:z.extractall(DATA)
        full=ds.ImageFolder(str(root/'train'),transform=transform)
        ann=[line.split()[:2] for line in (root/'val/val_annotations.txt').read_text().splitlines()]
        class TinyValidation(torch.utils.data.Dataset):
            def __len__(self):return len(ann)
            def __getitem__(self,i):
                from PIL import Image
                file,wnid=ann[i]
                return transform(Image.open(root/'val/images'/file).convert('RGB')),full.class_to_idx[wnid]
        test=TinyValidation();classes=200
    else:
        from datasets import load_dataset
        # Publicly available dataset. HTTP/gating errors remain visible.
        hf=load_dataset('clane9/imagenet-100',cache_dir=str(DATA/'hf'))
        class HFImages(torch.utils.data.Dataset):
            def __init__(self,s):self.s=s
            def __len__(self):return len(self.s)
            def __getitem__(self,i):
                row=self.s[i];return transform(row['image'].convert('RGB')),int(row['label'])
        full=HFImages(hf['train']);test=HFImages(hf['validation']);classes=100
        record(name,source='https://huggingface.co/datasets/clane9/imagenet-100',fingerprint=hf['train']._fingerprint,labels=hf['train'].features['label'].names)
    g=torch.Generator().manual_seed(42);order=torch.randperm(len(full),generator=g)
    cut=int(.9*len(full));tr=Subset(full,order[:cut].tolist());val=Subset(full,order[cut:].tolist())
    if replace_example:
        # Adjacent training datasets with identical validation/test sets.
        tr.indices[0]=tr.indices[1]
    record(name+'_split',training=len(tr),validation=len(val),test=len(test),split_seed=42,image_size=size)
    return tr,val,test,classes,size

def graph(name):
    if name=='ogbn-arxiv':
        from torch_geometric.data.data import Data,DataEdgeAttr,DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage,NodeStorage,EdgeStorage
        torch.serialization.add_safe_globals([Data,DataEdgeAttr,DataTensorAttr,GlobalStorage,NodeStorage,EdgeStorage])
        from ogb.nodeproppred import PygNodePropPredDataset
        from torch_geometric.utils import to_undirected
        d=PygNodePropPredDataset(name=name,root=str(DATA/'ogb'))
        data=d[0];data.edge_index=to_undirected(data.edge_index);data.y=data.y.flatten()
        s=d.get_idx_split()
        for key,out in [('train','train_mask'),('valid','val_mask'),('test','test_mask')]:
            mask=torch.zeros(data.num_nodes,dtype=torch.bool);mask[s[key]]=True;setattr(data,out,mask)
    else:
        from torch_geometric.datasets import Planetoid
        d=Planetoid(str(DATA/'Planetoid'),{'cora':'Cora','citeseer':'CiteSeer','pubmed':'PubMed'}[name],split='public');data=d[0]
    record(name,root=str(DATA),nodes=data.num_nodes,edges=data.num_edges,split='official public',features=data.num_features,classes=int(data.y.max())+1)
    return data,int(data.y.max())+1

def language(name):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    path=DATA/(name+'_tokens.pt')
    if path.exists():return torch.load(path,weights_only=True)
    tok=AutoTokenizer.from_pretrained('gpt2',cache_dir=str(DATA/'hf'))
    repo='roneneldan/TinyStories' if name=='tinystories_subset' else 'Salesforce/wikitext'
    config=None if name=='tinystories_subset' else 'wikitext-2-raw-v1' if name=='wikitext2' else 'wikitext-103-raw-v1'
    out={}
    # Subsets have a fixed token budget selected before any model run.
    for split in ['train','validation','test']:
        actual=split
        if name=='tinystories_subset' and split=='test':actual='validation'
        source=load_dataset(repo,config,split=actual,streaming=True,cache_dir=str(DATA/'hf'))
        limit=(10**12 if name=='wikitext2' else 1000000) if split=='train' else 65536
        ids=[]
        # TinyStories has no test split: partition official validation by row parity.
        for i,row in enumerate(source):
            if name=='tinystories_subset' and split in ('validation','test') and i%2!=(0 if split=='validation' else 1):continue
            ids.extend(tok.encode(row['text'],add_special_tokens=False));ids.append(tok.eos_token_id)
            if len(ids)>=limit:break
        out[split]=torch.tensor(ids[:limit],dtype=torch.long)
    out['vocab_size']=len(tok)
    torch.save(out,path)
    record(name,source=repo,config=config,tokenizer='gpt2',tokenizer_vocab=len(tok),tokens={k:len(v) for k,v in out.items() if isinstance(v,torch.Tensor)},sha256=hashlib.sha256(path.read_bytes()).hexdigest(),test_partition='TinyStories validation odd rows; validation even rows' if name=='tinystories_subset' else 'official')
    return out

def inr(name):
    g=torch.Generator().manual_seed(42)
    if name=='image':
        # Explicit synthetic image, not a natural-image benchmark claim.
        a=torch.linspace(-1,1,64);x,y=torch.meshgrid(a,a,indexing='ij');coords=torch.stack([x.flatten(),y.flatten()],1)
        target=(torch.sin(8*torch.pi*x)*torch.cos(6*torch.pi*y)+.5*torch.sin(14*torch.pi*(x+y))).flatten()[:,None]
    elif name=='sdf':
        coords=2*torch.rand(16384,3,generator=g)-1
        target=(coords.norm(dim=1)-.65)[:,None]
    else:
        coords=2*torch.rand(16384,3,generator=g)-1
        target=(torch.sin(4*torch.pi*coords[:,0])*torch.cos(6*torch.pi*coords[:,1])+.2*coords[:,2]**2)[:,None]
    order=torch.randperm(len(coords),generator=g);n=len(order);ds=TensorDataset(coords,target)
    return [Subset(ds,o.tolist()) for o in (order[:int(.8*n)],order[int(.8*n):int(.9*n)],order[int(.9*n):])],coords.shape[1]

def scientific(name):
    path=DATA/(name+'_manufactured.pt')
    if path.exists():return torch.load(path,weights_only=True)
    g=torch.Generator().manual_seed(42);theta=.1+.7*torch.rand(160,generator=g);n=32
    axis=torch.linspace(0,1,n)
    x,y=torch.meshgrid(axis,axis,indexing='ij')
    if name in ('poisson','darcy'):
        u=torch.sin(torch.pi*x)*torch.sin(torch.pi*y)
        if name=='poisson':
            sol=theta[:,None,None]*u;inp=2*torch.pi**2*sol
        else:
            a=1+theta[:,None,None]*torch.sin(2*torch.pi*x)*torch.sin(2*torch.pi*y)
            # -div(a grad u), a smooth positive variable coefficient.
            ax=2*torch.pi*theta[:,None,None]*torch.cos(2*torch.pi*x)*torch.sin(2*torch.pi*y)
            ay=2*torch.pi*theta[:,None,None]*torch.sin(2*torch.pi*x)*torch.cos(2*torch.pi*y)
            forcing=2*torch.pi**2*a*u-ax*torch.pi*torch.cos(torch.pi*x)*torch.sin(torch.pi*y)-ay*torch.pi*torch.sin(torch.pi*x)*torch.cos(torch.pi*y)
            inp=a;sol=u.expand(160,-1,-1).clone()
        out=dict(input=inp,solution=sol,coords=torch.stack([x,y],-1),theta=theta)
        if name=='darcy':out['forcing']=forcing
    else:
        # Cole--Hopf heat-kernel quadrature. Log weights avoid underflow in
        # the small-viscosity regime where a centered explicit solver fails.
        from scipy.special import softmax
        xx=np.linspace(-1,1,n);tt=np.linspace(0,1,n);nu=.01/np.pi
        yy=np.linspace(-1,1,1024,endpoint=False)
        yimages=np.concatenate([yy-2,yy,yy+2])
        distance=xx[:,None]-yimages[None,:]
        solutions=[]
        for amplitude in theta.numpy():
            initial_log=-amplitude*np.cos(np.pi*yimages)/(2*np.pi*nu)
            solution=[-amplitude*np.sin(np.pi*xx)]
            for t in tt[1:]:
                weights=softmax(initial_log[None,:]-distance**2/(4*nu*t),axis=1)
                velocity=(weights*distance/t).sum(1);velocity[[0,-1]]=0
                solution.append(velocity)
            solutions.append(np.stack(solution))
        tx,xxgrid=torch.meshgrid(axis,torch.linspace(-1,1,n),indexing='ij')
        out=dict(input=(-theta[:,None,None]*torch.sin(torch.pi*xxgrid)).expand(-1,n,-1).clone(),solution=torch.tensor(np.stack(solutions),dtype=torch.float32),coords=torch.stack([xxgrid,tx],-1),theta=theta)
    torch.save(out,path)
    record(name,source='deterministic manufactured solution' if name!='burgers' else 'Cole-Hopf periodic heat-kernel quadrature, 1024 points per period, three images, zero Dirichlet by odd symmetry',grid=n,train=128,validation=16,test=16,seed=42,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return out

if __name__=='__main__':
    import sys,time,traceback
    groups=sys.argv[1:] or ['cifar10','cifar100','cora','citeseer','pubmed','ogbn-arxiv','tinyimagenet','imagenet100','wikitext2','wikitext103_subset','tinystories_subset','poisson','burgers','darcy']
    status=json.loads((DATA/'readiness.json').read_text()) if (DATA/'readiness.json').exists() else {}
    for name in groups:
        try:
            if name in ('cifar10','cifar100','tinyimagenet','imagenet100'):vision(name)
            elif name in ('cora','citeseer','pubmed','ogbn-arxiv'):graph(name)
            elif name.startswith('wikitext') or name.startswith('tinystories'):language(name)
            else:scientific(name)
            status[name]={'status':'ready'}
        except Exception as e:status[name]={'status':'failed','error':repr(e)};traceback.print_exc()
        (DATA/'readiness.json').write_text(json.dumps(status,indent=2))
        print(name,status[name],flush=True)
