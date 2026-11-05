"""One complete, resumable run. Short runs are smoke tests, never paper results."""
import os,json,time,math,random,hashlib,argparse,traceback,shutil
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from grid import ROOT,run_id,encode
from geometry_torch import LowRankLinear,FactorOptimizer,project,diagnostic_project
from models_full import CoordinateMLP,vision_model,GraphNet,DeepONet,FNO,language_model
from chain import Chain,ChainOptimizer
import datasets_full as data

torch.set_num_threads(2)
SOURCE_FILES=['grid.py','run.py','geometry_torch.py','models_full.py','datasets_full.py','chain.py']
SOURCE_HASHES={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}

def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False

def loader(ds,batch,shuffle=False):return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0,generator=torch.Generator().manual_seed(42))

def target(n,r,condition,dev):
    u=torch.linalg.qr(torch.randn(n,r,device=dev),mode='reduced').Q;v=torch.linalg.qr(torch.randn(n,r,device=dev),mode='reduced').Q
    s=torch.logspace(0,-math.log10(condition),r,device=dev);return (u*s)@v.T

class Task:
    def __init__(self,c,dev):
        self.c=c;self.dev=dev;f=c['family'];self.graph=None;self.scientific=None;self.metadata={}
        if f=='matrix':
            n=c.get('size',128);r=c['rank'];self.truth=target(n,r,c.get('condition',100),dev)
            self.net=LowRankLinear(n,n,r,bias=False,frozen_base=False)
            kind=c['dataset']
            if kind=='sensing':
                N=5*r*(2*n-r);a=torch.randn(N,n);b=torch.randn(N,n);y=((a@self.truth.cpu())*b).sum(-1)
                self.tensor=(a.to(dev),b.to(dev),y.to(dev));self.order=torch.randperm(N,device=dev)
                self.metadata['measurement']='independent Gaussian rank-one a_i b_i^T; 5*r*(2*n-r) observations'
            elif kind=='completion':
                order=torch.randperm(n*n,device=dev);N=int(.5*n*n)
                self.order=order[:N];self.val_order=order[N:N+int(.1*n*n)];self.test_order=order[N+int(.1*n*n):]
            elif kind=='regression':
                self.features=torch.randn(4*n,n,device=dev);self.labels=self.features@self.truth.T
        elif f=='deep_linear':
            self.net=Chain(c);self.truth=target(128,c['rank'],c['condition'],dev)
            self.train_target=self.truth+c['noise']*torch.randn_like(self.truth)/128**.5
        elif f=='vision':
            tr,va,te,k,size=data.vision(c['dataset'],c.get('replace_example',False));self.net=vision_model(c['model'],k,size,c['rank'])
            self.tr=loader(tr,128,True);self.va=loader(va,256);self.te=loader(te,256)
        elif f=='graph':
            self.graph,k=data.graph(c['dataset']);self.graph=self.graph.to(dev);self.net=GraphNet(self.graph.num_features,k,c)
        elif f=='inr':
            splits,ind=data.inr(c['dataset']);self.net=CoordinateMLP(ind,c['rank'],c['model'])
            self.tr=loader(splits[0],1024,True);self.va=loader(splits[1],2048);self.te=loader(splits[2],2048)
        elif f=='sciml':
            self.scientific={k:v.to(dev) for k,v in data.scientific(c['dataset']).items()}
            if c['model']=='pinn':self.net=CoordinateMLP(3,c['rank'])
            elif c['model']=='deeponet':self.net=DeepONet(c['rank'])
            else:self.net=FNO(c['rank'])
        elif f=='lm':
            tokens=data.language(c['dataset']);self.net=language_model(c['model'],c['rank'],tokens['vocab_size'])
            self.tokens={k:v.to(dev) for k,v in tokens.items() if isinstance(v,torch.Tensor)}
            self.metadata['tokens']={k:len(v) for k,v in self.tokens.items()}
            self.metadata['sequence_length']=256
        else:raise ValueError(f)
        initialization_seed=c.get('initialization_seed',42)
        if initialization_seed!=42:
            if f!='matrix':raise ValueError('Matrix confirmation only')
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(initialization_seed)
                self.net=LowRankLinear(c['size'],c['size'],c['rank'],bias=False,frozen_base=False)
        self.metadata['data_and_shuffle_seed']=42
        self.metadata['initialization_seed']=initialization_seed
        self.net=self.net.to(dev)
        self.metadata['total_parameters']=sum(p.numel() for p in self.net.parameters())
        self.metadata['trainable_parameters']=sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.metadata['stored_weight_elements_including_frozen_base']=self.metadata['total_parameters']+sum(m.W0.numel() for m in self.net.modules() if isinstance(m,LowRankLinear))
        self.opt=ChainOptimizer(self.net,c) if f=='deep_linear' else FactorOptimizer(self.net,c)
        self.inject_gauge()

    @torch.no_grad()
    def inject_gauge(self):
        c=self.c
        if 'gauge_kind' not in c and c.get('ablation')!=13:return
        for m in self.net.modules():
            if not isinstance(m,LowRankLinear):continue
            r=m.B.shape[1];v=c.get('gauge_value',10.)
            if c.get('gauge_kind','dense')=='scalar':q=torch.eye(r,device=self.dev)*v
            else:
                gen=torch.Generator(device=self.dev).manual_seed(777+self.opt.step_count)
                u=torch.linalg.qr(torch.randn(r,r,device=self.dev,generator=gen)).Q
                w=torch.linalg.qr(torch.randn(r,r,device=self.dev,generator=gen)).Q
                q=(u*torch.logspace(0,math.log10(v),r,device=self.dev))@w.T
            m.B.copy_(m.B@q);m.A.copy_(torch.linalg.solve(q,m.A))
        if isinstance(self.net,Chain):
            r=c['rank'];q=torch.eye(r,device=self.dev)*c.get('gauge_value',10.)
            self.net.W[0].copy_(torch.linalg.solve(q,self.net.W[0]));self.net.W[1].copy_(self.net.W[1]@q)

    def update(self,loss):
        if not torch.isfinite(loss):raise ArithmeticError('nonfinite_loss')
        self.opt.zero_grad(self.net);loss.backward();self.opt.step();return float(loss.detach())

    def train_epoch(self):
        c=self.c;f=c['family'];self.net.train();total=0.;count=0
        if c.get('ablation')==13:self.inject_gauge()
        if f in ('matrix','deep_linear','graph'):
            self.opt.prepare()
            if f=='deep_linear':loss=.5*(self.net()-self.train_target).square().sum()
            elif f=='graph':
                d=self.graph;pred=self.net(d.x,d.edge_index);loss=nn.functional.cross_entropy(pred[d.train_mask],d.y[d.train_mask])
            else:
                X=self.net.B@self.net.A
                if c['dataset']=='sensing':
                    a,b,y=self.tensor;pred=((a@self.net.B)*(b@self.net.A.T)).sum(-1);loss=.5*(pred-y).square().mean()
                elif c['dataset']=='completion':loss=.5*(X-self.truth).flatten()[self.order].square().mean()*X.numel()
                elif c['dataset']=='regression':loss=.5*(self.features@X.T-self.labels).square().mean()*X.numel()
                else:loss=.5*(X-self.truth).square().sum()
            return self.update(loss)
        if f=='sciml':
            s=self.scientific;coords=s['coords'].reshape(-1,2);order=torch.randperm(128,device=self.dev)
            for ids in order.split(16):
                self.opt.prepare()
                if c['model']=='pinn':
                    n=512;inds=torch.randint(coords.shape[0],(len(ids),n),device=self.dev)
                    xy=coords[inds].detach().requires_grad_(True);theta=s['theta'][ids,None,None].expand(-1,n,1)
                    inp=torch.cat([xy,theta],-1);out=self.net(inp).squeeze(-1)
                    expected=s['solution'][ids].flatten(1).gather(1,inds)
                    # PDE term, with independent supervised boundary/reference term.
                    du=torch.autograd.grad(out.sum(),xy,create_graph=True)[0]
                    dxx=torch.autograd.grad(du[...,0].sum(),xy,create_graph=True)[0][...,0]
                    if c['dataset']=='burgers':res=du[...,1]+out*du[...,0]-(.01/torch.pi)*dxx
                    else:
                        dyy=torch.autograd.grad(du[...,1].sum(),xy,create_graph=True)[0][...,1]
                        x=xy[...,0];y=xy[...,1];th=theta[...,0]
                        if c['dataset']=='poisson':res=-(dxx+dyy)-2*torch.pi**2*th*torch.sin(torch.pi*x)*torch.sin(torch.pi*y)
                        else:
                            a=1+th*torch.sin(2*torch.pi*x)*torch.sin(2*torch.pi*y)
                            ax=2*torch.pi*th*torch.cos(2*torch.pi*x)*torch.sin(2*torch.pi*y);ay=2*torch.pi*th*torch.sin(2*torch.pi*x)*torch.cos(2*torch.pi*y)
                            u=torch.sin(torch.pi*x)*torch.sin(torch.pi*y)
                            forcing=2*torch.pi**2*a*u-ax*torch.pi*torch.cos(torch.pi*x)*torch.sin(torch.pi*y)-ay*torch.pi*torch.sin(torch.pi*x)*torch.cos(torch.pi*y)
                            res=-a*(dxx+dyy)-ax*du[...,0]-ay*du[...,1]-forcing
                    loss=(out-expected).square().mean()+res.square().mean()
                else:
                    out=self.net(s['input'][ids],coords);expected=s['solution'][ids].flatten(1)
                    loss=(out-expected).square().mean()
                total+=self.update(loss);count+=1
        elif f=='lm':
            tok=self.tokens['train'];n=(len(tok)-1)//256
            order=torch.randperm(n,device=self.dev)
            for inds in order.split(16):
                self.opt.prepare();offset=inds[:,None]*256+torch.arange(256,device=self.dev)
                logits=self.net(tok[offset]).logits;labels=tok[offset+1]
                loss=nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.flatten())
                total+=self.update(loss)*len(inds);count+=len(inds)
        else:
            for x,y in self.tr:
                x=x.to(self.dev);y=y.to(self.dev);self.opt.prepare();pred=self.net(x)
                loss=nn.functional.cross_entropy(pred,y) if f=='vision' else (pred-y).square().mean()
                total+=self.update(loss)*len(x);count+=len(x)
        return total/max(1,count)

    @torch.no_grad()
    def evaluate(self,split):
        self.net.eval();c=self.c;f=c['family']
        if f in ('matrix','deep_linear'):
            X=self.net().double() if f=='deep_linear' else self.net.B.double()@self.net.A.double()
            if f=='matrix' and c['dataset']=='completion':
                ids=self.order if split=='train' else self.val_order if split=='val' else self.test_order
                difference=(X.double()-self.truth.double()).flatten()[ids]
                truth=self.truth.double().flatten()[ids]
                return dict(mse=float(difference.square().mean()),relative_error=float(difference.norm()/truth.norm()),full_matrix_relative_error=float((X.double()-self.truth.double()).norm()/self.truth.double().norm()))
            return dict(relative_error=float((X.double()-self.truth.double()).norm()/self.truth.double().norm()))
        if f=='graph':
            d=self.graph;mask=getattr(d,{'train':'train_mask','val':'val_mask','test':'test_mask'}[split]);out=self.net(d.x,d.edge_index)
            return dict(accuracy=float((out[mask].argmax(1)==d.y[mask]).float().mean()),loss=float(nn.functional.cross_entropy(out[mask],d.y[mask])))
        if f=='sciml':
            s=self.scientific;ids=slice(0,128) if split=='train' else slice(128,144) if split=='val' else slice(144,160);xy=s['coords'].reshape(-1,2)
            if c['model']=='pinn':
                theta=s['theta'][ids];z=torch.cat([xy.expand(len(theta),-1,-1),theta[:,None,None].expand(-1,len(xy),1)],-1);out=self.net(z).squeeze(-1)
            else:out=self.net(s['input'][ids],xy)
            y=s['solution'][ids].flatten(1)
            return dict(relative_l2=float((out-y).norm()/y.norm()),mse=float((out-y).square().mean()))
        if f=='lm':
            tok=self.tokens['validation' if split=='val' else split];loss=0.;count=0
            for start in range(0,len(tok)-257,256*16):
                n=min(16,(len(tok)-1-start)//256);off=start+torch.arange(n,device=self.dev)[:,None]*256+torch.arange(256,device=self.dev)
                pred=self.net(tok[off]).logits;yy=tok[off+1];loss+=float(nn.functional.cross_entropy(pred.flatten(0,1),yy.flatten(),reduction='sum'));count+=yy.numel()
            loss/=max(1,count);return dict(nll=loss,perplexity=math.exp(min(loss,80)),tokens=count)
        ds=self.tr if split=='train' else self.va if split=='val' else self.te
        loss=0.;correct=0.;count=0
        for x,y in ds:
            x=x.to(self.dev);y=y.to(self.dev);p=self.net(x)
            if f=='vision':loss+=float(nn.functional.cross_entropy(p,y,reduction='sum'));correct+=int((p.argmax(1)==y).sum())
            else:loss+=float((p-y).square().sum())
            count+=len(y)
        if f=='vision':return dict(loss=loss/count,accuracy=correct/count)
        return dict(mse=loss/count,psnr=-10*math.log10(max(loss/count,1e-30)))

    @torch.no_grad()
    def diagnostics(self):
        # Higher precision is for measurements only. Training stays float32.
        if isinstance(self.net,Chain):return self.opt.last
        totals={};unavailable=[]
        for m,d in zip(self.opt.layers,self.opt.last):
            try:
                b,a,gb,ga,ub,ua=[d[k].double() for k in ['B','A','gB','gA','uB','uA']]
                hg,ha,_,_=diagnostic_project(b,a,gb,ga)
                hu,hua,_,res=diagnostic_project(b,a,ub,ua)
                sb=torch.linalg.svdvals(b);sa=torch.linalg.svdvals(a)
                qb,rb=torch.linalg.qr(b,mode='reduced');qa,ra=torch.linalg.qr(a.T,mode='reduced');sx=torch.linalg.svdvals(rb@ra.T)
                values=dict(raw_gradient_norm=(gb.square().sum()+ga.square().sum()).sqrt(),horizontal_gradient_norm=(hg.square().sum()+ha.square().sum()).sqrt(),vertical_gradient_norm=((gb-hg).square().sum()+(ga-ha).square().sum()).sqrt(),vertical_energy_ratio=((ub-hu).square().sum()+(ua-hua).square().sum())/(ub.square().sum()+ua.square().sum()+1e-300),factor_norm=(b.square().sum()+a.square().sum()).sqrt(),factor_condition_number=((sb[0]/sb[-1])*(sa[0]/sa[-1])).sqrt(),smallest_singular_value=sx[-1],balance_error=(b.T@b-a@a.T).norm()/(b.T@b).norm().clamp_min(1e-300),sylvester_residual=res.norm(),momentum_vertical_residual=(b.T@ub-ua@a.T).norm(),product_update_norm=(m.B.double()@m.A.double()-b@a).norm(),factor_update_norm=((m.B.double()-b).square().sum()+(m.A.double()-a).square().sum()).sqrt(),quotient_gradient_proxy=(hg.square().sum()+ha.square().sum()).sqrt())
                for k,v in values.items():
                    value=float(v)
                    if math.isfinite(value):totals[k]=totals.get(k,0.)+value
                    else:unavailable.append(k)
            except (RuntimeError,ArithmeticError) as error:
                unavailable.append(repr(error))
        N=max(1,len(self.opt.layers))
        return dict({k:v/N for k,v in totals.items()},diagnostic_unavailable=unavailable,diagnostic_precision='float64_svd_measurements_only')

def validation_key(c):
    return 'loss' if c['family']=='graph' else 'mse'

def save_checkpoint(path,task,epoch):
    if shutil.disk_usage(path.parent).free<5*1024**3:raise OSError('disk_full: preserve all existing checkpoints')
    generators={name:getattr(task,name).generator.get_state() for name in ('tr','va','te') if hasattr(task,name) and getattr(task,name).generator is not None}
    state=dict(model=task.net.state_dict(),optimizer=task.opt.state_dict(),epoch=epoch,torch_rng=torch.get_rng_state(),loader_generators=generators,numpy_rng=np.random.get_state(),python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)

def run(c,output=None,device=None,smoke_epochs=None):
    seed();dev=device or ('cuda' if torch.cuda.is_available() else 'cpu');rid=run_id(c)
    out=Path(output or ROOT/'runs')/rid;out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
    status=dict(status='initializing',run_id=rid,config=c);(out/'status.json').write_text(json.dumps(status))
    start=time.time();task=Task(c,dev);epoch0=0
    task.metadata['source_sha256']=SOURCE_HASHES
    task.metadata['initial_validation']=task.evaluate('val')
    checkpoints=list(out.glob('checkpoint_*.pt'))
    if checkpoints:
        latest=max(checkpoints,key=lambda p:int(p.stem.split('_')[-1]));s=torch.load(latest,map_location=dev,weights_only=False)
        task.net.load_state_dict(s['model']);task.opt.load_state_dict(s['optimizer']);epoch0=s['epoch']
        torch.set_rng_state(s['torch_rng'].cpu());np.random.set_state(s['numpy_rng']);random.setstate(s['python_rng'])
        for name,state in s.get('loader_generators',{}).items():getattr(task,name).generator.set_state(state.cpu())
        if s['cuda_rng'] is not None:torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda_rng']])
    else:save_checkpoint(out/'checkpoint_0.pt',task,0)
    horizon=smoke_epochs or c['epochs'];best=None;auc=0.;best_epoch=0
    log=out/'trajectory.jsonl'
    # A restart must not leave duplicate epochs after the last complete checkpoint.
    previous=[]
    if log.exists():
        previous=[json.loads(line) for line in log.read_text().splitlines() if json.loads(line)['epoch']<=epoch0]
        log.write_text(''.join(json.dumps(row)+'\n' for row in previous))
        auc=sum(row['train_loss'] for row in previous)
        for row in previous:
            v=row['validation'];key=validation_key(c)
            value=v[key];maximize=key in ('accuracy','psnr')
            if best is None or (value>best if maximize else value<best):best=value;best_epoch=row['epoch']
    gauge_products=[]
    if 'gauge_kind' in c and (out/'gauge_products_checkpoint.pt').exists():
        saved=torch.load(out/'gauge_products_checkpoint.pt',weights_only=True)
        gauge_products=saved['products'][:epoch0]
    for epoch in range(epoch0+1,horizon+1):
        loss=task.train_epoch()
        if any(not torch.isfinite(p).all() for p in task.net.parameters()):
            raise ArithmeticError('nonfinite_parameters_after_update')
        val=task.evaluate('val');test=None;diag=task.diagnostics()
        key=validation_key(c)
        metric=val[key];maximize=key in ('accuracy','psnr')
        if not math.isfinite(metric):raise ArithmeticError('nonfinite_validation_metric')
        if best is None or (metric>best if maximize else metric<best):
            best=metric;best_epoch=epoch
            save_checkpoint(out/'best_validation_state.pt',task,epoch)
            (out/'best_validation.json').write_text(json.dumps(dict(epoch=epoch,key=key,value=best,validation=val)))
        auc+=loss
        row=dict(epoch=epoch,train_loss=loss,validation=val,test=test,learning_rate=c['lr'],**diag,gauge_trajectory_disagreement=None,gauge_measurement='dedicated paired gauge runs only',wall_clock=time.time()-start,peak_memory=torch.cuda.max_memory_allocated() if str(dev).startswith('cuda') else None)
        with log.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        if 'gauge_kind' in c:
            with torch.no_grad():gauge_products.append((task.net() if isinstance(task.net,Chain) else torch.cat([(m.B@m.A).flatten() for m in task.opt.layers])).cpu())
        if epoch in (1,2,5,10,20,50,100,200,500,1000) or epoch%100==0 or epoch==horizon:
            save_checkpoint(out/f'checkpoint_{epoch}.pt',task,epoch)
            if gauge_products:torch.save(dict(epoch=epoch,products=gauge_products),out/'gauge_products_checkpoint.pt')
        if epoch%10==0 or epoch==horizon:
            (out/'status.json').write_text(json.dumps(dict(status='running',epoch=epoch,horizon=horizon,seconds=time.time()-start,run_id=rid)))
            print(rid,epoch,loss,val,flush=True)
    if gauge_products:torch.save(gauge_products,out/'gauge_products.pt')
    test=task.evaluate('test')
    final_model={k:v.detach().clone() for k,v in task.net.state_dict().items()}
    selected=torch.load(out/'best_validation_state.pt',map_location=dev,weights_only=False)
    task.net.load_state_dict(selected['model'])
    selected_test=task.evaluate('test')
    task.net.load_state_dict(final_model)
    train_metrics=task.evaluate('train')
    gap={key:test[key]-train_metrics[key] for key in test if key in train_metrics and isinstance(test[key],(int,float)) and key!='tokens'}
    result=dict(run_id=rid,config=c,status='smoke_only' if smoke_epochs else 'complete',completed_epochs=horizon,final_train_loss=loss,final_training_metrics=train_metrics,test_minus_train=gap,final_validation=val,final_test=test,selected_test=selected_test,selection_key=validation_key(c),best_validation=best,best_epoch=best_epoch,loss_auc=auc,metadata=task.metadata,seconds=time.time()-start,source_hash=SOURCE_HASHES['run.py'],diagnostics=diag)
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(out/'status.json').write_text(json.dumps(result))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--device');p.add_argument('--smoke-epochs',type=int);p.add_argument('--output');a=p.parse_args()
    c=json.loads(Path(a.config).read_text());print(json.dumps(run(c,a.output,a.device,a.smoke_epochs),indent=2))
