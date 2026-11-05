"""True depth-L chain and JOINT vertical projection, replacing pairwise heuristics."""
import torch
from torch import nn

class Chain(nn.Module):
    def __init__(self,c):
        super().__init__();r=c['rank'];n=128;L=c['depth']
        shapes=[(r,n)]+[(r,r)]*(L-2)+[(n,r)]
        self.W=nn.ParameterList([nn.Parameter(torch.randn(*s)/s[1]**.5*(.01 if i==L-1 else 1)) for i,s in enumerate(shapes)])
    def forward(self):
        x=self.W[0]
        for w in self.W[1:]:x=w@x
        return x

def vertical(w,om):
    v=[-om[0]@w[0]]
    v.extend(w[i]@om[i-1]-om[i]@w[i] for i in range(1,len(w)-1))
    v.append(w[-1]@om[-1]);return v
def adjoint(w,u):return torch.stack([w[i+1].T@u[i+1]-u[i]@w[i].T for i in range(len(w)-1)])

@torch.no_grad()
def joint_project(w,u,eps):
    rhs=adjoint(w,u);om=torch.zeros_like(rhs);res=rhs.clone();p=res.clone();rr=(res*res).sum()
    initial=rr.clone()
    for _ in range(min(1000,rhs.numel())):
        if rr<=1e-12*initial+1e-24:break
        ap=adjoint(w,vertical(w,p))+2*eps*p
        alpha=rr/(p.mul(ap).sum()+1e-30);om+=alpha*p;res-=alpha*ap
        nr=(res*res).sum();p=res+(nr/(rr+1e-30))*p;rr=nr
    v=vertical(w,om);return [a-b for a,b in zip(u,v)],float(rr.sqrt())

class ChainOptimizer:
    def __init__(self,net,c):
        self.net=net;self.c=c;self.t=0;self.mom=[torch.zeros_like(w) for w in net.W];self.last=[]
        self.base=None
        if c['method']=='adamw':self.base=torch.optim.AdamW(net.parameters(),lr=c['lr'],betas=(c['beta'],.999),weight_decay=0.)
        if c['method']=='sgdm':self.base=torch.optim.SGD(net.parameters(),lr=c['lr'],momentum=c['beta'])
    @torch.no_grad()
    def prepare(self):
        if self.base or self.t%self.c['balance_interval']:return
        w=self.net.W;X=self.net();u,s,vh=torch.linalg.svd(X,full_matrices=False);r=w[0].shape[0];L=len(w)
        s=s[:r]
        if s[-1]<=0:raise ArithmeticError('rank_boundary')
        power=s.pow(1/L);new=[power[:,None]*vh[:r]]+[torch.diag(power) for _ in range(L-2)]+[u[:,:r]*power]
        # Transport the state under the exact layerwise gauge action.
        qs=[];prev=None
        for i in range(L-1):
            rhs=w[i] if i==0 else w[i]@prev
            q=rhs@torch.linalg.pinv(new[i]);qs.append(q);prev=q
        transported=[torch.linalg.solve(qs[0],self.mom[0])]
        transported.extend(torch.linalg.solve(qs[i],self.mom[i]@qs[i-1]) for i in range(1,L-1))
        transported.append(self.mom[-1]@qs[-1]);self.mom=transported
        for wi,ni in zip(w,new):wi.copy_(ni)
    def zero_grad(self,model):model.zero_grad(set_to_none=True)
    @torch.no_grad()
    def step(self):
        w=self.net.W;grads=[p.grad.clone() for p in w]
        old=[p.clone() for p in w];residual=None
        if self.base:self.base.step()
        else:
            self.mom=[self.c['beta']*m+(1-self.c['beta'])*g for m,g in zip(self.mom,grads)]
            if self.c['method']=='quotientopt':self.mom,residual=joint_project(w,self.mom,self.c['eps'])
            for p,m in zip(w,self.mom):p.add_(m,alpha=-self.c['lr'])
        updates=[(a-b)/self.c['lr'] for a,b in zip(old,w)]
        projected,_=joint_project(old,updates,0.)
        self.last=dict(raw_gradient_norm=float(sum(g.square().sum() for g in grads).sqrt()),vertical_energy_ratio=float(sum((a-b).square().sum() for a,b in zip(updates,projected))/(sum(a.square().sum() for a in updates)+1e-30)),balance_error=float(sum((w[i+1].T@w[i+1]-w[i]@w[i].T).square().sum() for i in range(len(w)-1)).sqrt()),sylvester_residual=residual)
        self.t+=1
    def state_dict(self):return dict(t=self.t,mom=self.mom,base=self.base.state_dict() if self.base else None)
    def load_state_dict(self,s):
        self.t=s['t'];self.mom=s['mom']
        if self.base:self.base.load_state_dict(s['base'])
