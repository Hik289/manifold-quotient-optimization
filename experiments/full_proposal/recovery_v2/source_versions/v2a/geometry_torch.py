"""Correct section-before-backward geometry; no adaptive-state claims.

This is the proposal's Momentum algorithm. Archived Adam-style QO experiments
are retained separately and are not relabeled as runs of this algorithm.
"""
import torch
from torch import nn

class LowRankLinear(nn.Module):
    def __init__(self,ind,out,rank,bias=True,frozen_base=True):
        super().__init__()
        if rank>min(ind,out): raise ValueError(f'infeasible_rank: {rank}>{min(ind,out)}')
        self.B=nn.Parameter(.01*torch.randn(out,rank))
        self.A=nn.Parameter(torch.randn(rank,ind)/ind**.5)
        self.register_buffer('W0',torch.randn(out,ind)/ind**.5 if frozen_base else torch.zeros(out,ind))
        self.bias=nn.Parameter(torch.zeros(out)) if bias else None
    def forward(self,x): return torch.nn.functional.linear(x,self.W0+self.B@self.A,self.bias)

@torch.no_grad()
def balance(B,A):
    # Polar-balanced gauge close to identity, avoiding arbitrary SVD sign jumps.
    qb,rb=torch.linalg.qr(B.float(),mode='reduced')
    qa,ra=torch.linalg.qr(A.float().T,mode='reduced')
    u,s,vh=torch.linalg.svd(rb@ra.T,full_matrices=False)
    if not torch.isfinite(s).all() or s[-1]<=0: raise ArithmeticError('rank_boundary')
    bn=(qb@u)*s.sqrt()
    an=s.sqrt()[:,None]*(vh@qa.T)
    # Orthogonal Procrustes alignment to the previous representative.
    c=bn.T@B.float()+an@A.float().T
    pu,_,pvh=torch.linalg.svd(c)
    rot=pu@pvh
    bn=bn@rot; an=rot.T@an
    q=torch.linalg.lstsq(B.float(),bn).solution
    return bn,an,q

@torch.no_grad()
def project(B,A,uB,uA,eps=0.,mode='full'):
    gb=B.float().T@B.float(); ga=A.float()@A.float().T
    lb,ub=torch.linalg.eigh(gb); la,ua=torch.linalg.eigh(ga)
    c=B.T@uB-uA@A.T
    den=lb[:,None]+la[None,:]+2*eps
    if den.min()<=0: raise ArithmeticError('rank_boundary')
    omega=ub@((ub.T@c@ua)/den)@ua.T
    if mode=='scalar': omega=torch.eye(len(lb),device=B.device)*omega.trace()/len(lb)
    if mode=='diagonal': omega=torch.diag(omega.diag())
    hb=uB-B@omega; ha=uA+omega@A
    residual=B.T@hb-ha@A.T
    return hb,ha,omega,residual

class FactorOptimizer:
    def __init__(self,model,c):
        self.c=c; self.layers=[m for m in model.modules() if isinstance(m,LowRankLinear)]
        factors={id(p) for m in self.layers for p in (m.B,m.A)}
        other=[p for p in model.parameters() if p.requires_grad and id(p) not in factors]
        method=c['method']; beta=c['beta']; self.step_count=0
        # All non-factor weights share AdamW with no decay, across four arms.
        self.other=torch.optim.AdamW(other,lr=c['lr'],weight_decay=0.) if other else None
        self.factor_base=None
        if method in ('adamw','sgdm'):
            params=[p for m in self.layers for p in (m.B,m.A)]
            self.factor_base=(torch.optim.AdamW(params,lr=c['lr'],betas=(beta,.999),weight_decay=0.) if method=='adamw' else torch.optim.SGD(params,lr=c['lr'],momentum=beta))
        self.mom=[(torch.zeros_like(m.B),torch.zeros_like(m.A)) for m in self.layers]
        self.last=[]

    @torch.no_grad()
    def prepare(self):
        """Must run before the forward/backward pass, including at epoch zero."""
        c=self.c; method=c['method']; ab=c.get('ablation',0)
        if method not in ('balance_only','quotientopt'): return
        interval={6:5,7:25}.get(ab,c['balance_interval'])
        if ab==2 or (ab==12 and self.step_count>0) or self.step_count%interval: return
        for i,m in enumerate(self.layers):
            if ab in (9,10):
                if ab==9:
                    scale=(m.A.norm()/m.B.norm().clamp_min(1e-30)).sqrt()
                    q=torch.eye(m.B.shape[1],device=m.B.device)*scale
                else:
                    scale=(m.A.norm(dim=1)/m.B.norm(dim=0).clamp_min(1e-30)).sqrt()
                    q=torch.diag(scale)
                b=m.B@q;a=torch.linalg.solve(q,m.A)
            else:b,a,q=balance(m.B,m.A)
            mb,ma=self.mom[i]
            if ab!=4:
                mb=mb@q; ma=torch.linalg.solve(q,ma)
            m.B.copy_(b);m.A.copy_(a);self.mom[i]=(mb,ma)

    def zero_grad(self,model):
        model.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self):
        c=self.c; ab=c.get('ablation',0); self.last=[]
        beta=0. if ab==3 else c['beta']; eps=0. if ab==8 else c['eps']
        mode={9:'scalar',10:'diagonal'}.get(ab,'full')
        for i,m in enumerate(self.layers):
            if m.B.grad is None or m.A.grad is None: raise RuntimeError('unused factor')
            b=m.B.detach().clone();a=m.A.detach().clone()
            gb=m.B.grad.float();ga=m.A.grad.float()
            # Logging must never prevent an otherwise valid baseline update.
            hgb=hga=None
            if c['method'] in ('balance_only','quotientopt'):
                if ab==11: gb=gb+.01*b;ga=ga+.01*a
                mb,ma=self.mom[i]
                mb=beta*mb+(1-beta)*gb;ma=beta*ma+(1-beta)*ga
                if c['method']=='quotientopt' and ab not in (1,5): mb,ma,_,_=project(b,a,mb,ma,eps,mode)
                self.mom[i]=(mb,ma); ub,ua=mb,ma
                m.B.add_(ub,alpha=-c['lr']);m.A.add_(ua,alpha=-c['lr'])
            else:
                # Record the exact baseline update after the shared factor step.
                ub=ua=None
            self.last.append(dict(B=b,A=a,gB=gb,gA=ga,hgB=hgb,hgA=hga,uB=ub,uA=ua))
        if self.factor_base:
            self.factor_base.step()
            for m,d in zip(self.layers,self.last): d['uB']=(d['B']-m.B)/c['lr'];d['uA']=(d['A']-m.A)/c['lr']
        if self.other: self.other.step()
        self.step_count+=1

    def state_dict(self):
        return dict(step_count=self.step_count,mom=self.mom,other=self.other.state_dict() if self.other else None,factor_base=self.factor_base.state_dict() if self.factor_base else None)
    def load_state_dict(self,s):
        self.step_count=s['step_count'];self.mom=s['mom']
        if self.other:self.other.load_state_dict(s['other'])
        if self.factor_base:self.factor_base.load_state_dict(s['factor_base'])

@torch.no_grad()
def diagnostic_project(B,A,uB,uA):
    """Full-rank Euclidean projection, using float64 SVD instead of Gram eigh.

    Measurement only: never changes the optimizer state or update.
    Unavailable measurements are caught by the caller, not imputed as zero.
    """
    B,A,uB,uA=[x.double() for x in (B,A,uB,uA)]
    if any(not torch.isfinite(x).all() for x in (B,A,uB,uA)):
        raise ArithmeticError('nonfinite_diagnostic_input')
    _,sb,vhb=torch.linalg.svd(B,full_matrices=False)
    ua,sa,_=torch.linalg.svd(A,full_matrices=False)
    vb=vhb.T
    den=sb.square()[:,None]+sa.square()[None,:]
    if den.min()<=0:raise ArithmeticError('singular_diagnostic_factor')
    c=B.T@uB-uA@A.T
    omega=vb@((vb.T@c@ua)/den)@ua.T
    hb=uB-B@omega;ha=uA+omega@A
    return hb,ha,omega,B.T@hb-ha@A.T
