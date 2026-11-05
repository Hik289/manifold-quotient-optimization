#!/usr/bin/env python3
"""
CRITICAL observability test (survey finding #10 / failure criterion #1).

The raw Euclidean LoRA gradient (G_X A^T, B^T G_X) is EXACTLY horizontal
(vertical energy = 0). So where does measurable vertical (gauge) energy come
from?  Candidates:
  (a) Adam elementwise preconditioning m/sqrt(v)  -- NOT gauge-covariant
  (b) SGD momentum accumulation under a drifting gauge -- non-equivariant transport
  (c) factor weight decay  -- not orbit-invariant
  (d) LoRA-style asymmetric init (B=0 or A~N, B~0) driving factors off-balance
This script quantifies each on a matrix-sensing-like f(BA)=0.5||BA - X*||^2.
"""
import numpy as np
from scipy.linalg import solve_sylvester
import json
RNG=np.random.default_rng(42); F32=np.float32
def fn(M): return float(np.linalg.norm(M,'fro'))
def conn(B,A,xB,xA,eps=1e-6):
    H=(0.5*(B.T@B+A@A.T)).astype(np.float64); He=H+eps*np.eye(H.shape[0])
    C=(B.T@xB-xA@A.T).astype(np.float64)
    return solve_sylvester(He,He,C).astype(F32)
def vert_ratio(B,A,gB,gA,eps=1e-6):
    Om=conn(B,A,gB,gA,eps)
    hB=gB-B@Om; hA=gA+Om@A
    vB=gB-hB; vA=gA-hA
    den=fn(gB)**2+fn(gA)**2
    return (fn(vB)**2+fn(vA)**2)/den if den>0 else 0.0

m,d,r=64,48,8
# target low-rank X*
U=np.linalg.qr(RNG.standard_normal((m,r)))[0]; V=np.linalg.qr(RNG.standard_normal((d,r)))[0]
s=np.logspace(0,-2,r)  # cond=100
Xstar=((U*s)@V.T).astype(F32)

def grad(B,A):
    Gx=(B@A - Xstar).astype(F32)
    return (Gx@A.T).astype(F32),(B.T@Gx).astype(F32)

report={}

# ---- confirm raw gradient is exactly horizontal ----
B=(U*np.sqrt(s)).astype(F32); A=(np.sqrt(s)[:,None]*V.T).astype(F32)
gB,gA=grad(B,A)
report['raw_gradient_vert_ratio']=vert_ratio(B,A,gB,gA)

# ---- (a) Adam preconditioned update vertical energy over a short run ----
def run(optimizer, T=300, lr=1e-2, wd=0.0, init='balanced', beta1=0.9, beta2=0.999):
    if init=='balanced':
        B=(U*np.sqrt(s)+0.05*RNG.standard_normal((m,r))).astype(F32)
        A=(np.sqrt(s)[:,None]*V.T+0.05*RNG.standard_normal((r,d))).astype(F32)
    elif init=='lora':   # B=0, A~N (classic LoRA init) -> maximally unbalanced
        B=(0.01*RNG.standard_normal((m,r))).astype(F32)
        A=(RNG.standard_normal((r,d))/np.sqrt(d)).astype(F32)
    mB=np.zeros_like(B); mA=np.zeros_like(A)
    vB=np.zeros_like(B); vA=np.zeros_like(A)
    vr_update=[]; vr_grad=[]; bal=[]; loss=[]
    for t in range(1,T+1):
        gB,gA=grad(B,A)
        if wd>0: gB=gB+wd*B; gA=gA+wd*A
        vr_grad.append(vert_ratio(B,A,gB,gA))
        if optimizer=='sgdm':
            mB=beta1*mB+gB; mA=beta1*mA+gA
            uB,uA=mB,mA
        elif optimizer=='adam':
            mB=beta1*mB+(1-beta1)*gB; mA=beta1*mA+(1-beta1)*gA
            vB=beta2*vB+(1-beta2)*gB**2; vA=beta2*vA+(1-beta2)*gA**2
            mhB=mB/(1-beta1**t); mhA=mA/(1-beta1**t)
            vhB=vB/(1-beta2**t); vhA=vA/(1-beta2**t)
            uB=mhB/(np.sqrt(vhB)+1e-8); uA=mhA/(np.sqrt(vhA)+1e-8)
        # vertical energy of the *actual update direction*
        vr_update.append(vert_ratio(B,A,uB.astype(F32),uA.astype(F32)))
        B=(B-lr*uB).astype(F32); A=(A-lr*uA).astype(F32)
        bal.append(fn(B.T@B-A@A.T)/(fn(B.T@B)+1e-12))
        loss.append(0.5*fn(B@A-Xstar)**2)
    import numpy as _np
    def summ(x): return dict(mean=float(_np.mean(x)),final=float(x[-1]),max=float(_np.max(x)))
    return dict(update_vert=summ(vr_update),grad_vert=summ(vr_grad),
                balance=summ(bal),final_loss=float(loss[-1]))

report['sgdm_balanced_init']   = run('sgdm', init='balanced')
report['adam_balanced_init']   = run('adam', init='balanced')
report['adam_lora_init']       = run('adam', init='lora')
report['sgdm_lora_init']       = run('sgdm', init='lora')
report['adam_weightdecay_1e-2']= run('adam', init='balanced', wd=1e-2)
report['sgdm_weightdecay_1e-2']= run('sgdm', init='balanced', wd=1e-2)

print(json.dumps(report,indent=2,default=float))
