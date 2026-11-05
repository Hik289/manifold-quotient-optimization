#!/usr/bin/env python3
"""Root-cause verification: are §10.5/10.6/10.7 failures damping bias (theory)
or bugs?  Re-run the horizontal-projection checks with eps=0 (exact connection)
and eps=1e-6, and directly compare 10.5 residual against the predicted 2*eps*||Omega||.
Also fix 10.11 with a centered finite difference to show it was truncation error."""
import numpy as np
from scipy.linalg import solve_sylvester
import json
RNG=np.random.default_rng(42); F32=np.float32
def fn(M): return float(np.linalg.norm(M,'fro'))
def balance(B,A):
    B=B.astype(F32);A=A.astype(F32)
    QB,RB=np.linalg.qr(B); QA,RA=np.linalg.qr(A.T)
    U,S,Vt=np.linalg.svd((RB@RA.T).astype(F32)); S12=np.sqrt(S).astype(F32)
    return (QB@U@np.diag(S12)).astype(F32),(np.diag(S12)@Vt@QA.T).astype(F32)
def conn(B,A,xB,xA,eps):
    H=(0.5*(B.T@B+A@A.T)).astype(np.float64); He=H+eps*np.eye(H.shape[0])
    C=(B.T@xB-xA@A.T).astype(np.float64)
    Om=solve_sylvester(He,He,C)
    return Om.astype(F32)
def hlift(B,A,xB,xA,eps):
    Om=conn(B,A,xB,xA,eps)
    return (xB-B@Om).astype(F32),(xA+Om@A).astype(F32),Om
def make(m,d,r,cond,gauge=None):
    U=np.linalg.qr(RNG.standard_normal((m,r)))[0]; V=np.linalg.qr(RNG.standard_normal((d,r)))[0]
    s=np.logspace(0,-np.log10(cond),r) if r>1 else np.array([1.0])
    B=(U*np.sqrt(s)).astype(F32); A=(np.sqrt(s)[:,None]*V.T).astype(F32)
    if gauge is not None: B=(B@gauge).astype(F32); A=(np.linalg.inv(gauge)@A).astype(F32)
    return B,A
out={}
for cond in [1.,10.,100.,1000.]:
    g=(np.eye(8)+0.3*RNG.standard_normal((8,8))).astype(F32)
    B,A=make(64,48,8,cond,gauge=g); Bb,Ab=balance(B,A)
    gB=RNG.standard_normal((64,8)).astype(F32); gA=RNG.standard_normal((8,48)).astype(F32)
    rec={}
    for eps in [0.0,1e-6]:
        hB,hA,Om=hlift(Bb,Ab,gB,gA,eps)
        r105=fn(Bb.T@hB-hA@Ab.T)
        # idempotence
        hB2,hA2,_=hlift(Bb,Ab,hB,hA,eps); r106=fn(hB2-hB)+fn(hA2-hA)
        # pure vertical -> 0
        Om0=RNG.standard_normal((8,8)).astype(F32)
        vB=(Bb@Om0).astype(F32); vA=(-Om0@Ab).astype(F32)
        pB,pA,_=hlift(Bb,Ab,vB,vA,eps); r107=fn(pB)+fn(pA)
        rec[f'eps={eps:g}']=dict(r105=r105,r106=r106,r107=r107,
                                 omega_norm=fn(Om),pred_2eps_omega=2*eps*fn(Om))
    out[f'cond={cond:g}']=rec
# 10.11 centered FD, small t
Bb,Ab=balance(*make(64,48,8,1.0))
gB=RNG.standard_normal((64,8)).astype(np.float64); gA=RNG.standard_normal((8,48)).astype(np.float64)
Bb64=Bb.astype(np.float64); Ab64=Ab.astype(np.float64)
dpi=gB@Ab64+Bb64@gA
fd_res={}
for t in [1e-4,1e-5,1e-6]:
    fd=((Bb64+t*gB)@(Ab64+t*gA)-(Bb64-t*gB)@(Ab64-t*gA))/(2*t)  # centered, exact for bilinear
    fd_res[f't={t:g}']=fn(fd-dpi)/(fn(dpi)+1)
out['10.11_centered_fd_float64']=fd_res
print(json.dumps(out,indent=2,default=float))
