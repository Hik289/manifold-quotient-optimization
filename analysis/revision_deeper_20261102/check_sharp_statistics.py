"""Checks of equality cases and local statistical leading constants."""
import json
from pathlib import Path
import numpy as np
rng=np.random.default_rng(57021)
out={}
D=7;n=120
a=rng.normal(size=(D,D));H=a@a.T+.8*np.eye(D);s=rng.normal(size=D)/np.sqrt(n)
theta=-np.linalg.solve(H,s)
risk=.5*theta@H@theta
decrement=s@np.linalg.solve(H,s)
out['quadratic_Newton_risk_equality_error']=float(abs(risk-.5*decrement))
# Coordinate invariance of the effective statistical dimension.
b=rng.normal(size=(D,D));C=b@b.T
T=np.diag(np.linspace(.3,2.,D))+rng.normal(size=(D,D))*.05
invT=np.linalg.inv(T);Hy=invT.T@H@invT;Cy=invT.T@C@invT
out['covariance_curvature_trace_invariance_error']=float(abs(np.trace(np.linalg.solve(H,C))-np.trace(np.linalg.solve(Hy,Cy))))
# Laurent--Massart / HKZ Chernoff substitution, including an anisotropic proxy.
vv=np.array([.08,.2,.7,1.1,2.])
v=np.sum(vv**2);c=2*max(vv)
err=[]
for u in [.01,.3,1.,5.,12.]:
 t=np.sqrt(u)/(np.sqrt(v)+c*np.sqrt(u));x=2*np.sqrt(v*u)+c*u
 err.append(abs(t*x-v*t*t/(1-c*t)-u))
out['quadratic_tail_Chernoff_optimization_error']=float(max(err))
# Complete residual-gap spectrum in a strict-saddle example.
# A retained singular value 0.8 is below an omitted singular value 1.4.
m,d,r=4,5,2
spec=np.array([2.6,.8]);N=np.zeros((m-r,d-r));N[0,0]=1.4;N[1,1]=.3
B=np.vstack([np.diag(np.sqrt(spec)),np.zeros((m-r,r))])
A=np.hstack([np.diag(np.sqrt(spec)),np.zeros((r,d-r))])
X=B@A;target=X.copy();target[r:,r:]=N
def join(b,a):return np.r_[b.reshape(-1),a.reshape(-1)]
def split(p):return p[:m*r].reshape(m,r),p[m*r:].reshape(r,d)
nf=m*r+r*d
vert=np.stack([join(B@om,-om@A) for om in np.eye(r*r).reshape(-1,r,r)],axis=1)
Q=np.linalg.svd(vert,full_matrices=True)[0][:,r*r:]
def hess(dp):
 db,da=split(dp);g=X-target;dg=db@A+B@da
 return join(dg@A.T+g@da.T,db.T@g+B.T@dg)
h=Q.T@np.stack([hess(e) for e in Q.T],axis=1)
tau=np.linalg.svd(N,compute_uv=False)
pred=[x+y for x in spec for y in spec]
pred += [x+sg*t for x in spec for t in tau for sg in [-1,1]]
pred += list(np.repeat(spec,abs(m-d)))
measured=np.linalg.eigvalsh((h+h.T)/2)
out['strict_saddle_full_spectrum_error']=float(np.linalg.norm(measured-np.sort(pred)))
out['strict_saddle_negative_index']=int(np.sum(measured<-1e-9))
assert out['strict_saddle_negative_index']==sum(t>x for x in spec for t in tau)
# Matrix denoising ERM: sample average followed by the exact rank-one SVD.
# This verifies leading constants, not any task-family performance.
sigma=.2;trials=40000
truth=np.array([[2.5,0.],[0.,0.]])
results=[]
for sample_n in [64,256,1024]:
 mean=truth+rng.normal(size=(trials,2,2))*(sigma/np.sqrt(sample_n))
 u,sv,vt=np.linalg.svd(mean,full_matrices=False)
 hat=(u[:,:,:1]*sv[:,:1,None])@vt[:,:1,:]
 population=.5*np.sum((hat-truth)**2,axis=(1,2))
 fit_residual=.5*np.sum((hat-mean)**2,axis=(1,2))
 base_residual=.5*np.sum((truth-mean)**2,axis=(1,2))
 scaled=sample_n*population
 optimism=sample_n*(population-fit_residual)+2*sigma*sigma
 results.append({'n':sample_n,'n_mean_excess':float(np.mean(scaled)),
 'n_mean_training_improvement':float(np.mean(sample_n*(fit_residual-base_residual))),
 'n_mean_generalization':float(np.mean(optimism)),
 'excess_Monte_Carlo_SE':float(np.std(scaled,ddof=1)/np.sqrt(trials))})
out['rank1_denoising_local_ERM']=results
out['theoretical_n_excess_limit']=3*sigma*sigma/2
out['theoretical_n_generalization_limit']=3*sigma*sigma
for row in results:
 assert abs(row['n_mean_excess']-out['theoretical_n_excess_limit'])<6*row['excess_Monte_Carlo_SE']
# General identifiable design: covariance survives in Frobenius error
# but cancels exactly in the design-weighted prediction risk.
lf=np.stack([(db@A+B@da).reshape(-1) for db,da in map(split,np.eye(nf))],axis=1)
uu,sv0,vvt=np.linalg.svd(lf,full_matrices=False)
dd=int(np.sum(sv0>1e-10));uu=uu[:,:dd]
zz=rng.normal(size=(m*d,m*d));MM=zz.T@zz+.7*np.eye(m*d)
fisher=lf.T@MM@lf/(sigma*sigma)
prod=lf@np.linalg.pinv(fisher,rcond=1e-11)@lf.T
expected=sigma*sigma*uu@np.linalg.inv(uu.T@MM@uu)@uu.T
out['general_design_Fisher_covariance_error']=float(np.linalg.norm(prod-expected))
out['general_design_prediction_trace_error']=float(abs(np.trace(MM@prod)-sigma*sigma*dd))
# A globally valid RIP perturbation with delta = .2.
eigvec=np.linalg.qr(rng.normal(size=(m*d,m*d)))[0]
delta=.2
MMrip=eigvec@np.diag(np.linspace(1-delta,1+delta,m*d))@eigvec.T
bp=B+.004*rng.normal(size=B.shape);ap=A+.004*rng.normal(size=A.shape)
errmat=X-bp@ap;GG=(MMrip@errmat.reshape(-1)).reshape(m,d)
def rhess(dp):
 db,da=split(dp);dg=(MMrip@(db@A+B@da).reshape(-1)).reshape(m,d)
 return join(dg@A.T+GG@da.T,db.T@GG+B.T@dg)
HH=Q.T@np.stack([rhess(e) for e in Q.T],axis=1)
actual_eigs=np.linalg.eigvalsh((HH+HH.T)/2)
omega=np.linalg.norm(errmat,2)+delta*np.linalg.norm(errmat)
lower=(1-delta)*min(spec)-omega;upper=(1+delta)*2*max(spec)+omega
out['sharp_RIP_Hessian_interval']={'lower_bound':float(lower),'actual_min':float(min(actual_eigs)),
 'actual_max':float(max(actual_eigs)),'upper_bound':float(upper)}
assert lower<=min(actual_eigs) and max(actual_eigs)<=upper
for k,v in out.items():
 if k.endswith('_error'):assert v<1e-10,(k,v)
Path(__file__).with_name('sharp_statistics_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
