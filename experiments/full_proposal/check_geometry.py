"""Numerical regression checks of the nontrivial theory and coordinate fix."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_sylvester,null_space
import torch
from geometry_torch import LowRankLinear,FactorOptimizer
from grid import ROOT

rng=np.random.default_rng(42);m,d,r=7,6,3
u=np.linalg.qr(rng.normal(size=(m,r)))[0];v=np.linalg.qr(rng.normal(size=(d,r)))[0]
s=np.array([2.,.7,.2]);B=u*np.sqrt(s);A=np.sqrt(s)[:,None]*v.T;H=np.diag(s)
xB=rng.normal(size=B.shape);xA=rng.normal(size=A.shape)
C=B.T@xB-xA@A.T;omega=solve_sylvester(H,H,C);hB=xB-B@omega;hA=xA+omega@A
checks={}
checks['horizontal_residual']=float(np.linalg.norm(B.T@hB-hA@A.T))
checks['differential_preservation']=float(np.linalg.norm(hB@A+B@hA-(xB@A+B@xA)))
eta=.013
actual=(B-eta*xB)@(A-eta*xA)-(B-eta*hB)@(A-eta*hA)
pred=eta**2*(-hB@omega@A+B@omega@hA-B@omega@omega@A)
checks['finite_step_cross_term_identity']=float(np.linalg.norm(actual-pred))
eps=.03;om=solve_sylvester(H+eps*np.eye(r),H+eps*np.eye(r),C)
checks['damping_residual_identity']=float(np.linalg.norm(B.T@(xB-B@om)-(xA+om@A)@A.T-2*eps*om))
checks['factor_condition_lock']=float(abs(np.sqrt(np.linalg.cond(B)*np.linalg.cond(A))/np.sqrt(s[0]/s[-1])-1))
# Hessian of full-observation reconstruction at an exact solution is L*L.
n=B.size+A.size;columns=[]
for i in range(n):
    e=np.eye(n)[i];db=e[:B.size].reshape(B.shape);da=e[B.size:].reshape(A.shape)
    columns.append((db@A+B@da).ravel())
L=np.stack(columns,1);eig=np.linalg.eigvalsh(L.T@L)
checks['hessian_gauge_nullity']=int((abs(eig)<1e-10).sum())
checks['expected_gauge_nullity']=r*r
positive=np.linalg.eigvalsh(L@L.T);positive=positive[positive>1e-9]
expected=np.concatenate([(s[:,None]+s[None,:]).ravel(),np.repeat(s,d-r),np.repeat(s,m-r)])
checks['product_metric_spectrum_error']=float(np.max(np.abs(np.sort(positive)-np.sort(expected))))
# At fixed BA, sine activation does not distinguish balanced factors.
from geometry_torch import balance
torch.manual_seed(42);layer=LowRankLinear(6,7,3)
with torch.no_grad():layer.B.mul_(20);layer.A.div_(20)
z=torch.randn(10,6);before=torch.sin(30*layer(z)).detach();b,a,_=balance(layer.B,layer.A)
with torch.no_grad():layer.B.copy_(b);layer.A.copy_(a)
checks['siren_product_invariance_error']=float((torch.sin(30*layer(z))-before).abs().max())
# A gradient at a new section must come from a new backward pass.
net=LowRankLinear(6,7,3);c=dict(method='quotientopt',beta=.9,lr=.001,eps=1e-6,balance_interval=1)
opt=FactorOptimizer(net,c);opt.prepare();loss=(net(z)**2).mean();loss.backward()
checks['fresh_gradient_horizontal_residual']=float((net.B.T@net.B.grad-net.A.grad@net.A.T).norm())
checks['all_pass']=all(checks[k]<1e-10 for k in ['horizontal_residual','differential_preservation','finite_step_cross_term_identity','damping_residual_identity','factor_condition_lock','product_metric_spectrum_error']) and checks['hessian_gauge_nullity']==r*r and checks['siren_product_invariance_error']<1e-4 and checks['fresh_gradient_horizontal_residual']<1e-4
(ROOT/'geometry_checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
if not checks['all_pass']:raise SystemExit(1)
