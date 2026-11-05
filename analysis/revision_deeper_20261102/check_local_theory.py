"""Independent local-dynamics and stochastic-efficiency checks, float64."""
from pathlib import Path
import json
import numpy as np
rng=np.random.default_rng(11903)
checks={}
def sqrt_spd(x):
 e,v=np.linalg.eigh((x+x.T)/2);return (v*np.sqrt(e))@v.T
m,d,r=5,4,2
u=np.linalg.qr(rng.normal(size=(m,m)))[0]
v=np.linalg.qr(rng.normal(size=(d,d)))[0]
spec=np.array([3.2,.7])
B=u[:,:r]@np.diag(np.sqrt(spec));A=np.diag(np.sqrt(spec))@v[:,:r].T
X=B@A
target=X+.04*u[:,r:]@rng.normal(size=(m-r,d-r))@v[:,r:].T
def join(b,a):return np.r_[b.reshape(-1),a.reshape(-1)]
def split(p):return p[:m*r].reshape(m,r),p[m*r:].reshape(r,d)
p0=join(B,A);Ndim=len(p0)
def vertical(p):
 b,a=split(p)
 return np.stack([join(b@om,-om@a) for om in np.eye(r*r).reshape(-1,r,r)],axis=1)
vv=vertical(p0)
orth=np.linalg.svd(vv,full_matrices=True)[0]
Vb=orth[:,:r*r];Hb=orth[:,r*r:];D=Hb.shape[1]
def balance(p):
 b,a=split(p);g=b.T@b;h=a@a.T
 sg=sqrt_spd(g);ig=np.linalg.inv(sg)
 pp=ig@sqrt_spd(sg@h@sg)@ig;q=sqrt_spd(pp)
 return join(b@q,np.linalg.solve(q,a)),q
def gf(p):
 b,a=split(p);g=b@a-target
 return join(g@a.T,b.T@g)
def dg(dp):
 db,da=split(dp);g=X-target;gg=db@A+B@da
 return join(gg@A.T+g@da.T,db.T@g+B.T@gg)
H=Hb.T@np.stack([dg(e) for e in Hb.T],axis=1)
H=(H+H.T)/2
lam,Q=np.linalg.eigh(H)
assert lam[0]>0
eta=.13;beta=.82
J=np.block([[np.eye(D)-eta*(1-beta)*H,-eta*beta*np.eye(D)],[(1-beta)*H,beta*np.eye(D)]])
def actual(state,eps):
 xp,mp,mv=state[:D],state[D:2*D],state[2*D:]
 p,_=balance(p0+Hb@xp)
 mem=Hb@mp+Vb@mv
 vg=vertical(p);normal=vg.T@vg
 if eps is None:project=np.eye(Ndim)
 else:project=np.eye(Ndim)-vg@np.linalg.solve(normal+2*eps*np.eye(r*r),vg.T)
 direction=project@(beta*mem+(1-beta)*gf(p))
 nxt=p-eta*direction;pn,qq=balance(nxt)
 mb,ma=split(direction)
 mom=join(mb@qq,np.linalg.solve(qq,ma))
 return np.r_[Hb.T@(pn-p0),Hb.T@mom,Vb.T@mom]
for eps in [0.,.2,None]:
 z=np.zeros(2*D+r*r);hh=2e-5
 deriv=np.stack([(actual(z+hh*e,eps)-actual(z-hh*e,eps))/(2*hh) for e in np.eye(len(z))],axis=1)
 if eps is None:pv=np.eye(r*r)
 else:
  pv=Vb.T@(np.eye(Ndim)-vv@np.linalg.solve(vv.T@vv+2*eps*np.eye(r*r),vv.T))@Vb
 exact=np.zeros_like(deriv);exact[:2*D,:2*D]=J;exact[2*D:,2*D:]=beta*pv
 key='balance_only' if eps is None else 'eps_'+str(eps)
 checks['actual_Jacobian_'+key+'_relative_error']=float(np.linalg.norm(deriv-exact)/np.linalg.norm(exact))
 checks['linear_radius_'+key]=float(np.max(np.abs(np.linalg.eigvals(exact))))
# Compare the complete residual-gap spectrum, including rectangular leftovers.
normal=u[:,r:].T@(target-X)@v[:,r:]
tau=np.linalg.svd(normal,compute_uv=False)
pred=[a+b for a in spec for b in spec]
pred += [a+sign*t for a in spec for t in tau for sign in [-1,1]]
pred += list(np.repeat(spec,abs(m-d)))
checks['complete_residual_gap_spectrum_error']=float(np.linalg.norm(lam-np.sort(pred)))
mu,La=lam[0],lam[-1]
qq=(np.sqrt(La)-np.sqrt(mu))/(np.sqrt(La)+np.sqrt(mu))
bet=qq**2;et=1/np.sqrt(mu*La);eta_opt=et
radii=[]
for ll in np.linspace(mu,La,501):
 jm=np.array([[1-et*(1-bet)*ll,-et*bet],[(1-bet)*ll,bet]])
 radii.append(max(abs(np.linalg.eigvals(jm))))
checks['optimal_interval_radius_error']=float(abs(max(radii)-qq))
for ll,z0 in [(mu,qq),(La,-qq)]:
 jm=np.array([[1-et*(1-bet)*ll,-et*bet],[(1-bet)*ll,bet]])
 k=11;formula=z0**k*np.eye(2)+k*z0**(k-1)*(jm-z0*np.eye(2))
 checks['Jordan_endpoint_'+str(round(ll,3))+'_error']=float(np.linalg.norm(np.linalg.matrix_power(jm,k)-formula))
# A noncommuting noise covariance in a three-mode quadratic system.
hh=np.diag([.7,1.6,3.1]);dd=3
cc0=rng.normal(size=(dd,dd));cc=cc0@cc0.T+.2*np.eye(dd)
be=.73;et=.22;al=et*(1-be)
jj=np.block([[np.eye(dd)-al*hh,-et*be*np.eye(dd)],[(1-be)*hh,be*np.eye(dd)]])
nn=np.vstack([-al*np.eye(dd),(1-be)*np.eye(dd)])
rhs=nn@cc@nn.T;sz=2*dd
ss=np.linalg.solve(np.eye(sz*sz)-np.kron(jj,jj),rhs.reshape(-1,order='F')).reshape(sz,sz,order='F')
expected=np.array([et*(1+be)*cc[i,i]/(hh[i,i]*(2*(1+be)-et*(1-be)*hh[i,i])) for i in range(dd)])
checks['stationary_position_variance_error']=float(np.linalg.norm(np.diag(ss[:dd,:dd])-expected))
mv=np.array([2*(1-be)*cc[i,i]/(2*(1+be)-et*(1-be)*hh[i,i]) for i in range(dd)])
checks['stationary_momentum_variance_error']=float(np.linalg.norm(np.diag(ss[dd:,dd:])-mv))
cross=-(1-be)*np.diag(hh)/(1+be)*expected
checks['stationary_cross_covariance_error']=float(np.linalg.norm(np.diag(ss[:dd,dd:])-cross))
pp=np.linalg.solve(np.eye(sz*sz)-np.kron(jj.T,jj.T),np.eye(sz).reshape(-1,order='F')).reshape(sz,sz,order='F')
checks['Lyapunov_identity_error']=float(np.linalg.norm(jj.T@pp@jj-pp+np.eye(sz)))
assert np.linalg.eigvalsh(pp)[0]>.99
qp=np.sqrt(1-1/np.linalg.eigvalsh(pp)[-1])
for _ in range(100):
 a0=rng.normal(size=sz);a1=jj@a0
 assert np.sqrt(a1@pp@a1/(a0@pp@a0))<=qp+1e-12
# Exact averaging boundary identity.
xx=rng.normal(size=dd);mm=rng.normal(size=dd);xinit=xx.copy();minit=mm.copy();sumx=np.zeros(dd);sumz=np.zeros(dd)
for t in range(123):
 zz=rng.normal(size=dd);sumz+=zz;sumx+=xx
 mm=be*mm+(1-be)*(hh@xx+zz);xx=xx-et*mm
checks['averaging_boundary_identity_error']=float(np.linalg.norm(hh@sumx+sumz-(xinit-xx)/et-be/(1-be)*(mm-minit)))
filter0=np.linalg.solve(np.eye(sz)-jj,nn)[:dd]
checks['zero_frequency_filter_error']=float(np.linalg.norm(filter0+np.linalg.inv(hh)))
# Gauge-singular Fisher covariance from the full factor differential.
def Lop(p):
 b,a=split(p)
 return np.stack([(db@a+b@da).reshape(-1) for db,da in map(split,np.eye(Ndim))],axis=1)
l0=Lop(p0);PT=l0@np.linalg.pinv(l0)
for name,qg in [('balanced',np.eye(r)),('nonorthogonal',np.array([[4.,1.],[0.,.4]]))]:
 pg=join(B@qg,np.linalg.solve(qg,A));ll=Lop(pg)
 identifiable=ll@np.linalg.pinv(ll.T@ll,rcond=1e-11)@ll.T
 checks['Fisher_product_projector_'+name+'_error']=float(np.linalg.norm(identifiable-PT))
 checks['Fisher_product_trace_'+name]=float(np.trace(identifiable))
 assert np.linalg.matrix_rank(ll)==D
# Nonlinear local trajectory, with a nonzero ambient normal residual.
beta=bet;eta=eta_opt;p=p0.copy();p,_=balance(p0+Hb@(rng.normal(size=D)*1e-4));mem=np.zeros(Ndim)
b,a=split(p);finit=.5*np.linalg.norm(b@a-target)**2-.5*np.linalg.norm(X-target)**2
for t in range(60):
 vp=vertical(p);pr=np.eye(Ndim)-vp@np.linalg.solve(vp.T@vp,vp.T)
 direction=pr@(beta*mem+(1-beta)*gf(p))
 p,qq=balance(p-eta*direction);mb,ma=split(direction);mem=join(mb@qq,np.linalg.solve(qq,ma))
b,a=split(p)
checks['nonlinear_local_final_product_error']=float(np.linalg.norm(b@a-X))
checks['intrinsic_dimension']=D
for k,val in checks.items():
 if k.endswith(('error','relative_error')):assert val<2e-7,(k,val)
assert checks['nonlinear_local_final_product_error']<1e-10
Path(__file__).with_name('local_theory_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
