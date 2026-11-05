"""Independent float64 checks of the manuscript's new differential identities."""
import json
from pathlib import Path
import numpy as np
rng=np.random.default_rng(918)
out={}
m,d,r=5,4,2
U=np.linalg.qr(rng.normal(size=(m,m)))[0]
V=np.linalg.qr(rng.normal(size=(d,d)))[0]
s=np.array([3.2,.7])
X=U[:,:r]@np.diag(s)@V[:,:r].T
Uf,Vf=U[:,:r],V[:,:r]
B=Uf@np.diag(np.sqrt(s)); A=np.diag(np.sqrt(s))@Vf.T
def vec(x):return x.reshape(-1)
def mat(x):return x.reshape(m,d)
def op(fun):
 return np.stack([vec(fun(mat(e))) for e in np.eye(m*d)],axis=1)
def KP(x):
 u,sv,vt=np.linalg.svd(x,full_matrices=True)
 sl=(u[:,:r]*sv[:r])@u[:,:r].T
 sr=(vt[:r].T*sv[:r])@vt[:r]
 return op(lambda y:sl@y+y@sr)
K=KP(X); AK=np.linalg.pinv(K,rcond=1e-10); P=K@AK
basis=[]
for i in range(m):
 for j in range(d):
  if i<r or j<r:
   e=np.zeros((m,d));e[i,j]=1
   basis.append(vec(U@e@V.T))
E=np.stack(basis,axis=1)
def blocks(z):return U.T@z@V
def curve(z,t):
 z0=blocks(z);c=np.diag(s)+t*z0[:r,:r];dd=t*z0[:r,r:];ee=t*z0[r:,:r]
 return U@np.block([[c,dd],[ee,ee@np.linalg.solve(c,dd)]])@V.T
def DK(z):
 z0=blocks(z);zz=z0[:r,:r]
 tl=(zz*s[None,:]+s[:,None]*zz.T)/(s[:,None]+s[None,:])
 tr=(zz.T*s[None,:]+s[:,None]*zz)/(s[:,None]+s[None,:])
 dl=U@np.block([[tl,z0[r:,:r].T],[z0[r:,:r],np.zeros((m-r,m-r))]])@U.T
 dr=V@np.block([[tr,z0[:r,r:]],[z0[:r,r:].T,np.zeros((d-r,d-r))]])@V.T
 return op(lambda y:dl@y+y@dr)
def dP(z,y):
 z0=blocks(z)
 pu=U[:,r:]@(z0[r:,:r]/s[None,:])@Uf.T
 pu=pu+pu.T
 pv=V[:,r:]@(z0[:r,r:].T/s[None,:])@Vf.T
 pv=pv+pv.T
 return pu@y@(np.eye(d)-Vf@Vf.T)+(np.eye(m)-Uf@Uf.T)@y@pv
def dA(z):
 dk=DK(z)
 return -AK@dk@AK+AK@AK@dk@(np.eye(m*d)-P)+(np.eye(m*d)-P)@dk@AK@AK
def nablaA(z):return P@dA(z)@P
def conn(z,y):
 zv,yv=vec(z),vec(y)
 q=E@np.array([zv@nablaA(mat(e))@yv for e in E.T])
 return mat(.5*K@(nablaA(z)@yv+nablaA(y)@zv-q))
z=mat(E@rng.normal(size=E.shape[1]));z/=np.linalg.norm(z)
y=mat(E@rng.normal(size=E.shape[1]));y/=np.linalg.norm(y)
step=1e-5
kp,km=KP(curve(z,step)),KP(curve(z,-step))
out["DK_finite_difference_relative"]=float(np.linalg.norm((kp-km)/(2*step)-DK(z))/np.linalg.norm(DK(z)))
out["D_pseudoinverse_finite_difference_relative"]=float(np.linalg.norm((np.linalg.pinv(kp,rcond=1e-10)-np.linalg.pinv(km,rcond=1e-10))/(2*step)-dA(z))/np.linalg.norm(dA(z)))
out["D_projector_finite_difference_error"]=float(np.linalg.norm(((kp@np.linalg.pinv(kp,rcond=1e-10)-km@np.linalg.pinv(km,rcond=1e-10))/(2*step))@vec(y)-vec(dP(z,y))))
gmat=mat(AK@vec(z));xiB=gmat@A.T;xiA=B.T@gmat
out["retraction_covariant_acceleration"]=float(np.linalg.norm(mat(P@vec(2*xiB@xiA))+conn(z,z)))
G=rng.normal(size=(m,d))
hess=mat(P@(DK(z)@vec(G)+K@vec(z)))+conn(z,mat(K@vec(G)))
bilinear=vec(z)@vec(y)+np.sum(G*(dP(z,y)-conn(z,y)))
out["Hessian_bilinear_identity_error"]=float(abs(vec(hess)@AK@vec(y)-bilinear))
out["connection_torsion_error"]=float(np.linalg.norm(conn(z,y)-conn(y,z)))
derivmetric=vec(y)@nablaA(z)@vec(y)
out["connection_metric_compatibility_error"]=float(abs(derivmetric-2*vec(conn(z,y))@AK@vec(y)))
om=rng.normal(size=(r,r));eta=.013
uB=xiB+B@om;uA=xiA-om@A
drift=(B-eta*uB)@(A-eta*uA)-(B-eta*xiB)@(A-eta*xiA)
expr=eta**2*(-xiB@om@A+B@om@xiA-B@om@om@A)
out["mixed_finite_step_identity_error"]=float(np.linalg.norm(drift-expr))
for L in [2,3,5,9]:
 rr=3; ss=np.array([4.,1.4,.3]); dd=ss**(1/L); D=np.diag(dd)
 ws=[D.copy() for _ in range(L)]
 def vertical(om):
  return [-om[0]@D]+[D@om[i-1]-om[i]@D for i in range(1,L-1)]+[D@om[-1]]
 def Napply(om):
  v=vertical(om)
  return np.stack([D@v[i+1]-v[i]@D for i in range(L-1)])
 size=(L-1)*rr*rr
 nmat=np.stack([Napply(e.reshape(L-1,rr,rr)).reshape(-1) for e in np.eye(size)],axis=1)
 lam=np.array([dd[a]**2+dd[b]**2-2*dd[a]*dd[b]*np.cos(k*np.pi/L) for k in range(1,L) for a in range(rr) for b in range(rr)])
 out[f"L{L}_normal_eigenspectrum_error"]=float(np.max(np.abs(np.linalg.eigvalsh(nmat)-np.sort(lam))))
 out[f"L{L}_condition_relative_error"]=float(abs(np.linalg.cond(nmat)/((ss[0]/ss[-1])**(2/L)/np.tan(np.pi/(2*L))**2)-1))
 sine=np.sqrt(2/L)*np.sin(np.arange(1,L)[:,None]*np.arange(1,L)[None,:]*np.pi/L)
 rhs=rng.normal(size=(L-1,rr,rr))
 inv=np.zeros_like(rhs)
 for a in range(rr):
  for b in range(rr):
   ev=dd[a]**2+dd[b]**2-2*dd[a]*dd[b]*np.cos(np.arange(1,L)*np.pi/L)
   inv[:,a,b]=sine@((sine.T@rhs[:,a,b])/ev)
 out[f"L{L}_DST_residual"]=float(np.linalg.norm(Napply(inv)-rhs)/np.linalg.norm(rhs))
 def jj(g):
  return sum(np.linalg.matrix_power(D,L-i-1)@np.linalg.matrix_power(D,L-i-1)@g@np.linalg.matrix_power(D,i)@np.linalg.matrix_power(D,i) for i in range(L))
 eig=[]
 for a in range(rr):
  for b in range(rr):
   e=np.zeros((rr,rr));e[a,b]=1
   exact=sum(ss[a]**(2*(L-i-1)/L)*ss[b]**(2*i/L) for i in range(L))
   eig.append(abs(jj(e)[a,b]-exact))
 out[f"L{L}_product_eigenvalue_error"]=float(max(eig))
delta=np.linspace(1e-7,.5,5000)
assert np.all(np.sqrt(1+delta/2)/(1-delta/16)<=np.sqrt(1+delta))
assert np.all(np.sqrt(1-delta/2)-delta/16*np.sqrt(1+delta)>=np.sqrt(1-delta))
out["Gaussian_RIP_scalar_inequalities"]=True
rho=np.sqrt(s[-1])/3
smallest_ratio=1e6
for _ in range(100):
 cs=rng.normal(size=(2,E.shape[1]))
 cs*=rho/(2*np.linalg.norm(cs,axis=1)[:,None])
 lifts=[]
 for c in cs:
  zz=mat(E@c); zzlift=mat(AK@vec(zz))
  xb=zzlift@A.T;xa=B.T@zzlift
  q=np.sqrt(np.sum(xb*xb)+np.sum(xa*xa));xb*=rho*.8/q;xa*=rho*.8/q
  lifts.append((xb,xa))
 (xb,xa),(yb,ya)=lifts
 ratio=np.linalg.norm((B+xb)@(A+xa)-(B+yb)@(A+ya))/np.sqrt(np.sum((xb-yb)**2)+np.sum((xa-ya)**2))
 smallest_ratio=min(smallest_ratio,ratio)
assert smallest_ratio>=np.sqrt(s[-1])-rho
out["retraction_inverse_sampled_min_ratio"]=float(smallest_ratio)
out["retraction_inverse_required_lower"]=float(np.sqrt(s[-1])-rho)
tolerance=1e-7
for name,value in out.items():
 if name.endswith(("error","relative","residual","acceleration")):
  assert value<tolerance,(name,value)
p=Path(__file__).with_name("derivation_checks.json")
p.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
