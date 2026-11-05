"""Finite-difference check of smooth balance and the complete state Jacobian."""
import json
from pathlib import Path
import numpy as np
rng=np.random.default_rng(241)
m,d,r=5,4,2
def sq(x):
 v,u=np.linalg.eigh(x);return (u*np.sqrt(v))@u.T
def syl(h,c):
 v,u=np.linalg.eigh(h);return u@((u.T@c@u)/(v[:,None]+v[None,:]))@u.T
def join(b,a):return np.r_[b.reshape(-1),a.reshape(-1)]
def split(p):return p[:m*r].reshape(m,r),p[m*r:].reshape(r,d)
def vert(p):
 b,a=split(p)
 return np.stack([join(b@om,-om@a) for om in np.eye(r*r).reshape(-1,r,r)],axis=1)
def proj(p):
 v=vert(p);return np.eye(len(p))-v@np.linalg.solve(v.T@v,v.T)
def balance(p):
 b,a=split(p);gb=b.T@b;ga=a@a.T
 sr=sq(gb);ir=np.linalg.inv(sr);pp=ir@sq(sr@ga@sr)@ir;q=sq(pp)
 return join(b@q,np.linalg.solve(q,a)),q,pp
def dq(p,dp):
 b,a=split(p);db,da=split(dp);gb=b.T@b;ga=a@a.T
 _,q,pp=balance(p);sr=sq(gb);ir=np.linalg.inv(sr);tt=sr@pp@sr
 dr=da@a.T+a@da.T-pp@(b.T@db+db.T@b)@pp
 dc=syl(tt,sr@dr@sr);dpp=ir@dc@ir
 return syl(q,dpp)
def dbal(p,dp):
 b,a=split(p);db,da=split(dp);_,q,_=balance(p);qq=dq(p,dp);iq=np.linalg.inv(q)
 return join(db@q+b@qq,iq@da-iq@qq@iq@a)
def dproj(p,dp):
 v=vert(p);dv=vert(dp);pi=proj(p);inv=np.linalg.solve(v.T@v,v.T)
 return -pi@dv@inv-inv.T@dv.T@pi
b=rng.normal(size=(m,r));a=rng.normal(size=(r,d));p=join(b,a)
dp=rng.normal(size=len(p));dp/=np.linalg.norm(dp)
hh=1e-5
results={}
_,q,_=balance(p)
results['SPD_gauge_differential_error']=float(np.linalg.norm((balance(p+hh*dp)[1]-balance(p-hh*dp)[1])/(2*hh)-dq(p,dp)))
results['balance_differential_error']=float(np.linalg.norm((balance(p+hh*dp)[0]-balance(p-hh*dp)[0])/(2*hh)-dbal(p,dp)))
results['horizontal_projector_differential_error']=float(np.linalg.norm((proj(p+hh*dp)-proj(p-hh*dp))/(2*hh)-dproj(p,dp)))
p=balance(p)[0];b,a=split(p);target=rng.normal(size=(m,d))
beta=.9;eta=.007
mem=proj(p)@rng.normal(size=len(p))
dm=rng.normal(size=len(p));dm/=np.linalg.norm(dm)
def grad(p):
 b,a=split(p);g=b@a-target
 return join(g@a.T,b.T@g)
def dgrad(p,dp):
 b,a=split(p);db,da=split(dp);g=b@a-target;dg=db@a+b@da
 return join(dg@a.T+g@da.T,db.T@g+b.T@dg)
def act(p,mem):
 vv=proj(p)@(beta*mem+(1-beta)*grad(p))
 zz=p-eta*vv;pp,qq,_=balance(zz);vb,va=split(vv)
 w=join(vb@qq,np.linalg.solve(qq,va))
 return np.r_[pp,proj(pp)@w]
u=beta*mem+(1-beta)*grad(p);vv=proj(p)@u
dv=dproj(p,dp)@u+proj(p)@(beta*dm+(1-beta)*dgrad(p,dp))
zz=p-eta*vv;dz=dp-eta*dv;pp,qq,_=balance(zz);dqq=dq(zz,dz);dpp=dbal(zz,dz)
vb,va=split(vv);dvb,dva=split(dv);iq=np.linalg.inv(qq)
w=join(vb@qq,iq@va)
dw=join(dvb@qq+vb@dqq,iq@dva-iq@dqq@iq@va)
dmp=dproj(pp,dpp)@w+proj(pp)@dw
exact=np.r_[dpp,dmp]
numeric=(act(p+hh*dp,mem+hh*dm)-act(p-hh*dp,mem-hh*dm))/(2*hh)
results['complete_state_Jacobian_relative_error']=float(np.linalg.norm(exact-numeric)/np.linalg.norm(exact))
for name,value in results.items():assert value<1e-7,(name,value)
(Path(__file__).with_name('state_derivation_checks.json')).write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
