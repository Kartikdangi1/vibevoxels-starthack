import sys; sys.path.insert(0,'/work/verification')
from common import *
case=load_public(8); a,p,x,ref=case['arrays'],case['ports'],case['params'],jnp.asarray(case['ref'])
set_backend("host")
L=jax.jit(lambda q: uifo_loss(q,a,p,ref)[0]); g=np.asarray(jax.grad(lambda q: uifo_loss(q,a,p,ref)[0])(x))
for i in (15,131,31,17,2): print(i,case['pairs'][i],repr(float(x[i])),g[i])
print("x range", float(x.min()), float(x.max()), "|x|", float(jnp.linalg.norm(x)), "|g|", np.linalg.norm(g))
for i in (15,131):
  for h in (1e-4,1e-6,1e-8,1e-10,1e-12):
    print(i,h,(float(L(x.at[i].add(h)))-float(L(x.at[i].add(-h))))/(2*h), "fwd",(float(L(x.at[i].add(h)))-float(L(x)))/h)
d=jnp.asarray(g)/np.linalg.norm(g)
for t in (1e-12,1e-10,1e-9,1e-8,1e-7): print("step",t,float(L(x-t*d))-float(L(x)), "pred",-t*np.linalg.norm(g))
