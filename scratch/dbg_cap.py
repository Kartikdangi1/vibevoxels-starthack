import sys; sys.path.insert(0,'verification')
from common import *
import scipy.sparse as sps, scipy.sparse.linalg as spl
S,opt=uifo(2,random=True,random_seed=201)
f=jnp.asarray(np.geomspace(5,5000,4))
dense,sparse,_=build_arrays(S,[("f","frequency")],f,opt)
cap=[]
orig=sim.sparse_solve
def wrap(values,rows,cols,b,c):
    out=orig(values,rows,cols,b,c); cap.append((values,rows,cols,b,c,out)); return out
sim.sparse_solve=wrap
set_backend("jax")
res=sim.simulate_sparse_jax(**sparse)
print("noise",float(np.asarray(res[2]).ravel()[0]))
for k,(v,r,c_,b,cc,out) in enumerate(cap):
    n=int(max(np.asarray(r).max(),np.asarray(c_).max())+1)
    print("call",k,"values",v.shape,"b",b.shape,"c",cc.shape,"n",n)
    if k==1:
        A=np.zeros((n,n),complex); A[np.asarray(r),np.asarray(c_)]=np.asarray(v[0])
        x,w=np.asarray(out[0]),np.asarray(out[1])
        print(" x shape",x.shape,"w shape",w.shape)
        cvec=np.asarray(cc)[0,0]
        wt=spl.splu(sps.csc_matrix(A)).solve(cvec.astype(complex),trans="H")
        print(" w[d=0,f=0] vs true: rel diff",np.abs(w[0,0]-wt).max()/np.abs(wt).max(), "| residual of returned w", np.abs(A.conj().T@w[0,0]-cvec).max(), "true resid",np.abs(A.conj().T@wt-cvec).max())
        print(" |c|max",np.abs(cvec).max())
