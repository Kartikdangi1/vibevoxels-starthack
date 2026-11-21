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
acap=[]
orig_a=sim._assemble
def wa(p,v,g):
    o=orig_a(p,v,g); acap.append((p,v,g,o)); return o
sim._assemble=wa
set_backend("jax")
res=sim.simulate_sparse_jax(**sparse)
print("noise",float(np.asarray(res[2]).ravel()[0]))


p,v,g,nv=acap[2]; nv=np.asarray(nv)
w=np.asarray(cap[1][5][1]); rows,cols=np.asarray(sparse["noise_rows"]),np.asarray(sparse["noise_cols"])
vv,rr,cc_,b,cs_,out=cap[1]
n=367
A=np.zeros((n,n),complex); A[np.asarray(rr),np.asarray(cc_)]=np.asarray(vv[0])
cvec=np.asarray(cs_)[0,0]
wt=spl.splu(sps.csc_matrix(A)).solve(cvec.astype(complex),trans="H")
N=np.zeros((n,n),complex); N[rows,cols]=nv[0]
def qs(w_): return np.sum(np.real(nv[0]*np.conj(w_[rows])*w_[cols]))
def qd(w_): return np.real(np.vdot(w_,N@w_))
for f_ in range(4):
    print("f",f_,"q_sparse(captured w)",qs(w[f_,0]),"q_dense(captured w)",qd(w[f_,0]),"max|w-wt|",np.abs(w[f_,0]-wt).max())
print("true wt: q_sparse",qs(wt),"q_dense",qd(wt))
print("c equal across f?",[np.abs(np.asarray(cs_)[f_,0]-cvec).max() for f_ in range(4)])
print("A equal across f?",[float(np.abs(np.asarray(vv[f_])-np.asarray(vv[0])).max()) for f_ in range(4)])
