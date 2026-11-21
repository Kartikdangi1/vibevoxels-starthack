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

print("assemble calls",len(acap))
p,v,g,nv=acap[2]
print("noise nvals shape",nv.shape,"param rows",p.shape)
w=np.asarray(cap[1][5][1])           # (F,D,n)
rows,cols=np.asarray(sparse["noise_rows"]),np.asarray(sparse["noise_cols"])
q=np.sum(np.real(np.asarray(nv)[:,None,:]*np.conj(w[:,:,rows])*w[:,:,cols]),axis=2)
print("q from captured nvals+w:",q.ravel()[:2],"-> noise",np.sqrt(2*q*float(sim.UNIT_VACUUM*sim.H_PLANCK*sim.F0)*0.25).ravel()[:2])
# the same with nvals computed from DENSE-style N
N=np.zeros((367,367),complex); N[rows,cols]=np.asarray(nv)[0]
print("nnz of captured N",np.count_nonzero(N), "max|N|",np.abs(N).max())
# dense-style N
from jax import vmap
d=dense
P=d["parameters"].at[[0],d["optimized_parameter_indices"]].set(d["optimized_parameters"][d["optimized_value_indices"]])
Nd=np.asarray(vmap(sim.update,in_axes=(0,None,None,None,None,None))(sim.expand_parameters(P,d["signal_changing_parameter_indices"],d["signal_changing_values"]),d["noise_matrix"],d["noise_function_input_indices"],d["noise_output_indices"],d["noise_system_matrix_indices"],d["noise_function_indices"]))[0]
print("dense N vs captured N max diff",np.abs(N-Nd).max())
bad=np.argwhere(np.abs(N-Nd)>1e-12)
print("differing entries",len(bad),[ (int(i),int(j),N[i,j],Nd[i,j]) for i,j in bad[:6]])
