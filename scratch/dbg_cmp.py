import sys; sys.path.insert(0,'verification')
from common import *
import mpmath as mp
from jax import lax, vmap
mp.mp.dps=50
S,opt=uifo(2,random=True,random_seed=201)
f=jnp.asarray(np.geomspace(5,5000,8))
dense,sparse,_=build_arrays(S,[("f","frequency")],f,opt)
d=dense; up=sim.update
P=d["parameters"].at[[0],d["optimized_parameter_indices"]].set(d["optimized_parameters"][d["optimized_value_indices"]])
P=P.at[[0],d["indices_to_link"]].set(vmap(lambda i,x: lax.switch(i,sim.LINKING_FUNCTIONS,x))(d["linking_function_indices"],P[0,d["linked_indices"]]))
qhd=jnp.exp(1j*jnp.radians(P[0,d["qhd_parameter_indices"]]))
P=sim.expand_parameters(P,d["carrier_changing_parameter_indices"],d["carrier_changing_values"])
cm=vmap(up,in_axes=(0,None,None,None,None,None))(P,d["carrier_matrix"],d["carrier_function_input_indices"],d["carrier_output_indices"],d["carrier_matrix_indices"],d["carrier_function_indices"])
carrier=vmap(sim.solve)(cm[:,:,:-1],cm[:,:,-1])
P=sim.expand_parameters(P,d["signal_changing_parameter_indices"],d["signal_changing_values"])
sm=vmap(up,in_axes=(0,None,None,None,None,None))(P,d["signal_matrix"],d["signal_function_input_indices"],d["signal_output_indices"],d["signal_matrix_indices"],d["signal_function_indices"])
pi=d["signal_carrier_placing_indices"]
sm=sm.at[:,pi[0],pi[1]].set(-sm[:,pi[0],pi[1]]*carrier[:,d["signal_carrier_indices"].flatten()])
cn=carrier.shape[1]
sm=sm.at[:,cn:2*cn,cn:2*cn].set(jnp.conjugate(sm[:,cn:2*cn,cn:2*cn]))
sm=sm.at[:,2*cn:,:cn].set(jnp.conjugate(sm[:,2*cn:,:cn]))
sm=sm.at[:,cn:2*cn,2*cn:].set(jnp.conjugate(sm[:,cn:2*cn,2*cn:]))
nsv=lax.scatter_mul(d["noise_selection_vectors"],d["qhd_placing_indices"].reshape(-1,3),qhd,lax.ScatterDimensionNumbers(update_window_dims=(),inserted_window_dims=(0,1,2),scatter_dims_to_operand_dims=(0,1,2)),unique_indices=True)
nsv=jnp.tile(nsv,(1,carrier.shape[0],1))
up_=nsv.at[:,:,:cn].set(nsv[:,:,:cn]*carrier)
lo=jnp.conjugate(up_[:,:,cn:2*cn]*carrier)
cs=up_.at[:,:,cn:2*cn].set(lo)           # (D,F,n)
Nm=vmap(up,in_axes=(0,None,None,None,None,None))(P,d["noise_matrix"],d["noise_function_input_indices"],d["noise_output_indices"],d["noise_system_matrix_indices"],d["noise_function_indices"])

import scipy.sparse as sps, scipy.sparse.linalg as spl
cap=[]
orig=sim.sparse_solve
def wrap(values,rows,cols,b,c_):
    out=orig(values,rows,cols,b,c_); cap.append((values,rows,cols,b,c_,out)); return out
sim.sparse_solve=wrap
set_backend("jax"); res=sim.simulate_sparse_jax(**sparse)
vv,rr,cc_,bb,cs_,out=cap[1]
n=367
Asp=np.zeros((vv.shape[0],n,n),complex); Asp[:,np.asarray(rr),np.asarray(cc_)]=np.asarray(vv)
Ad=np.asarray(sm)[:,:,:-1]
print("A: sparse-captured vs dense-style max diff",np.abs(Asp-Ad).max(),"(nF",Asp.shape[0],Ad.shape[0],")")
print("c: captured shape",np.asarray(cs_).shape,"dense-style cs shape",np.asarray(cs).shape)
cdense=np.transpose(np.asarray(cs),(1,0,2))
print("c max diff",np.abs(np.asarray(cs_)-cdense).max(), "max|c|",np.abs(cdense).max())
bd=np.asarray(bb); bdense=np.asarray(sm)[:,:,-1]
print("b max diff",np.abs(bd-bdense).max())
# true quad per frequency with dense-style vs captured
Nn=np.asarray(Nm)[0]
def q(Af,cf): 
    w=spl.splu(sps.csc_matrix(Af)).solve(cf.astype(complex),trans="H"); return np.real(np.vdot(w,Nn@w))
for f_ in range(Asp.shape[0]):
    print("f",f_,"q dense-style",q(Ad[f_],cdense[f_,0]),"q captured",q(Asp[f_],np.asarray(cs_)[f_,0]))
