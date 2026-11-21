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
A=np.asarray(sm)[0,:,:-1]; c=np.asarray(cs)[:,0,:]; N=np.asarray(Nm)[0]

import scipy.sparse as sps, scipy.sparse.linalg as spl, scipy.linalg as sl
UV=float(sim.UNIT_VACUUM*sim.H_PLANCK*sim.F0)
def noise_of(w): return float(np.sqrt(2*np.real(np.vdot(w, N@w))*UV*0.25))   # vdot conj(w).(N w)
AH=A.conj().T
cc=c[0]
w_lapack=np.linalg.solve(AH,cc)
print("LAPACK  noise",noise_of(w_lapack),"resid",np.abs(AH@w_lapack-cc).max())
for perm in ("COLAMD","MMD_AT_PLUS_A","MMD_ATA","NATURAL"):
    lu=spl.splu(sps.csc_matrix(A),permc_spec=perm)
    w=lu.solve(cc.astype(complex),trans="H")
    r=cc-AH@w
    print(perm.ljust(14),"noise",noise_of(w),"resid",np.abs(r).max(),"| w rel diff vs lapack",np.abs(w-w_lapack).max()/np.abs(w_lapack).max())
    for k in range(3):
        w=w+lu.solve(cc-AH@w,trans="H"); 
    print("   after 3 refines noise",noise_of(w),"resid",np.abs(cc-AH@w).max())
print("|w| max",np.abs(w_lapack).max(),"min nonzero",np.abs(w_lapack[np.abs(w_lapack)>0]).min())
