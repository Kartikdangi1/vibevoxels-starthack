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

sp=sparse
Pc=sim.expand_parameters(P0:=None or P, sp["carrier_changing_parameter_indices"], sp["carrier_changing_values"]) if False else None
# rebuild sparse signal matrix exactly as simulate_sparse_jax does (parameters P already expanded for carrier+signal)
svals=sim._assemble(P,sp["signal_values"],sp["signal_groups"])
svals=svals.at[:,sp["signal_place_k"]].set(-svals[:,sp["signal_place_k"]]*carrier[:,sp["signal_carrier_indices"].flatten()])
svals=jnp.where(sp["signal_conj_mask"],jnp.conjugate(svals),svals)
n_sig=sp["signal_rows"].shape[0]
Ssp=np.zeros(np.asarray(sm).shape,complex)
r,c_=np.asarray(sp["signal_rows"]),np.asarray(sp["signal_cols"])
Ssp[:,r,c_]=np.asarray(svals[:,:n_sig])
rr=np.asarray(sp["signal_rhs_rows"]); Ssp[:,rr,-1]=np.asarray(svals[:,n_sig:])
D=np.asarray(sm)
diff=np.abs(D-Ssp)[0]
print("A max|dense|",np.abs(D).max(),"max diff",diff.max())
bad=np.argwhere(diff>1e-9*np.abs(D).max())
print("mismatching entries",len(bad),"of",D.shape[1]*D.shape[2])
for (i,j) in bad[:15]: print((i,j),"dense",D[0,i,j],"sparse",Ssp[0,i,j],"cn",cn)
