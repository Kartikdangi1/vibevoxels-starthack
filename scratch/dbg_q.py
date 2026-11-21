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

sp=sparse
# sparse code path, copied
qhd_phase_values=jnp.exp(1j*jnp.radians(sp["parameters"].at[[0],sp["optimized_parameter_indices"]].set(sp["optimized_parameters"][sp["optimized_value_indices"]])[0,sp["qhd_parameter_indices"]]))
print("qhd dense vs sparse(before linking)",np.asarray(qhd),np.asarray(qhd_phase_values))
selections=lax.scatter_mul(sp["noise_selection_vectors"],sp["qhd_placing_indices"].reshape(-1,3),qhd_phase_values,
  lax.ScatterDimensionNumbers(update_window_dims=(),inserted_window_dims=(0,1,2),scatter_dims_to_operand_dims=(0,1,2)),unique_indices=True)
selections=jnp.tile(selections,(1,carrier.shape[0],1))
upper=selections.at[:,:,:cn].set(selections[:,:,:cn]*carrier)
lower=jnp.conjugate(upper[:,:,cn:2*cn]*carrier)
selections=upper.at[:,:,cn:2*cn].set(lower)
print("selections max diff vs dense cs",float(jnp.abs(selections-cs).max()),"shape",selections.shape)
# quadratic form: dense vs sparse formula using TRUE weights
import scipy.sparse as sps, scipy.sparse.linalg as spl
A=np.asarray(sm)[0,:,:-1]; N=np.asarray(Nm)[0]; cc=np.asarray(cs)[0,0]
w=spl.splu(sps.csc_matrix(A)).solve(cc.astype(complex),trans="H")
rows,cols=np.asarray(sp["noise_rows"]),np.asarray(sp["noise_cols"])
nv=np.asarray(sim._assemble(P,sp["noise_values"],sp["noise_groups"]))[0]
q_sparse=np.sum(np.real(nv*np.conj(w[rows])*w[cols])); q_dense=np.real(np.sum((N@w)*np.conj(w)))
print("quad dense",q_dense,"quad sparse-formula",q_sparse, "ratio",q_sparse/q_dense)
print("noise-matrix nnz base",np.count_nonzero(N),"sparse structure",len(rows),"unique",len(set(zip(rows.tolist(),cols.tolist()))))
miss=[(i,j) for i,j in np.argwhere(N!=0) if (i,j) not in set(zip(rows.tolist(),cols.tolist()))]
print("nonzeros of N missing from sparse structure:",len(miss))
