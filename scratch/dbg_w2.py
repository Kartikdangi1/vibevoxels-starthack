import sys; sys.path.insert(0,'verification')
from common import *
from jax import lax, vmap
S,opt=uifo(2,random=True,random_seed=201)
f=jnp.asarray(np.geomspace(5,5000,8))
dense,sparse,_=build_arrays(S,[("f","frequency")],f,opt)
d=dense; up=sim.update
P=d["parameters"].at[[0],d["optimized_parameter_indices"]].set(d["optimized_parameters"][d["optimized_value_indices"]])
P=P.at[[0],d["indices_to_link"]].set(vmap(lambda i,x: lax.switch(i,sim.LINKING_FUNCTIONS,x))(d["linking_function_indices"],P[0,d["linked_indices"]]))
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
A=np.asarray(sm)[:,:,:-1]
for k in (0,4,7):
    sv=np.linalg.svd(A[k],compute_uv=False); print("freq",k,"cond %.2e"%(sv[0]/sv[-1]),"smin %.2e smax %.2e"%(sv[-1],sv[0]))
# which entry is bad: compare the three noise values per detector/frequency
h=run_backend("host",dense,sparse)[2]; dn=run_backend("dense",dense,sparse)[2]
print("noise host",np.asarray(h).ravel()[:3],"dense",np.asarray(dn).ravel()[:3])
print("noise shapes",np.asarray(h).shape,np.asarray(dn).shape)
np.savez("/root/dbg_sm.npz",A=A,carrier=np.asarray(carrier))
