import sys; sys.path.insert(0,'verification')
from common import *
from jax import lax, vmap
S,opt=uifo(2,random=True,random_seed=201)
f=jnp.asarray(np.geomspace(5,5000,8))
dense,sparse,_=build_arrays(S,[("f","frequency")],f,opt)
d=dense
P=d["parameters"].at[[0],d["optimized_parameter_indices"]].set(d["optimized_parameters"][d["optimized_value_indices"]])
P=P.at[[0],d["indices_to_link"]].set(vmap(lambda i,x: lax.switch(i,sim.LINKING_FUNCTIONS,x))(d["linking_function_indices"],P[0,d["linked_indices"]]))
P=sim.expand_parameters(P,d["carrier_changing_parameter_indices"],d["carrier_changing_values"])
P=sim.expand_parameters(P,d["signal_changing_parameter_indices"],d["signal_changing_values"])
Nd=vmap(sim.update,in_axes=(0,None,None,None,None,None))(P,d["noise_matrix"],d["noise_function_input_indices"],d["noise_output_indices"],d["noise_system_matrix_indices"],d["noise_function_indices"])
vals=sim._assemble(P,sparse["noise_values"],sparse["noise_groups"])
Ns=np.zeros(Nd.shape,complex)
r,c=np.asarray(sparse["noise_rows"]),np.asarray(sparse["noise_cols"])
for k in range(Nd.shape[0]): Ns[k][r,c]=np.asarray(vals[k])
Nd=np.asarray(Nd)
print("shape",Nd.shape,"max|Nd|",np.abs(Nd).max(),"max diff",np.abs(Nd-Ns).max())
bad=np.argwhere(np.abs(Nd-Ns).max(0)>1e-12*np.abs(Nd).max())
print("mismatching entries",len(bad))
for (i,j) in bad[:12]: print((i,j),"dense",Nd[0,i,j],"sparse",Ns[0,i,j])
# is NaN/zero entry involved?
print("dense nonzero",np.count_nonzero(Nd[0]),"sparse nonzero",np.count_nonzero(Ns[0]))
