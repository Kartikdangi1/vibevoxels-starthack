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
AH=A.conj().T
def to_mp(x): return mp.mpc(float(x.real),float(x.imag))
nz=np.argwhere(AH!=0); AHm=[(i,j,to_mp(AH[i,j])) for i,j in nz]
nzN=np.argwhere(N!=0); Nmp=[(i,j,to_mp(N[i,j])) for i,j in nzN]
lu=__import__("scipy.linalg",fromlist=["x"]).lu_factor(AH)
from scipy.linalg import lu_solve
UV=float(sim.UNIT_VACUUM*sim.H_PLANCK*sim.F0)
def noise_from(w):
    tot=mp.mpf(0)
    Nw=[mp.mpc(0)]*len(w)
    for i,j,v in Nmp: Nw[i]+=v*w[j]
    for i in range(len(w)): tot+=mp.re(Nw[i]*mp.conj(w[i]))
    return mp.sqrt(2*tot*mp.mpf(UV)*mp.mpf(0.25))
for dn in range(c.shape[0]):
    cm_=[to_mp(x) for x in c[dn]]
    w=[to_mp(x) for x in lu_solve(lu,c[dn])]
    for it in range(12):
        r=list(cm_)
        for i,j,v in AHm: r[i]-=v*w[j]
        rn=max(abs(x) for x in r)
        dw=lu_solve(lu,np.array([complex(float(mp.re(x)),float(mp.im(x))) for x in r]))
        # scale-safe: r may be tiny; solve scaled
        w=[w[k]+to_mp(dw[k]) for k in range(len(w))]
        print("det",dn,"refine it",it,"residual",mp.nstr(rn,3),"noise_hp",mp.nstr(noise_from(w),12),flush=True)
        if rn<mp.mpf(10)**-40: break
h=np.asarray(run_backend("host",dense,sparse)[2]); dn_=np.asarray(run_backend("dense",dense,sparse)[2])
print("HOST noise",h.ravel()[0],"DENSE(original) noise",dn_.ravel()[0])
