import sys; sys.path.insert(0,'verification')
from common import *
S,opt=uifo(2,random=True,random_seed=201)
f=jnp.asarray(np.geomspace(5,5000,8))
dense,sparse,_=build_arrays(S,[("f","frequency")],f,opt)
for kind in ("carrier","signal","noise"):
    key=f"{kind}_matrix_indices" if kind!="noise" else "noise_system_matrix_indices"
    idx=np.asarray(dense[key]).reshape(2,-1)
    shape=np.asarray(dense[f"{kind}_matrix"]).shape
    flat=(idx[0]%shape[0])*shape[1]+(idx[1]%shape[1])
    u,cnt=np.unique(flat,return_counts=True)
    dup=u[cnt>1]
    print(kind,"writes",flat.size,"unique",u.size,"duplicated positions",dup.size)
    if dup.size and kind=="noise":
        fi=np.asarray(dense["noise_function_indices"]); fin=np.asarray(dense["noise_function_input_indices"]); oi=np.asarray(dense["noise_output_indices"]).reshape(2,-1)
        for d in dup[:5]:
            w=np.flatnonzero(flat==d); print("  pos",divmod(int(d),shape[1]),"written by output slots",w, "output idx rows/cols",oi[:,w].T.tolist())
