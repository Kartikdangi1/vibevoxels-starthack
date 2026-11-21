import sys; sys.path.insert(0,'verification')
from common import *
S,opt=uifo(2,random=True,random_seed=201)
f=jnp.asarray(np.geomspace(5,5000,8))
dense,sparse,_=build_arrays(S,[("f","frequency")],f,opt)
M=np.asarray(dense["signal_matrix"])
print("signal_matrix",M.shape)
A=M[:,:,:-1]
for k in (0,3,7):
    sv=np.linalg.svd(A[k],compute_uv=False); print("freq",k,"cond",sv[0]/sv[-1],"smallest sv",sv[-1],"largest",sv[0])
c=np.asarray(dense["carrier_matrix"]); 
sv=np.linalg.svd(c[0][:, :-1] if c.shape[-1]==c.shape[-2]+1 else c[0],compute_uv=False); print("carrier cond",sv[0]/sv[-1])
