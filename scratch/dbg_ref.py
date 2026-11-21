import sys; sys.path.insert(0,'verification')
from common import *
S,opt=uifo(2,random=True,random_seed=201)
f=jnp.asarray(np.geomspace(5,5000,4))
dense,sparse,_=build_arrays(S,[("f","frequency")],f,opt)
print("noise[0]:", float(np.asarray(run_backend(sys.argv[1],dense,sparse)[2]).ravel()[0]), "(true 1.98129943947e-25)")
