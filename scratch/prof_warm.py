import os, sys
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
sys.path.insert(0,'/root/hk_v2/verification')
from common import *
import cProfile, pstats, io
case=load_public(None)
a,p,x,ref=case['arrays'],case['ports'],case['params'],jnp.asarray(case['ref'])
set_backend("host")
vg=jax.value_and_grad(lambda q: uifo_loss(q,a,p,ref)[0])
for _ in range(3): vg(x)
pr=cProfile.Profile(); pr.enable()
for _ in range(10): jax.block_until_ready(vg(x))
pr.disable()
s=io.StringIO(); pstats.Stats(pr,stream=s).sort_stats("tottime").print_stats(14); 
out=s.getvalue().split("\n")
print("\n".join(l[:150] for l in out[:34]))
