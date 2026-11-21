import json, os, sys, time
os.environ["JAX_PLATFORMS"]="cpu"
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True); jax.config.update("jax_enable_compilation_cache", False)
import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces
S=sys.modules["differometor.simulate"]
fx=json.load(open('examples/data/uifo_800_3000.json')); rf=json.load(open('examples/data/uifo_800_3000_test_data.json'))
setup=df.Setup.from_data(fx); pairs=constrain_inter_grid_cell_spaces(setup.parameters,["reflectivity","tuning","db","angle","power","mass","length"])
arrays,ports,*_=df.run_build_step(setup,[("f","frequency")],jnp.asarray(rf['frequencies']),pairs)
p0=jnp.asarray([(fx["nodes"]|fx["edges"])[(pr[0] if isinstance(pr[0],list) else pr)[0]]["properties"][(pr[0] if isinstance(pr[0],list) else pr)[1]] for pr in pairs],dtype=jnp.float64)
def phases(name):
    fwd=lambda p,arr: jax.vjp(lambda q: S.simulate(**dict(arr,optimized_parameters=q)),p)
    t0=time.time(); l=jax.jit(fwd).lower(p0,arrays); t1=time.time(); c=l.compile(); t2=time.time()
    out,pb=c(p0,arrays); jax.block_until_ready(out)
    ct=jax.tree_util.tree_map(jnp.ones_like,out)
    bw=lambda pb,ct: pb(ct)[0]
    t3=time.time(); l2=jax.jit(bw).lower(pb,ct); t4=time.time(); l2.compile(); t5=time.time()
    print(f"{name}: fwd lower {t1-t0:.2f} compile {t2-t1:.2f} | bwd lower {t4-t3:.2f} compile {t5-t4:.2f}",flush=True)
phases("kernel")
real=S.evaluate_components
def cheap(fid,x):
    r=jnp.real(x).astype(jnp.float64); return (r[:, :4]*(1.0+0.1*fid[:,None])).astype(jnp.complex128)
S.evaluate_components=cheap
phases("trivial kernel (glue+callbacks only)")
