import json, os, sys, time
os.environ["JAX_PLATFORMS"]="cpu"
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True); jax.config.update("jax_enable_compilation_cache", False)
import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces
S=sys.modules["differometor.simulate"]
PROPS=["reflectivity","tuning","db","angle","power","mass","length"]
fx=json.load(open('examples/data/uifo_800_3000.json')); rf=json.load(open('examples/data/uifo_800_3000_test_data.json'))
setup=df.Setup.from_data(fx); pairs=constrain_inter_grid_cell_spaces(setup.parameters,PROPS)
f=jnp.asarray(rf['frequencies'])
arrays,ports,*_=df.run_build_step(setup,[("f","frequency")],f,pairs)
p0=jnp.asarray([(fx["nodes"]|fx["edges"])[(pr[0] if isinstance(pr[0],list) else pr)[0]]["properties"][(pr[0] if isinstance(pr[0],list) else pr)[1]] for pr in pairs],dtype=jnp.float64)
def t(name):
    fn=lambda p,arr: jax.vjp(lambda q: S.simulate(**dict(arr,optimized_parameters=q)),p)
    t0=time.time(); l=jax.jit(fn).lower(p0,arrays); t1=time.time(); l.compile(); t2=time.time()
    print(f"{name}: lower {t1-t0:.2f} compile {t2-t1:.2f}",flush=True)
orig=(S._CARRIER_DISPATCH,S._NOISE_DISPATCH,S._SIGNAL_DISPATCH)
dummy=S._dispatcher((S.dummy_function,))
t("baseline")
S._SIGNAL_DISPATCH=dummy; t("signal->dummy"); S._SIGNAL_DISPATCH=orig[2]
S._CARRIER_DISPATCH=dummy; S._NOISE_DISPATCH=dummy; t("carrier+noise->dummy"); S._CARRIER_DISPATCH,S._NOISE_DISPATCH=orig[0],orig[1]
S._CARRIER_DISPATCH=dummy;S._NOISE_DISPATCH=dummy;S._SIGNAL_DISPATCH=dummy; t("all->dummy (glue only)")
