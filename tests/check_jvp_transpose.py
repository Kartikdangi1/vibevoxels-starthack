import os
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "true")
import jax, jax.numpy as jnp, numpy as np
import time
H = __import__("differometor.host", fromlist=["x"])
from differometor.components import FUNCTIONS
rng = np.random.default_rng(0)
B=13
fns = list(FUNCTIONS) + [jnp.sqrt, lambda x: x**2 + 3*x, lambda x: jnp.abs(x) * jnp.exp(1j*x), lambda x: jnp.real(x)*jnp.conj(x)]
worst=0
for k, fn in enumerate(fns):
    shape = (7,) if k < len(FUNCTIONS) else ()
    x = (rng.uniform(0.2, 2.0, (B, *shape)) + 1j*rng.uniform(-0.5,0.5,(B,*shape))*(k>=len(FUNCTIONS))).astype(np.complex128)
    t0=time.perf_counter()
    try: tf = H.JvpTransposedFunction(fn, shape, jnp.complex128); kind="jvpT"
    except H._Unsupported as ex: tf = H.TracedFunction(fn, shape, jnp.complex128); kind="vjp:"+str(ex)
    dt=time.perf_counter()-t0
    y = tf(x); ybar = (rng.normal(size=y.shape)+1j*rng.normal(size=y.shape))
    yref, pb = jax.vjp(jax.vmap(fn), jnp.asarray(x)); gref = np.asarray(pb(jnp.asarray(ybar, yref.dtype))[0])
    g = tf.vjp(x, ybar)
    ey = np.abs(np.asarray(y)-np.asarray(yref)).max()/max(np.abs(np.asarray(yref)).max(),1e-300)
    eg = np.abs(g-gref).max()/max(np.abs(gref).max(),1e-300)
    worst=max(worst, eg, ey)
    print(f"{getattr(fn,'__name__','lambda'):40s} {kind:12s} trace {dt*1000:5.1f} ms  fwd rel {ey:.1e}  vjp rel {eg:.1e}")
print("WORST", worst)
