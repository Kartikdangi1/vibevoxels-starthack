import os, time, json
os.environ["JAX_PLATFORMS"]="cpu"
import numpy as np, jax, jax.numpy as jnp
from jax import vmap
jax.config.update("jax_enable_x64", True); jax.config.update("jax_enable_compilation_cache", False)
from differometor.components import evaluate_components
K=679; V=101
rng=np.random.default_rng(0)
ids=jnp.asarray(rng.integers(0,19,K)); X=jnp.asarray(rng.uniform(0.1,0.9,(V,K,7)))
def timeit(name,fn,*a):
    t=time.time(); l=jax.jit(fn).lower(*a); t1=time.time(); l.compile(); print(f"{name}: lower {t1-t:.2f} compile {time.time()-t1:.2f}",flush=True)
f=lambda X: vmap(lambda x: evaluate_components(ids,x))(X)
timeit("kernel fwd",f,X)
timeit("kernel vjp(fwd+residuals)",lambda X: jax.vjp(f,X),X)
g=lambda X,ct: jax.grad(lambda X: jnp.sum(jnp.real(f(X))*ct))(X)
timeit("kernel grad",g,X,jnp.ones((V,K,4)))
txt=jax.jit(f).lower(X).compile().as_text()
import collections,re
ops=collections.Counter(re.findall(r"= \w+\[[^\]]*\]\S* (\w[\w-]*)\(",txt)); print(sum(ops.values()),"ops;", ops.most_common(12))
