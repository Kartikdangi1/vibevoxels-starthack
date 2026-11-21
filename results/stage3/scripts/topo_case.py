"""One topology x one mode per process (memory-safe replacement for the crashed generality.py).

Usage (cwd = worktree under test): topo_case.py CASE MODE OUT.npz
MODE: native (host path, native kernels) | traced (host path, JAX-traced kernels) | jax (in-graph sparse path)
"""
import os, sys, json, time
CASE, MODE, OUT = sys.argv[1:4]
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["DIFFEROMETOR_NATIVE"] = "none" if MODE == "traced" else "all"
os.environ["DIFFEROMETOR_BACKEND"] = "jax" if MODE == "jax" else "host"
sys.path.insert(0, os.getcwd())
import jax, jax.numpy as jnp, numpy as np
import differometor as df
from differometor.setups import aligo, voyager, uifo, constrain_inter_grid_cell_spaces
from differometor.components import signal_detector

PROPS = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length", "loss", "phase", "alpha"]


def make(name):
    kind, _, arg = name.partition(":")
    if kind == "aligo": return aligo(arg)[0]
    if kind == "voyager": return voyager(arg)[0]
    if kind == "uifo":
        size, seed = map(int, arg.split(","))
        return uifo(size, random=True, random_seed=seed)[0]
    if kind == "file":
        with open(f"examples/data/{arg}.json") as f:
            return df.Setup.from_data(json.load(f))
    raise ValueError(name)


setup = make(CASE)
pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPS)
data = setup.to_data()
init = []
for pair in pairs:
    cname, prop = pair[0] if isinstance(pair[0], list) else pair
    props = data["nodes" if cname in data["nodes"] else "edges"][cname]["properties"]
    init.append(float(props[prop]))
freqs = jnp.asarray(np.logspace(np.log10(3), np.log10(3000), 20))
arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], freqs, pairs)
p0 = jnp.asarray(init, dtype=jnp.float64)
ports = np.asarray(ports)


def loss(p):
    c, s, n = df.simulate(**dict(arrays, optimized_parameters=p))
    sig = signal_detector(c, s)[ports]
    det = sig.squeeze() if ports.shape[0] == 1 else sig[0] - sig[1]
    sens = (n / jnp.abs(det)).squeeze()
    return jnp.mean(jnp.log10(sens)), (c, s, n, sens)


t = time.perf_counter()
(l, (c, s, n, sens)), g = jax.jit(jax.value_and_grad(loss, has_aux=True))(p0)
np.savez(OUT, c=np.asarray(c), s=np.asarray(s), n=np.asarray(n), sens=np.asarray(sens), g=np.asarray(g), l=float(l))
print(f"{CASE} {MODE} params={len(init)} t={time.perf_counter()-t:.2f}s", file=sys.stderr)
