"""Finite differences with steps relative to the parameter itself, for near-zero parameters."""
import os, sys, json
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcase, jax, jax.numpy as jnp, numpy as np
case, names = sys.argv[1], sys.argv[2:]
arrays, ports, p0, f, expected, pairs, _ = vcase.build(case)
def loss(p):
    return vcase.readout(vcase.df.simulate(**dict(arrays, optimized_parameters=p)), ports, expected)[0]
L = jax.jit(loss); g = np.asarray(jax.jit(jax.grad(loss))(p0)); P = np.asarray(p0)
out = []
for name in names:
    i = [vcase.name_of(q) for q in pairs].index(name); rows = []
    prev = None
    for rel in (1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4, 3.125e-4, 1.5625e-4):
        h = rel * abs(P[i])
        fd = (float(L(jnp.asarray(P).at[i].add(h))) - float(L(jnp.asarray(P).at[i].add(-h)))) / (2 * h)
        rich = (4 * fd - prev) / 3 if prev is not None else None; prev = fd
        rows.append({"rel_step": rel, "h": h, "fd": fd, "rel_err": abs(fd - g[i]) / abs(g[i]),
                     "richardson_rel_err": None if rich is None else abs(rich - g[i]) / abs(g[i])})
    out.append({"parameter": name, "value": float(P[i]), "autodiff": float(g[i]), "steps": rows})
    print(name, "p", P[i], "ad", g[i], "best", min(min(r["rel_err"], r["richardson_rel_err"] or 9) for r in rows))
json.dump(out, open(f"/root/validation/artifacts/fd/{case}_near_zero_fd.json", "w"), indent=2)
