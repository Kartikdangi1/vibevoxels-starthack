#!/usr/bin/env python3
"""Time N optimisation steps on the public benchmark problem; one JSON line per step.

Copy into the root of a Differometor checkout and run it from there, so `import differometor`
picks up that checkout (baseline or final):

    taskset -c 72 python race_optimizer.py --steps 100 --out race_final.jsonl

The objective is benchmark.py's: mean(log10(sensitivity / public reference)) over all 101
frequencies, with its gradient w.r.t. all 193 physical parameters. Each step is one jitted
value_and_grad, then Adam (lr 1e-10; reflectivities clipped inside (0, 1)), as in
verification/verify_generalization.py. Timestamps count from before `import jax`, so step 1
includes imports, setup and every compilation, like the benchmark's first-run total.
"""
from time import perf_counter

T0 = perf_counter()
import argparse
import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import jax
import jax.numpy as jnp
import optax
import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=Path.cwd().name)
    a = ap.parse_args()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)

    data = Path("examples/data")
    fixture = json.loads((data / "uifo_800_3000.json").read_text())
    reference = json.loads((data / "uifo_800_3000_test_data.json").read_text())
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    comps = fixture["nodes"] | fixture["edges"]
    init, is_r = [], []
    for pair in pairs:
        name, prop = pair[0] if isinstance(pair[0], list) else pair
        init.append(comps[name]["properties"][prop])
        is_r.append(prop == "reflectivity")
    x = jnp.asarray(init, dtype=jnp.float64)
    ref = jnp.asarray(reference["sensitivities"], dtype=jnp.float64)
    freqs = jnp.asarray(reference["frequencies"], dtype=jnp.float64)
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], freqs, pairs)
    arrays["optimized_parameters"] = x
    ports = jnp.asarray(ports)
    is_r = jnp.asarray(is_r)
    lo, hi = jnp.where(is_r, 1e-12, -jnp.inf), jnp.where(is_r, 1 - 1e-12, jnp.inf)

    def loss_fn(p, arrs):
        carrier, signal, noise = df.simulate(**dict(arrs, optimized_parameters=p))
        sig = df.signal_detector(carrier, signal)[ports]
        detected = sig.squeeze() if ports.shape[0] == 1 else sig[0] - sig[1]
        return jnp.mean(jnp.log10((noise / jnp.abs(detected)).squeeze() / ref))

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    opt = optax.adam(1e-10)
    state = opt.init(x)
    t_setup = perf_counter() - T0
    with open(a.out, "w") as f:
        f.write(json.dumps({"label": a.label, "event": "setup_done", "t": t_setup}) + "\n")
        for step in range(1, a.steps + 1):
            loss, g = value_and_grad(x, arrays)
            loss = float(loss)  # waits for the result
            upd, state = opt.update(g, state)
            x = jnp.clip(optax.apply_updates(x, upd), lo, hi)
            row = {"label": a.label, "step": step, "t": perf_counter() - T0, "loss": loss,
                   "grad_norm": float(jnp.linalg.norm(g))}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"step {step:3d}  t={row['t']:9.3f} s  loss={loss:.10e}", flush=True)


if __name__ == "__main__":
    main()
