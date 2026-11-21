#!/usr/bin/env python3
"""Time the main pieces of one forward simulation in isolation (warm, 1 CPU).

Usage (inside the container): python tools/profile_parts.py
"""
import os

os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"

import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax import vmap

import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces
from differometor.simulate import update, solve

jax.config.update("jax_enable_compilation_cache", False)
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]


def timeit(label, function, *args, repeats=3):
    compiled = jax.jit(function).lower(*args).compile()
    jax.block_until_ready(compiled(*args))
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        jax.block_until_ready(compiled(*args))
        samples.append(perf_counter() - start)
    print(f"{label:55s} min {min(samples):8.3f} s", flush=True)
    return compiled(*args)


data = Path(__file__).resolve().parent.parent / "examples" / "data"
fixture = json.loads((data / "uifo_800_3000.json").read_text())
reference = json.loads((data / "uifo_800_3000_test_data.json").read_text())
setup = df.Setup.from_data(fixture)
pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], jnp.asarray(reference["frequencies"]), pairs)
components = fixture["nodes"] | fixture["edges"]
values = []
for pair in pairs:
    name, prop = pair[0] if isinstance(pair[0], list) else pair
    values.append(components[name]["properties"][prop])
arrays["optimized_parameters"] = jnp.asarray(values, dtype=jnp.float64)

V, N = 101, arrays["signal_matrix"].shape[0]
key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (V, N, N)) + 1j * jax.random.normal(key, (V, N, N)) + 30 * jnp.eye(N)
rhs = jnp.ones((V, N), dtype=jnp.complex128)

timeit("full forward simulate", lambda a: df.simulate(**a), arrays)
timeit("forward + vjp residuals (as benchmark)", lambda a: jax.vjp(lambda p: df.simulate(**dict(a, optimized_parameters=p)), a["optimized_parameters"])[0], arrays)
timeit(f"dense LU solve only, {V} x {N}x{N} complex", vmap(solve), dense, rhs)
timeit(f"LU factor only, {V} x {N}x{N}", lambda m: jax.lax.linalg.lu(m)[0], dense)

params = jnp.tile(arrays["parameters"], (V, 1))
timeit(
    "signal matrix fill (vmap update / lax.switch)",
    lambda p, m: vmap(update, in_axes=(0, None, None, None, None, None))(
        p, m, arrays["signal_function_input_indices"], arrays["signal_output_indices"],
        arrays["signal_matrix_indices"], arrays["signal_function_indices"]),
    params, arrays["signal_matrix"],
)
timeit(
    "noise matrix fill (vmap update / lax.switch)",
    lambda p, m: vmap(update, in_axes=(0, None, None, None, None, None))(
        p, m, arrays["noise_function_input_indices"], arrays["noise_output_indices"],
        arrays["noise_system_matrix_indices"], arrays["noise_function_indices"]),
    params, arrays["noise_matrix"],
)
timeit("copy/conj of one (V, N, N+1) complex block", lambda m: jnp.conjugate(m) + 0, jnp.broadcast_to(arrays["signal_matrix"], (V, N, N + 1)))
