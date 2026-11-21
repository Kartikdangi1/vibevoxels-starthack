#!/usr/bin/env python3
"""Measure trace/lower and compile time of the benchmark's forward+pullback and its pieces.

Usage (inside the container): python tools/profile_compile.py [piece ...]
Pieces: full, fills, solve. XLA_FLAGS from the environment apply as usual.
"""
import os
import sys

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
from differometor.simulate import update, solve_with_adjoint

jax.config.update("jax_enable_compilation_cache", False)
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]

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
parameters = jnp.asarray(values, dtype=jnp.float64)
arrays["optimized_parameters"] = parameters
V, N = 101, arrays["signal_matrix"].shape[0]


def measure(label, function, *args):
    start = perf_counter()
    lowered = jax.jit(function).lower(*args)
    lower_seconds = perf_counter() - start
    start = perf_counter()
    lowered.compile()
    compile_seconds = perf_counter() - start
    ops = lowered.as_text().count("\n")
    print(f"{label:45s} lower {lower_seconds:6.2f} s  compile {compile_seconds:6.2f} s  stablehlo lines {ops}", flush=True)


def with_pullback(function):
    """Like benchmark.forward_with_pullback: return outputs plus the AD pullback."""
    return lambda p, *rest: jax.vjp(lambda q: function(q, *rest), p)


def fills(p, a):
    a = dict(a, optimized_parameters=p)
    params = a["parameters"].at[[0], a["optimized_parameter_indices"]].set(p[a["optimized_value_indices"]])
    params = jnp.tile(params, (V, 1))
    up = vmap(update, in_axes=(0, None, None, None, None, None))
    s = up(params, a["signal_matrix"], a["signal_function_input_indices"], a["signal_output_indices"],
           a["signal_matrix_indices"], a["signal_function_indices"])
    n = up(params, a["noise_matrix"], a["noise_function_input_indices"], a["noise_output_indices"],
           a["noise_system_matrix_indices"], a["noise_function_indices"])
    return s, n


def solves(m):
    sel = jnp.ones((V, 1, N), dtype=jnp.complex128)
    return vmap(solve_with_adjoint)(m[:, :, :-1], m[:, :, -1], sel)


pieces = sys.argv[1:] or ["full", "fills", "solve"]
print("XLA_FLAGS:", os.environ.get("XLA_FLAGS"), flush=True)
if "full" in pieces:
    measure("full forward + pullback (benchmark)", with_pullback(lambda p, a: df.simulate(**dict(a, optimized_parameters=p))), parameters, arrays)
if "fills" in pieces:
    measure("matrix fills only + pullback", with_pullback(fills), parameters, arrays)
if "solve" in pieces:
    matrix = jnp.broadcast_to(arrays["signal_matrix"], (V, N, N + 1)) + jnp.eye(N, N + 1)
    measure("signal solve + adjoint only + pullback", lambda m: jax.vjp(solves, m), matrix)
