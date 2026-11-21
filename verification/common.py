"""Shared helpers for the verification suite.

Every check script is runnable on its own (`python verification/verify_xxx.py`), exits 0 on PASS
and 1 on FAIL, and uses `Report` for uniform output. `run_all.py` runs them all.

Reference hierarchy used throughout (each level is independent of the one it checks):
  analytic closed forms  >  upstream dense solver (`simulate_dense`)  >  in-graph sparse JAX
  >  default host (SciPy SuperLU) path  >  finite differences for gradients.
"""
import json
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import differometor as df  # noqa: E402
import importlib  # noqa: E402

sim = importlib.import_module("differometor.simulate")  # the package re-exports a function named `simulate`
from differometor.build import build, pairs_to_arrays, prepare_arrays  # noqa: E402
from differometor.setups import aligo, constrain_inter_grid_cell_spaces, uifo, voyager  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_compilation_cache", False)

DATA = ROOT / "examples" / "data"
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]


class Report:
    """Collects named checks, prints them, and `finish()` returns the process exit code."""

    def __init__(self, title):
        self.title, self.rows, self.t0 = title, [], perf_counter()
        print(f"=== {title} ===", flush=True)

    def check(self, name, ok, detail=""):
        ok = bool(ok)
        self.rows.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
        return ok

    def close(self, name, got, want, rtol=0.0, atol=0.0, detail=""):
        """Check |got-want| <= atol + rtol*|want| elementwise; reports the worst violation."""
        got, want = np.asarray(got), np.asarray(want)
        if got.shape != want.shape:
            return self.check(name, False, f"shape {got.shape} != {want.shape}")
        err = np.abs(got - want)
        budget = atol + rtol * np.abs(want)
        finite = bool(np.isfinite(got).all())
        used = float(np.max(err / np.maximum(budget, 1e-300))) if err.size else 0.0
        worst = float(err.max()) if err.size else 0.0
        return self.check(name, finite and bool(np.all(err <= budget)),
                          f"max_err={worst:.3e} (budget used {used:.2f}x) {detail}")

    def finish(self):
        bad = [r for r in self.rows if not r[1]]
        print(f"--- {self.title}: {len(self.rows) - len(bad)}/{len(self.rows)} passed in "
              f"{perf_counter() - self.t0:.1f}s ---", flush=True)
        return 1 if bad else 0


def set_backend(name):
    os.environ["DIFFEROMETOR_BACKEND"] = name


# ---------------------------------------------------------------- setups
def cavity(r0=0.99, r1=0.991, loss=0.0, length=1.0):
    S = df.Setup()
    S.add("laser", "l0", power=1)
    S.add("mirror", "m0", reflectivity=r0, loss=loss)
    S.add("mirror", "m1", reflectivity=r1, loss=loss)
    S.space("l0", "m0", length=length)
    S.space("m0", "m1", length=length)
    S.add("detector", "refl", target="m0", port="left", direction="out")
    S.add("detector", "circ", target="m1", port="left", direction="in")
    S.add("detector", "trns", target="m1", port="right", direction="out")
    return S


def public_frequencies(n=None):
    f = np.asarray(json.loads((DATA / "aligo_test_data.json").read_text())["frequencies"], dtype=np.float64)
    return f if n is None else f[:: max(1, len(f) // n)]


def named_setup(name):
    """(setup, sweep_pairs, sweep_values) for a named case."""
    if name == "cavity":
        return cavity(), [("m0", "tuning")], jnp.linspace(-180, 180, 40)
    if name in ("aligo", "voyager"):
        S, _ = (aligo if name == "aligo" else voyager)()
        return S, [("f", "frequency")], jnp.asarray(public_frequencies(12))
    if name.startswith("uifo"):
        size = int(name[4:] or 3)
        S, _ = uifo(size, random=True, random_seed=size)
        return S, [("f", "frequency")], jnp.asarray(np.geomspace(5, 5000, 8))
    raise KeyError(name)


# ---------------------------------------------------------------- build / solve
def build_arrays(setup, pairs, values, opt_pairs=None):
    """Return (dense_arrays, sparse_arrays, metadata). Dense = what simulate_dense (upstream) consumes."""
    instructions, matrices, metadata = build(setup)
    to_prep, carrier_a, signal_a, noise_a = pairs_to_arrays(*instructions, opt_pairs, pairs)
    prepared = prepare_arrays(*to_prep, values, parameters=matrices[0])
    names = sim.simulate_dense.__code__.co_varnames[: sim.simulate_dense.__code__.co_argcount]
    allv = (*prepared, *matrices, *carrier_a, *signal_a, *noise_a)
    dense = {n: (v if v is None else jnp.asarray(v)) for n, v in zip(names, allv)}
    if dense["optimized_parameters"] is None:
        # Start the optimized vector at the setup's own values (what an optimizer would start from).
        table = np.asarray(dense["parameters"])
        table = table[0] if table.ndim == 2 else table
        idx, val = np.asarray(dense["optimized_parameter_indices"]), np.asarray(dense["optimized_value_indices"])
        x0 = np.zeros(int(val.max()) + 1)
        x0[val] = np.real(table[idx])
        dense["optimized_parameters"] = jnp.asarray(x0)
    return dense, sim.compress_arrays(dense), metadata


def run_backend(backend, dense, sparse, optimized_parameters=None):
    """Run 'dense' (upstream), 'jax' (in-graph sparse) or 'host' (default) -> (carrier, signal, noise)."""
    if backend == "dense":
        arrays = dict(dense)
        if optimized_parameters is not None:
            arrays["optimized_parameters"] = optimized_parameters
        return sim.simulate_dense(**arrays)
    set_backend(backend)
    arrays = dict(sparse)
    if optimized_parameters is not None:
        arrays["optimized_parameters"] = optimized_parameters
    return sim.simulate(**arrays)


def scalar_objective(outs):
    """Position-weighted real scalar of all outputs, so no gradient component cancels by symmetry."""
    total = 0.0
    for o in outs[:3]:
        if o.size:
            w = jnp.linspace(0.5, 1.5, o.size).reshape(o.shape)
            total = total + jnp.sum(jnp.real(o) * w) + 0.37 * jnp.sum(jnp.imag(o) * w)
    return total


# ---------------------------------------------------------------- public UIFO
def load_public(n_freq=None):
    fixture = json.loads((DATA / "uifo_800_3000.json").read_text())
    reference = json.loads((DATA / "uifo_800_3000_test_data.json").read_text())
    freqs = np.asarray(reference["frequencies"], dtype=np.float64)
    ref = np.asarray(reference["sensitivities"], dtype=np.float64)
    idx = (np.arange(len(freqs)) if n_freq is None
           else np.unique(np.linspace(0, len(freqs) - 1, n_freq).round().astype(int)))
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    comps = fixture["nodes"] | fixture["edges"]
    init = []
    for pair in pairs:
        name, prop = pair[0] if isinstance(pair[0], list) else pair
        init.append(comps[name]["properties"][prop])
    params = jnp.asarray(init, dtype=jnp.float64)
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], jnp.asarray(freqs[idx]), pairs)
    arrays["optimized_parameters"] = params
    return dict(arrays=arrays, ports=jnp.asarray(ports), params=params, freqs=freqs[idx],
                ref=ref[idx], idx=idx, pairs=pairs)


def uifo_loss(params, arrays, ports, ref):
    """Same objective as benchmark.py: mean(log10(sensitivity / reference))."""
    carrier, signal, noise = df.simulate(**dict(arrays, optimized_parameters=params))
    sig = df.signal_detector(carrier, signal)[ports]
    detected = sig.squeeze() if ports.shape[0] == 1 else sig[0] - sig[1]
    sens = (noise / jnp.abs(detected)).squeeze()
    return jnp.mean(jnp.log10(sens / ref)), sens
