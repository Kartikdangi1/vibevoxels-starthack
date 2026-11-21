"""Native C++ sparse LU vs SciPy SuperLU on the host path, across topologies.

For every case: sensitivities (log) and the gradient of mean(log10 sensitivity) with the
native solver and with SuperLU (`differometor.native` disabled), plus the in-graph JAX
path as an independent third implementation. Prints the worst disagreements.

    python tests/check_native_solver.py
"""
import json
import sys
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
import differometor as df  # noqa: E402
from differometor import native  # noqa: E402
from differometor.setups import aligo, voyager, uifo, constrain_inter_grid_cell_spaces  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "examples" / "data"
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]


def sweep(setup_and_pairs, freqs):
    S, pairs = setup_and_pairs
    return S, pairs, freqs


def fixture_case(name):
    fixture = json.loads((DATA / name).read_text())
    S = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(S.parameters, PROPERTIES)
    return S, pairs


def cases():
    aligo_f = jnp.asarray(json.loads((DATA / "aligo_test_data.json").read_text())["frequencies"])[::5]
    uifo_f = jnp.asarray(json.loads((DATA / "uifo_800_3000_test_data.json").read_text())["frequencies"])
    log_f = jnp.logspace(np.log10(3), np.log10(3000), 20)
    yield "aligo-space", sweep(aligo("space_modulation"), aligo_f)
    yield "aligo-amplitude", sweep(aligo("amplitude_modulation"), aligo_f)
    yield "aligo-frequency", sweep(aligo("frequency_modulation"), aligo_f)
    yield "voyager-space", sweep(voyager("space_modulation"), log_f)
    yield "voyager-amplitude", sweep(voyager("amplitude_modulation"), log_f)
    yield "voyager-frequency", sweep(voyager("frequency_modulation"), log_f)
    yield "uifo_800_3000", sweep(fixture_case("uifo_800_3000.json"), uifo_f)
    yield "uifo_20_5000", sweep(fixture_case("uifo_20_5000.json"), log_f)
    for seed in range(4):
        yield f"uifo-random-3x3-seed{seed}", sweep(uifo(3, random=True, random_seed=seed), log_f)


def evaluate(arrays, ports):
    def loss(p):
        carrier, signal, noise = df.simulate(**dict(arrays, optimized_parameters=p))
        signals = df.signal_detector(carrier, signal)[ports]
        detected = signals.squeeze() if ports.shape[0] == 1 else signals[0] - signals[1]
        sens = (noise / jnp.abs(detected)).squeeze()
        return jnp.mean(jnp.log10(sens)), sens

    (value, sens), grad = jax.value_and_grad(loss, has_aux=True)(arrays["optimized_parameters"])
    return np.log(np.asarray(sens)), np.asarray(grad)


def with_solver(use_native, arrays, ports):
    saved = list(native._LIB)
    if not use_native:
        native._LIB[:] = [None]
    try:
        return evaluate(arrays, ports)
    finally:
        native._LIB[:] = saved


def main():
    assert native.available(), "build the native solver first: python differometor/native/build.py"
    worst = {}
    for name, (S, pairs, freqs) in cases():
        arrays, ports, *_ = df.run_build_step(S, [("f", "frequency")], freqs, pairs)
        data = S.to_data()
        components = data["nodes"] | data["edges"]
        values = []
        for pair in pairs:
            comp, prop = pair[0] if isinstance(pair[0], list) else pair
            values.append(components[comp]["properties"][prop])
        arrays["optimized_parameters"] = jnp.asarray(values, dtype=jnp.float64)
        ports = jnp.asarray(ports)
        t = perf_counter()
        ls_n, g_n = with_solver(True, arrays, ports)
        t_n = perf_counter() - t
        ls_s, g_s = with_solver(False, arrays, ports)
        import os
        os.environ["DIFFEROMETOR_BACKEND"] = "jax"
        try:
            ls_j, g_j = evaluate(arrays, ports)
        finally:
            del os.environ["DIFFEROMETOR_BACKEND"]
        scale = max(np.max(np.abs(g_s)), 1e-300)
        row = {
            "native_vs_superlu_logS": float(np.max(np.abs(ls_n - ls_s))),
            "native_vs_superlu_grad": float(np.max(np.abs(g_n - g_s)) / scale),
            "superlu_vs_jax_logS": float(np.max(np.abs(ls_s - ls_j))),
            "native_vs_jax_logS": float(np.max(np.abs(ls_n - ls_j))),
            "native_vs_jax_grad": float(np.max(np.abs(g_n - g_j)) / scale),
            "finite": bool(np.all(np.isfinite(ls_n)) and np.all(np.isfinite(g_n))),
        }
        print(f"{name:26s} " + " ".join(f"{k}={v:.1e}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
              + f" t_native={t_n:.2f}s", flush=True)
        for k, v in row.items():
            if isinstance(v, float):
                worst[k] = max(worst.get(k, 0.0), v)
    print("worst:", {k: f"{v:.1e}" for k, v in worst.items()})


if __name__ == "__main__":
    sys.exit(main())
