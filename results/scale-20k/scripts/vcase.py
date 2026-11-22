"""Independent outputs + gradient + finite differences for one optical setup.

    cd <code dir>   # optimised branch or the official reference implementation
    python vcase.py run CASE OUT.npz [--fd FD.json]

CASE: dev   (official 3x3 grid uifo_800_3000, 101 f, 193 vars, public reference)
      aligo (official aLIGO space_modulation, 101 f, 29 vars, public reference)
      uifo20 (public uifo_20_5000 grid fixture, no reference; 101 log-spaced f, 20-5000 Hz)

Inputs are built exactly as the official benchmark builds them, and the readout/loss is
the official `readout` (mean log10 sensitivity relative to the reference; ones when no
reference exists). The full gradient comes from jax.value_and_grad of that loss.
"""
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vprobe  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import differometor as df  # noqa: E402
from differometor.setups import aligo, constrain_inter_grid_cell_spaces, voyager  # noqa: E402

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
DATA = Path(os.getcwd()) / "examples" / "data"


def readout(outputs, detector_ports, reference):  # copied verbatim from official benchmark.py
    carrier, signal, noise = outputs
    signals = df.signal_detector(carrier, signal)[detector_ports]
    detected = signals.squeeze() if detector_ports.shape[0] == 1 else signals[0] - signals[1]
    sensitivity = (noise / jnp.abs(detected)).squeeze()
    loss = jnp.mean(jnp.log10(sensitivity / reference))
    return loss, (sensitivity, df.power_detector(carrier))


def build(case):
    if case == "aligo":
        setup, _ = aligo(mode="space_modulation")
        fixture = deepcopy(setup.to_data())
        pairs = [list(pair) for pair in setup.parameters if pair[1] in PROPERTIES]
        reference = json.loads((DATA / "aligo_test_data.json").read_text())
    elif case.startswith("voyager_"):
        mode = {"voyager_fm": "frequency_modulation", "voyager_am": "amplitude_modulation",
                "voyager_space": "space_modulation"}[case]
        setup, pairs = voyager(mode)  # same construction as tests/check_native_solver.py
        pairs = [list(p) for p in pairs]
        fixture = deepcopy(setup.to_data())
        reference = None
    else:
        name = {"dev": "uifo_800_3000.json", "uifo20": "uifo_20_5000.json"}[case]
        fixture = json.loads((DATA / name).read_text())
        setup = df.Setup.from_data(fixture)
        pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
        reference = json.loads((DATA / "uifo_800_3000_test_data.json").read_text()) if case == "dev" else None
    initial = []
    for pair in pairs:
        members = pair if isinstance(pair[0], list) else [pair]
        values = [fixture["nodes" if n in fixture["nodes"] else "edges"][n]["properties"][p] for n, p in members]
        initial.append(values[0])
    parameters = jnp.asarray(initial, dtype=jnp.float64)
    if reference is not None:
        frequencies = jnp.asarray(reference["frequencies"], dtype=jnp.float64)
        expected = jnp.asarray(reference["sensitivities"], dtype=jnp.float64)
    else:
        f = np.geomspace(3.0, 3000.0, 20) if case.startswith("voyager_") else np.geomspace(20.0, 5000.0, 101)
        frequencies = jnp.asarray(f)
        expected = jnp.ones(f.shape[0], dtype=jnp.float64)
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    arrays["optimized_parameters"] = parameters
    return arrays, jnp.asarray(ports), parameters, frequencies, expected, pairs, reference is not None


def name_of(pair):
    members = pair if isinstance(pair[0], list) else [pair]
    return "+".join(f"{n}.{p}" for n, p in members)


def main():
    _, case, out = sys.argv[1:4]
    fd_path = sys.argv[sys.argv.index("--fd") + 1] if "--fd" in sys.argv else None
    probe_path = str(Path(out).with_suffix(".probe.json"))
    vprobe.install(probe_path)

    t0 = time.perf_counter()
    arrays, ports, parameters, frequencies, expected, pairs, has_ref = build(case)

    def full(p):
        outputs = df.simulate(**dict(arrays, optimized_parameters=p))
        loss, (sens, powers) = readout(outputs, ports, expected)
        return loss, (outputs, sens, powers)

    (loss, (outputs, sens, powers)), grad = jax.jit(jax.value_and_grad(full, has_aux=True))(parameters)
    carrier, signal, noise = jax.device_get(outputs)
    record = {
        "case": case, "seconds_build_compile_run": time.perf_counter() - t0,
        "frequency_count": int(frequencies.shape[0]), "design_variables": int(parameters.shape[0]),
        "gradient_length": int(grad.shape[0]), "loss": float(loss),
        "dtypes": {k: str(v.dtype) for k, v in (("carrier", carrier), ("signal", signal), ("noise", noise), ("gradient", grad))},
        "finite": {k: bool(np.all(np.isfinite(np.asarray(v)))) for k, v in
                   (("carrier", carrier), ("signal", signal), ("noise", noise), ("sensitivity", sens),
                    ("powers", powers), ("loss", loss), ("gradient", grad))},
        "sensitivity_positive": bool(np.all(np.asarray(sens) > 0)),
        "gradient_nonzero": int(np.count_nonzero(np.asarray(grad))),
        "pairs": [name_of(p) for p in pairs],
    }
    if has_ref:
        record["max_abs_log_sensitivity_error_vs_public_reference"] = float(np.max(np.abs(np.log(np.asarray(sens)) - np.log(np.asarray(expected)))))
    if "--jitter" in sys.argv:
        # Conditioning probe: relative 1e-12 perturbation of every parameter (fixed seed).
        rng = np.random.default_rng(0)
        pj = np.asarray(parameters) * (1 + 1e-12 * rng.standard_normal(parameters.shape[0]))
        _, (_, sens_j, _) = jax.jit(full)(jnp.asarray(pj))
        record["jitter_1e-12_max_abs_dlogS"] = float(np.max(np.abs(np.log(np.asarray(sens_j)) - np.log(np.asarray(sens)))))
    np.savez(out, carrier=carrier, signal=signal, noise=noise, sensitivity=np.asarray(sens), powers=np.asarray(powers),
             gradient=np.asarray(grad), loss=np.asarray(loss), parameters=np.asarray(parameters), frequencies=np.asarray(frequencies))

    if fd_path:
        loss_only = jax.jit(lambda p: full(p)[0])
        g = np.asarray(grad)
        # Representative parameters: per property type, the largest-|gradient| entry and the first entry.
        chosen = []
        for prop in PROPERTIES:
            idx = [i for i, p in enumerate(pairs) if (p[0] if isinstance(p[0], list) else p)[1] == prop]
            if idx:
                for i in sorted({idx[0], max(idx, key=lambda i: abs(g[i]))}):
                    chosen.append(i)
        checks = []
        p0 = np.asarray(parameters)
        for i in chosen:
            prop = (pairs[i][0] if isinstance(pairs[i][0], list) else pairs[i])[1]
            x = p0[i]
            base = max(abs(x), 1.0)
            if prop == "reflectivity":  # stay inside (0, 1)
                base = min(base, 0.5 * min(x, 1.0 - x) / 1e-3) if 0 < x < 1 else base
            results = []
            previous = None
            for k in range(16):  # step halves each time: 1e-3 ... 3e-8 of the base scale
                h = 1e-3 * base / 2 ** k
                up = float(loss_only(jnp.asarray(p0).at[i].add(h)))
                dn = float(loss_only(jnp.asarray(p0).at[i].add(-h)))
                fd = (up - dn) / (2 * h)
                row = {"step": h, "central_difference": fd, "abs_err": abs(fd - g[i]),
                       "rel_err": abs(fd - g[i]) / max(abs(g[i]), 1e-12)}
                if previous is not None:  # Richardson: cancels the O(h^2) truncation term
                    rich = (4 * fd - previous) / 3
                    row.update({"richardson": rich, "richardson_abs_err": abs(rich - g[i]),
                                "richardson_rel_err": abs(rich - g[i]) / max(abs(g[i]), 1e-12)})
                previous = fd
                results.append(row)
            candidates = [(r["abs_err"], r["rel_err"]) for r in results]
            candidates += [(r["richardson_abs_err"], r["richardson_rel_err"]) for r in results if "richardson" in r]
            best_abs, best_rel = min(candidates)
            best = {"abs_err": best_abs, "rel_err": best_rel}
            checks.append({"parameter": name_of(pairs[i]), "index": i, "property": prop, "autodiff": float(g[i]),
                           "steps": results, "best_abs_err": best["abs_err"], "best_rel_err": best["rel_err"]})
        Path(fd_path).write_text(json.dumps({"case": case, "gradient_inf_norm": float(np.max(np.abs(g))), "checks": checks}, indent=2) + "\n")
        record["fd_checks"] = len(checks)
    Path(out).with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "pairs"}))


if __name__ == "__main__":
    main()
