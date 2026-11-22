#!/usr/bin/env python3
"""Extra benchmark cases, timed with the same phases as benchmark.py / benchmark_aligo.py.

    python benchmark_suite.py --list
    python benchmark_suite.py --case voyager_space --cpus 1 --output out/voyager_space.json --save-arrays out/voyager_space.npz

There is no public reference sensitivity for these setups, so correctness is checked by running the
SAME case on the unmodified baseline and on the optimised code and comparing the saved arrays
(see tools/compare_suite.py). Locally each run only checks finiteness, positivity and one
central-difference gradient spot check (recorded, not fatal).
"""
from time import perf_counter

IMPORT_START = perf_counter()
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"

cpu_parser = argparse.ArgumentParser(add_help=False)
cpu_parser.add_argument("--cpus", type=int)
cpu_options, _ = cpu_parser.parse_known_args()
if cpu_options.cpus is not None:
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[:cpu_options.cpus])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(cpu_options.cpus)

import jax
import jax.numpy as jnp
import numpy as np
import differometor as df
from differometor.setups import aligo, constrain_inter_grid_cell_spaces, uifo, voyager

IMPORT_SECONDS = perf_counter() - IMPORT_START
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
TARGET = 1e-22  # arbitrary fixed positive reference; the loss is mean(log10(sensitivity / TARGET))

# name -> (constructor, number of frequencies, tie grid lengths?)
CASES = {
    "voyager_space":     (lambda: voyager(mode="space_modulation")[0], 101, False),
    "voyager_amplitude": (lambda: voyager(mode="amplitude_modulation")[0], 101, False),
    "voyager_frequency": (lambda: voyager(mode="frequency_modulation")[0], 101, False),
    "aligo_amplitude":   (lambda: aligo(mode="amplitude_modulation")[0], 101, False),
    "aligo_frequency":   (lambda: aligo(mode="frequency_modulation")[0], 101, False),
    "aligo_21f":         (lambda: aligo(mode="space_modulation")[0], 21, False),
    "aligo_301f":        (lambda: aligo(mode="space_modulation")[0], 301, False),
    "uifo2_space":       (lambda: uifo(2, random=True, random_seed=3, mode="space_modulation")[0], 101, True),
    "uifo3_space":       (lambda: uifo(3, random=True, random_seed=3, mode="space_modulation")[0], 101, True),
    "uifo3_amplitude":   (lambda: uifo(3, random=True, random_seed=5, mode="amplitude_modulation")[0], 101, True),
    # extra candidates, screened for conditioning before entering SUITE
    "uifo2_seed1":       (lambda: uifo(2, random=True, random_seed=1, mode="space_modulation")[0], 101, True),
    "uifo2_seed2":       (lambda: uifo(2, random=True, random_seed=2, mode="space_modulation")[0], 101, True),
    "uifo2_seed4":       (lambda: uifo(2, random=True, random_seed=4, mode="space_modulation")[0], 101, True),
    "uifo2_amplitude":   (lambda: uifo(2, random=True, random_seed=3, mode="amplitude_modulation")[0], 101, True),
    "uifo2_frequency":   (lambda: uifo(2, random=True, random_seed=3, mode="frequency_modulation")[0], 101, True),
    "voyager_21f":       (lambda: voyager(mode="space_modulation")[0], 21, False),
    "voyager_301f":      (lambda: voyager(mode="space_modulation")[0], 301, False),
    "aligo_5f":          (lambda: aligo(mode="space_modulation")[0], 5, False),
}


# Final suite: cases that passed the conditioning probe (a 1e-9 relative nudge moves log sensitivity < 1e-5)
# and whose unmodified baseline fits in 15 GB. Screened out as ill-conditioned: voyager_{amplitude,frequency},
# uifo3_space, uifo2_frequency, aligo_{amplitude,frequency}.
SUITE = ["aligo_5f", "aligo_21f", "aligo_301f", "voyager_21f", "voyager_space", "voyager_301f",
         "uifo2_space", "uifo2_amplitude", "uifo2_seed2", "uifo3_amplitude"]


def forward_with_pullback(parameters, arrays):
    return jax.vjp(lambda p: df.simulate(**dict(arrays, optimized_parameters=p)), parameters)


def readout(outputs, detector_ports, reference):
    carrier, signal, noise = outputs
    signals = df.signal_detector(carrier, signal)[detector_ports]
    detected = signals.squeeze() if detector_ports.shape[0] == 1 else signals[0] - signals[1]
    sensitivity = (noise / jnp.abs(detected)).squeeze()
    return jnp.mean(jnp.log10(sensitivity / reference)), (sensitivity, df.power_detector(carrier))


def sensitivity_with_pullback(outputs, detector_ports, reference):
    return jax.vjp(lambda values: readout(values, detector_ports, reference), outputs, has_aux=True)


def backward(simulation_pullback, readout_pullback, cotangent):
    return simulation_pullback(readout_pullback(cotangent)[0])[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=sorted(CASES))
    parser.add_argument("--list", action="store_true", help="print the final suite")
    parser.add_argument("--list-all", action="store_true", help="print every screened candidate")
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-arrays", type=Path)
    args = parser.parse_args()
    if args.list_all:
        print("\n".join(sorted(CASES)))
        return
    if args.list or not args.case:
        print("\n".join(SUITE))
        return

    phases = []

    def timed(key, function):
        start = perf_counter()
        result = function()
        jax.block_until_ready(result)
        phases.append({"key": key, "seconds": perf_counter() - start})
        print(f"{key}: {phases[-1]['seconds']:.6f} s", flush=True)
        return result

    def compile_phase(key, function, *values):
        lowered = timed(key + "_lower", lambda: jax.jit(function).lower(*values))
        return timed(key + "_compile", lowered.compile)

    phases.append({"key": "imports", "seconds": IMPORT_SECONDS})

    def initialize():
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_enable_compilation_cache", False)
        return jax.devices("cpu")

    timed("initialize", initialize)
    constructor, n_freq, tied = CASES[args.case]

    def build_inputs():
        setup = constructor()
        fixture = deepcopy(setup.to_data())
        if tied:
            pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
        else:
            pairs = [list(pair) for pair in setup.parameters if pair[1] in PROPERTIES]
        initial = []
        for pair in pairs:
            members = pair if isinstance(pair[0], list) else [pair]
            values = [fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop] for name, prop in members]
            initial.append(values[0])
        parameters = jnp.asarray(initial, dtype=jnp.float64)
        frequencies = jnp.asarray(np.logspace(np.log10(20.0), np.log10(5000.0), n_freq), dtype=jnp.float64)
        arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
        arrays["optimized_parameters"] = parameters
        return arrays, jnp.asarray(ports), parameters, frequencies, pairs, jnp.asarray(1.0, dtype=jnp.float64)

    arrays, ports, parameters, frequencies, pairs, one = timed("build", build_inputs)
    reference = jnp.asarray(TARGET, dtype=jnp.float64)
    print(f"Case {args.case}: {len(frequencies)} frequencies; {len(parameters)} design variables", flush=True)

    forward = compile_phase("forward", forward_with_pullback, parameters, arrays)
    outputs, simulation_pullback = timed("forward", lambda: forward(parameters, arrays))
    sensitivity_fn = compile_phase("sensitivity", sensitivity_with_pullback, outputs, ports, reference)
    loss, readout_pullback, (sensitivity, powers) = timed("sensitivity", lambda: sensitivity_fn(outputs, ports, reference))
    gradient_fn = compile_phase("gradient", backward, simulation_pullback, readout_pullback, one)
    gradient = timed("gradient", lambda: gradient_fn(simulation_pullback, readout_pullback, one))
    host = timed("host_transfer", lambda: jax.device_get((outputs, sensitivity, powers, loss, gradient)))
    first = list(phases)
    total = sum(p["seconds"] for p in first)
    print(f"Complete first-run total: {total:.6f} s", flush=True)

    samples = []
    for index in range(args.repeats):
        del simulation_pullback, readout_pullback
        phases.clear()
        outputs, simulation_pullback = timed("forward", lambda: forward(parameters, arrays))
        loss, readout_pullback, (sensitivity, powers) = timed("sensitivity", lambda: sensitivity_fn(outputs, ports, reference))
        gradient = timed("gradient", lambda: gradient_fn(simulation_pullback, readout_pullback, one))
        host = timed("host_transfer", lambda: jax.device_get((outputs, sensitivity, powers, loss, gradient)))
        samples.append({p["key"]: p["seconds"] for p in phases})
    del simulation_pullback, readout_pullback

    # Validation (outside the timers)
    host_outputs, host_sensitivity, host_powers, host_loss, host_gradient = host
    finite = all(bool(np.isfinite(v).all()) for v in jax.tree_util.tree_leaves(host))
    positive = bool(np.all(host_sensitivity > 0))
    nonzero = bool(np.any(host_gradient != 0))
    check = {"checked": False}
    conditioning = {"checked": False}
    if finite and nonzero:
        evaluate = jax.jit(lambda p, a, d, t: readout(df.simulate(**dict(a, optimized_parameters=p)), d, t))
        scale = jnp.maximum(jnp.abs(parameters), 1.0)
        direction = jnp.asarray(np.random.default_rng(0).standard_normal(parameters.shape)) * scale
        direction = direction / jnp.linalg.norm(direction)
        # (1) conditioning probe: a ~1e-9 relative nudge must not move the log sensitivity noticeably.
        # If it does, no solver can reproduce this case to benchmark accuracy in double precision.
        nudged = evaluate(parameters + 1e-9 * direction * scale, arrays, ports, reference)[1][0]
        sens_shift = float(np.max(np.abs(np.log(np.asarray(nudged)) - np.log(host_sensitivity))))
        conditioning = {"checked": True, "nudge_relative": 1e-9, "max_abs_dlog_sensitivity": sens_shift,
                        "well_conditioned": sens_shift < 1e-5}
        # (2) directional central-difference check of the full gradient
        autodiff = float(jnp.dot(host_gradient, direction))
        errors = {}
        for step in (1e-6, 1e-7):
            up = float(evaluate(parameters + step * direction, arrays, ports, reference)[0])
            dn = float(evaluate(parameters - step * direction, arrays, ports, reference)[0])
            estimate = (up - dn) / (2 * step)
            errors[step] = abs(autodiff - estimate) / max(abs(estimate), 1e-30)
        check = {"checked": True, "autodiff_directional": autodiff, "rel_error_by_step": {str(k): v for k, v in errors.items()},
                 "passed": min(errors.values()) < 1e-3}
    keys = ["forward", "sensitivity", "gradient", "host_transfer"]
    totals = [sum(s.values()) for s in samples]
    report = {
        "case": args.case, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "frequencies": len(frequencies), "design_variables": len(parameters),
        "phases_first_run": first,
        "timing_seconds": {
            "end_to_end": total,
            "trace_lower_and_compile": sum(p["seconds"] for p in first if p["key"].endswith(("_lower", "_compile"))),
            "runtime_median": statistics.median(totals),
            "phase_runtime_medians": {k: statistics.median(s[k] for s in samples) for k in keys},
        },
        "validation": {"conditioning": conditioning, "finite": finite, "positive_sensitivity": positive, "gradient_nonzero": nonzero, "gradient_check": check,
                       "loss": float(host_loss)},
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    if args.save_arrays:
        np.savez(args.save_arrays, sensitivity=host_sensitivity, powers=host_powers, gradient=host_gradient,
                 parameters=np.asarray(parameters), loss=np.asarray(host_loss))
    print(f"RESULT {args.case}: end-to-end {total:.3f} s, runtime median {statistics.median(totals) * 1e3:.2f} ms, "
          f"finite={finite} well_conditioned={conditioning.get('well_conditioned')} (nudge {conditioning.get('max_abs_dlog_sensitivity')}) grad_check_passed={check.get('passed')}", flush=True)


if __name__ == "__main__":
    main()
