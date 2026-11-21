"""Fast correctness check: one run, no repeats, optionally on a subset of frequencies.

Each frequency is solved independently, so a subset is checked against the matching
reference values. Far cheaper (time and RAM) than benchmark.py.

    python tools/check.py                       # 5 evenly spaced frequencies
    python tools/check.py --n-freq 20 --save-arrays results/cand.npz
    python tools/check.py --full                # all 101 frequencies (needs ~14 GiB RAM)

Exit code 0 = PASS, 1 = FAIL. Compare saved arrays with tools/compare.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-freq", type=int, default=5, help="number of evenly spaced frequencies (default 5)")
    ap.add_argument("--full", action="store_true", help="use all 101 frequencies")
    ap.add_argument("--save-arrays", type=Path, help="save outputs/gradient to .npz for tools/compare.py")
    ap.add_argument("--atol", type=float, default=1e-5, help="max |log(sens) - log(ref)|")
    args = ap.parse_args()

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"
    t0 = perf_counter()
    import jax
    import jax.numpy as jnp
    import numpy as np
    import differometor as df
    from differometor.setups import constrain_inter_grid_cell_spaces

    jax.config.update("jax_enable_compilation_cache", False)

    data = ROOT / "examples" / "data"
    fixture = json.loads((data / "uifo_800_3000.json").read_text())
    reference = json.loads((data / "uifo_800_3000_test_data.json").read_text())
    all_freqs = np.asarray(reference["frequencies"], dtype=np.float64)
    all_ref = np.asarray(reference["sensitivities"], dtype=np.float64)
    idx = np.arange(len(all_freqs)) if args.full else np.unique(np.linspace(0, len(all_freqs) - 1, args.n_freq).round().astype(int))
    freqs, ref = jnp.asarray(all_freqs[idx]), jnp.asarray(all_ref[idx])

    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    comps = fixture["nodes"] | fixture["edges"]
    vals = []
    for pair in pairs:
        name, prop = pair[0] if isinstance(pair[0], list) else pair
        vals.append(comps[name]["properties"][prop])
    params = jnp.asarray(vals, dtype=jnp.float64)
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], freqs, pairs)
    ports = jnp.asarray(ports)

    def evaluate(p, arrays, ports, ref):
        carrier, signal, noise = df.simulate(**dict(arrays, optimized_parameters=p))
        sig = df.signal_detector(carrier, signal)[ports].squeeze()
        sens = noise / jnp.abs(sig)
        loss = jnp.mean(jnp.log10(sens / ref))
        return loss, (carrier, signal, noise, sens, df.power_detector(carrier))

    t1 = perf_counter()
    fn = jax.jit(jax.value_and_grad(evaluate, has_aux=True))
    compiled = fn.lower(params, arrays, ports, ref).compile()
    t2 = perf_counter()
    (loss, outs), grad = jax.block_until_ready(compiled(params, arrays, ports, ref))
    t3 = perf_counter()
    print(f"frequencies: {len(idx)}/{len(all_freqs)}  setup {t1 - t0:.1f}s  compile {t2 - t1:.1f}s  run {t3 - t2:.1f}s")

    errors = []
    for i, leaf in enumerate(jax.tree_util.tree_leaves((loss, outs, grad))):
        if not np.isfinite(np.asarray(leaf)).all():
            errors.append(f"non-finite value in output leaf {i}")
    sens = np.asarray(outs[3])
    if sens.shape != ref.shape or np.any(sens <= 0):
        errors.append("sensitivity must be positive, one per frequency")
    else:
        worst = float(np.max(np.abs(np.log(sens) - np.log(np.asarray(ref)))))
        print(f"max |log sens - log ref|: {worst:.3e} (limit {args.atol:g})")
        if worst > args.atol:
            errors.append(f"sensitivity error {worst:.3e} > {args.atol:g}")
    if grad.shape != params.shape or not np.any(np.asarray(grad) != 0):
        errors.append("gradient must have 193 entries and not be all zero")

    if args.save_arrays:
        args.save_arrays.parent.mkdir(parents=True, exist_ok=True)
        carrier, signal, noise, sens_, powers = (np.asarray(x) for x in outs)
        np.savez(args.save_arrays, freq_idx=idx, carrier=carrier, signal=signal, noise=noise,
                 sensitivity=sens_, powers=powers, gradient=np.asarray(grad), loss=np.asarray(loss),
                 setup_s=t1 - t0, compile_s=t2 - t1, run_s=t3 - t2)
        print(f"saved {args.save_arrays}")

    if errors:
        print("FAIL: " + "; ".join(errors))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
