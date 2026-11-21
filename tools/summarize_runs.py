#!/usr/bin/env python3
"""Summarise a multi-variant benchmark session (several runs per variant) into one comparison table.

    python tools/summarize_runs.py DIR baseline,cpp,stage3,fastreadout,v2_head,final,final_fast [--runs 3] [--pairs a:b,...]

DIR holds `<variant>_<i>.json` / `.npz` written by `benchmark.py --output ... --save-arrays ...`.
Reported per variant: median (min-max) over runs of the first-run total, the trace/lower+compile
sum and the warm median, speedups against the first variant, and per-phase first-run medians.
Numeric agreement uses run 1 of each variant against run 1 of the first variant. `--pairs`
additionally checks bit-identity between variants (e.g. `v2_head:final`).
Only needs numpy.
"""

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np


def load(d, name, runs):
    return [json.loads((d / f"{name}_{i}.json").read_text()) for i in range(1, runs + 1)]


def med(xs):
    return st.median(xs), min(xs), max(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("variants")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--pairs", default="")
    a = ap.parse_args()
    names = a.variants.split(",")
    data = {n: load(a.dir, n, a.runs) for n in names}
    base = names[0]

    def stat(n, key):
        return med([r["timing_seconds"][key] for r in data[n]])

    b_first, b_warm = stat(base, "end_to_end")[0], stat(base, "runtime_median")[0]
    env = data[base][0]["environment"]
    print(f"cpu: {env['cpu_model']} ({env['cpu_affinity_count']} cpu), python {env['python']}, "
          f"jax {env['packages']['jax']}, runs per variant: {a.runs}, warm repeats per run: "
          f"{len(data[base][0]['timing_seconds']['runtime_samples'])}\n")
    print(f"{'variant':<12}{'first run s (min-max)':>26}{'speedup':>9}{'compile s':>11}"
          f"{'warm med s (min-max)':>26}{'speedup':>9}{'max|dlogS| ref':>16}  PASS  xla_flags")
    for n in names:
        f, c, w = stat(n, "end_to_end"), stat(n, "trace_lower_and_compile"), stat(n, "runtime_median")
        err = max(r["validation"]["max_absolute_log_sensitivity_error"] for r in data[n])
        ok = all(r["validation"]["upstream_sensitivity_check_passed"] and r["validation"]["gradient_check"]["passed"]
                 for r in data[n])
        print(f"{n:<12}{f[0]:>10.3f} ({f[1]:.3f}-{f[2]:.3f}){b_first / f[0]:>8.1f}x{c[0]:>11.3f}"
              f"{w[0]:>10.4f} ({w[1]:.4f}-{w[2]:.4f}){b_warm / w[0]:>8.1f}x{err:>16.3e}  {'yes' if ok else 'NO '}"
              f"   {data[n][0]['environment'].get('xla_flags')}")

    keys = [p["key"] for p in data[base][0]["phases"]]
    print("\nfirst-run phase medians (seconds):")
    print(f"{'phase':<22}" + "".join(f"{n:>13}" for n in names))
    for k in keys:
        row = []
        for n in names:
            vals = [p["seconds"] for r in data[n] for p in r["phases"] if p["key"] == k]
            row.append(st.median(vals) if vals else float("nan"))
        print(f"{k:<22}" + "".join(f"{v:>13.4f}" for v in row))

    print("\nwarm phase medians (seconds, median over runs of each run's median):")
    for k in data[base][0]["timing_seconds"]["phase_runtime_medians"]:
        print(f"{k:<22}" + "".join(
            f"{st.median(r['timing_seconds']['phase_runtime_medians'][k] for r in data[n]):>13.4f}" for n in names))

    if not (a.dir / f"{base}_1.npz").exists():  # .npz files are gitignored; timings above need only the JSONs
        print("\n(no .npz arrays found: numeric agreement skipped)")
        return
    ref = np.load(a.dir / f"{base}_1.npz")
    print(f"\nnumeric agreement with '{base}' (saved arrays, run 1):")
    print(f"{'variant':<12}{'max|dlogS|':>12}{'grad rel.err':>14}{'grad cosine':>15}")
    for n in names[1:]:
        z = np.load(a.dir / f"{n}_1.npz")
        dl = np.max(np.abs(np.log(z["sensitivity"]) - np.log(ref["sensitivity"])))
        g, g0 = z["gradient"], ref["gradient"]
        rel = np.linalg.norm(g - g0) / np.linalg.norm(g0)
        cos = float(g @ g0 / (np.linalg.norm(g) * np.linalg.norm(g0)))
        print(f"{n:<12}{dl:>12.3e}{rel:>14.3e}{cos:>15.10f}")

    for pair in filter(None, a.pairs.split(",")):
        x, y = pair.split(":")
        zx, zy = np.load(a.dir / f"{x}_1.npz"), np.load(a.dir / f"{y}_1.npz")
        diff = {k: float(np.max(np.abs(zx[k] - zy[k]) / np.maximum(np.abs(zx[k]), 1e-300)))
                for k in zx.files if not np.array_equal(zx[k], zy[k])}
        print(f"\n{x} vs {y}: " + ("bit-identical on every saved array" if not diff else
                                  "max relative difference " + ", ".join(f"{k} {v:.2e}" for k, v in diff.items())))


if __name__ == "__main__":
    main()
