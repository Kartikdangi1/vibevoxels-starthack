#!/usr/bin/env python3
"""Compare two benchmark.py --save-arrays archives (baseline vs candidate).

Usage: python tools/compare_arrays.py benchmarks/runs/baseline.npz benchmarks/runs/candidate.npz

Prints, per array, the max absolute and max relative difference, and for
sensitivity the max |log(candidate) - log(baseline)| (the quantity the
official check bounds by 1e-5 against the public reference).
"""
import sys

import numpy as np


def main(baseline_path, candidate_path):
    baseline = np.load(baseline_path)
    candidate = np.load(candidate_path)
    for name in baseline.files:
        a, b = baseline[name], candidate[name]
        if a.shape != b.shape:
            print(f"{name:12s} SHAPE MISMATCH {a.shape} vs {b.shape}")
            continue
        diff = np.abs(a - b)
        scale = np.maximum(np.abs(a), np.finfo(float).tiny)
        print(f"{name:12s} shape={str(a.shape):16s} max_abs={diff.max():.3e} max_rel={np.max(diff / scale):.3e}")
    log_diff = np.abs(np.log(candidate["sensitivity"]) - np.log(baseline["sensitivity"]))
    print(f"sensitivity max |log(candidate) - log(baseline)| = {log_diff.max():.3e}")
    g_a, g_b = baseline["gradient"], candidate["gradient"]
    print(f"gradient relative L2 difference = {np.linalg.norm(g_a - g_b) / np.linalg.norm(g_a):.3e}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
