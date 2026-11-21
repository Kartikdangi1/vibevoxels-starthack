"""Compare benchmark arrays from two runs, including every gradient entry.

Example:
    python tools/compare_arrays.py baseline.npz candidate.npz
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


CORE_KEYS = ("carrier", "signal", "noise", "gradient")


def compare(reference_path, candidate_path, rtol: float, atol: float) -> bool:
    with np.load(reference_path) as reference, np.load(candidate_path) as candidate:
        missing = [key for key in CORE_KEYS if key not in reference or key not in candidate]
        if missing:
            print(f"FAIL: required arrays missing from one or both files: {', '.join(missing)}")
            return False

        passed = True
        for key in CORE_KEYS:
            expected = np.asarray(reference[key])
            actual = np.asarray(candidate[key])
            if expected.shape != actual.shape:
                print(f"FAIL {key}: shape {actual.shape}, expected {expected.shape}")
                passed = False
                continue

            difference = np.abs(actual - expected)
            max_absolute = float(np.max(difference, initial=0.0))
            denominator = np.maximum(np.abs(expected), atol)
            max_relative = float(np.max(difference / denominator, initial=0.0))
            finite = bool(np.isfinite(actual).all())
            close = bool(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False))
            status = "PASS" if finite and close else "FAIL"
            print(
                f"{status} {key}: shape={actual.shape}, max_abs={max_absolute:.6g}, "
                f"max_scaled_rel={max_relative:.6g}, finite={finite}"
            )
            passed &= finite and close

        for key in ("sensitivity", "powers", "parameters", "frequencies"):
            if key in reference and key in candidate:
                expected = np.asarray(reference[key])
                actual = np.asarray(candidate[key])
                if expected.shape != actual.shape:
                    print(f"FAIL {key}: shape {actual.shape}, expected {expected.shape}")
                    passed = False
                    continue
                finite = bool(np.isfinite(actual).all())
                close = bool(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False))
                difference = np.abs(actual - expected)
                print(
                    f"{'PASS' if finite and close else 'FAIL'} {key}: "
                    f"max_abs={float(np.max(difference, initial=0.0)):.6g}"
                )
                passed &= finite and close

    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="baseline .npz produced by benchmark.py --save-arrays")
    parser.add_argument("candidate", help="candidate .npz produced by benchmark.py --save-arrays")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()
    if args.rtol < 0 or args.atol < 0:
        parser.error("Tolerances must be non-negative.")
    return 0 if compare(args.reference, args.candidate, args.rtol, args.atol) else 1


if __name__ == "__main__":
    sys.exit(main())
