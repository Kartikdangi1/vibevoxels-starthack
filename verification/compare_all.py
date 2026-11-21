"""Compare several benchmark.py JSON reports (and optionally their saved .npz arrays) side by side.

    python verification/compare_all.py baseline=base.json ours=ours.json cpp=cpp.json ...
The FIRST entry is the reference for speedups and for numeric differences (when a sibling .npz exists).
"""
import json
import sys
from pathlib import Path

import numpy as np


def main(args):
    runs = [a.split("=", 1) for a in args]
    reports = {n: json.loads(Path(p).read_text()) for n, p in runs}
    arrays = {}
    for n, p in runs:
        npz = Path(p).with_suffix(".npz")
        if npz.exists():
            arrays[n] = np.load(npz)
    ref = runs[0][0]
    t = lambda n: reports[n]["timing_seconds"]
    print(f"{'run':<14}{'first run s':>12}{'speedup':>9}{'warm med s':>12}{'speedup':>9}{'compile s':>11}{'max|dlogS| vs ref data':>24}")
    for n, _ in runs:
        v = reports[n]["validation"]
        print(f"{n:<14}{t(n)['end_to_end']:>12.3f}{t(ref)['end_to_end'] / t(n)['end_to_end']:>8.1f}x"
              f"{t(n)['runtime_median']:>12.4f}{t(ref)['runtime_median'] / t(n)['runtime_median']:>8.1f}x"
              f"{t(n)['trace_lower_and_compile']:>11.3f}{v['max_absolute_log_sensitivity_error']:>24.3e}")
    if ref in arrays:
        print(f"\nnumeric agreement with '{ref}' (saved arrays):")
        print(f"{'run':<14}{'max|dlogS|':>12}{'grad rel.err':>14}{'grad cosine':>13}{'loss diff':>12}")
        a = arrays[ref]
        for n, _ in runs[1:]:
            if n not in arrays:
                continue
            b = arrays[n]
            dlog = float(np.max(np.abs(np.log(b["sensitivity"]) - np.log(a["sensitivity"]))))
            ga, gb = a["gradient"], b["gradient"]
            rel = float(np.linalg.norm(gb - ga) / np.linalg.norm(ga))
            cos = float(np.dot(ga, gb) / (np.linalg.norm(ga) * np.linalg.norm(gb)))
            dl = abs(reports[n]["validation"]["loss"] - reports[ref]["validation"]["loss"])
            print(f"{n:<14}{dlog:>12.3e}{rel:>14.3e}{cos:>13.10f}{dl:>12.3e}")
    print("\nphase medians (warm, seconds):")
    keys = list(t(ref)["phase_runtime_medians"])
    print(f"{'phase':<16}" + "".join(f"{n:>14}" for n, _ in runs))
    for k in keys:
        print(f"{k:<16}" + "".join(f"{t(n)['phase_runtime_medians'].get(k, float('nan')):>14.4f}" for n, _ in runs))
    print("\ncpu:", {n: reports[n]["environment"]["cpu_model"] + f" ({reports[n]['environment']['cpu_affinity_count']} cpu)" for n, _ in runs})


if __name__ == "__main__":
    main(sys.argv[1:])
