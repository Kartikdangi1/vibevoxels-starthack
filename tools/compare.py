"""Compare a baseline and a candidate run.

    python tools/compare.py benchmarks/cpu-baseline.json benchmark.json     # timing per phase
    python tools/compare.py base.npz cand.npz                               # numeric differences (tools/check.py or
                                                                           #   benchmark.py --save-arrays output)
Only needs numpy; works on Windows or in the container.
"""

import json
import sys
from pathlib import Path

import numpy as np


def compare_json(a, b):
    ja, jb = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    pa = {p["key"]: p["seconds"] for p in ja["phases"]}
    pb = {p["key"]: p["seconds"] for p in jb["phases"]}
    print(f"{'phase':<28}{'base s':>10}{'cand s':>10}{'speedup':>9}")
    for k in pa:
        if k in pb:
            print(f"{k:<28}{pa[k]:>10.3f}{pb[k]:>10.3f}{pa[k] / pb[k] if pb[k] else float('inf'):>8.2f}x")
    ta, tb = sum(pa.values()), sum(pb.values())
    print(f"{'TOTAL first run':<28}{ta:>10.3f}{tb:>10.3f}{ta / tb:>8.2f}x")
    ma, mb = ja.get("timing_seconds", {}).get("runtime_median"), jb.get("timing_seconds", {}).get("runtime_median")
    if ma and mb:
        print(f"{'warm run median':<28}{ma:>10.3f}{mb:>10.3f}{ma / mb:>8.2f}x")


def compare_npz(a, b):
    za, zb = np.load(a), np.load(b)
    if "freq_idx" in za and "freq_idx" in zb and not np.array_equal(za["freq_idx"], zb["freq_idx"]):
        sys.exit("Different frequency subsets; rerun both with the same --n-freq.")
    print(f"{'array':<14}{'max abs diff':>14}{'max rel diff':>14}")
    for k in sorted(set(za.files) & set(zb.files)):
        if k.endswith("_s") or k == "freq_idx":
            continue
        x, y = np.asarray(za[k]), np.asarray(zb[k])
        if x.shape != y.shape:
            print(f"{k:<14}  shape mismatch {x.shape} vs {y.shape}")
            continue
        d = np.abs(x - y)
        rel = d / np.maximum(np.abs(x), 1e-300)
        print(f"{k:<14}{d.max():>14.3e}{rel.max():>14.3e}")
    if "sensitivity" in za and "sensitivity" in zb:
        worst = np.max(np.abs(np.log(za["sensitivity"]) - np.log(zb["sensitivity"])))
        print(f"max |log sens diff|: {worst:.3e}  ({'OK' if worst <= 1e-5 else 'EXCEEDS 1e-5'})")
    for k in ("setup_s", "compile_s", "run_s"):
        if k in za and k in zb:
            print(f"{k}: {float(za[k]):.2f} -> {float(zb[k]):.2f}")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = sys.argv[1:]
    if a.endswith(".json") and b.endswith(".json"):
        compare_json(a, b)
    elif a.endswith(".npz") and b.endswith(".npz"):
        compare_npz(a, b)
    else:
        sys.exit("Give two .json or two .npz files.")


if __name__ == "__main__":
    main()
