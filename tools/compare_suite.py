#!/usr/bin/env python3
"""Compare benchmark_suite.py runs of the unmodified baseline vs the optimised code.

    python tools/compare_suite.py BASE_DIR NEW_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

base_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for new_json in sorted(new_dir.glob("*.json")):
    case = new_json.stem
    new = json.loads(new_json.read_text())
    base_json = base_dir / f"{case}.json"
    if not base_json.exists():
        rows.append((case, new, None, None))
        continue
    base = json.loads(base_json.read_text())
    a, b = np.load(base_dir / f"{case}.npz"), np.load(new_dir / f"{case}.npz")
    log_err = float(np.max(np.abs(np.log(a["sensitivity"]) - np.log(b["sensitivity"]))))
    ga, gb = a["gradient"], b["gradient"]
    grad_err = float(np.max(np.abs(ga - gb)) / max(float(np.max(np.abs(ga))), 1e-300))
    rows.append((case, new, base, (log_err, grad_err)))

print(f"{'case':<18}{'freq':>5}{'vars':>5} | {'base e2e':>9}{'new e2e':>9}{'x':>7} | {'base run':>10}{'new run':>10}{'x':>7} | {'|dlog sens|':>11}{'rel dgrad':>10}  ok")
for case, new, base, err in rows:
    n = new["timing_seconds"]
    if base is None:
        print(f"{case:<18}{new['frequencies']:>5}{new['design_variables']:>5} | baseline did not complete (new e2e {n['end_to_end']:.2f} s, runtime {n['runtime_median'] * 1e3:.1f} ms)")
        continue
    b = base["timing_seconds"]
    cond = new["validation"]["conditioning"].get("well_conditioned") and base["validation"]["conditioning"].get("well_conditioned")
    gcheck = new["validation"]["gradient_check"].get("passed") and base["validation"]["gradient_check"].get("passed")
    ok = err[0] <= 1e-5 and err[1] <= 1e-3 and cond and gcheck
    print(f"{case:<18}{new['frequencies']:>5}{new['design_variables']:>5} | {b['end_to_end']:>8.2f}s{n['end_to_end']:>8.2f}s{b['end_to_end'] / n['end_to_end']:>6.1f}x"
          f" | {b['runtime_median'] * 1e3:>8.1f}ms{n['runtime_median'] * 1e3:>8.1f}ms{b['runtime_median'] / n['runtime_median']:>6.1f}x"
          f" | {err[0]:>11.1e}{err[1]:>10.1e}  {'PASS' if ok else 'CHECK'}")
