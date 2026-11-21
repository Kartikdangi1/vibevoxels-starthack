"""Run the whole verification suite and print one summary table. Exit 0 only if everything passes.

    python verification/run_all.py            # full suite
    python verification/run_all.py --quick    # smaller sweeps, a couple of minutes
    python verification/run_all.py --only analytic gradients

Each stage runs in its own process so a crash or leaked global state in one cannot hide a failure
in another. A JSON summary is written to verification/last_run.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
STAGES = {
    "analytic": (["verify_analytic.py"], ["verify_analytic.py"]),
    "vs_dense": (["verify_vs_dense.py"], ["verify_vs_dense.py", "--quick"]),
    "invariants": (["verify_invariants.py"], ["verify_invariants.py"]),
    "gradients": (["verify_gradients.py", "--n-freq", "16", "--samples", "20"],
                  ["verify_gradients.py", "--n-freq", "6", "--samples", "8"]),
    "reference": (["verify_reference.py"], ["verify_reference.py", "--n-freq", "21"]),
    "generalization": (["verify_generalization.py"], ["verify_generalization.py", "--quick"]),
    "scaling": (["verify_scaling.py"], ["verify_scaling.py", "--quick"]),
    "pytest": (["-m", "pytest", "-q", str(HERE.parent / "tests")], ["-m", "pytest", "-q", str(HERE.parent / "tests")]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="+", choices=list(STAGES))
    args = ap.parse_args()

    results = []
    for name in args.only or STAGES:
        full, quick = STAGES[name]
        argv = quick if args.quick else full
        cmd = [sys.executable] + ([str(HERE / argv[0])] + argv[1:] if argv[0].endswith(".py") else argv)
        print(f"\n##### {name}: {' '.join(argv)}", flush=True)
        t0 = perf_counter()
        code = subprocess.call(cmd, cwd=HERE.parent)
        results.append({"stage": name, "passed": code == 0, "seconds": round(perf_counter() - t0, 1)})

    print("\n==================== SUMMARY ====================")
    for r in results:
        print(f"  {'PASS' if r['passed'] else 'FAIL':<5}{r['stage']:<12}{r['seconds']:>8.1f}s")
    ok = all(r["passed"] for r in results)
    print("  " + ("ALL VERIFICATION STAGES PASSED" if ok else "VERIFICATION FAILED"))
    (HERE / "last_run.json").write_text(json.dumps({"quick": args.quick, "passed": ok, "stages": results}, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
