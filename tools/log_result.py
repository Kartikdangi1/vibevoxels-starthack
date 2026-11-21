"""Append a benchmark.py result to the shared results/leaderboard.csv.

    python tools/log_result.py benchmark.json --note "single LU factorization" --status PASS

Records git commit, who, first-run total, median warm run, compile time and your note.
Commit leaderboard.csv so the team sees which ideas won.
"""

import argparse
import csv
import getpass
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "leaderboard.csv"
FIELDS = ["utc", "who", "commit", "dirty", "first_run_s", "compile_s", "warm_median_s", "status", "note"]


def git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path, help="benchmark.json from benchmark.py")
    ap.add_argument("--note", default="", help="what you changed")
    ap.add_argument("--status", choices=["PASS", "FAIL"], required=True, help="did benchmark.py print PASS?")
    args = ap.parse_args()

    r = json.loads(args.report.read_text())
    phases = r["phases"]
    row = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "who": git("config", "user.name") or getpass.getuser(),
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": "yes" if git("status", "--porcelain", "--", "differometor") else "no",
        "first_run_s": round(sum(p["seconds"] for p in phases), 3),
        "compile_s": round(sum(p["seconds"] for p in phases if p.get("category") == "compile"), 3),
        "warm_median_s": r.get("timing_seconds", {}).get("runtime_median", ""),
        "status": args.status,
        "note": args.note,
    }
    CSV.parent.mkdir(exist_ok=True)
    new = not CSV.exists()
    with CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"logged to {CSV}: {row}")


if __name__ == "__main__":
    main()
