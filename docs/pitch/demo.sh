#!/usr/bin/env bash
# Live demo for the pitch: run the scored benchmark once, show the first-run total and PASS.
#
#   DEMO_SSH="ssh -p <port> root@<host>" ./docs/pitch/demo.sh   # on the AMD EPYC (the measured machine)
#   ./docs/pitch/demo.sh                                       # fallback: local dev container
#
# The cloud copy lives in /root/hk_final (final build) with the venv /root/hk_v2/venv, pinned to
# core 72 like results/amd-epyc-final/. Takes ~6 s end to end there (incl. validation).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
filter() { grep -E "^CPU:|^Workload|first-run total|^Repeat|^PASS|FAIL|Error" || true; }
if [[ -n "${DEMO_SSH:-}" ]]; then
  $DEMO_SSH 'cd /root/hk_final && taskset -c 72 ../hk_v2/venv/bin/python benchmark.py --cpus 1 --repeats 3 --output /tmp/demo.json' 2>&1 | filter
else
  ./container.bash exec python benchmark.py --cpus 1 --repeats 3 --output /tmp/demo.json 2>&1 | filter
fi
