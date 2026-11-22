#!/usr/bin/env bash
# Live demo for the pitch: run the scored benchmark once, show the first-run total and PASS.
#
#   DEMO_SSH="ssh -p <port> root@<host>" ./docs/pitch/demo.sh   # on the AMD EPYC (the measured machine)
#   DEMO_SETUP=aligo DEMO_SSH=... ./docs/pitch/demo.sh          # the official aLIGO benchmark instead
#   ./docs/pitch/demo.sh                                       # fallback: local dev container
#
# The cloud copy is /root/validation/opt_prebuilt: the shipped commit c8066b8 with only the prebuilt
# solver (nothing compiled) and the official scoring scripts from feyerlabs 83f8d2f, as validated in
# FEYER_FINAL_VALIDATION.md. Venv /root/hk_v2/venv, pinned to core 72. About 6 s end to end there.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
script=benchmark.py
[[ "${DEMO_SETUP:-grid}" == "aligo" ]] && script=benchmark_aligo.py
filter() { grep -E "^CPU:|^Workload|first-run total|^Repeat|^PASS|FAIL|Error" || true; }
if [[ -n "${DEMO_SSH:-}" ]]; then
  $DEMO_SSH "cd /root/validation/opt_prebuilt && taskset -c 72 /root/hk_v2/venv/bin/python $script --cpus 1 --repeats 3 --output /tmp/demo.json" 2>&1 | filter
else
  ./container.bash exec python "$script" --cpus 1 --repeats 3 --output /tmp/demo.json 2>&1 | filter
fi
