"""Run an official benchmark script UNCHANGED, in-process, with the backend probe attached.

    python run_official.py PROBE_JSON SCRIPT [benchmark args...]

The script's own code (affinity, imports, timers, validation) runs exactly as with
`python SCRIPT args`. The probe only wraps solver entry points to count calls. It adds a
few microseconds per wrapped call; timing evidence comes from runs without the probe.
"""
import os
import runpy
import sys

probe_json, script, *args = sys.argv[1:]
here = os.getcwd()
sys.path.insert(0, here)
# The benchmark pins CPUs before importing JAX; importing differometor first would import
# JAX early. Probe runs are for backend evidence, not timing, so this is acceptable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vprobe  # noqa: E402

# The tools folder also holds copies of the official scripts; drop it so that
# `from benchmark import main` (inside benchmark_aligo.py) resolves to the code under test.
sys.path.remove(os.path.dirname(os.path.abspath(__file__)))
vprobe.install(probe_json)
sys.argv = [script, *args]
runpy.run_path(script, run_name="__main__")
