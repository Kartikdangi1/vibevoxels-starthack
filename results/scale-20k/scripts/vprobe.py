"""Count which simulation backend actually runs, without modifying the simulator.

Wraps (from outside, after import) the entry points of every solver path:
  - differometor.simulate.simulate_sparse_jax   (in-graph JAX fallback of the host path)
  - differometor.host.make_host_simulate        (host path; None result = fallback)
  - differometor.native.NativeBatchedLU          (C++ solver)
  - differometor.host.SuperLUBatchedLU           (SciPy SuperLU solver)
  - differometor.host.RefinedLU                  (SuperLU single-system factorisation)
On the official reference code none of the host/native modules exist; that is recorded.
"""
import atexit
import json
import os

COUNTS = {}


def _wrap(module, name, key, on_result=None):
    original = getattr(module, name)

    if isinstance(original, type):
        class Counted(original):  # noqa: D401 - subclass keeps isinstance checks valid
            def __init__(self, *args, **kwargs):
                COUNTS[key] = COUNTS.get(key, 0) + 1
                super().__init__(*args, **kwargs)
        Counted.__name__ = original.__name__
        setattr(module, name, Counted)
        return

    import functools

    @functools.wraps(original)  # keeps inspect.signature() intact (run_build_step checks it)
    def wrapper(*args, **kwargs):
        COUNTS[key] = COUNTS.get(key, 0) + 1
        result = original(*args, **kwargs)
        if on_result:
            on_result(result)
        return result

    setattr(module, name, wrapper)


def install(report_path=None):
    import importlib
    import sys

    importlib.import_module("differometor.simulate")
    # `differometor.simulate` the attribute is the simulate() function; take the module itself.
    sim = sys.modules["differometor.simulate"]

    info = {"differometor_file": None, "native_library": None, "host_module": False, "native_module": False}
    import differometor
    info["differometor_file"] = differometor.__file__
    if hasattr(sim, "simulate_sparse_jax"):
        _wrap(sim, "simulate_sparse_jax", "simulate_sparse_jax (JAX fallback)")
    # simulate_dense is not wrapped: run_build_step reads simulate_dense.__code__ directly.
    try:
        importlib.import_module("differometor.host")
        host = sys.modules["differometor.host"]
        info["host_module"] = True

        def host_result(result):
            key = "make_host_simulate -> host path" if result[0] is not None else "make_host_simulate -> None (fallback)"
            COUNTS[key] = COUNTS.get(key, 0) + 1

        _wrap(host, "make_host_simulate", "make_host_simulate calls", host_result)
        _wrap(host, "SuperLUBatchedLU", "SuperLUBatchedLU (SciPy)")
        _wrap(host, "RefinedLU", "RefinedLU (SciPy SuperLU)")
    except ImportError:
        pass
    try:
        importlib.import_module("differometor.native")
        native = sys.modules["differometor.native"]
        info["native_module"] = True
        _wrap(native, "NativeBatchedLU", "NativeBatchedLU (C++)")
        path = native.library_path() if hasattr(native, "library_path") else None
        info["native_library"] = str(path) if path else None
        info["native_available"] = native.available()
    except ImportError:
        pass

    def dump():
        record = {"info": info, "counts": COUNTS}
        text = json.dumps(record, indent=2)
        if report_path:
            with open(report_path, "w") as fh:
                fh.write(text + "\n")
        print("BACKEND-PROBE " + json.dumps(record), flush=True)

    atexit.register(dump)
    return info
