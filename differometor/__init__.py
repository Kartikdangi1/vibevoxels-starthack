import os as _os


def _configure_xla():
    """Cheaper XLA:CPU compilation for the small XLA programs left around the host path.

    The simulation itself runs on the host (``differometor.host``); XLA only compiles the
    callback stubs and the caller's readout (detector indexing, abs, log, mean and their
    VJPs). For those few tiny fusions the per-fusion code generation dominates: the
    MLIR fusion emitters cost ~0.1 s more than XLA's classic emitters on the benchmark.

    XLA reads ``XLA_FLAGS`` when the CPU client is created, so this must run before the
    first JAX computation (the benchmark initialises the backend after importing us).
    Flags the user already set are never overridden.

    DIFFEROMETOR_XLA_FLAGS:
      ``exact`` (default) classic emitters only; outputs bit-identical to XLA's default.
      ``fast``  also LLVM -O0 for the generated kernels: another ~0.1 s less compile,
                but the readout's float rounding can change (~1e-11 relative on the loss).
      ``0``     leave XLA's defaults alone.
    """
    mode = _os.environ.get("DIFFEROMETOR_XLA_FLAGS", "exact").lower()
    if mode in ("0", "off", "none", ""):
        return
    wanted = {"xla_cpu_use_fusion_emitters": "false"}
    if mode == "fast":
        wanted["xla_backend_optimization_level"] = "0"
    current = _os.environ.get("XLA_FLAGS", "")
    extra = [f"--{k}={v}" for k, v in wanted.items() if f"--{k}" not in current]
    if extra:
        _os.environ["XLA_FLAGS"] = " ".join([current, *extra]).strip()


_configure_xla()

from differometor.setups import Setup  # noqa: E402
from differometor.simulate import run, run_build_step, run_simulation_step, simulate  # noqa: E402
from differometor.components import power_detector, signal_detector  # noqa: E402

__all__ = [
    "Setup",
    "run",
    "run_build_step",
    "run_simulation_step",
    "simulate",
    "visualize_setup",
    "power_detector",
    "signal_detector",
]


def __getattr__(name):
    # Plotly's graph_objects import is expensive and plotting is not part of the
    # simulation or benchmark path. Preserve the public API, but load it only
    # when a caller requests visualization.
    if name == "visualize_setup":
        from differometor.plot import visualize_setup

        globals()[name] = visualize_setup
        return visualize_setup
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
