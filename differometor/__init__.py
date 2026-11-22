import os as _os

# XLA compile settings. Differometor's numerical work runs on the host (NumPy and the
# native sparse solver), so XLA compiles only small glue programs: callbacks plus a few
# elementwise readouts. For those, LLVM optimisation and parallel codegen splitting are
# almost all of the compile time: about 20 ms per generated kernel at the defaults,
# against about 3 ms with the settings below. The settings change how fast XLA compiles,
# not what it computes. Fast-math stays off. The legacy (non-MLIR) emitters may round a
# complex abs or divide differently in the last bit. On the public benchmark the readout
# changes by 2.5e-17 relative and the gradient by 9e-11, the usual amplification of 1 ulp
# through the adjoint solves. Simulation outputs are bit-identical.
# They are defaults only:
# - a flag the user already put in XLA_FLAGS is kept;
# - DIFFEROMETOR_XLA_FAST_COMPILE=0 disables them;
# - they have no effect if a JAX backend was initialised before this import (XLA reads
#   XLA_FLAGS once).
_FAST_COMPILE_FLAGS = (
    "--xla_backend_optimization_level=0",
    "--xla_cpu_use_fusion_emitters=false",
    "--xla_cpu_parallel_codegen_split_count=1",
)


def _configure_xla():
    if _os.environ.get("DIFFEROMETOR_XLA_FAST_COMPILE", "1") == "0":
        return
    current = _os.environ.get("XLA_FLAGS", "")
    names = {token.split("=", 1)[0] for token in current.split()}
    extra = [flag for flag in _FAST_COMPILE_FLAGS if flag.split("=", 1)[0] not in names]
    if extra:
        _os.environ["XLA_FLAGS"] = " ".join([current, *extra]).strip()


_configure_xla()

from differometor.setups import Setup
from differometor.simulate import run, run_build_step, run_simulation_step, simulate
from differometor.components import power_detector, signal_detector

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
