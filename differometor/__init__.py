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
