from importlib import import_module as _import_module

from differometor.setups import Setup
from differometor.simulate import run, run_build_step, run_simulation_step, simulate
from differometor.components import power_detector, signal_detector

__all__ = [
    "Setup",
    "build",
    "components",
    "plot",
    "power_detector",
    "run",
    "run_build_step",
    "run_simulation_step",
    "setups",
    "signal_detector",
    "simulate",
    "visualize_setup",
]


def __getattr__(name):
    # Defer the plotting dependency until a caller asks to visualize a setup.
    if name in ("plot", "visualize_setup"):
        plot_module = _import_module(".plot", __name__)
        globals()["visualize_setup"] = plot_module.visualize_setup
        return plot_module if name == "plot" else plot_module.visualize_setup
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
