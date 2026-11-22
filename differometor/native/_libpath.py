"""Choose which compiled dfsparse library to load (no NumPy, so setup.py can use it too).

Order, first that exists and loads wins:
1. ``DIFFEROMETOR_NATIVE_LIB`` (explicit path);
2. a library built on this machine by ``build.py`` (next to this file);
3. a prebuilt library shipped in ``prebuilt/<os>-<arch>/``, best CPU level first.

Prebuilt libraries need no compiler and no build step. A prebuilt level is offered only
if the CPU reports every instruction-set feature it needs, so a library can never stop
the process with an illegal instruction. All levels are built with FMA contraction off
and give bit-identical results; higher levels are only faster.
"""

import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREBUILT = HERE / "prebuilt"

# x86-64 microarchitecture levels (psABI), as Linux /proc/cpuinfo flag names.
_V2 = {"cx16", "lahf_lm", "popcnt", "sse4_1", "sse4_2", "ssse3"}
_V3 = _V2 | {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave"}
_V4 = _V3 | {"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"}

# (file suffix, required flags), fastest first. "v1" is baseline x86-64 (SSE2).
X86_LEVELS = [("v4", _V4), ("v3", _V3), ("v1", set())]


def local_name():
    if sys.platform == "win32":
        return "dfsparse.dll"
    if sys.platform == "darwin":
        return "libdfsparse.dylib"
    return "libdfsparse.so"


def platform_tag():
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    system = {"win32": "windows", "darwin": "macos"}.get(sys.platform, sys.platform)
    if system.startswith("linux"):
        system = "linux"
    return f"{system}-{machine}"


def _cpu_flags():
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def prebuilt_candidates():
    tag = platform_tag()
    folder = PREBUILT / tag
    if tag == "linux-x86_64":
        flags = _cpu_flags()
        return [folder / f"libdfsparse-{level}.so" for level, needs in X86_LEVELS if needs <= flags]
    return []


def candidates():
    paths = []
    if os.environ.get("DIFFEROMETOR_NATIVE_LIB"):
        paths.append(Path(os.environ["DIFFEROMETOR_NATIVE_LIB"]))
    paths.append(HERE / local_name())
    paths.extend(prebuilt_candidates())
    return [path for path in paths if path.is_file()]


def loadable_prebuilt():
    """First prebuilt library for this machine that loads and has the expected ABI."""
    import ctypes

    for path in prebuilt_candidates():
        if not path.is_file():
            continue
        try:
            if ctypes.CDLL(str(path)).dfs_version() == 1:
                return path
        except (OSError, AttributeError):
            continue
    return None
