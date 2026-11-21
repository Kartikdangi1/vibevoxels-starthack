"""Compile the native sparse solver (`dfsparse.cpp`) into a shared library next to this file.

Run once at install time (``setup.py`` does it automatically, or run
``python differometor/native/build.py``). The simulator never compiles anything at
run time: if the library is missing it uses SciPy's SuperLU instead.

Compiler: $CXX if set, else the first of c++/g++/clang++ on PATH, else the
clang toolchain from the ``ziglang`` pip package (no system headers needed).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "dfsparse.cpp"
FLAGS = ["-O3", "-std=c++17", "-shared", "-fPIC", "-fno-math-errno"]
# Tune for the CPU that runs the benchmark (the library is built at install time on that
# machine). -ffp-contract=off forbids fused multiply-add contraction, so results stay
# bit-identical to the portable build; measured ~17% faster factorisation on the benchmark.
# DIFFEROMETOR_NATIVE_PORTABLE=1 skips these (e.g. when building for another machine).
ARCH_FLAGS = ["-march=native", "-ffp-contract=off"]


def library_name():
    if sys.platform == "win32":
        return "dfsparse.dll"
    if sys.platform == "darwin":
        return "libdfsparse.dylib"
    return "libdfsparse.so"


def compilers():
    if os.environ.get("CXX"):
        yield os.environ["CXX"].split()
    for name in ("c++", "g++", "clang++"):
        path = shutil.which(name)
        if path:
            yield [path]
    try:
        import ziglang  # noqa: F401
    except ImportError:
        return
    zig = [sys.executable, "-m", "ziglang", "c++"]
    if sys.platform.startswith("linux"):
        # zig ships its own glibc headers; target an old glibc so the library loads anywhere
        import platform
        zig += ["-target", f"{platform.machine()}-linux-gnu.2.28"]
    yield zig


def _clean_env():
    # pip's isolated build sets PYTHONPATH/PYTHONNOUSERSITE for its own interpreter; a
    # Python-based compiler driver (ziglang) must not inherit them.
    return {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}


def build(output=None, verbose=True):
    output = Path(output) if output else HERE / library_name()
    extra = os.environ.get("DIFFEROMETOR_NATIVE_FLAGS", "").split()
    errors = []
    portable = os.environ.get("DIFFEROMETOR_NATIVE_PORTABLE", "0") not in ("", "0")
    variants = [[]] if portable else [ARCH_FLAGS, []]
    for compiler in compilers():
        strip = ["-s"] if sys.platform.startswith("linux") else []
        for arch in variants:  # a compiler that rejects the tuning flags retries without them
            command = [*compiler, *FLAGS, *arch, *strip, *extra, "-o", str(output), str(SOURCE)]
            try:
                result = subprocess.run(command, capture_output=True, text=True, env=_clean_env())
            except OSError as error:
                errors.append(f"{' '.join(compiler)}: {error}")
                break
            if result.returncode == 0 and output.exists():
                if verbose:
                    print(f"dfsparse: built {output} with {' '.join(compiler + arch)}")
                return output
            errors.append(f"{' '.join(command)}\n{result.stderr.strip()}")
    raise RuntimeError("dfsparse: no working C++ compiler found:\n" + "\n".join(errors))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else None)
