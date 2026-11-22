"""Compile the native sparse solver (`dfsparse.cpp`) into a shared library next to this file.

Run once at install time (``setup.py`` does it automatically, or run
``python differometor/native/build.py``). The simulator never compiles anything at
run time: if the library is missing it uses SciPy's SuperLU instead.

Compiler: $CXX if set, else the first of c++/g++/clang++ on PATH, else the
clang toolchain from the ``ziglang`` pip package (no system headers needed).

Normally nothing needs building: the package ships prebuilt libraries
(``prebuilt/linux-x86_64/libdfsparse-{v1,v3,v4}.so``) and ``setup.py`` skips the
compiler when one of them loads. Maintainers regenerate them with
``python differometor/native/build.py --prebuilt`` (needs ``pip install ziglang``).
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


def _libpath():
    # Loaded by file path: this script also runs standalone and from setup.py.
    import importlib.util

    spec = importlib.util.spec_from_file_location("_dfsparse_libpath", HERE / "_libpath.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def library_name():
    return _libpath().local_name()


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


# Prebuilt Linux x86-64 libraries: zig's clang against glibc 2.17 with libc++ linked in,
# so they load on any x86-64 Linux from 2014 on and need no compiler or C++ runtime.
# -ffp-contract=off keeps every level bit-identical; only the vector width differs.
PREBUILT_X86 = {"v1": "x86_64", "v3": "x86_64_v3", "v4": "x86_64_v4"}
PREBUILT_GLIBC = "2.17"


def build_prebuilt(verbose=True):
    folder = HERE / "prebuilt" / "linux-x86_64"
    folder.mkdir(parents=True, exist_ok=True)
    zig = [sys.executable, "-m", "ziglang", "c++"]
    outputs = []
    for level, cpu in PREBUILT_X86.items():
        output = folder / f"libdfsparse-{level}.so"
        command = [*zig, "-target", f"x86_64-linux-gnu.{PREBUILT_GLIBC}", f"-mcpu={cpu}",
                   *FLAGS, "-ffp-contract=off", "-s", "-o", str(output), str(SOURCE)]
        subprocess.run(command, check=True, env=_clean_env())
        if verbose:
            print(f"dfsparse: built {output.relative_to(HERE)} ({cpu}, glibc {PREBUILT_GLIBC})")
        outputs.append(output)
    return outputs


def ensure(verbose=True):
    """Install-time entry point: use a fitting prebuilt library, else compile one.

    DIFFEROMETOR_NATIVE_BUILD=1 always compiles a library tuned for this machine.
    """
    if os.environ.get("DIFFEROMETOR_NATIVE_BUILD", "0") in ("", "0"):
        path = _libpath().loadable_prebuilt()
        if path is not None:
            if verbose:
                print(f"dfsparse: using prebuilt {path.relative_to(HERE)}; nothing to compile")
            return path
    return build(verbose=verbose)


if __name__ == "__main__":
    if sys.argv[1:] == ["--prebuilt"]:
        build_prebuilt()
    else:
        build(sys.argv[1] if len(sys.argv) > 1 else None)
