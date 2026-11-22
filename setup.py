import importlib.util
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithNative(build_py):
    """Make sure a native sparse solver is available before collecting package files.

    A shipped prebuilt library that fits this machine is used as is, so a normal install
    compiles nothing. Otherwise the library is compiled into the source tree (so editable
    installs see it). A failed build is not an error: Differometor then uses SciPy's
    SuperLU.
    """

    def run(self):
        path = Path(__file__).resolve().parent / "differometor" / "native" / "build.py"
        spec = importlib.util.spec_from_file_location("_dfsparse_build", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            module.ensure()
        except Exception as error:  # noqa: BLE001 - optional component
            print(f"warning: native solver not built, falling back to SuperLU ({error})")
        super().run()


setup(
    cmdclass={"build_py": BuildPyWithNative},
    package_data={"differometor.native": ["*.so", "*.dylib", "*.dll", "*.cpp", "prebuilt/*/*.so"]},
)
