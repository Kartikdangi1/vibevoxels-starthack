import importlib.util
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithNative(build_py):
    """Compile the optional native sparse solver before collecting package files.

    The library is written into the source tree (so editable installs see it) and is
    shipped as package data. A failed build is not an error: Differometor then uses
    SciPy's SuperLU.
    """

    def run(self):
        path = Path(__file__).resolve().parent / "differometor" / "native" / "build.py"
        spec = importlib.util.spec_from_file_location("_dfsparse_build", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            module.build()
        except Exception as error:  # noqa: BLE001 - optional component
            print(f"warning: native solver not built, falling back to SuperLU ({error})")
        super().run()


setup(
    cmdclass={"build_py": BuildPyWithNative},
    package_data={"differometor.native": ["*.so", "*.dylib", "*.dll", "*.cpp"]},
)
