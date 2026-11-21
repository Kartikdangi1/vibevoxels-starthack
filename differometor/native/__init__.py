"""ctypes binding of the native batched sparse LU (`dfsparse.cpp`).

`available()` is False when the compiled library is missing or disabled with
``DIFFEROMETOR_NATIVE=0``; callers then use SciPy's SuperLU. Nothing is compiled
at import or run time (see ``build.py``).
"""

import os
import sys
import weakref
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NAMES = {"win32": "dfsparse.dll", "darwin": "libdfsparse.dylib"}
_LIB = []

# Pivoting: threshold 1 = partial pivoting (SciPy SuperLU's default) for every fresh
# factorisation. Frequencies after the first reuse the previous pivot order when each
# reused pivot is at least REUSE_TOL times the largest entry below it (threshold
# partial pivoting, as in UMFPACK's default); otherwise that frequency is pivoted afresh.
PIVOT_TOL = 1.0
REUSE_TOL = 0.1
# Column order: 0 natural, 1 minimum degree on A + A^T, 2 minimum degree on A^T A.
# 2 gives ~30% less fill on the benchmark, but loses accuracy on ill-conditioned setups
# (Voyager space modulation: |dlog S| 1.5e-2 vs 1e-10 with order 1, see
# tests/check_native_solver.py), so order 1 is used.
ORDER = 1
REFINE_STEPS = 1
_TRANS = {"N": 0, "T": 1, "H": 2}


def _load():
    if _LIB:
        return _LIB[0]
    lib = None
    if os.environ.get("DIFFEROMETOR_NATIVE", "1") != "0":
        path = _HERE / _NAMES.get(sys.platform, "libdfsparse.so")
        if path.exists():
            import ctypes
            try:
                lib = ctypes.CDLL(str(path))
                ptr, cint, cdouble = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
                lib.dfs_factor.restype = ptr
                lib.dfs_factor.argtypes = [cint, cint, ptr, ptr, ptr, cint, cdouble, cint, cdouble, ptr]
                lib.dfs_solve.restype = None
                lib.dfs_solve.argtypes = [ptr, cint, cint, ptr, cint]
                lib.dfs_free.restype = None
                lib.dfs_free.argtypes = [ptr]
                lib.dfs_refactored.restype = cint
                lib.dfs_refactored.argtypes = [ptr]
                if lib.dfs_version() != 1:
                    lib = None
            except (OSError, AttributeError):
                lib = None
    _LIB.append(lib)
    return lib


def available():
    return _load() is not None


class NativeBatchedLU:
    """F same-pattern sparse systems factorised in C++; same interface as host.BatchedLU.

    pattern: a host.SparsePattern (sorted CSC order of the (rows, cols) entries).
    values: (F, nnz) in the (rows, cols) order. solve(b): b (F, n) or (F, n, k).
    Each solve applies one step of iterative refinement, like host.RefinedLU.
    Singular systems give NaN solutions; the others are unaffected.
    """

    def __init__(self, values, pattern):
        lib = _load()
        values = np.asarray(values, dtype=np.complex128)
        self.F, self.n = values.shape[0], pattern.n
        data = np.ascontiguousarray(values[:, pattern.order])
        self.singular = np.zeros(self.F, np.intc)
        handle = lib.dfs_factor(self.n, self.F, pattern.indptr.ctypes.data, pattern.indices.ctypes.data,
                                data.ctypes.data, ORDER, PIVOT_TOL, 1, REUSE_TOL,
                                self.singular.ctypes.data)
        self._handle = handle
        self._lib = lib
        weakref.finalize(self, lib.dfs_free, handle)

    @property
    def refactored(self):
        return self._lib.dfs_refactored(self._handle)

    def solve(self, b, trans="N"):
        b = np.asarray(b, dtype=np.complex128)
        if b.ndim == 3 and b.shape[2] == 0:
            return np.zeros(b.shape, np.complex128)
        x = np.array(b, dtype=np.complex128, order="C", copy=True)
        k = 1 if x.ndim == 2 else x.shape[2]
        self._lib.dfs_solve(self._handle, _TRANS[trans], k, x.ctypes.data, REFINE_STEPS)
        return x
