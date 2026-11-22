"""Time and check dfs_factor / dfs_solve on the dumped benchmark systems (scratch/systems.npz)."""
import sys, time, ctypes, numpy as np
sys.path.insert(0, ".")
from differometor import native
import differometor.host as H
lib = native._load(); assert lib is not None, "native library missing"
lib.dfs_stats.argtypes = [ctypes.c_void_p] * 3
d = np.load("scratch/systems.npz")
vals, rows, cols, n = d["signal_values"], d["signal_rows"], d["signal_cols"], int(d["signal_n"])
F = vals.shape[0]
import os
REUSE = int(os.environ.get('REUSE', '1'))
pat = H._pattern(rows, cols, n)
data = np.ascontiguousarray(vals[:, pat.order])
sing = np.zeros(F, np.intc)
def factor(reps=7):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        h = lib.dfs_factor(n, F, pat.indptr.ctypes.data, pat.indices.ctypes.data, data.ctypes.data, data.shape[1], None, 1, 1.0, REUSE, 0.1, sing.ctypes.data)
        best = min(best, time.perf_counter() - t)
        if _ < reps - 1: lib.dfs_free(h)
    return h, best
h, tf = factor(int(os.environ.get("REPS", "7")))
rng = np.random.default_rng(0)
b = rng.normal(size=(F, n, 2)) + 1j * rng.normal(size=(F, n, 2))
out, ts = {}, 0
for trans, name in ((0, "N"), (1, "T"), (2, "H")):
    best = 1e9
    for _ in range(5):
        x = np.ascontiguousarray(b.copy()); t = time.perf_counter(); lib.dfs_solve(h, trans, 2, x.ctypes.data, 1); best = min(best, time.perf_counter() - t)
    out[name] = x; ts += best
print(f"factor {tf*1e3:.2f} ms (refactored {lib.dfs_refactored(h)}/{F-1}), N+T+H solves (2 rhs, refined) {ts*1e3:.2f} ms")
if len(sys.argv) > 1 and sys.argv[1] == "save":
    np.savez("scratch/refactor_ref.npz", **out); print("saved reference solutions")
else:
    ref = np.load("scratch/refactor_ref.npz")
    for k in out:
        print(f"  {k}: max rel diff vs reference build {np.abs(out[k] - ref[k]).max() / np.abs(ref[k]).max():.2e}")
