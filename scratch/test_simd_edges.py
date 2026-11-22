"""Edge cases of the SIMD-batched refactorisation vs the sequential path (mode 1) and SciPy SuperLU."""
import sys, ctypes, numpy as np
sys.path.insert(0, ".")
from differometor import native
import differometor.host as H
lib = native._load()
d = np.load("scratch/systems.npz")
vals0, rows, cols, n = d["signal_values"], d["signal_rows"], d["signal_cols"], int(d["signal_n"])
pat = H._pattern(rows, cols, n)

def run(vals, mode):
    F = vals.shape[0]
    data = np.ascontiguousarray(vals[:, pat.order])
    sing = np.zeros(F, np.intc)
    h = lib.dfs_factor(n, F, pat.indptr.ctypes.data, pat.indices.ctypes.data, data.ctypes.data, data.shape[1], None, 1, 1.0, mode, 0.1, sing.ctypes.data)
    rng = np.random.default_rng(1)
    b = rng.normal(size=(F, n, 1)) + 1j * rng.normal(size=(F, n, 1))
    out = {}
    for t, name in ((0, "N"), (1, "T"), (2, "H")):
        x = np.ascontiguousarray(b.copy()); lib.dfs_solve(h, t, 1, x.ctypes.data, 1); out[name] = x
    refac = lib.dfs_refactored(h); lib.dfs_free(h)
    return out, sing, refac, b

def residual_ok(vals, out, b, skip=()):
    worst = 0
    for f in range(vals.shape[0]):
        if f in skip: continue
        for name in "NTH":
            r = b[f, :, 0] - pat.matvec(vals[f], out[name][f, :, 0], name)
            lu = H.RefinedLU(vals[f], rows, cols, n)
            y = lu.solve(b[f, :, 0], name)
            r2 = b[f, :, 0] - pat.matvec(vals[f], y, name)
            worst = max(worst, np.abs(r).max() / max(np.abs(r2).max(), 1e-300))
    return worst

# 1. an exactly singular system in the middle of a group: zero one column of system 5
vals = vals0.copy()
vals[5, cols == cols[0]] = 0
for mode in (1, 2):
    out, sing, refac, b = run(vals, mode)
    nan_sys = sorted({f for f in range(vals.shape[0]) if np.isnan(out["N"][f]).any()})
    print(f"singular test mode={mode}: singular flags {np.flatnonzero(sing).tolist()}, NaN systems {nan_sys}, refactored {refac}, "
          f"resid ratio vs SuperLU (others) {residual_ok(vals, out, b, skip={5}):.2f}")

# 2. a system whose reused pivots fail the threshold check: swap two rows' values in system 11
vals = vals0.copy()
r0, r1 = 0, int(rows[np.argmax(np.abs(vals0[11]))])
m0, m1 = rows == r0, rows == r1
perm = vals.copy()
for f in (11,):
    perm[f] = vals[f] * np.where(m0 | m1, 1e-9, 1.0)  # shrink two rows: old pivots become tiny
out1, _, refac1, b = run(perm, 1)
out2, _, refac2, _ = run(perm, 2)
print(f"pivot-failure test: refactored mode1 {refac1}, mode2 {refac2}; "
      f"max rel diff mode2 vs mode1 {max(np.abs(out2[k] - out1[k]).max() / np.abs(out1[k]).max() for k in out1):.1e}; "
      f"resid ratio vs SuperLU {residual_ok(perm, out2, b):.2f}")

# residual detail: relative residuals ||b - A x|| / (||A|| ||x||) (backward error) for ours and SuperLU
out, _, _, b = run(vals0, 2)
worst = []
for f in range(vals0.shape[0]):
    for name in "NTH":
        x = out[name][f, :, 0]
        y = H.RefinedLU(vals0[f], rows, cols, n).solve(b[f, :, 0], name)
        nA = np.abs(vals0[f]).max()
        e1 = np.abs(b[f, :, 0] - pat.matvec(vals0[f], x, name)).max() / (nA * np.abs(x).max())
        e2 = np.abs(b[f, :, 0] - pat.matvec(vals0[f], y, name)).max() / (nA * np.abs(y).max())
        worst.append((e1, e2, f, name))
print("worst backward error ours %.1e (SuperLU on same system %.1e), worst SuperLU %.1e" % (
    max(w[0] for w in worst), max(worst)[1], max(w[1] for w in worst)))
