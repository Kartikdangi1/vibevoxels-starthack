"""
Sparse direct solvers for the signal / noise / carrier systems, with hand-written VJPs.

The system matrices built by `differometor.build` are extremely sparse (official uifo_800_3000
fixture: signal system 751x751 with ~0.4% nonzeros, one matrix per signal frequency), so we solve
them with SciPy's SuperLU on the host (via `jax.pure_callback`) instead of dense O(N^3) LU.

What is different from a plain "swap `jnp.linalg.solve` for `splu`" (see
docs/notes/eidothea_literature_verdicts.md for the literature behind each point):

1. Factorization reuse (Verdict 2). One `splu(A(f))` per frequency serves every solve against that
   matrix: the forward system solve (`A x = b`), the adjoint/noise solve (`A^T y = c`), and the
   reverse-mode solves (`A^T lam = g`, `A mu = g`) - `SuperLU.solve(rhs, trans='N'|'T')` handles all
   of them. Factorizations live in a small host-side LRU cache keyed by a hash of the matrix
   *content*, so forward and backward callbacks find each other's factors without any token
   plumbing; a cache miss simply refactorizes, so results never depend on the cache.
2. Values-only sparse assembly and VJP (Verdict 3). The signal matrix is never materialised densely
   on the JAX side. The solvers take a static dense `base` (N, N+1) shared by all frequencies plus
   a COO list of per-frequency entries `(rows, cols, vals[V, E])` that overwrite `base` (last entry
   wins, matching `.at[].set`). Column index N is the right-hand side. The matrix cotangent is
   evaluated only at those entries, `-lam[r] * x[c]`, instead of as a dense outer product.
3. The host builds the CSC structure once per call from `base` + `(rows, cols)` and only refreshes
   `data` per frequency; no per-frequency scan of 564k mostly-zero entries.

Gradient conventions (JAX convention for holomorphic linear maps: plain transpose, no conjugates):
    x = A^{-1} b      : db = A^{-T} g,           dA = -outer(db, x)
    z = A^{-T} y      : dy = A^{-1} g,           dA = -outer(z, dy)
Both verified against `jnp.linalg.solve` autodiff in `_self_check` (python -m differometor.sparse_solve).
"""

from collections import OrderedDict
from hashlib import blake2b

import numpy as np
import jax
import jax.numpy as jnp
import scipy.sparse
import scipy.sparse.linalg


# --------------------------------------------------------------------------------------
# Host side: structure + factorization cache
# --------------------------------------------------------------------------------------

_CACHE_SIZE = 6  # factorization sets kept alive (each is V SuperLU objects of a few 100 kB)
_cache: "OrderedDict[bytes, _Factored]" = OrderedDict()
_dense_cache: OrderedDict = OrderedDict()  # dense-in `sparse_solve` (carrier system)


class _Factored:
    """CSC structure + per-frequency LU factors + right-hand sides for one (batch item of a) call."""

    __slots__ = ("n", "rows", "cols", "winner", "lus", "rhs")

    def __init__(self, base, rows, cols, vals):
        n = base.shape[0]
        n_freq, n_entries = vals.shape
        self.n = n
        self.rows = rows
        self.cols = cols
        in_a = cols < n

        # Slot of every entry in the union pattern (base nonzeros + entry positions), column-major
        # so that the sorted order *is* CSC order.
        base_r, base_c = np.nonzero(base[:, :n])
        lin_base = base_c.astype(np.int64) * n + base_r
        lin_ent = cols[in_a].astype(np.int64) * n + rows[in_a]
        lin_all, inverse = np.unique(np.concatenate([lin_base, lin_ent]), return_inverse=True)
        slot_base, slot_ent = inverse[: lin_base.size], inverse[lin_base.size :]
        size = lin_all.size
        csc_rows = (lin_all % n).astype(np.int32)
        csc_cols = lin_all // n
        indptr = np.concatenate([[0], np.cumsum(np.bincount(csc_cols, minlength=n))]).astype(np.int32)

        # Which entries survive duplicate resolution (last one wins, like `.at[].set`).
        self.winner = np.ones(n_entries, dtype=bool)
        last = np.full(size, -1, dtype=np.int64)
        last[slot_ent] = np.nonzero(in_a)[0]
        self.winner[in_a] = last[slot_ent] == np.nonzero(in_a)[0]
        rhs_idx = np.nonzero(~in_a)[0]
        if rhs_idx.size:
            last_rhs = np.full(n, -1, dtype=np.int64)
            last_rhs[rows[rhs_idx]] = rhs_idx
            self.winner[rhs_idx] = last_rhs[rows[rhs_idx]] == rhs_idx

        data = np.zeros((n_freq, size), dtype=np.complex128)
        data[:, slot_base] = base[base_r, base_c]
        data[:, slot_ent] = vals[:, in_a]  # repeated slots: last assignment wins

        rhs = np.broadcast_to(base[:, n], (n_freq, n)).copy()
        rhs[:, rows[rhs_idx]] = vals[:, rhs_idx]
        self.rhs = rhs

        self.lus = [
            scipy.sparse.linalg.splu(scipy.sparse.csc_matrix((data[f], csc_rows, indptr), shape=(n, n)))
            for f in range(n_freq)
        ]


def _digest(*arrays) -> bytes:
    h = blake2b(digest_size=16)
    for a in arrays:
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).data)
    return h.digest()


def _factored(base, rows, cols, vals) -> _Factored:
    key = _digest(base, rows, cols, vals)
    hit = _cache.get(key)
    if hit is not None:
        _cache.move_to_end(key)
        return hit
    hit = _Factored(base, rows, cols, vals)
    _cache[key] = hit
    while len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
    return hit


def _batch_items(base, rows, cols, vals):
    """Broadcast the (possibly vmap-expanded) operands to a common leading batch shape and yield
    per-batch-item numpy views. `vals` is (..., V, E); base (..., N, N+1); rows/cols (..., E)."""
    base, rows, cols, vals = (np.asarray(a) for a in (base, rows, cols, vals))
    lead = vals.shape[:-2]
    base = np.broadcast_to(base, lead + base.shape[-2:]).reshape((-1,) + base.shape[-2:])
    rows = np.broadcast_to(rows, lead + rows.shape[-1:]).reshape((-1, rows.shape[-1]))
    cols = np.broadcast_to(cols, lead + cols.shape[-1:]).reshape((-1, cols.shape[-1]))
    cols = np.where(cols < 0, cols + base.shape[-1], cols)  # negative = Python-style wrap
    vals = vals.reshape((-1,) + vals.shape[-2:])
    return lead, [(base[i], rows[i], cols[i], vals[i]) for i in range(vals.shape[0])]


# --- system solve:  A x = b  (b is the RHS column of the assembled system) -----------------------

def _host_system_solve(base, rows, cols, vals):
    lead, items = _batch_items(base, rows, cols, vals)
    out = np.empty((len(items),) + (vals.shape[-2], base.shape[-2]), dtype=np.complex128)
    for i, item in enumerate(items):
        fac = _factored(*item)
        for f, lu in enumerate(fac.lus):
            out[i, f] = lu.solve(fac.rhs[f])
    return out.reshape(lead + out.shape[1:])


def _host_system_solve_bwd(base, rows, cols, vals, x, g):
    lead, items = _batch_items(base, rows, cols, vals)
    x = np.broadcast_to(x, lead + x.shape[-2:]).reshape((-1,) + x.shape[-2:])
    g = np.broadcast_to(g, lead + g.shape[-2:]).reshape((-1,) + g.shape[-2:])
    out = np.zeros((len(items),) + vals.shape[-2:], dtype=np.complex128)
    for i, (b, r, c, v) in enumerate(items):
        fac = _factored(b, r, c, v)
        n = fac.n
        lam = np.stack([lu.solve(g[i, f], trans="T") for f, lu in enumerate(fac.lus)])
        in_a = c < n
        win_a = fac.winner & in_a
        win_b = fac.winner & ~in_a
        out[i][:, win_a] = -lam[:, r[win_a]] * x[i][:, c[win_a]]
        out[i][:, win_b] = lam[:, r[win_b]]
    return out.reshape(lead + out.shape[1:])


# --- transposed solve:  z = A^{-T} y  for D right-hand sides per frequency ------------------------

def _host_transpose_solve(base, rows, cols, vals, y):
    lead, items = _batch_items(base, rows, cols, vals)
    y = np.broadcast_to(y, lead + y.shape[-3:]).reshape((-1,) + y.shape[-3:])  # (B, D, V, N)
    out = np.empty(y.shape, dtype=np.complex128)
    for i, item in enumerate(items):
        fac = _factored(*item)
        for f, lu in enumerate(fac.lus):
            out[i, :, f] = lu.solve(np.ascontiguousarray(y[i, :, f].T), trans="T").T
    return out.reshape(lead + out.shape[1:])


def _host_transpose_solve_bwd(base, rows, cols, vals, z, g):
    lead, items = _batch_items(base, rows, cols, vals)
    z = np.broadcast_to(z, lead + z.shape[-3:]).reshape((-1,) + z.shape[-3:])
    g = np.broadcast_to(g, lead + g.shape[-3:]).reshape((-1,) + g.shape[-3:])
    d_vals = np.zeros((len(items),) + vals.shape[-2:], dtype=np.complex128)
    d_y = np.empty(g.shape, dtype=np.complex128)
    for i, (b, r, c, v) in enumerate(items):
        fac = _factored(b, r, c, v)
        mu = np.empty(g.shape[1:], dtype=np.complex128)  # (D, V, N) = A^{-1} g
        for f, lu in enumerate(fac.lus):
            mu[:, f] = lu.solve(np.ascontiguousarray(g[i, :, f].T)).T
        d_y[i] = mu
        win_a = fac.winner & (c < fac.n)
        d_vals[i][:, win_a] = -np.einsum("dve,dve->ve", z[i][:, :, r[win_a]], mu[:, :, c[win_a]])
    return d_vals.reshape(lead + d_vals.shape[1:]), d_y.reshape(lead + d_y.shape[1:])


# --------------------------------------------------------------------------------------
# JAX side
# --------------------------------------------------------------------------------------

def _cb(fn, shape, *args):
    return jax.pure_callback(fn, jax.ShapeDtypeStruct(shape, jnp.complex128), *args, vmap_method="expand_dims")


def _prep(base, rows, cols, vals):
    return (
        base.astype(jnp.complex128),
        rows.astype(jnp.int64),
        cols.astype(jnp.int64),
        vals.astype(jnp.complex128),
    )


@jax.custom_vjp
def sparse_system_solve(base, rows, cols, vals):
    """Solve A(f) x(f) = b(f) for every frequency f.

    The system is `base` (N, N+1) with the COO entries `(rows[e], cols[e]) <- vals[f, e]` written on
    top (last wins). Columns 0..N-1 are A, column N (or -1) is b. Returns x of shape (V, N)."""
    base, rows, cols, vals = _prep(base, rows, cols, vals)
    return _cb(_host_system_solve, (vals.shape[0], base.shape[0]), base, rows, cols, vals)


def _system_fwd(base, rows, cols, vals):
    x = sparse_system_solve(base, rows, cols, vals)
    return x, (base, rows, cols, vals, x)


def _system_bwd(res, g):
    base, rows, cols, vals, x = res
    base, rows, cols, vals = _prep(base, rows, cols, vals)
    d_vals = _cb(_host_system_solve_bwd, vals.shape, base, rows, cols, vals, x, g)
    return jnp.zeros_like(base), None, None, d_vals


sparse_system_solve.defvjp(_system_fwd, _system_bwd)


@jax.custom_vjp
def sparse_system_transpose_solve(base, rows, cols, vals, y):
    """Solve A(f)^T z = y for D right-hand sides per frequency; y is (D, V, N), result (D, V, N).
    Uses the same factorization as `sparse_system_solve` (cache hit). The RHS column is ignored."""
    base, rows, cols, vals = _prep(base, rows, cols, vals)
    return _cb(_host_transpose_solve, y.shape, base, rows, cols, vals, y.astype(jnp.complex128))


def _transpose_fwd(base, rows, cols, vals, y):
    z = sparse_system_transpose_solve(base, rows, cols, vals, y)
    return z, (base, rows, cols, vals, z)


def _transpose_bwd(res, g):
    base, rows, cols, vals, z = res
    base, rows, cols, vals = _prep(base, rows, cols, vals)
    d_vals = jax.ShapeDtypeStruct(vals.shape, jnp.complex128)
    d_y = jax.ShapeDtypeStruct(z.shape, jnp.complex128)
    d_vals, d_y = jax.pure_callback(
        _host_transpose_solve_bwd, (d_vals, d_y), base, rows, cols, vals, z, g.astype(jnp.complex128),
        vmap_method="expand_dims",
    )
    return jnp.zeros_like(base), None, None, d_vals, d_y


sparse_system_transpose_solve.defvjp(_transpose_fwd, _transpose_bwd)


# --- dense-in drop-in (used for the small carrier system) -----------------------------------------

def _host_dense_solve(matrix, rhs, trans):
    matrix = np.asarray(matrix)
    rhs = np.asarray(rhs)
    n = rhs.shape[-1]
    lead = rhs.shape[:-1]
    flat_matrix = np.broadcast_to(matrix, lead + (n, n)).reshape((-1, n, n))
    flat_rhs = rhs.reshape((-1, n))
    out = np.empty(flat_rhs.shape, dtype=np.complex128)
    for i in range(flat_rhs.shape[0]):
        key = _digest(flat_matrix[i])
        lu = _dense_cache.get(key)
        if lu is None:
            lu = scipy.sparse.linalg.splu(scipy.sparse.csc_matrix(flat_matrix[i]))
            _dense_cache[key] = lu
            while len(_dense_cache) > _CACHE_SIZE * 8:
                _dense_cache.popitem(last=False)
        else:
            _dense_cache.move_to_end(key)
        out[i] = lu.solve(flat_rhs[i], trans="T" if trans else "N")
    return out.reshape(lead + (n,))


def _dense_cb(matrix, rhs, trans):
    return jax.pure_callback(
        lambda m, r: _host_dense_solve(m, r, trans),
        jax.ShapeDtypeStruct(rhs.shape, jnp.complex128),
        matrix.astype(jnp.complex128),
        rhs.astype(jnp.complex128),
        vmap_method="expand_dims",
    )


@jax.custom_vjp
def sparse_solve(matrix: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """Drop-in for a dense solve: matrix (..., N, N), rhs (..., N) -> (..., N), both complex128.
    The forward factorization is cached and reused by the backward `A^T` solve."""
    return _dense_cb(matrix, rhs, False)


def _sparse_solve_fwd(matrix, rhs):
    x = _dense_cb(matrix, rhs, False)
    return x, (matrix.astype(jnp.complex128), x)


def _sparse_solve_bwd(res, g):
    matrix, x = res
    d_rhs = _dense_cb(matrix, g, True)  # cache hit: same LU, trans='T'
    return -d_rhs[..., :, None] * x[..., None, :], d_rhs


sparse_solve.defvjp(_sparse_solve_fwd, _sparse_solve_bwd)


# --------------------------------------------------------------------------------------
# Self check
# --------------------------------------------------------------------------------------

def _self_check() -> None:
    """Values and gradients of all three solvers vs. `jnp.linalg.solve` autodiff on random sparse
    complex systems (including duplicate entries and RHS-column entries)."""
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(0)
    n, n_freq, n_ent, n_det = 14, 3, 40, 2

    base = np.zeros((n, n + 1), dtype=complex)
    base[np.arange(n), np.arange(n)] = 4.0 + 1j
    base[rng.integers(0, n, 10), rng.integers(0, n, 10)] += 1.0
    rows = rng.integers(0, n, n_ent)
    cols = rng.integers(0, n + 1, n_ent)
    cols[:5] = n  # RHS entries
    rows[-3:], cols[-3:] = rows[3], cols[3]  # duplicates: last must win
    vals = rng.normal(size=(n_freq, n_ent)) + 1j * rng.normal(size=(n_freq, n_ent))
    y = rng.normal(size=(n_det, n_freq, n)) + 1j * rng.normal(size=(n_det, n_freq, n))
    base, rows, cols, vals, y = map(jnp.asarray, (base, rows, cols, vals, y))

    def dense(base, rows, cols, vals):
        m = jnp.broadcast_to(base, (n_freq, n, n + 1))
        return m.at[:, rows, cols].set(vals)  # last duplicate wins (as in the real code path)

    def ref_x(vals):
        m = dense(base, rows, cols, vals)
        return jax.vmap(jnp.linalg.solve)(m[:, :, :n], m[:, :, n])

    def ref_z(vals, y):
        m = dense(base, rows, cols, vals)
        return jax.vmap(lambda a, yy: jax.vmap(lambda v: jnp.linalg.solve(a.T, v))(yy), in_axes=(0, 1), out_axes=1)(
            m[:, :, :n], y
        )

    def check(name, new, ref, *args):
        v_new, v_ref = new(*args), ref(*args)
        err = float(jnp.max(jnp.abs(v_new - v_ref)))
        w = jax.random.normal(jax.random.PRNGKey(1), v_ref.shape) + 1j * jax.random.normal(jax.random.PRNGKey(2), v_ref.shape)
        loss = lambda f: (lambda *a: jnp.sum(jnp.abs(f(*a) * w) ** 2))
        argnums = tuple(range(len(args)))
        g_new = jax.grad(loss(new), argnums=argnums)(*args)
        g_ref = jax.grad(loss(ref), argnums=argnums)(*args)
        gerr = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(g_new, g_ref))
        print(f"[{name}] forward max|diff| = {err:.2e}   gradient max|diff| = {gerr:.2e}")
        assert err < 1e-9 and gerr < 1e-8, name

    check("system solve", lambda v: sparse_system_solve(base, rows, cols, v), ref_x, vals)
    check("transpose solve", lambda v, yy: sparse_system_transpose_solve(base, rows, cols, v, yy), ref_z, vals, y)

    # dense-in drop-in
    mats = jax.vmap(lambda v: dense(base, rows, cols, v)[:, :, :n])(vals[None])[0]
    rhs = y[0]
    check("dense drop-in", sparse_solve, lambda m, r: jax.vmap(jnp.linalg.solve)(m, r), mats, rhs)

    # vmap over an outer batch (what run_with_parameter_sets style code does)
    batched = jnp.stack([vals, 2 * vals])
    out = jax.vmap(lambda v: sparse_system_solve(base, rows, cols, v))(batched)
    ref = jax.vmap(ref_x)(batched)
    assert float(jnp.max(jnp.abs(out - ref))) < 1e-9
    print("[vmap] OK")
    print("All checks passed.")


if __name__ == "__main__":
    _self_check()
