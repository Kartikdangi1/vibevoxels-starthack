"""
Drop-in replacement for `differometor.simulate.solve` that exploits the extreme sparsity of the
signal/noise system matrices (measured on the official uifo_800_3000 fixture: 751x751, ~0.47%
nonzero, ~3.6 nonzeros/row) instead of `jnp.linalg.solve`'s dense O(N^3) LU factorization.

Differentiability is preserved with a hand-written `jax.custom_vjp`: for the linear system
A x = b, the VJP is solver-independent (it follows purely from implicit differentiation of
A x = b, not from how x was computed), so the sparse forward solve can be swapped in without
changing what `jax.grad` returns - only how long it takes to get there.

For x = solve(A, b), with upstream cotangent g, JAX's own `jnp.linalg.solve` VJP (empirically
pinned down below against `jax.vjp` on random complex matrices, since the convention is easy to
get subtly wrong - no conjugates at all, just the plain transpose):
    db = solve(A.T, g)
    dA = -outer(db, x)          where outer(u, v)[i, j] = u[i] * v[j]
Verified exactly (max|diff| ~1e-16) against `jax.vjp(jnp.linalg.solve)` before trusting this
anywhere near the real simulation; see `_self_check` below.

The forward solve itself runs on the host via `jax.pure_callback` (SciPy's SuperLU has no JAX/XLA
binding), batched: one callback per batch call, looping in SciPy/C over the batch on the host
rather than one callback per vmap element, so Python-call overhead is paid once, not per
frequency.
"""

import numpy as np
import jax
import jax.numpy as jnp
import scipy.sparse
import scipy.sparse.linalg


def _host_batched_sparse_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    matrix: (..., N, N) complex128 (dense ndarray in - most entries are zero; sparsity is
            rediscovered here via scipy.sparse rather than assumed statically, so this is
            correct for any matrix, just fast specifically when it is sparse, as ours is).
    rhs:    (..., N) complex128
    returns (..., N) complex128

    Any number of leading batch dimensions is accepted (0, 1, or several stacked by nested
    vmap calls - JAX's `pure_callback(vmap_method="expand_dims")` prepends one new leading axis
    per vmap level on top of whatever shape this function is given, rather than folding into an
    axis the caller already created, so this has to tolerate an arbitrary, not fixed, rank).
    """
    matrix = np.asarray(matrix)
    rhs = np.asarray(rhs)
    n = rhs.shape[-1]
    leading_shape = rhs.shape[:-1]
    flat_matrix = matrix.reshape((-1, n, n))
    flat_rhs = rhs.reshape((-1, n))
    batch = flat_rhs.shape[0]
    out = np.empty((batch, n), dtype=np.complex128)
    for i in range(batch):
        a = scipy.sparse.csc_matrix(flat_matrix[i])
        lu = scipy.sparse.linalg.splu(a)
        out[i] = lu.solve(flat_rhs[i])
    return out.reshape(leading_shape + (n,))


def _sparse_solve_batched(matrix: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """Signature matches `solve(matrix, rhs)`: matrix (..., N, N), rhs (..., N) -> (..., N).
    Leading dims (none, one from a single vmap, or several from nested vmaps) are handled
    inside the host callback itself - see `_host_batched_sparse_solve`."""
    matrix = matrix.astype(jnp.complex128)
    rhs = rhs.astype(jnp.complex128)
    return jax.pure_callback(
        _host_batched_sparse_solve,
        jax.ShapeDtypeStruct(rhs.shape, jnp.complex128),
        matrix,
        rhs,
        vmap_method="expand_dims",
    )


@jax.custom_vjp
def sparse_solve(matrix: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """Drop-in replacement for `differometor.simulate.solve`, same contract: matrix (..., N, N),
    rhs (..., N) -> solution (..., N), both cast to complex128."""
    return _sparse_solve_batched(matrix, rhs)


def _sparse_solve_fwd(matrix, rhs):
    x = _sparse_solve_batched(matrix, rhs)
    return x, (matrix.astype(jnp.complex128), x)


def _sparse_solve_bwd(residuals, cotangent):
    matrix, x = residuals
    matrix_t = jnp.swapaxes(matrix, -1, -2)
    d_rhs = _sparse_solve_batched(matrix_t, cotangent)
    d_matrix = -d_rhs[..., :, None] * x[..., None, :]
    return d_matrix, d_rhs


sparse_solve.defvjp(_sparse_solve_fwd, _sparse_solve_bwd)


def _self_check() -> None:
    """Confirms sparse_solve matches jnp.linalg.solve's values AND gradients exactly (within
    tolerance), on both a dense-random and a genuinely sparse random complex system, before this
    is trusted anywhere near the real simulation. Run directly: `python -m differometor.sparse_solve`."""
    from jax.test_util import check_grads

    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(0)

    def _reference_solve_one(matrix, rhs):
        return jnp.linalg.solve(matrix.astype(jnp.complex128), rhs.astype(jnp.complex128))

    def reference_solve(matrix, rhs):
        # matches the real call sites' pattern: vmap(solve) over unbatched (N,N)/(N,) pairs,
        # not a raw batched call (which recent JAX disallows as ambiguous - see simulate.solve).
        return jax.vmap(_reference_solve_one)(matrix, rhs)

    for label, density in [("dense random", 1.0), ("sparse-like (5% nnz)", 0.05)]:
        n, batch = 12, 3
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        real = jax.random.normal(k1, (batch, n, n))
        imag = jax.random.normal(k2, (batch, n, n))
        mask = (jax.random.uniform(k3, (batch, n, n)) < density).astype(real.dtype)
        mask = mask + jnp.eye(n)[None]  # keep the diagonal so the system stays nonsingular
        matrix = (real + 1j * imag) * jnp.clip(mask, 0, 1) + 3.0 * jnp.eye(n)[None]
        rhs = jax.random.normal(k4, (batch, n)) + 1j * jax.random.normal(k4, (batch, n))

        x_ref = reference_solve(matrix, rhs)
        x_new = sparse_solve(matrix, rhs)
        value_error = float(jnp.max(jnp.abs(x_new - x_ref)))
        print(f"[{label}] forward max|diff| = {value_error:.3e}")
        assert value_error < 1e-8, f"forward mismatch for {label}: {value_error}"

        check_grads(lambda m, b: sparse_solve(m, b), (matrix, rhs), order=1, modes=["rev"], atol=1e-5, rtol=1e-5)
        print(f"[{label}] gradient check vs. finite differences: OK")

        g_ref = jax.grad(lambda m: jnp.sum(jnp.abs(reference_solve(m, rhs)) ** 2))(matrix)
        g_new = jax.grad(lambda m: jnp.sum(jnp.abs(sparse_solve(m, rhs)) ** 2))(matrix)
        grad_error = float(jnp.max(jnp.abs(g_new - g_ref)))
        print(f"[{label}] grad max|diff| vs. jnp.linalg.solve = {grad_error:.3e}")
        assert grad_error < 1e-8, f"gradient mismatch for {label}: {grad_error}"

    print("All checks passed: sparse_solve matches jnp.linalg.solve's values and gradients.")


if __name__ == "__main__":
    _self_check()
