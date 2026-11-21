# `sparse_solve`: replacing dense LU with sparse direct solve

## The finding

The official fixture's `signal_matrix`/`noise_matrix` are **751x751 but only ~0.47% nonzero**
(~3.6 nonzeros/row — an optical-component connectivity graph, not a dense system). `solve()`
called `jnp.linalg.solve` (dense LAPACK LU, O(N^3)) on it roughly 300 times (3 solve call sites x
101 frequencies). Measured dense-vs-sparse on a representative 751x751/101-batch case:

```
dense  total: 163.4s   (1617.8 ms/solve)
sparse total: 0.70s    (6.96 ms/solve)
speedup: 232.6x
max solution diff: 8.3e-14
```

## Implementation

`differometor/sparse_solve.py`: `sparse_solve(matrix, rhs)`, a drop-in for `solve()` (same
`(..., N, N)`/`(..., N)` contract), backed by SciPy's SuperLU (C) via `jax.pure_callback`
instead of XLA's dense LAPACK path. Differentiability is preserved with a hand-written
`jax.custom_vjp` - the adjoint of a linear solve doesn't depend on how the forward solve was
computed, only on `A`, `b` and `x`.

**Getting the complex-number convention right was the hard part**, and wasn't guessed at: it was
pinned down empirically against `jax.vjp(jnp.linalg.solve)` on small random complex matrices
before ever touching the real pipeline (see `_self_check` in that file, run with
`python -m differometor.sparse_solve`):

```
db = solve(A.T, g)      # plain transpose, NOT conjugate-transpose
dA = -outer(db, x)      # no conjugates anywhere
```

Wired into `differometor/simulate.py`: `solve = sparse_solve` by default, `_dense_solve` kept as
a fallback (`DIFFEROMETOR_SOLVER=dense` env var) in case something needs re-checking later.

## Verification against the actual graded quantities (not just synthetic matrices)

Ran the real fixture + real reference file (`uifo_800_3000_test_data.json`), all 193 design
variables, on a 3-frequency subset of the real grid (full 101 can't run locally - see below):

| | sparse | dense |
|---|---|---|
| sensitivity vs. official reference (log-error, need <=1e-5) | 1.03e-6 | 1.03e-6 |
| sensitivity, sparse vs. dense | 8.4e-8 | - |
| gradient finite / non-zero | yes | yes |
| gradient, sparse vs. dense (relative max diff) | 7.3e-6 | - |

## A red herring worth recording

The raw `signal` output (not `sensitivity`) differs **hugely** between solvers (~1e15) at some
indices. This is NOT a bug: that system matrix is genuinely singular to machine precision there
(condition number ~1e30 - likely a physically-unused/decoupled port in the optical graph). Both
LAPACK and SuperLU produce "equally meaningless but finite" values in that one unstable
direction. It doesn't matter for grading: `noise` doesn't depend on `signal` at all (only on
`signal_matrix` directly), and `sensitivity` only indexes `signal` at the actual detector ports,
which land in a well-conditioned direction (confirmed by the sensitivity comparison above).
`benchmark.py`'s correctness check only requires values to be *finite*, which they are.

## What's still unverified: full 101-frequency scale

This laptop's Docker environment OOMs on the full official benchmark regardless of solver choice
(confirmed: swapping in `sparse_solve` did not move the OOM point - it dies at the same ~10s into
the "Forward: compile" phase either way, meaning the compile-time memory bottleneck is something
else entirely - most likely autodiff through the ~532-branch `lax.switch` over component transfer
functions, batched over 101 frequencies - not `solve()` itself). This machine only gives Docker
7.6 GiB (host has 15.7 GiB total, ~2 GiB free); the workload needs ~14.4 GiB peak per the upstream
README. **Full-scale validation needs a machine with real headroom (the AMD Radeon Cloud instance,
or similar ~32 GB box)** - the sparse-solve win itself is solid, but the official before/after
benchmark numbers can only come from there.
