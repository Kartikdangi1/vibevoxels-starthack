# Combined solution (`final/combined`)

## What ships

`differometor/` is the **feyer-sparse-host-solver** implementation. It already contains the ideas of
the other branches, so the result is that branch plus everything else that was independently useful:

| Branch | Idea | In the final code |
|---|---|---|
| `feature/kartik` | adjoint noise form, SuperLU shared per frequency, values-only sparse VJP, per-system dispatch | yes, as the in-graph sparse path (`DIFFEROMETOR_BACKEND=jax`) |
| `perf/shared-lu-grouped-fill` | one LU per frequency, grouped component functions, sparse noise | yes, same ideas (`group_by_function`) in the host and JAX paths |
| `feyer-sparse-host-solver` | whole simulation on the host in one `custom_vjp`, NumPy jaxpr interpreter, block-diagonal SuperLU | **default path** |
| `bhuvanedits` | `tools/` correctness/compare/leaderboard scripts | merged unchanged |

From the non-core branches we also keep: profiling scripts and approach notes (`perf`), the Schur
patch that did not pay off (`docs/notes/step3-schur-complement.patch`), the literature notes and
sparse-solver notes (`kartik`). The three superseded test files from `kartik` targeted internals that
no longer exist and were replaced by `tests/test_backends.py`.

## Bug found while combining

The host path assumed every component function reads 7 inputs. Groups can be narrower (the carrier-side
`space` reads 3), so aLIGO with a frequency sweep crashed with `cannot reshape array of size 3 into shape (7)`.
The benchmark's UIFO never hits it. Fixed in `differometor/host.py` (`_assemble`, `_assemble_backward`) by
zero-padding narrow groups and slicing the gradient back.

## Verification

- `tests/test_backends.py`: host path vs in-graph JAX path, values and reverse-mode gradients, on aLIGO,
  Voyager and a swept cavity: passes.
- `tests/check_jvp_transpose.py`: worst error 2.7e-16.
- `python benchmark.py --cpus 1` (Docker, `results/combined-final.json`): first run 3.97 s, warm median
  0.41 s, trace/lower/compile 1.52 s, max |dlog S| 5.374e-6 (same as the upstream code), PASS.
  Baseline on the official Xeon is 44.2 s; the machines differ, so compare loosely.

Known limits are those in `OPTIMIZATION_LOG.md` (no forward-mode AD on the host path; gradients only
w.r.t. `optimized_parameters`). `DIFFEROMETOR_BACKEND=jax` restores full JAX autodiff.
