# Combined solution (`final/combined-v2`)

## What ships

`final/combined-v2` unifies all four optimization workstreams into the default host path, while preserving SuperLU (`DIFFEROMETOR_NATIVE=0`) and the in-graph sparse JAX path (`DIFFEROMETOR_BACKEND=jax`) as fallbacks:

| Branch / Module | Idea | Role in `final/combined-v2` |
|---|---|---|
| `feature/kartik` | adjoint noise form, shared LU per frequency, values-only sparse VJP, per-system dispatch | in-graph sparse path (`DIFFEROMETOR_BACKEND=jax`) |
| `perf/shared-lu-grouped-fill` | one LU per frequency, grouped component functions, sparse noise | `group_by_function` in both host and JAX paths |
| `feyer-sparse-host-solver` | whole simulation on host in one `custom_vjp`, NumPy `jaxpr` + JVP-transpose, primal caching, block-diagonal SuperLU | **default host framework** (`differometor/host.py`) |
| `cpp-implementation` | C++ Gilbert–Peierls batched sparse LU, $A+A^T$ min-degree, KLU-style pivot reuse (`REUSE_TOL=0.1`), 1-step refinement | **default sparse LU solver** (`differometor/native/dfsparse.cpp`) |
| `feyer-stage3-native-kernels` | 19 closed-form NumPy Wirtinger-dual component kernels eliminating startup `make_jaxpr` + self-check | **default built-in component evaluator** (`differometor/native_kernels.py`) |
| `bhuvan/fast-readout` | host `signal_detector` and `power_detector` behind `jax.custom_vjp` | **default readout detectors** (`differometor/components.py`) |
| `bhuvanedits` | `tools/` correctness/compare/leaderboard scripts | merged unchanged |

From the non-core branches we also keep: profiling scripts and approach notes (`perf`), the Schur patch that did not pay off (`docs/notes/step3-schur-complement.patch`), the 10-year and 15-year Eidothea literature notes (`docs/notes/eidothea_literature_verdicts.md`, `docs/notes/eidothea_literature_verdicts_15y.md`), and the explainer report (`docs/report/main.tex`, `docs/report/main.pdf`).

## Bug fixes during combination

- **Narrow-group host assembly (`final/combined`):** `host.py` assumed every component function reads 7 inputs; carrier-side `space` reads 3, which crashed frequency-swept setups like aLIGO. Fixed in `_assemble` and `_assemble_backward` by zero-padding narrow groups and slicing the gradient back.
- **Stage 3 + cached host path (`final/combined-v2`):** Renamed `differometor/native.py` to `differometor/native_kernels.py` (avoiding collision with the `differometor/native/` C++ package), added `forward_with_cache` / `vjp_from_cache` to `NativeFunction`, renamed the kernel toggle to `DIFFEROMETOR_KERNELS` (leaving `DIFFEROMETOR_NATIVE` for the C++ solver), and re-pinned the `components.py` SHA-256 in `tests/check_native.py`.

## Verification

- `tests/test_backends.py`: host path vs in-graph JAX path, values and reverse-mode gradients, on aLIGO, Voyager and a swept cavity.
- `tests/test_host_detectors.py`: host `signal_detector` / `power_detector` vs JAX detectors (forward $\le 10^{-14}$, gradients $\le 10^{-12}$).
- `tests/check_jvp_transpose.py`: JVP transposition vs `jax.vjp` (worst error $\le 2.7\times 10^{-16}$).
- `tests/check_native.py`: all 19 Wirtinger-dual kernels vs `jax.vjp` (worst relative error $6.9\times 10^{-15}$ on component parameters; `components.py` hash pinned).
- `tests/check_native_solver.py`: C++ `dfsparse` vs SuperLU vs JAX across 12 topologies.
- `results/amd-epyc/comparison.txt` (AMD EPYC 9334, 1 logical CPU, apples-to-apples):
  - Baseline: **35.829 s** first run, **16.952 s** warm median, **19.301 s** compile.
  - v1 (`ours_cpu1_b`): **1.038 s** first run (34.5×), **0.1602 s** warm median (105.8×).
  - `+ Stage 3 kernels` (`stage3_cpu1`): **0.941 s** first run (38.1×), **0.1472 s** warm median (115.2×).
  - `+ Host readout` (`fastreadout_cpu1`): **0.895 s** first run (40.0×), **0.1727 s** warm median (98.1×).
  - `+ C++ LU` (`cpp_cpu1`): **0.964 s** first run (37.2×), **0.0969 s** warm median (**174.9×**).
  - All achieve `max |dlog S| = 5.374e-6` (identical to baseline) and gradient cosine similarity `1.0000000000` vs baseline.
- `benchmark-v2.json` (i7-1355U WSL2, 1 CPU): **3.180 s** first run, **0.865 s** compile, `max |dlog S| = 5.374e-6`, PASS.

Known limits are those in `OPTIMIZATION_LOG.md` (no forward-mode AD on the host path; gradients only w.r.t. `optimized_parameters`). `DIFFEROMETOR_BACKEND=jax` restores full JAX autodiff.
