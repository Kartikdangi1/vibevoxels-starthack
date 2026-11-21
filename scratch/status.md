## 6. Implementation status (verified in the container, 1 pinned CPU)

| Verdict | Status | Where |
|---|---|---|
| 1 — drop the 2nd quantum-noise solve (`w^H N w`) | **Done**, exact identity | `simulate._signal_and_noise_sparse` |
| 2 — one sparse LU per frequency for all solves (fwd, `A^T`, bwd) | **Done** (content-hash LRU cache of SuperLU objects; a cache miss just refactorizes, so results never depend on it) | `sparse_solve.py` |
| 3 — values-only sparse assembly + sparse VJP (`-lam[r]·x[c]` on nonzeros) | **Done**; no `(101,751,751)` array is built or differentiated | `sparse_solve.py`, `simulate.py` |
| 4 — noise matrix evaluated once, contracted on its nonzeros | **Done** | `simulate._signal_and_noise_sparse` |
| 5 — kill the 19-way `vmap(lax.switch)` | **Done**: one branch-free kernel (`components.evaluate_components`) evaluates the component functions of carrier, signal and noise in a *single* flat call. Every function is `amplitude · exp(i·phase)` per output slot, selected by static lookup tables, so XLA compiles and differentiates one small kernel instead of 19 branches × 3 call sites. | `components.py`, `simulate._component_outputs` |
| 6 — Schur / APF, static hoisting | **Not implemented.** The note itself ranks it secondary and, after 1-5, the 101 sparse LUs take milliseconds; static hoisting additionally needs frequency-independence known at trace time, but `benchmark.py` passes every index array as a traced `jit` argument. | — |

Beyond the six verdicts (found by profiling, not by the literature scan):
- `build.py` ran eager `jnp` scalar ops and `jnp.array(host, dtype=...)` conversions; each distinct primitive/shape
  costs an XLA compile on first use, ~2.7 s of the first run. Build-time formulas now use NumPy
  (`components._reflectivity_left_np` & co., `space_entry_np`, `build._arr`): build 2.85 s → 0.19 s.
- `lax.optimization_barrier` around the kernel's inputs/outputs stops XLA from fusing (and duplicating) the kernel
  into every consumer: -3 s forward compile.

Measured on an i7-1355U pinned to one CPU (`benchmark.py --cpus 1`), first-run total (laptop timing noise is ~±10%):

| Stage | First-run total |
|---|---:|
| Verdicts 1-4 only (19-way switch everywhere) | 66.5 s |
| + per-call-site switch subsets | 24.4 s |
| + unified kernel, evaluated for all three systems at once | ~18 s |
| + NumPy build, optimization barriers (final) | **14.1-16.1 s** (warm forward+gradient ~0.6 s) |

The committed pre-change code (dense `(101,751,751)` assembly) was killed by the OOM killer in this 7.6 GB container, so
there is no same-machine "before"; the Xeon baseline in `benchmarks/cpu-baseline.json` is 44.2 s. What is left is
~10 s of XLA compile (forward ~6.3 s, gradient ~3.5 s: ~1,500 HLO ops of glue plus the kernel), ~2.6 s imports of the
benchmark harness itself, ~1 s lowering and ~1.2 s execution.

Where this note and the code disagree (the code was followed): the noise matrix has 272 updated entries (not ~760);
`signal_matrix` has 751×752 entries including the RHS column; the 19 component functions are not the ones listed in
Verdict 5 (see `components.FUNCTIONS`); quantum noise is `sqrt(2·w^H N w·h·f0/4)`; some systems get a 3-wide
placeholder input row from `build`, not 7-wide.

Validation (`python -m pytest tests -q`):
- `test_components.py`: the kernel equals every `FUNCTIONS[i]` row-by-row (values and gradients, random rows and the real
  fixture rows) and gradients are finite (no NaN poisoning from masked-out candidates).
- `test_sparse_pipeline.py`: full uifo pipeline, new path vs. the original dense + 19-way-switch path (values, gradients),
  plus the solver self-check against `jnp.linalg.solve` (duplicates, RHS-column entries, vmap).
- `test_example_setups.py`: aLIGO, Voyager and the carrier-swept cavity vs. the original path.
- `benchmark.py`'s reference-sensitivity and finite-difference gradient checks, `start_here.py`, `examples/*.py`.

On Voyager the new noise result is bit-identical to the previously committed sparse solver and differs from the dense
LU by 6e-5 (relative to max): LU roundoff on an ill-conditioned system, not the refactor.
Escape hatches: `DIFFEROMETOR_SOLVER=dense` (original dense assembly + `jnp.linalg.solve`) and
`DIFFEROMETOR_DISPATCH=sites|full` (per-system `lax.switch` subsets / original 19-way switch).
Assumption worth knowing: the base `noise_matrix` is empty (always true as built by `build.py`; the tests assert it).
