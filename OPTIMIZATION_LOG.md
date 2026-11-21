# Optimization log — Feyer Faster Physics

All timings: Apple M2, macOS, `python benchmark.py --repeats 3` (the official `--cpus 1`
affinity flag is Linux-only), complete first run as reported by `benchmark.py`
(imports → host transfer, validation excluded). Raw reports are in [`results/`](results/).
The machine was not idle (load average 4–6 from other processes), so single runs vary by
~±0.1 s at the sub-second level and by up to 2× on multi-second runs. Each experiment
was run at least twice; all runs are listed.

| ID | Change | First run (s) | Warm median (s) | Max \|Δlog S\| | Decision | Commit |
|---|---|---|---|---|---|---|
| V0 | upstream baseline | 24.889 | 12.130 | 5.374e-6 | baseline | 4c5f703 |
| V1 | adjoint quadratic form for quantum noise (drop 2nd solve) | 22.239 | 11.393 | 5.374e-6 | INTEGRATED | 734e137 |
| E1 | sparse compressed assembly + SuperLU + custom adjoint VJP | 7.106 | 0.231 | 5.374e-6 | superseded by E2 | — |
| E2 | + component functions grouped by type (no `lax.switch`) | 3.61 / 3.98 (6.64 under load) | 0.09 | 5.374e-6 | INTEGRATED | 4e2b3c3 |
| E3 | whole simulation on the host in one `custom_vjp`; NumPy jaxpr interpreter | 1.456 / 1.647 | 0.114 | 5.374e-6 | INTEGRATED | 560b5cc |
| E4 | NumPy build-time constants; one unbatched trace per function | 0.806 / 1.013 | 0.126 | 5.374e-6 | INTEGRATED | 9ecce4d |
| E5 | iterative refinement on every solve; singular-safe LU; real sweep detection | 0.730 / 0.781 / 0.786 | 0.125 | 5.374e-6 | INTEGRATED | 9505c3d |
| E6 | SciPy SuperLU extension called directly (no `scipy.sparse` import) | 0.734 / 0.842 / 0.854 (load 14) | 0.13–0.17 | 5.374e-6 | INTEGRATED | ebfafcd |
| E7 | all frequencies factorised as one block-diagonal system | (runtime 0.17 → 0.13 s) | — | 5.374e-6 | INTEGRATED | bf7d4c2 |
| E8 | VJP by transposing JAX's JVP program (cheaper tracing) | 0.591 / 0.632 | 0.13 | 5.374e-6 | INTEGRATED | 0c64f2b |
| E9 | Challenger fixes: rank alignment, real cotangents, self-check, referenced links only | 0.616 / 0.647 / 0.732 | 0.12 | 5.374e-6 | INTEGRATED | bc8ad3c |
| E10 | native C++ batched sparse LU (`differometor/native`), branch `cpp-implementation` | WSL i5: see E10 section | WSL i5: 0.26–0.28 → 0.14–0.16 | 5.374e-6 | CANDIDATE | — |

Overall: **24.889 s → ~0.62 s (≈40×)** first run; warm 12.13 s → 0.12 s (≈100×).
The 5.374e-6 public error equals the upstream code's own error against the reference.

## E5 phase breakdown (run 3, 0.730 s)

| Phase | s | Notes |
|---|---|---|
| imports | 0.276 | `jax` itself ≈ 0.26 s (imported by the benchmark before our package) |
| build | 0.047 | was 1.40 s (V0) |
| forward trace/lower | 0.141 | jaxpr tracing of ~14 component functions |
| forward compile | 0.009 | was 2.87 s |
| forward | 0.074 | was 8.80 s |
| sensitivity lower+compile | 0.069 | benchmark's readout graph |
| gradient lower+compile | 0.056 | was 1.87 s |
| gradient | 0.055 | was 5.98 s |

## What changed and why

**V1 — adjoint quadratic form.** `w = A⁻ᴴc` ⇒ `cᴴA⁻¹ = wᴴ`, so `Re(cᴴA⁻¹Nw) = Re(wᴴNw)`.
This holds exactly for any invertible A and any N (Hermitian or not), so it is valid for every topology.
Challenger: **APPROVE**. Checked on aLIGO, Voyager (qhd), random UIFO, multi-detector and carrier-sweep setups; old and new agree to ≤ 2.5e-10.

**E1/E2 — sparsity.** Measured on the public setup: carrier 286² with 680 nonzeros, signal 751²
with 2537 nonzeros (0.45 %, same pattern at all 101 frequencies), noise 751² with 272 nonzeros.
The dense code built 101×751×752 complex arrays (≈ 0.9 GB each) and ran dense LU.

- **Compressed assembly.** The build step maps every structurally relevant (row, col) to a flat
  index: base nonzeros ∪ every write position, with negative indices normalised. The block
  conjugation rules become a boolean mask.
- **One factorisation per frequency.** SuperLU (`scipy.sparse.linalg.splu`, COLAMD) factorises A
  once per frequency and serves `x = A⁻¹b` and `w = A⁻ᴴc` (`trans="H"`).
- **Adjoint (JAX complex convention).** Verified numerically against `jax.vjp` of `jnp.linalg.solve`:
  - `b̄ = A⁻ᵀx̄`
  - `c̄ = conj(A⁻¹ conj w̄)`
  - `Ā = −b̄xᵀ − conj(w)(A⁻¹ conj w̄)ᵀ`, evaluated only at the nonzeros

  The backward pass reuses the forward LU objects through a token-keyed cache, and checks that the
  cached values are equal before reusing them.
- **Grouped component functions.** Entries are grouped by component function at build time under
  static dict keys, which removes the vmapped `lax.switch` over all 19 functions (compile time).

Challenger (E2): **CONDITIONAL**.
- Correct to round-off on all of its cases.
- Demonstrated: SuperLU raised on a singular matrix (fixed in E5).
- Demonstrated: the 'c' grouping was dead (fixed in E5).
- No forward-mode AD (by design; the benchmark is reverse mode).

**E3 — host execution.** With E2 most of the remaining time was XLA compilation of the assembly
graph and its reverse pass. The whole simulation now runs in NumPy/SciPy inside one
`jax.custom_vjp`, so XLA compiles only two callback stubs.

- **Exact derivatives, no rewrite.** Component and linking functions are not re-implemented. Each
  is traced to a jaxpr of `x ↦ (f(x), vjp_f(x)(ct))` and run by a small NumPy jaxpr interpreter.
  That keeps JAX's own differentiation rules and complex conventions.
- **Hand-written glue adjoint.** Parameter insertion, linking, sweep expansion, carrier scaling,
  conjugation, qhd phases, detector selections, solves and the noise form have a hand-written
  adjoint, validated against JAX autodiff of the in-graph path.
- **Fallback.** If a function uses a primitive the interpreter lacks, `simulate` falls back to the
  in-graph sparse path. `DIFFEROMETOR_BACKEND=jax` forces that path.

**E4 — setup and tracing.**
- **Setup.** The builder evaluated scalar optics helpers with eager `jnp`, which compiled about 64
  tiny XLA kernels (0.33 s). Those helpers now use NumPy for concrete inputs, and traced calls are
  unchanged. The builder's arrays are NumPy.
- **Tracing.** Each component function is traced once, unbatched, with dead-code elimination
  splitting off the forward part. The interpreter carries a leading batch axis.

**E5 — robustness found by validation.**
- **Voyager noise error (demonstrated).** On Voyager (space modulation, f ≈ 357 Hz, cond(A) ≈ 1e31)
  sparse noise was off by 3.2e-3 in log while dense was exact. A 60-digit reference showed dense
  error 5e-15 and SuperLU error 3.2e-3, although w was normwise accurate to 1.5e-14: the quadratic
  form depends on tiny components. One step of fixed-precision iterative refinement (sparse
  residual) gives 1.1e-14. It is applied to every forward and adjoint solve.
- **Singular matrices** now give NaN, matching dense.
- **Sweep detection.** Sweeps with a single value no longer count as swept, so sweep-independent
  groups are evaluated once.

**E6 — direct SuperLU binding.** Importing `scipy.sparse.linalg` cost about 0.1 s. The compiled
`_superlu` extension needs only NumPy, so it is loaded directly (registered under its real module
name; the regular import is the fallback), and `gstrf` is called with `splu`'s default options on
a sorted CSC built once per pattern. Outputs are bit-identical to E5.

**E7 — block-diagonal batching.** All 101 frequencies are factorised as one 75,851² block-diagonal
system: one SuperLU call instead of 101, and one solve per right-hand-side set instead of about 800.
Pivoting cannot cross blocks, so the factors are the same. A singular block triggers per-frequency
refactorisation, which keeps the NaNs local to that frequency. COLAMD (the default) was the fastest
ordering; MMD/NATURAL were slower or gave more fill.

**E8 — cheaper tracing.** Tracing `jax.jvp` costs 35 ms for all functions, against 93 ms for
`jax.vjp`. The interpreter now transposes only the linear tangent equations of JAX's JVP program,
so derivative formulas are still JAX's. This matches `jax.vjp` to ≤ 2.7e-16 on all 19 component
functions (`tests/check_jvp_transpose.py`). Unsupported tangent programs fall back to VJP tracing.

**E9 — Challenger audit of E3–E5 (CONDITIONAL → fixed).** It reported two demonstrated bugs. Both
were outside the shipped component set, and both are now fixed:
- **(A) Rank-mismatched operands.** Elementwise ops with mixed per-example ranks (e.g.
  `mul c128[4] c128[]`) were mis-broadcast with a batch axis. Batched operands are now aligned.
- **(B) Complex cotangent into a real output.** A real-valued custom linking function received a
  complex cotangent (JAX passes the real part); it now receives a real one.

Every traced function is also self-checked once, at about 19 ms in total:
- batched evaluation against batch-1 rows;
- for JVP transposition, the dot-product test against JAX's JVP program.

Any failure routes `simulate` to the in-graph JAX fallback. All Challenger reproducers now match JAX
or fall back. One finding was rejected as out of scope: gradients with respect to inputs other than
`optimized_parameters`, which is documented.

**E10 — native C++ sparse LU (branch `cpp-implementation`).** After E9 the largest host-side
cost was SciPy SuperLU: the factorisation (≈0.1 s), refined solves, and loading the extension
(≈25 ms). `differometor/native/dfsparse.cpp` (plain C ABI, loaded with `ctypes`) replaces
`BatchedLU` when the library is built; otherwise SuperLU is used unchanged (`DIFFEROMETOR_NATIVE=0`
forces SuperLU).

- **Algorithm.** Left-looking Gilbert–Peierls LU with partial pivoting (CSparse/KLU style).
  Minimum-degree column order on A + Aᵀ is computed once per pattern.
- **Pivot reuse (KLU refactor).** Frequency 0 is pivoted; each later frequency reuses the previous
  pivot order and L/U pattern. The reuse is accepted only if every reused pivot is ≥ 0.1 × the
  largest entry below it (threshold partial pivoting); otherwise that frequency is pivoted afresh.
  On the benchmark all 100 later frequencies reuse it.
- **Solves.** N/T/H solves with one refinement step (residual from the original A), as in
  `RefinedLU`. A singular system gives NaN for that frequency only.
- **Signal system, 101 frequencies, 1 CPU (WSL i5-1145G7).** Factorisation 76 ms (SuperLU) →
  ≈20 ms. Refined solves are ≈4× faster. Benchmark phases over 3 runs each: forward 0.21–0.28 →
  0.10–0.11 s; gradient 0.10–0.11 → 0.08–0.11 s; warm runtime median 0.26–0.28 → 0.14–0.16 s. The
  end-to-end first run on this machine is dominated by import-time noise (2.5–3.8 s).
- **Accuracy** (`tests/check_native_solver.py`, 12 topologies: aLIGO ×3 modes, Voyager ×3,
  both UIFO fixtures, 4 random UIFOs). Native vs SuperLU agrees about as closely as SuperLU vs the
  in-graph JAX path. The exceptions:
  - Voyager AM: 4e-6 log (stagnates with more refinement; at the float64 floor shown in E5).
  - Voyager FM: below float64 resolution for every method (E5). Here native is closer to JAX than
    SuperLU is.
  - Public benchmark: |Δlog S| 5.374e-6, identical to SuperLU.
- **Rejected.** Minimum degree on AᵀA (COLAMD-like) has 30 % less fill and factorises in 13 ms, but
  its error is 1.5e-2 on Voyager space modulation and 2e-6 on uifo_20_5000, even without pivot
  reuse. The ordering affects the tiny solution components the noise form depends on.
- **Build.** `pip install -e .` compiles the library (`setup.py` → `differometor/native/build.py`):
  `$CXX`, `c++`, `g++` or `clang++`, else the `ziglang` pip package. If no compiler is found, the
  install still succeeds and SuperLU is used. Nothing is compiled at run time.

## Generality evidence (E5; unchanged in E9, [`results/generality-e9.jsonl`](results/generality-e9.jsonl))

20 log-spaced frequencies (3 Hz – 3 kHz) per case. The comparisons are host path vs original
dense code, and host path vs in-graph sparse path differentiated by JAX.

| Case | vs dense: Δlog S | vs dense: grad rel | vs JAX-AD: grad rel |
|---|---|---|---|
| aLIGO (3 modulation modes) | ≤ 2.1e-7 | ≤ 8.8e-9 | ≤ 5.4e-10 |
| Voyager space modulation (qhd) | 9.7e-12 | 5.2e-10 | 4.4e-12 |
| uifo_20_5000 fixture | 2.6e-10 | 4.7e-10 | 3.1e-10 |
| random UIFO 3×3 seed 0 | 9.4e-10 | 9.1e-11 | 1.2e-12 |
| random UIFO 3×3 seed 1 | 7.9e-4 † | 1.6e-4 † | 8.8e-11 |
| Voyager amplitude modulation | 1.9e-4 † | 1.3e-6 | 6.3e-8 |
| Voyager frequency modulation | 2.2 † | 1.3 † | 1.4e-3 |

† In these cases the dense code is the inaccurate one. Measured against 50-digit references at the
worst frequency:

| Case | Dense log error | Ours log error |
|---|---|---|
| UIFO 3×3 seed 1 | 1.1e-3 | 2.2e-10 |
| Voyager AM | 1.8e-4 | 2.2e-6 |
| Voyager FM | 2.13 | 6.8e-3 |

In Voyager FM the detected signal is about 1e-13·‖x‖, which is below float64 resolution for any
method. Noise agrees with dense to ≤ 2.4e-10 in every case.

## Known limitations

- **No forward-mode AD on the host path.** Forward-mode autodiff (`jacfwd`, `hessian`) is not
  supported on the default host path. Reverse mode, `jit`, `vmap` (sequential) and `checkpoint`
  work. `DIFFEROMETOR_BACKEND=jax` uses the in-graph sparse path, which supports full JAX autodiff.
- **Gradients only with respect to `optimized_parameters`.** On the host path, other `simulate`
  inputs are treated as constants.

## Failed / rejected ideas

- **Splitting groups by frequency dependence (E2).** No benefit at first, because padding index 0
  marked everything as swept. Fixed in E5.
- **Removing `lax.switch` (E2).** Cut gradient compile from 1.68 s to 0.79 s, but forward compile
  stayed about 1.5 s. Stubbing the component functions showed they were about 1.4 s of compile,
  which motivated E3.

## Reproduce

```bash
python -m pip install -r requirements.txt && python -m pip install -e .
python benchmark.py --cpus 1 --repeats 3 --output benchmark-check.json   # Linux
python benchmark.py --repeats 3 --output benchmark-check.json            # macOS
DIFFEROMETOR_BACKEND=jax python benchmark.py --repeats 3 --output check-jax.json  # in-graph fallback path
DIFFEROMETOR_NATIVE=0 python benchmark.py --repeats 3 --output check-superlu.json  # host path with SuperLU instead of the C++ solver
python tests/check_native_solver.py  # C++ solver vs SuperLU vs JAX on 12 topologies
```

## Challenger — final audit of the integrated submission (HEAD after E9): **APPROVE**

No demonstrated bugs in E6–E9 or their combination.

- **E6 (direct SuperLU binding).** Solves and `perm_c` are bitwise identical to `scipy.sparse.linalg.splu`.
  On Linux, scipy 1.18.1 ships exactly one `_superlu.*.so`; it depends only on NumPy, and a later
  `import scipy.sparse.linalg` reuses the same module.
- **E7 (block-diagonal LU).** Results match per-block LU within 1e-15. NaN/inf in one block does not
  affect the others. Fill is identical even on an 8×8 UIFO (n = 4533).
- **E8 (JVP transposition).** On real component inputs captured from aLIGO, Voyager, UIFO and the
  fixture, all 19 component functions match `jax.vjp` to ≤ 4e-16, including scaled, sign-flipped and
  zeroed inputs.
- **Integrated gradients, host vs JAX autodiff.**
  - aLIGO and Voyager, all modes: ≤ 3e-11.
  - UIFO 2/3/4: ≤ 2e-11.
  - Carrier sweeps: ≤ 4e-14.
  - Perturbed parameters: ≤ 7e-14.
  - Process-state cases (eviction, re-jit, vmap, repeated builds) all correct.
- **benchmark.py:** PASS (max log error 5.37e-6).

Residual hypothetical risks:
- The pattern caches never evict, which only matters for many large setups in one process.
- The self-check samples inputs only in [0.3, 1.7].
- **HEAD has not yet been run on Linux with `--cpus 1`.** Recommended before submission.
