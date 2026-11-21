# Faster Differometor: approach and correctness

## Result

Full `benchmark.py --cpus 1` runs (10 warm repetitions), same machine and container:

| | Before | After | Change |
| --- | ---: | ---: | ---: |
| **Complete first-run total** | **55.14 s** | **31.15 s** | **−44%** |
| Trace/lower + compile | 26.42 s | 16.01 s | −39% |
| Forward simulation + AD residuals | 20.47 s | 9.58 s | −53% |
| Reverse-mode gradient | 6.59 s | 3.84 s | −42% |
| Median warm evaluation | 22.69 s | 13.18 s | −42% |
| Max \|log(sensitivity) − log(reference)\| | 5.37e-6 | 5.37e-6 | unchanged |
| Gradient spot check (`mb11.tuning`) | passed | passed | |

Reports and plots: [`benchmarks/results/`](../../benchmarks/results/)
(`before-original.*`, `after-shared-lu-grouped-fill.*`).

Hardware: Intel Core Ultra 7 155H laptop, Docker Desktop (WSL2), pinned to one logical CPU,
15.4 GiB available to the container. This is slower and noisier than the official Xeon 8468
baseline (44.2 s there): identical original-code runs here ranged 52.7–63.8 s. Before/after
comparisons were therefore always run back to back on the same machine.

All changes are in [`differometor/simulate.py`](../../differometor/simulate.py). The physics, the
setup, all 101 frequencies, all 193 design variables and complex128/float64 precision are unchanged.

## What we changed and why

We measured before changing anything (`tools/profile_parts.py`, `tools/profile_compile.py`). The
signal system is 101 dense 751×751 complex matrices that are under 1% nonzero. Two things dominated:
dense factorizations of those matrices, and a matrix-fill step that was expensive to compile.

### 1. One LU factorization per frequency instead of three

The original code factorized the signal matrix M three times per frequency. That was once each for
the signal (`M⁻¹b`), the detector back-propagation (`w = M⁻ᴴs`) and the noise propagation
(`M⁻¹ Vᵢ w`).

- **The noise solve was redundant.** The noise power is `Re(sᴴ M⁻¹ Vᵢ w)`, and since
  `sᴴ M⁻¹ = (M⁻ᴴ s)ᴴ = wᴴ`, it equals `Re(wᴴ Vᵢ w)`. That needs no solve at all.
- **The signal and adjoint solves now share one LU** (`solve_with_adjoint`), using `lu_solve`
  with `trans=0` and `trans=2`. Derivatives go through `lax.custom_linear_solve`, which is how
  `jnp.linalg.solve` itself is implemented. Reverse mode therefore reuses the factors and does not
  differentiate through the LU.

### 2. A smaller compiled program

- **No `lax.switch` under `vmap`.** Matrix entries came from `vmap(lax.switch(i, FUNCTIONS, x))`.
  Under `vmap`, the switch becomes a select over all 19 component functions, so every branch (and
  its derivative) was evaluated for every entry and compiled into the program. The setup only uses
  3, 11 and 3 distinct functions for the carrier, signal and noise systems. The build step now groups
  rows by function (`group_by_function`), and each used function is called once on its own batch.
  This gives the same values, and the forward program shrank from 7,568 to 3,698 StableHLO lines.
- **Sparse input noise.** The noise matrix Vᵢ was a dense 101×751×751 array (~0.9 GB) holding 272
  entries per frequency. When the base noise matrix is all zero and no entry is written twice
  (checked at build time; otherwise the dense path is kept), the noise is evaluated directly as
  `Σₑ conj(w[rₑ]) · valueₑ · w[cₑ]`.

## Correctness checks

Besides the benchmark's own checks (PASS), every candidate was compared array by array against the
saved outputs of the unmodified code (`benchmark.py --save-arrays`, then
`tools/compare_arrays.py`):

| Output | Max difference vs. unmodified code |
| --- | --- |
| carrier, signal, powers | bit-identical |
| noise, sensitivity | 2.1e-8 relative (log-sensitivity 2.1e-8; the budget is 1e-5) |
| gradient (193 values) | 5.7e-8 relative L2 |

## What we tried that did not help

- **Exact block elimination (Schur complement).** 345 of the 751 unknowns have a constant unit
  diagonal and no couplings to each other, so they can be eliminated exactly, leaving a 406×406
  LU (about 16% of the factorization work). We confirmed that the reduced solve has the same
  backward error as the full LU (~1e-29 relative residual).
  - The forward pass did get faster. But the gathers and scatters around the reduced system made
    the reverse pass roughly 3× slower, so the full benchmark totals were worse (38.8 s and 41.6 s,
    against 31.2 s).
  - Not shipped; kept as [`step3-schur-complement.patch`](step3-schur-complement.patch).
- **SciPy SuperLU via `pure_callback`** (team branch `feature/kartik`). It passes at full scale
  (49.5–50.0 s first-run total), so it is correct but barely faster. The dense solve was not the
  main cost, and its custom backward pass builds a dense outer product per solve.
- **Note on the `signal` output.** The system is numerically singular in some directions (field
  values up to ~1e22). Any two correct solvers can therefore give different raw `signal` entries in
  those directions, while the detector signal, noise, sensitivity and gradient agree.

## Reproduce

```bash
./container.bash                                   # or .\container.ps1 on Windows
python benchmark.py --cpus 1 --output benchmark-after.json --save-arrays after.npz
# Optional: the unmodified simulator for comparison (simulate.py is the only changed file;
# acd8293 is the repo's setup commit with the untouched upstream code)
git checkout acd8293 -- differometor/simulate.py
python benchmark.py --cpus 1 --output benchmark-before.json --save-arrays before.npz
git checkout HEAD -- differometor/simulate.py
python tools/compare_arrays.py before.npz after.npz
```

## Possible next steps

- The remaining time is mostly compilation (16 s). Candidates are building the signal matrix with
  one scatter instead of five full-size `.at[].set` passes (which also shrinks the AD residuals),
  and trying XLA optimization flags. `--xla_backend_optimization_level=0` cut compile time from
  17.5 s to 12.0 s in an isolated test, but we have not measured its effect on runtime.
- The block elimination above, with a cheaper reverse pass (for example a custom VJP built on the
  reduced factors).
