# Feyer Faster Physics: final validation of `final/combined-v2`

**Verdict: GO.** The new aLIGO setup passes the unchanged official benchmark with all 29 gradients. Every output array and the full gradient match the official implementation to about 1e-10. All required solves ran on the native C++ backend, with no SciPy or JAX fallback. The development benchmark and a second public grid variant pass as well. The one case where outputs disagree (Voyager frequency modulation) is ill-conditioned in the problem itself: the official code also changes its answer by 9.1 in log-sensitivity under a 1e-12 input perturbation. Section H covers it. Limits on what was checked are listed in section I.

All evidence is in [`results/final-validation/`](results/final-validation/): `artifacts/` holds the raw JSON reports, logs, saved arrays and probe output, and `tools/` holds every script used. The submission code was not modified. Nothing was pushed or merged.

## A. Source revisions

| Repository | Ref | Commit |
|---|---|---|
| Optimised solver, `github.com/Kartikdangi1/vibevoxels-starthack` | `final/combined-v2` | `c8066b864024c704c230d6653d6f49f318f233da` |
| Official Feyer, `github.com/feyerlabs/start-hack-munich-2026` | `main` | `83f8d2f40fc75218a1d8ec7908cc0bf1c8f99a75` |

- **Branch name.** The brief says `final/combined_v2` (underscore). No such branch exists. `final/combined-v2` (hyphen) is the submission branch, and that is what was validated. The other similar branches (`final/combined`, `final/combined-v2-petr`) were not used.
- **Scoring scripts.** The official `benchmark.py`, `benchmark_aligo.py` and `plot_timings.py` from `83f8d2f` were copied unchanged into the optimised tree. The judges score with their own scripts, and our branch still carries the older `benchmark.py`. Every input fixture is byte-identical between the two trees (sha256 checked).
- **Reference simulator.** The official simulator code and data are unchanged since the upstream baseline `b0dc7c7`. Official HEAD is therefore the numerical reference.

## B. Execution environment

| Item | Value |
|---|---|
| Machine | AMD EPYC 9334 (x86_64), 128 logical CPUs, 503 GB RAM |
| OS | Ubuntu 22.04.5 LTS container, Linux 6.8.0-79, glibc 2.35 |
| CPU limit | every run pinned with `taskset -c 20` plus the official `--cpus 1` |
| Compiler | g++ 11.4.0 |
| Python environment, optimised | Python 3.12.15, jax 0.9.0, jaxlib 0.9.0, numpy 2.5.5, scipy 1.18.1 |
| Python environment, reference | separate venv, identical versions |
| Environment variables | none set (no XLA_FLAGS, JAX_*, DIFFEROMETOR_* or thread limits) |

Both venvs already existed on the machine with the pinned challenge versions. They were reused instead of created fresh, because the machine reaches PyPI only through a slow proxy.

## C. C++ backend

- **Build.** `differometor/native/build.py` ran `/usr/bin/c++ -O3 -std=c++17 -shared -fPIC -fno-math-errno -march=native -ffp-contract=off` and exited 0 ([build.log](results/final-validation/artifacts/build/build.log)).
- **Load.** `native.available()` is True, and the loaded library is `differometor/native/libdfsparse.so`.
- **Backend actually used.** `tools/run_official.py` executes the official script unchanged, in-process, and counts calls into every solver entry point. That counts the host path vs the JAX fallback, the C++ `NativeBatchedLU` vs SciPy `SuperLUBatchedLU`/`RefinedLU`, and `simulate_sparse_jax`. The counts are not inferred from a PASS.

| Run (official script, `--repeats 3`) | Host path | JAX fallback | C++ solves | SciPy solves | Library loaded |
|---|---:|---:|---:|---:|---|
| Development, locally built library | 2 of 2 | 0 | 16 | 0 | `libdfsparse.so` (g++) |
| aLIGO, locally built library | 2 of 2 | 0 | 16 | 0 | `libdfsparse.so` (g++) |
| Development, prebuilt only (no compile) | 2 of 2 | 0 | 16 | 0 | `prebuilt/linux-x86_64/libdfsparse-v4.so` |
| aLIGO, prebuilt only (no compile) | 2 of 2 | 0 | 16 | 0 | `prebuilt/linux-x86_64/libdfsparse-v4.so` |

Every independent `vcase` run (section F) also used the host path and the C++ solver only (`artifacts/vcase/*_opt.probe.json`).

## D. Development benchmark (101 frequencies, 193 design variables)

Command: `python benchmark.py --cpus 1 --repeats 3` (official `83f8d2f` script, unchanged).

| | Optimised (built) | Optimised (prebuilt, no compile) | Official reference |
|---|---:|---:|---:|
| Result | **PASS** | **PASS** | PASS |
| Complete first run | **0.587 s** | 0.602 s | 35.221 s |
| Setup / compilation | 0.329 / 0.171 s | 0.334 / 0.183 s | 0.974 / 19.299 s |
| Forward / sensitivity / gradient | 48.3 / 1.8 / 36.6 ms | 47.4 / 1.5 / 36.1 ms | 12.03 s / 0.9 ms / 2.92 s |
| Warm median | 0.0837 s | 0.0807 s | 15.40 s |
| Max \|Δlog S\| vs public reference | 5.374e-6 | n/a (not saved) | 5.374e-6 |
| Official gradient check (`mb11.tuning`) | AD 7.6162e-5, FD 7.6167e-5 / 7.6157e-5, passed | passed | AD 7.6162e-5, FD 7.5676e-5 / 7.6092e-5, passed |

All 101 sensitivities are positive. Carrier, signal, noise, sensitivity, powers, loss and gradient are all finite and double precision (complex128 / float64). The 193-entry gradient is nonzero.

## E. New optical setup: aLIGO

- **Identification.** Commit `83f8d2f` (21 Nov 2026, 23:40) added `benchmark_aligo.py`: the public simplified Advanced LIGO setup, `differometor.setups.aligo(mode="space_modulation")`, with mirror mechanics and squeezed light.
- **Dimensions.** 101 frequencies (20–5000 Hz) from `examples/data/aligo_test_data.json`, and **29 design variables**: reflectivity, tuning, squeezing dB and angle, laser power, mass and independent lengths. The official tolerance is \|Δlog S\| ≤ 1e-6.
- **Setup check.** The script also pins the constructed optical input by hash, and that check passed.

Command: `python benchmark_aligo.py --cpus 1` (official script, default 10 repeats).

| | Optimised (built) | Optimised (prebuilt, no compile) | Official reference |
|---|---:|---:|---:|
| Result | **PASS** | **PASS** | PASS |
| Complete first run | **0.490 s** | 0.491 s | 20.782 s |
| Setup / compilation | 0.303 / 0.172 s | 0.305 / 0.172 s | 0.892 / 19.715 s |
| Forward / sensitivity / gradient / transfer | 7.1 / 0.8 / 6.6 / 0.13 ms | 7.2 / 0.8 / 6.6 / 0.12 ms | 124.6 / 0.4 / 49.4 / 0.28 ms |
| Warm median | 0.0116 s | 0.0114 s | 0.122 s |
| Max \|Δlog S\| vs public reference | **2.88e-11** (limit 1e-6) | | 1.23e-10 |
| Official gradient check (`sq1.db`) | AD −0.0129066103, FD −0.0129066103 (both steps), passed | passed | passed |

**Caveat on the final evaluation.** The official README (`2bb437a`) says the final evaluation uses "a slight variation of the supplied example", which is the 3×3 grid. aLIGO is the published second benchmark for generality. The grid variant `uifo_20_5000` was therefore also validated (sections F and G).

## F. Numerical comparison against the official implementation

Both codes ran on the same machine with identical inputs.
- **Official benchmark arrays:** from `--save-arrays`.
- **Independent arrays:** from `tools/vcase.py`, which builds inputs exactly as the official benchmark does and differentiates the official readout with `jax.value_and_grad`.
- **Metrics.** "Normwise" means max\|opt − ref\| / max\|ref\|. "Elementwise" is the max relative error over entries larger than 1e-6 × max\|ref\|, because `signal` spans about 30 orders of magnitude and its near-zero entries make plain elementwise ratios meaningless.

| Setup (f, vars) | carrier | signal | noise | sensitivity | powers | gradient | max \|Δlog S\| | gradient cosine |
|---|---|---|---|---|---|---|---|---|
| **aLIGO** (101, 29) | 6.6e-13 / 6.6e-12 | 2.4e-11 / 5.2e-10 | 3.9e-12 / 1.2e-10 | 3.7e-12 / 1.2e-10 | 1.3e-12 / 1.0e-11 | 2.8e-10 / 5.6e-10 | **1.18e-10** | 1.000000000000 |
| **Development grid** (101, 193) | 3.6e-12 / 1.6e-11 | 1.3e-7 / 1.6e-7 | 4.1e-8 / 7.8e-8 | 5.2e-9 / 6.7e-8 | 5.8e-12 / 1.3e-11 | 8.9e-8 / 7.8e-7 | **6.72e-8** | 1.000000000000 |
| **uifo_20_5000 grid** (101, 192) | 1.1e-12 / 1.9e-11 | 3.1e-10 / 2.1e-8 | 4.7e-11 / 5.2e-11 | 3.5e-11 / 3.8e-11 | 2.1e-12 / 4.2e-12 | 2.2e-10 / 3.5e-9 | **3.76e-11** | 1.000000000000 |
| Voyager space modulation (20, 74) | 4.2e-12 / 1.6e-9 | 2.9e-9 / 3.5e-6 | 8.9e-12 / 8.9e-12 | 5.4e-12 / 9.7e-12 | 8.3e-12 / 8.8e-12 | 6.8e-10 / 6.1e-8 | **9.7e-12** | 1.000000000000 |

Each cell is normwise / elementwise. Scalar loss differences: aLIGO 5.0e-12, development 6.9e-10, uifo_20_5000 1.2e-12 (absolute). No array is bit-identical, which is expected: the sparse LU orders its floating-point operations differently from dense LU.

Only the development grid uses the official 1e-5 tolerance. Its difference from the official code (6.7e-8) is 150× smaller than that tolerance, and both codes sit at the same 5.374e-6 from the public reference. aLIGO, with its own 1e-6 tolerance, agrees to 1.2e-10. No official tolerance exists for the other arrays. They agree at the level of the floating-point conditioning measured in section H.

Sources: `artifacts/compare/*.json`, `artifacts/vcase/*.json`.

## G. Gradient validation

| Setup | Variables | Gradient length | Finite | Nonzero entries | vs official autodiff (rel. 2-norm) |
|---|---:|---:|---|---:|---:|
| aLIGO | 29 | 29 | yes | 29 | 2.78e-10 |
| Development grid | 193 | 193 | yes | 193 | 8.6e-8 (official arrays), 9.8e-8 (independent) |
| uifo_20_5000 grid | 192 | 192 | yes | 192 | 1.91e-10 |
| Voyager space modulation | 74 | 74 | yes | 74 | 5.30e-10 |

**Independent central finite differences.** For every property type, `tools/vcase.py` tests the first parameter and the parameter with the largest gradient: 11 aLIGO, 14 development, 14 uifo_20_5000 and 11 Voyager space checks. Each runs 16 halving step sizes plus Richardson extrapolation (`artifacts/fd/*_opt_fd.json`).

- **aLIGO, all 11 converge, worst 2.0e-8 relative.**
  - `etmx.tuning` (90°, high-finesse arm cavity) needs steps below 1e-4°. Richardson then reaches 1.5e-8.
  - `etmx.reflectivity` (0.999995) reaches 2.7e-6.
- **Development grid, 13 of 14 agree.** They range from 2.5e-10 to 1.8e-3 relative, with the larger values only on gradients below 1e-9 (absolute error ≤ 6e-7 of the largest gradient entry). The exception is `boundary01.power`, whose value is 1.0e-12, essentially zero. Steps larger than the value cross zero power. Steps smaller than it change the loss by less than the setup's noise floor, which is about 7e-7 in log S under a 1e-12 perturbation in both codes. Finite differences cannot resolve it in float64. The same check on the official code is worse (values from −1.4e5 to +8.4e5; `artifacts/fd/dev_near_zero_fd_ref.json`). Two independent derivative implementations agree on it: our adjoint gives −29.61287, and JAX autodiff of the official dense solver gives −29.61302 (5e-6 relative).
- **uifo_20_5000 grid, all converge.** `mt33.reflectivity` (value 3.2e-16) converges to 5.1e-5 with steps scaled to the value. The largest relative error, 2.9e-2, is on a gradient of 5e-12, i.e. 3e-21 of the largest gradient entry.
- **Voyager space modulation, all agree** except two gradients of size 1e-20 and 1e-23, which are roundoff (1e-15 of the largest gradient entry).

The official per-benchmark gradient spot checks pass on both codes (section D and E tables).

## H. Generality and fallbacks

| Setup | Topology vs development case | Result |
|---|---|---|
| aLIGO (space modulation) | different topology: Fabry–Perot arms, recycling, squeezer, suspensions; 36-node carrier and 82-node signal systems; 29 variables | agrees to 1e-10 |
| uifo_20_5000 | 3×3 grid variant with different parameters and sparsity (296/769 systems), 192 variables, 20–5000 Hz | agrees to 4e-11 |
| Voyager space modulation | different topology, 74 variables, 3–3000 Hz | agrees to 1e-11 |
| Voyager amplitude modulation | as above, amplitude-modulated signal | Δlog S 7.0e-5 vs official; the official code itself moves by 8.5e-3 under a 1e-12 perturbation |
| Voyager frequency modulation | as above, frequency-modulated signal | **outputs disagree (Δlog S 1.43, gradient cosine −1.0)**, but the official code itself moves by **9.1** in log S under a 1e-12 perturbation (ours 9.2) |

**On Voyager frequency modulation.** The sensitivity of that setup is not determined in double precision. A relative change of 1e-12 in the inputs changes the official dense result by a factor of about e^9. Neither implementation's output is meaningful there, so the disagreement says nothing about our solver's correctness. The repository's own `tests/check_native_solver.py` shows SciPy SuperLU and the in-graph JAX path disagreeing just as much (6.5e-2) on that case. It is not a benchmark setup.

**Fallbacks.** No fallback occurred on any setup tested: no JAX path, no SciPy solve. `tests/check_native_solver.py` (12 topologies: aLIGO ×3 modes, Voyager ×3, both grid fixtures, 4 random 3×3 grids) exited 0. Native vs SuperLU agrees to ≤ 7e-6 in log S on all but Voyager frequency modulation (see above).

**Supported features exercised:** different graph topologies and sparsity patterns, carrier/signal system sizes 36–296 / 82–769, 29–193 variables, tied and independent lengths, squeezing, suspensions/mechanics, 5–301 frequencies, frequency ranges from 3 Hz to 5 kHz, and near-singular systems (the ill-conditioned Voyager modes ran without failure and stayed finite).

## I. Submission verdict

**GO.**

- The new aLIGO setup passes the official benchmark unchanged, with all 29 gradients, at 0.490 s against the reference's 20.78 s.
- Outputs and gradients match the official implementation to about 1e-10, and every finite-difference check converges.
- The C++ backend was used for every solve, both when built locally and with the prebuilt library alone. The development benchmark and two further well-conditioned topologies, including a grid variant like the announced final setup, agree as well.

**Not covered, and why it does not block GO:**
1. **The judges' Intel Xeon Platinum 8468 was not available.** All runs were on AMD EPYC. The prebuilt AVX-512 library is selected only if the CPU reports the needed flags, and otherwise AVX2 or baseline. All levels give bit-identical output, measured earlier on this machine.
2. **Python environments were the existing pinned venvs, not freshly created ones** (slow package proxy). The versions match `requirements.txt`.
3. **The actual final setup is unknown.** Two grid variants and three other topologies were validated in its place.
4. **Recommended housekeeping before submitting.** The branch's `benchmark.py` and `plot_timings.py` are older copies than the official `83f8d2f` versions, and its `benchmark_aligo.py` is our self-contained variant. The judges use their own scripts, and all results above use the official ones. Syncing the three files to `83f8d2f` would avoid confusion.
