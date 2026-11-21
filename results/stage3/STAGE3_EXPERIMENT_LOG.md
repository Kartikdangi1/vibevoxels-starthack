# Stage 3 experiment log

Machine: Apple M2, 8 GB RAM, macOS 15.4, Python 3.12.10, JAX 0.9.0, NumPy 2.5.5, SciPy 1.18.1.
Benchmark: official `benchmark.py --repeats 3`, unchanged; no thread env vars set (`OMP/OPENBLAS/MKL/XLA_FLAGS` unset, as recorded in each JSON); persistent compilation cache off.
Recovery context: [STAGE3_RECOVERY_REPORT.md](STAGE3_RECOVERY_REPORT.md).

## S3-0 Baseline (interrupted session, 19:43 CET)
ad9603c, 5 runs: 1.437 (cold), 0.681, 0.739, 0.773, 0.954 s. Artifacts lost in reboot; numbers from transcript.

## S3-A Candidate A: native component value+VJP kernels
**Hypothesis.** ~82 ms of first run is `Plan` construction: `jax.make_jaxpr` of each component JVP (≈48 ms) + `_self_check` (≈38 ms) for 16 functions. Closed-form NumPy kernels with an exact Wirtinger-dual derivative remove it without changing numerics.

**Implementation.** `differometor/native.py` (19 kernels, same operation order as `components.py`), hooked in `Plan.__init__`; falls back to JAX tracing when a `FUNCTIONS` entry is not the original object. Linking functions still traced. Commits on `experiment/local-component-plan`: `d025ae0` (recovered WIP), `51b1896` (validated + evidence).

**Component validation** (`tests/check_native.py`, re-run post-crash, 281 MB peak): 19/19 functions match `jax.vjp` to ≤ 1e-10 (fwd ≤ 1e-12) on complex, physical and boundary inputs, plus a real-direction FD check. Only exception: `space_modulation(_lower)` d/d(frequency) (column 0) differs from JAX by 1.2e-5 relative. Against a cancellation-free Taylor reference (u = aω ≤ 6e-9), native errs 1.9e-6 and JAX 1.2e-5 — native is the more accurate of the two; the column is the swept frequency, never one of the optimised parameters. Columns 1–2: 3e-16.

**Topology validation** (11 setups × {native, traced, in-graph JAX}, 20 frequencies, one process per run, RSS guard 1.5 GB; peak 417 MB):

| Setup | P | native vs traced (fwd / grad normrel) | native vs in-graph JAX dlogS |
|---|---|---|---|
| aLIGO SM / AM / FM | 38 | 0 / ≤ 2e-16 | 4e-11 / 1e-8 / 3.5e-9 |
| Voyager SM / AM / FM | 60 | 0 / ≤ 1.4e-16 | 2e-10 / 6.1e-6 / 5.8e-2 † |
| UIFO 3×3 seeds 0,1,2 | 255–260 | 0 / ≤ 1.2e-16 | ≤ 3.4e-10 |
| UIFO 2×2 seed 5 | 130 | 0 / 1.7e-16 | 7e-15 |
| uifo_20_5000 | 255 | 0 / 2.2e-15 | 2.2e-9 |

† Pre-existing host-vs-in-graph difference, identical for ad9603c (native ≡ traced bitwise); documented in OPTIMIZATION_LOG (Voyager FM detected signal below float64 resolution; E9 measured 0.29).
Raw: `vv-exp/results/stage3/topo/topo_compare.json`.

**Benchmark** (6 ABBA-interleaved clean processes per arm, 20:07 CET, load avg ≈ 2):

| | baseline ad9603c | candidate 51b1896 | Δ |
|---|---|---|---|
| runs (s) | 1.008, 0.644, 0.636, 0.634, 0.652, 0.641 | 0.542, 0.530, 0.541, 0.534, 0.531, 0.540 | |
| **median complete first run** | **642.5 ms** | **537.1 ms** | **−105.4 ms (−16.4 %)** |
| imports | 219.2 | 211.8 | −7.4 |
| forward_lower | 92.6 | 10.5 | −82.1 |
| forward | 88.0 | 81.7 | −6.3 |
| gradient | 51.3 | 41.5 | −9.8 |
| sensitivity_compile / gradient_compile | 55.9 / 47.7 | 56.8 / 47.4 | ≈ 0 |
| warm runtime median | 123.0 | 111.2 | −11.8 |
| peak RSS | 652–692 MB | 644–669 MB | |

All 12 runs PASS (max |Δlog S| = 5.37e-6, same as baseline). Saved arrays: carrier, signal, noise, sensitivity, powers identical bit-for-bit; gradient normrel 1.6e-16 (maxabs 7.3e-12).
Every candidate run is faster than every baseline run. Raw: `vv-exp/results/stage3/bench/`.

**Decision.** Keep. Pending Challenger review.

## S3-B Candidate B: KLU symbolic reuse — BLOCKED
No KLU binding in the venv (`scikit-sparse`, `kvxopt`, `pypardiso`, … absent). Adopting it requires installing SuiteSparse + a wrapper and adding it to the official Docker image — needs explicit authorisation. Remaining target size: `forward` 82 ms on the candidate (SuperLU factor + solves, all frequencies in one block-diagonal factorisation since E7), of which only the symbolic part could be saved.

## Remaining first-run budget (candidate, median)
imports 212 (mostly the benchmark's own `import jax`, outside our control) · forward 82 · sensitivity_compile 57 · build 47 · gradient_compile 47 · gradient 42 · sensitivity_lower 17 · gradient_lower 11 · forward_lower 10 · forward_compile 8.

## S3-A Challenger review — APPROVE
Independent read-only audit. It ran only guarded checks, peaked at 284 MB, and no benchmarks overlapped with it.
- Verified kernel parity for all 19 functions line by line, the Dual product/quotient/conj rules, the VJP transpose convention, interface parity (batch 0/1, dtypes, `unsupported`), the identity-based fallback, that no cache persists across runs, and benchmark fairness (identical starter/setup/components hashes across all 12 JSONs).
- Non-blocking, now fixed: `tests/check_native.py` printed FAIL but exited 0. It now exits non-zero on any failure, holds the space_modulation frequency column to its own 1e-4 bound, and pins the `components.py` SHA-256, so an edit there forces a kernel re-check (finding 2).
- Informational: `np.errstate(all="ignore")` suppresses overflow warnings that the traced path used to print. Results are unaffected.
