# E11/E12: cheaper XLA compilation and a CPU-tuned solver build

**Summary.** After E10, the first run spent most of its controllable time compiling
the small XLA programs that wrap the host simulation. Two build/configuration changes,
with no algorithm changes, reduce that time:

| ID | Change | Where | Effect (same machine, fresh processes) | Outputs |
|---|---|---|---|---|
| E11 | XLA:CPU classic emitters (`--xla_cpu_use_fusion_emitters=false`) | `differometor/__init__.py` | trace/lower + compile 0.621 → 0.453 s | bit-identical |
| E11-fast | E11 + LLVM `-O0` for XLA kernels (`DIFFEROMETOR_XLA_FLAGS=fast`, opt-in) | `differometor/__init__.py` | trace/lower + compile 0.621 → 0.196 s | 1 of 101 sensitivities differs by 1 ulp; gradient bit-identical |
| E12 | C++ solver built with `-march=native -ffp-contract=off` | `differometor/native/build.py` | factorisation 29.7 → 24.3 ms, N/T/H solves ~15 % faster | bit-identical |

The numbers come from the dev container (Docker Desktop on an Intel i7 laptop, WSL2 backend,
`taskset -c 0`). The machine was noisy, so each setting was run in at least four fresh
processes and the medians are reported. Final `benchmark.py` before/after numbers are in
[the results section](#benchmark-before--after).

## Why compilation was still the target

The E10 phase breakdown on the AMD EPYC 9334 (1 CPU, `results/amd-epyc/v2_cpu1_1.json`,
first run 0.656 s):

| Phase | s | Controllable from `differometor/`? |
|---|---|---|
| imports | 0.264 | barely: `import jax` is ~0.22 s and the benchmark imports it before us |
| build | 0.064 | partly: ~half of it is the process's first XLA compile, triggered by the benchmark's own `jnp.asarray(..., dtype=float64)` |
| forward trace/lower + compile | 0.018 | already minimal: a single host callback |
| **sensitivity trace/lower + compile** | **0.150** | **yes, see below** |
| **gradient trace/lower + compile** | **0.067** | **yes, see below** |
| forward + sensitivity + gradient execution | 0.089 | yes (C++ solver: E12) |

The sensitivity program is the benchmark's readout: `signal_detector` → index ports →
`abs` → divide → `log10` → `mean`, plus `power_detector` and the VJP of all of it. After
E10 the detectors are host callbacks, so the compiled HLO is tiny: 234 lines, 9 fusions
and 2 custom calls (the gradient program has 158 lines, 5 fusions and 2 custom calls). It
still took ~0.2 s to compile. That cost is fixed per fusion (kernel code generation and
LLVM), not proportional to graph size. Making the graph smaller therefore could not help,
but making each kernel cheaper to generate could.

## E11: XLA flags set from the package

### What

`differometor/__init__.py` calls `_configure_xla()` before importing anything else. It
appends to `XLA_FLAGS`:

| `DIFFEROMETOR_XLA_FLAGS` | Flags added | Meaning |
|---|---|---|
| `exact` (default) | `--xla_cpu_use_fusion_emitters=false` | Use XLA's classic LLVM IR emitters instead of the newer MLIR fusion emitters. Same math, cheaper to compile for tiny fusions. |
| `fast` | the above + `--xla_backend_optimization_level=0` | Also skip LLVM's optimisation pipeline for the generated kernels. |
| `0` / `off` | none | XLA defaults (behaviour before E11). |

Flags already present in the user's `XLA_FLAGS` are never overridden: a flag is only
appended if its name does not appear there yet.

### Why it works from inside the package

XLA reads `XLA_FLAGS` when the CPU client is created, not when `jax` is imported. The
benchmark runs `import jax` → `import differometor` → `initialize()` (`jax.devices("cpu")`),
so flags set during `import differometor` reach the backend. If a program already ran a
JAX computation before importing differometor, the flags have no effect and nothing else
changes. The benchmark JSON records the active flags under `environment.xla_flags`, so the
setting is visible in every report.

### Evidence

Compile seconds (`scratch`-style probe that reproduces the benchmark up to each stage), 5
fresh processes per setting, medians:

| XLA flags | sensitivity compile | gradient compile | sum |
|---|---|---|---|
| defaults | 0.231 | 0.090 | 0.321 |
| `--xla_cpu_use_fusion_emitters=false` | 0.155 | 0.068 | 0.223 |
| + `--xla_backend_optimization_level=0` | 0.057 | 0.028 | 0.085 |

Flags that were tried and **did not help** (don't repeat):
- `--xla_llvm_disable_expensive_passes=true`: no change.
- `--xla_backend_optimization_level=1`: no change, and the loss changed in the 15th digit.
- `--xla_cpu_use_thunk_runtime=false`: slower.
- `--xla_cpu_use_xnnpack=false`: no change.
- `--xla_cpu_enable_concurrency_optimized_scheduler=false`: no change.
- `--xla_backend_optimization_level=0` alone (fusion emitters on): only about half the
  saving of the combination.

### Correctness

Full `benchmark.py --cpus 1 --repeats 3 --save-arrays`, 4 alternating rounds per mode, all
12 runs `PASS`. The saved arrays (carrier, signal, noise, sensitivity, powers, gradient)
were compared with `np.array_equal` against mode `0`:

- `exact`: every array bit-identical.
- `fast`: everything bit-identical except one of the 101 sensitivities, which differs by
  exactly 1 ulp (2.2e-16 relative). The gradient is bit-identical. The correctness bar is
  1e-5 in log sensitivity.

Runtime is unaffected: the readout kernels operate on 101 numbers, and the warm sensitivity
phase stays at ~1 ms.

### Why `exact` is the default

`fast` is not bit-identical, so it stays opt-in, following our rule that a default change
must keep outputs identical. Switching the default to `fast` is a one-word change in
`_configure_xla()`. On the EPYC it saves another 0.076 s on the first run
(0.585 → 0.509 s, 60× → 69×) for a 1-ulp difference, which is a team decision for the submission.

## E12: CPU-tuned, bit-identical solver build

### What

`differometor/native/build.py` first tries to compile `dfsparse.cpp` with
`-march=native -ffp-contract=off` in addition to the existing flags. If a compiler
rejects those flags, the same compiler is retried without them, so the install never
fails because of the tuning. `DIFFEROMETOR_NATIVE_PORTABLE=1` skips the tuning, for
example when building a library for a different machine.

### Why `-ffp-contract=off` matters

`-march=native` lets the compiler use AVX2/AVX-512 and FMA. With FMA contraction allowed,
`a*b + c` is computed with one rounding instead of two, and results change. On random
right-hand sides for the benchmark's signal matrices the difference reached 5.3e-4
relative, because these systems are ill-conditioned. With contraction off, the wider
instructions still apply (vectorised loads, stores and independent operations), but every
floating-point operation rounds exactly as in the portable build. The x86-64 baseline has
no FMA, so the portable build never contracted.

### Why `-march=native` is safe here

The library is compiled by `pip install -e .` on the machine that runs the benchmark (see
`setup.py`), not shipped as a binary. A library copied to an older CPU could hit an
illegal instruction; that is what `DIFFEROMETOR_NATIVE_PORTABLE=1` is for.

### Evidence

Real signal matrices of the benchmark (n = 751, nnz = 2597, F = 101), 9 repetitions each,
medians, `taskset -c 0`:

| Build | factor (101 systems) | N + T + H solves, k = 3 | vs portable |
|---|---|---|---|
| `-O3` (portable, before) | 29.7 ms | 46.5 ms | reference |
| `-O3 -march=native -ffp-contract=off` (**E12**) | 24.3 ms | 39.7 ms | **bit-identical** |
| `-O3 -march=native` (FMA allowed) | 19.8 ms | 41.4 ms | differs up to 5.3e-4: rejected |
| `-O2` | 27.7 ms | 67.5 ms | bit-identical, slower |

The library that `build.py` produced is byte-identical to the variant measured as E12.

## Benchmark before / after

One session on the AMD EPYC 9334 cloud machine (`results/amd-epyc-final/`, script `run_final.sh`):
every variant on the same core (`taskset -c 72`, `benchmark.py --cpus 1 --repeats 10`),
interleaved over 3 rounds, same Python 3.12.15 / JAX 0.9.0 / NumPy 2.5.5 / SciPy 1.18.1.
All 21 runs `PASS`. Medians of 3 runs (min–max):

| Variant | First run s (scored) | vs baseline | XLA compile s | Warm median s | vs baseline |
|---|---|---|---|---|---|
| upstream baseline | 35.283 (35.223–35.307) | 1.0× | 19.234 | 15.477 | 1.0× |
| combined-v2, committed (HEAD 303428a) | 0.656 (0.649–0.664) | 53.8× | 0.238 | 0.0834 | 185.5× |
| **final** (+ E11 `exact`, E12, `bincount` scatter) | **0.585 (0.583–0.595)** | **60.3×** | **0.171** | **0.0813** | **190.3×** |
| final, `DIFFEROMETOR_XLA_FLAGS=fast` | 0.509 (0.508–0.515) | 69.3× | 0.097 | 0.0817 | 189.4× |

The milestone branches (C++ LU 0.950 s, native kernels 0.923 s, host readout 0.886 s) and the
per-phase breakdown are in [`results/amd-epyc-final/comparison.txt`](../../results/amd-epyc-final/comparison.txt).

Where the 0.071 s from committed v2 to final comes from (first-run phase medians):

| Phase | committed v2 | final | fast |
|---|---|---|---|
| sensitivity compile | 0.130 | 0.088 | 0.033 |
| gradient compile | 0.060 | 0.034 | 0.016 |
| forward + gradient execution | 0.086 | 0.084 | 0.083 |

On EPYC, the solver build (E12) barely shows up in the end-to-end numbers: execution is already
short there, and most of the remaining first-run time is `import jax` (0.26 s) and setup.

Numeric agreement with the baseline's saved arrays (run 1): max |Δlog S| 6.717e-8, gradient
relative error 8.613e-8, cosine 1.0000000000. Final vs committed v2: the only array that
differs is the gradient (8.0e-10 relative), which comes from the `np.bincount` scatter's summation
order in `host._scatter_add`, not from E11/E12. Final vs fast: one sensitivity differs by 1 ulp
(2.2e-16).

Regenerate the table: `python tools/summarize_runs.py results/amd-epyc-final
baseline,cpp,stage3,fastreadout,v2_head,final,final_fast --pairs v2_head:final,final:final_fast`
(needs the `.npz` files, which are gitignored; the JSONs are enough for the timing part, and
`docs/report/figures/scripts/generate_all_figures.py` draws fig. 2 from them).

## Measured but not changed

| Idea | Measurement | Verdict |
|---|---|---|
| Cache the minimum-degree column order per pattern (C++ recomputes it on every `dfs_factor` call) | ~1 ms per call; the first run factorises each pattern only once | not worth the code |
| One batched `jax.device_put` instead of 98 separate calls in `compress_arrays` | 8 → 4 ms on the laptop (~1–2 ms on EPYC) | too small for now |
| Persistent XLA compilation cache | the benchmark disables it, and it would count as precomputation | not allowed |
| Numba for the Python kernels | importing numba alone costs more than the kernels | hurts the first run |
| A newer Python or another interpreter | the judges compare on matching software | not allowed |
| pip's editable-install finder (`__editable___differometor_*_finder`, 65 ms on the Docker bind mount) | mostly bind-mount file-system cost; small on a local disk | environment, not code |

## Next candidates (not implemented)

Warm-run profile after E12 (laptop, per forward + readout + gradient): C++ factorisation +
solves ≈ 60 %, NumPy glue in `host._backward` / `_assemble_backward` / `_scatter_add` ≈ 25 %,
Python dual-number kernels (`native_kernels.py`) ≈ 15 %.

1. **Vectorise the LU across frequencies.** All 101 frequencies reuse one pivot order and
   one L/U pattern (KLU refactor), so the numeric refactorisation and the triangular solves
   can store values frequency-innermost and loop over frequencies with SIMD. Each
   frequency's own sequence of operations is unchanged, so results stay bit-identical with
   `-ffp-contract=off`. Frequencies that fail the pivot threshold keep the current scalar
   path. Expected 2–4× on the largest remaining runtime block.
2. **Move the backward glue and the dual-number kernels into the same C library.** This
   removes ~40 % of the warm time that is still Python/NumPy dispatch.
3. Both mainly reduce execution time (~0.09 s of the EPYC first run). The first run's
   remaining floor is `import jax` plus the process's first XLA compile, which belong to
   the benchmark.

## Reproduce

```bash
pip install -e .                                   # builds dfsparse with the E12 flags (prints them)
python benchmark.py --cpus 1 --repeats 3 --output after.json --save-arrays after.npz           # E11 + E12
DIFFEROMETOR_XLA_FLAGS=0 python benchmark.py --cpus 1 --repeats 3 --output before.json --save-arrays before.npz
DIFFEROMETOR_XLA_FLAGS=fast python benchmark.py --cpus 1 --repeats 3 --output fast.json --save-arrays fast.npz
DIFFEROMETOR_NATIVE_PORTABLE=1 python differometor/native/build.py   # portable solver build (pre-E12)
python -c "import numpy as np; a, b = np.load('before.npz'), np.load('after.npz'); print(all(np.array_equal(a[k], b[k]) for k in a.files))"
```
