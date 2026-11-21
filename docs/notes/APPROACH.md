# Optimization approach

## Kept changes

- Group matrix updates by component function in a static JAX plan. This removes a traced `lax.switch` for every matrix update and lets frequency-independent signal updates run once.
- Factor each frequency's signal matrix once. Reuse its LU factors for signal, conjugate-transpose detector, and covariance solves.
- Use `lax.custom_linear_solve` around the factored solves, with LU factors stopped from differentiating. This follows the implicit derivative of a linear solve while avoiding differentiation through LU factorization.
- Map one frequency at a time and checkpoint its body to bound reverse-mode memory. Removing checkpointing passed correctness checks but made the cold first run slower, so checkpointing stays enabled.
- Avoid carrier parameter expansion when there is one changing column; remove redundant complex casts; use int32 plan indices and NumPy for eagerly evaluated space phases during build.
- Defer Plotly import until visualization is requested and replace repeated name/function lookups with maps.
- Pin test-data JSON line endings to LF in `.gitattributes`; this makes its digest match the benchmark's expected fixture on Windows checkouts too.

## One-CPU measurements

All measurements below used the WSL2 CPU backend with one logical CPU requested and one CPU in affinity. Host: Intel Core i5-1145G7, Python 3.14.4, JAX 0.9.0.

The official `benchmark.py` run with the factored implicit solve passed the reference sensitivity and finite-difference gradient checks. Cold first-run total: **20.36 s**. Its report is [`bench-custom-solve-cpu1.json`](../../bench-custom-solve-cpu1.json); the report records `requested_logical_cpus: 1` and `cpu_affinity_count: 1`.

A no-checkpoint variant also passed correctness, but its cold first-run total was **21.08 s**, so it was reverted. An identity-matrix noise shortcut was also tried and reverted after its cold run took **21.79 s**. Repeated-run timings are not the target for this hackathon and are not used to choose the changes.

The user-provided earlier baseline was 68.637 s (setup 4.806 s, compilation 16.519 s, simulation/sensitivity/gradient 47.312 s). Against that reported value, 20.36 s is about 70% lower, but the baseline and current measurements may differ by host or environment. The old checked-in reference JSON had CRLF line endings and a different raw digest; `.gitattributes` now fixes the checkout format. The current official benchmark ran without a temporary harness.

## Reproduction

From the repository root, use one CPU explicitly:

```sh
python benchmark.py --cpus 1 --repeats 3 --output bench-custom-solve-cpu1.json
```

The derivative spot check for the factored solver is:

```sh
python tools/check_factored_solve.py
```

It checks complex solution values and first derivatives against `jnp.linalg.solve` for normal and conjugate-transpose solves.

## Not retained

A topology motif cache was slower on a cold first run (2.862 ms versus 1.414 ms) and was removed. A cell-wise Schur prototype also passed correctness but increased cold compile/first-run time, so it was removed. Sparse noise and an identity-noise shortcut did not improve the measured cold total. Input-port elimination, sparse KLU, and compiler flags remain unimplemented and require profiling or platform-specific evaluation. The custom implicit solve is retained because it passed the benchmark's sensitivity and finite-difference gradient checks while the official one-CPU benchmark completed in 20.36 s.
