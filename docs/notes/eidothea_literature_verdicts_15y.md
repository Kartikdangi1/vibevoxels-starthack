# 15-Year Literature Verdicts, Cross-Domain Optimisations & Audit (2011–2026)

*Eidothea physics taxonomy v2 · START Hack Munich 2026 — Feyer: Faster Physics · Differometor*

## Evidence labels

- **`[S]`** computed from the Eidothea harvest in `data/physics15_probes` and `data/physics15_graph`.
- **`[A]`** abstract read in `abstracts.jsonl.gz` (not the full paper).
- **`[R]`** verified against the repository or a benchmark run of the final `final/combined` solution.
- **`[K]`** a general technique from outside this harvest; **not verified here**, so treat as a lead, not a result.

Nothing in this document was verified from full text. Every paper claim is `[A]` or weaker.

---

## 1. What changed in Eidothea

### 1.1 Taxonomy (`src/eidothea/taxonomy/physics_taxonomy.py`): 13 → 27 subfields, 4 → 6 groups

Groups A–D are unchanged. Two groups are new:

| Group | Subfield | What it captures | Papers |
|---|---|---|---:|
| **E host & numerics** | `host_exec` | running traced programs outside XLA: host callbacks, jaxpr interpreters, staging | 50 |
| | `jvp_transpose` | linearize then transpose: reverse mode derived from the forward (JVP) program | 39 |
| | `iter_refine` | iterative refinement, mixed precision, ill-conditioned sparse solves | 102 |
| | `compile_cold_start` | compile latency, cold start, control flow under vectorisation | 41 |
| | `iter_recycling` | Krylov recycling and preconditioner reuse across related systems | 28 |
| **F cross-domain** | `xd_circuit` | SPICE, modified nodal analysis, AC and noise analysis, harmonic balance | 37 |
| | `xd_power_grid` | power flow, contingency analysis, Kron reduction | 106 |
| | `xd_fdfd_geo` | FDFD, Helmholtz, RCWA, frequency-domain full-waveform inversion | 112 |
| | `xd_struct_dyn` | frequency response functions, dynamic substructuring, component mode synthesis | 16 |
| | `xd_quantum_transport` | NEGF, recursive Green's function, selected inversion | 110 |
| | `xd_photonic_cascade` | scattering-matrix cascading, Redheffer star product, photonic circuit simulators | 51 |
| | `xd_mor_control` | LTI model reduction, balanced truncation, Loewner, descriptor systems | 81 |
| | `xd_ordering_graph` | treewidth, nested dissection, multifrontal, partitioning | 107 |
| | `surrogate_ml` | neural surrogates and operators as an alternative to exact solvers | 132 |

`batched_sparse` was widened to include block-diagonal batching. The taxonomy tests pass (9/9).

**Tagging quality.** The tagger is a keyword matcher. 503 of 2,342 papers (21.5%) match no subfield. Two patterns had to be fixed after a first pass because common words collided: "SPICE" is also a line-spectral-estimation algorithm, and "circuit simulator" matches quantum-circuit simulators. Counts in the new groups are an upper bound on relevance, and the verdicts below come from reading abstracts, not from tag counts.

### 1.2 Probes (`scan_arxiv_physics.py`): 20 → 37 probes, 10 → 15 years

The scan window is now 2011-01 to 2026-11 in 7 sub-windows. 17 probes were added: 6 for lessons from the implementation (`num_`, `host_`, `jvp_`, `cold_`, `iter_`) and 11 cross-domain (`xd_`). The harvest used arXiv's public API at ≥3.5 s per request with a generic user agent. One probe hit HTTP 429 and recovered after backoff.

### 1.3 Harvest and graph `[S]`

| | Before (10 y) | Now (15 y) |
|---|---:|---:|
| Unique papers | 1,067 | **2,342** |
| Window | 2016-01 → 2026-11 | **2011-01 → 2026-11** |
| arXiv reference edges | 20,915 | **37,826** |
| Graph nodes / edges | 1,080 / 1,900 | **2,369 / 4,177** |
| Subfields | 13 | 27 |

Papers per year: 2011: 33 · 2012: 39 · 2013: 54 · 2014: 63 · 2015: 73 · 2016: 87 · 2017: 72 · 2018: 102 · 2019: 117 · 2020: 149 · 2021: 160 · 2022: 158 · 2023: 163 · 2024: 208 · 2025: 330 · 2026: 534 (partial year). The harvest is relevance-ranked (top 100 per probe), so counts reflect what the probes retrieve, not the size of each field.

Louvain communities over the 27 topics:
1. **Sparse linear algebra, reuse and ordering:** `sparse_lu`, `low_rank_update`, `batched_sparse`, `iter_refine`, `iter_recycling`, `jvp_transpose`, plus the `xd_fdfd_geo`, `xd_ordering_graph`, `xd_power_grid` and `xd_quantum_transport` analogues.
2. **Differentiable programming and compilers:** `jax_xla`, `autodiff_opt`, `host_exec`, `compile_cold_start`, `implicit_diff`, `diff_optics`, `surrogate_ml`.
3. **Frequency sweeps, adjoints and reduction:** `freq_sweep`, `adj_quantum_cov`, `xd_circuit`, `xd_mor_control`, `xd_struct_dyn`.
4. **Interferometry:** `ifo_sim`, `gw_detector`, `quantum_noise`, `sideband_mod`, `xd_photonic_cascade`.

Reading: the cross-domain fields that share a community with `sparse_lu` (power grids, FDFD/geophysics, quantum transport, ordering) are where reusable sparse-solver methods live. Circuit simulation, model reduction and structural dynamics share a community with `freq_sweep` and `adj_quantum_cov`, which is where the frequency-sweep and adjoint-noise analogues live.

---

## 2. Audit of the previous verdicts file `[A]`

`docs/notes/eidothea_literature_verdicts.md` (10-year version) cites several papers with the wrong title or authors. These were checked against the catalog:

| arXiv id | Previous note says | Catalog says |
|---|---|---|
| `2101.02270` | Anzt et al., *Batched sparse direct solvers and LU refactorization on fixed sparsity patterns* | Wang & Wende-von Berg, *Fast Parallel Newton-Raphson Power Flow Solver for Large Number of System Calculations with CPU and GPU* |
| `2610.41324` | RapidDOC: sparse KLU refactorization, "69× speedup" | *RapidDOC: Riccati-Preconditioned Implicit Differentiation for Differentiable Optimal Control* (Zhu, Chen) |
| `2610.12336` | asdex: fast Jacobian *sparsity detection* (Hill) | *asdex: Automatic Sparse Differentiation in JAX* (Hill, Dalle) |
| `2406.05027` | Lohoff & Neftci, *Avoiding expression growth in JAX AD* (NeurIPS 2024) | *Optimizing Automatic Differentiation with Deep Reinforcement Learning* (Lohoff, Neftci) |
| `2003.13202` | Miao, Smith & Evans, *completing the square…* | catalog lists Yamada, Enomoto et al. as authors |
| `2509.24740` | SymBoltz.jl: "differentiable linear solver" | an Einstein-Boltzmann solver for cosmology |

**Consequence.** The "69× speedup" attributed to RapidDOC and the "batched sparse LU refactorization" attributed to `2101.02270` are not supported by those papers' abstracts. Technical claims that rest only on those citations should be treated as unsupported. The ideas themselves (shared LU, adjoint reuse) stand on the repository evidence `[R]`, not on those citations. I did not edit the old file; it should be corrected or retired.

---

## 3. What the literature confirms about what we built

| What the final solution does `[R]` | Independent support in the harvest `[A]` |
|---|---|
| Quantum noise as `wᴴNw`, one adjoint solve (no second solve) | Adjoint-mode differentiation in a circuit simulator: ADSPICE `2611.23571`; transient forward harmonic adjoint `2401.13496`. The same adjoint idea is standard in circuit analysis. |
| All 101 frequencies factored as one block-diagonal sparse system | **SABLE** `2606.07099` (power flow): "block-diagonal embedding that reformulates batched three-dimensional Jacobians as a fixed-pattern two-dimensional sparse template". The same trick, arrived at independently. Also `2101.02270` for repeated power flows on one sparsity pattern. |
| Reverse mode from the JVP program (linearize, then transpose) | *You Only Linearize Once* `2204.10923`; *Decomposing reverse-mode AD* `2105.09469`. |
| Sparse VJP evaluated only on the nonzero pattern | *Differentiating Through Linear Solvers* `2404.17039`; APF-based inverse design `2306.09257`. |
| Whole simulation on the host via a `custom_vjp` and a jaxpr interpreter | **No direct literature.** The `host_jaxpr_interpreter` probe returned 5 papers total. This is an engineering choice, not a published method. |

---

## 4. New candidate optimisations (not yet implemented)

The final solution runs the official benchmark in about 4 s first run and 0.41 s warm (`results/combined-final.json`) `[R]`. The solve is no longer the dominant cost, so most items below have a **small** ceiling on the benchmark metric. Expected gains are deliberately not quantified: they have not been measured.

### V-A — Reuse a stale LU across neighbouring frequencies: INVESTIGATE
- **Idea:** factor every k-th frequency exactly and refine the others with that LU as a preconditioner, instead of factoring all 101.
- **Support:** *Sparse Generalized-Admittance AC Power Flow* `2609.14132` reuses the initial sparse LU factorizations across iterations `[A]`. *Recycling BiCGSTAB for parametric MOR* `1406.2831` `[A]`. Krylov recycling `[K]`.
- **Fit:** the host path already runs a refinement step per solve `[R]`, so the infrastructure exists.
- **Risk:** high. Voyager-type systems reach cond(A) ≈ 1e31 `[R]`, where a stale LU may not converge. It would need a residual check with automatic refactor fallback.
- **Ceiling:** limited to the factorization share of the 0.41 s warm run.

### V-B — Exact Schur / Kron elimination of pass-through unknowns: INVESTIGATE
- **Idea:** eliminate unknowns with unit diagonal and no mutual couplings before factorization. Kron reduction is the Schur complement of a subset of nodes `[A]` (`1102.2950`, ~925 citations `[S]`).
- **Prior measurement `[R]`:** on the public setup, 345 of 751 unknowns can be eliminated exactly, leaving a 406×406 system (~16% of the factorization work). In the in-graph JAX path the reverse pass got ~3× slower and the benchmark total got worse (`docs/notes/approach.md`, `step3-schur-complement.patch`).
- **Why retry:** that penalty came from gathers and scatters inside XLA. In the NumPy host path they are cheap, so the trade-off may differ. **Untested.**

### V-C — Certified adaptive frequency sweep (Loewner / reduced basis): DO NOT SHIP for the challenge
- **Idea:** solve ~10–20 greedily chosen frequencies and reconstruct the rest by rational interpolation with an error indicator.
- **Support:** certified greedy Loewner `2303.01015`; reduced-basis a-posteriori error estimation for microwave sweeps `2110.05925`; adaptive Loewner-state sweep for EM `2504.09942`; vibro-acoustic interpolation `2201.06518` `[A]`.
- **Problem:** the challenge requires all 101 frequencies and exact physics, and a log-sensitivity error ≤ 1e-5. An interpolated response is an approximation, and gradients would need their own treatment. It is a legitimate technique for the *design loop* (many evaluations), not for this benchmark.

### V-D — Augmented partial factorization for multi-input/output systems: HOLD
- **Idea:** augment the matrix with all source and probe profiles, then one partial factorization yields the full scattering matrix via a Schur complement. `2205.07887` and `2306.09257` report that forward-plus-adjoint for M input channels can be done in one simulation `[A]`.
- **Fit:** the benchmark has one detector and one drive, so there is little to gain. It matters for multi-detector or multi-port designs.

### Rejected for this problem

| Idea | Source `[A]` | Why rejected |
|---|---|---|
| Selected inversion | PSelInv `1404.0447`, `1708.04539`; Serinv `2503.17528` | The gradient needs only `λ_r x_c` on the pattern, with no inverse entries. |
| GPU sparse LU | GLU3.0 `1908.00204`; cuDSS in SABLE | The benchmark is 1 logical CPU. |
| Hierarchical / HSS direct solvers | `1403.5337`, `1110.3105`, `1502.07405` | Built for large 2D/3D PDE systems. Here nnz ≈ 2.5k, so overhead dominates. |
| Mixed-precision LU with refinement | `2007.06614`, `2011.01850` | Voyager cond(A) ≈ 1e31 `[R]`; lower precision would lose the answer. |
| Neural surrogates | `2511.19364`, `2209.10098`, `2411.03527` | Violates the exactness and gradient-correctness bar. Useful only as a pre-screen. |

### Cold start and compile cost
The harvest found essentially **no relevant literature** for this (`compile_cold_start` has 41 papers, none specific to the problem). Remaining first-run cost is tracing (~0.14 s in the earlier log), imports and process start `[R]`. Any gain there is plain engineering, for example precomputed derivatives instead of tracing, and is not supported by this review.

---

## 5. Limits of this review

- Abstracts only; no full-text reading.
- arXiv only. Much circuit-simulation, power-systems and HPC-solver work is published elsewhere (journals, SIAM, IEEE) and is invisible here.
- Relevance-ranked top-100 per probe over 15 years, so older work is under-sampled.
- Keyword tagging: 21.5% untagged, and tag counts include false positives.
- Citation counts come from Semantic Scholar and are unreliable for 2026 papers.
- No candidate in section 4 has been implemented or measured.
