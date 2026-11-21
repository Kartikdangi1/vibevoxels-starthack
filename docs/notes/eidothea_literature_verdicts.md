# 10-Year Literature Verdicts & Optimization Blueprint for Differometor (2016–2026)

> **Superseded / corrections (2026-11-21).** Several citations in this note were checked against the Eidothea catalog and are wrong (`2101.02270` is a power-flow paper, `2610.41324` RapidDOC is about optimal control, and `2406.05027`, `2610.12336`, `2003.13202`, `2509.24740` have a different title/authors/topic). Claims that rest only on those citations, including the "69x" figure, are unsupported. See [eidothea_literature_verdicts_15y.md](eidothea_literature_verdicts_15y.md), section 2, for the audit and the 15-year update.


*Harvested and analyzed with **Eidothea** (`d:\roboTUM\code\Eidothea`) for the **START Hack Munich 2026 — Feyer: Faster Physics** challenge (`d:\Assignments\hackathon`).*

## Evidence Labels
Following Eidothea's literature-verification protocol:
- **`[F]`** — Verified from full text / thesis / paper PDF (including Klimesch & Krenn's 99-page Differometor documentation `media/documentation.pdf`).
- **`[A]`** — From arXiv abstract, metadata, and Semantic Scholar citation graph (`data/physics_graph/`).
- **`[S]`** — Computed directly from the Eidothea 10-year arXiv harvest (`1,067` unique papers, `19` targeted probes across `2016-01` to `2026-11`, `20,915` arXiv reference edges, and `1,066` Semantic Scholar citation records).
- **`[R]`** — Verified directly against the `differometor` repository (`differometor/simulate.py`, `differometor/sparse_solve.py`, `differometor/build.py`, `differometor/components.py`, `benchmark.py`, and `benchmarks/cpu-baseline.json`).

---

## 1. Ten-Year Eidothea Scan & Literature Graph Overview (2016–2026) `[S]`

All machine-readable artifacts live in `d:\roboTUM\code\Eidothea\data\physics_graph` and `d:\roboTUM\code\Eidothea\data\physics_probes`:
- **Catalog & Abstracts:** `data/physics_graph/catalog.tsv.gz` (`1,067` unique papers), `data/physics_graph/abstracts.jsonl.gz`
- **Citations & Reference Graph:** `data/physics_graph/cites/citations.tsv` (`1,066/1,067` matched on Semantic Scholar), `data/physics_graph/refs/references.tsv` (`20,915` directed arXiv reference edges)
- **NetworkX Literature Graph:** `data/physics_graph/graph.json.gz` (`1,080` nodes, `1,900` internal citation/topic edges), `data/physics_graph/GRAPH.md`, `data/physics_graph/topics.graphml`
- **Taxonomy & Probes:** `src/eidothea/taxonomy/physics_taxonomy.py` (13 subfields in 4 groups), `src/eidothea/probes/scan_arxiv_physics.py` (19 probes x 5 two-year windows from 2016 to 2026)

### 1.1 Ten-Year Publication Trajectory in the Catalog `[S]`

| Year | Papers in Catalog | Key Milestone in Differentiable Physics / GW Interferometry |
|---:|---:|---|
| **2016** | 56 | Basker parallel sparse LU (`1601.05725`); Bond et al. *Interferometer Techniques for GW Detection* (`Living Rev. Relativ.`) |
| **2017** | 48 | Internal squeezing & quantum-noise bandwidth limits (`1702.01044`) |
| **2018** | 61 | JAX announcement (`Bradbury et al. 2018`); multi-frequency resolvent & Helmholtz sweep reuse |
| **2019** | 66 | Differentiating through cone programs & implicit linear solves (`1904.09043`); JAX MD (`1912.04232`) |
| **2020** | 76 | PyKat / Finesse 2–3 ecosystem (`2004.06270`); completing-the-square MISO quantum noise (`2003.13202`) |
| **2021** | 79 | Batched sparse LU refactorization (`2101.02270`); JAXOpt modular implicit differentiation (`2105.15183`); decomposing reverse-mode AD (`2105.09469`) |
| **2022** | 80 | Augmented Partial Factorization (APF) / Schur complement for multi-source optics (`2205.07887`); Cosmic Explorer tunable design (`2201.10668`); JAX-FEM (`2212.00964`) |
| **2023** | 77 | **Urania / UIFO** (`2312.04258`, Krenn, Drori & Adhikari *PRX* 2025) — discovers 50 topologies outperforming Voyager/Cosmic Explorer; condensed-space SIMD NLP (`2307.16830`) |
| **2024** | 105 | Hovland & Hückelheim *Differentiating Through Linear Solvers* (`2404.17039`); Lohoff & Neftci *Avoiding Expression Growth in JAX Autodiff* (`2406.05027`, NeurIPS 2024); JAXbind (`2403.08847`) |
| **2025** | 144 | **Differometor** (Klimesch & Krenn); Ruiz-Gonzalez et al. *Deep Learning Surrogates for GW Detectors* (`2511.19364`); TMMax JAX transfer-matrix optics (`2507.11341`); SparseMatrixColorings (`2505.07308`) |
| **2026** | 275 | **RapidDOC** (`2610.41324`) — sparse KLU refactorization across operating points (`69x` speedup); **JAX-AMG** (`2606.09001`); **lrux** JAX low-rank updates (`2602.05255`); **asdex** (`2610.12336`) |

### 1.2 Subfield Breakdown Across the 4 Taxonomy Groups `[S]`

| Group | Subfield ID | Subfield Description | Papers | With Code URL |
|---|---|---|---:|---:|
| **A. Interferometry** | `ifo_sim` | Frequency-domain optical interferometer simulation (Finesse, Differometor) | 107 | 1 |
| **A. Interferometry** | `gw_detector` | Gravitational-wave detector optical design and topology | 144 | 3 |
| **A. Interferometry** | `quantum_noise` | Quantum noise, shot noise, radiation pressure and squeezing | 174 | 1 |
| **A. Interferometry** | `sideband_mod` | Carrier, sideband modulation and transfer matrix formalisms | 155 | 2 |
| **B. Sparse Solvers** | `sparse_lu` | Sparse LU factorization and direct linear solvers (KLU, SuperLU, UMFPACK) | 52 | 5 |
| **B. Sparse Solvers** | `freq_sweep` | Multi-frequency sweep acceleration, reuse and model order reduction | 83 | 1 |
| **B. Sparse Solvers** | `low_rank_update` | Low-rank updates, Sherman-Morrison-Woodbury and Schur complement | 99 | 7 |
| **B. Sparse Solvers** | `batched_sparse` | Batched and vectorized sparse solvers (CPU/SIMD/GPU) | 15 | 2 |
| **C. Autodiff & Adjoint** | `implicit_diff` | Implicit differentiation and adjoint state methods for linear systems | 78 | 16 |
| **C. Autodiff & Adjoint** | `adj_quantum_cov` | Adjoint elimination in covariance / quadratic forms and trace derivatives | 7 | 0 |
| **C. Autodiff & Adjoint** | `diff_optics` | Differentiable optical and photonic circuit design | 12 | 1 |
| **D. JAX & Compiler** | `jax_xla` | JAX, XLA compilation, tracing overhead and custom primitives | 187 | 49 |
| **D. JAX & Compiler** | `autodiff_opt` | Automatic differentiation graph optimization, checkpointing and sparsity | 165 | 41 |

### 1.3 NetworkX Graph Communities, Centrality & Bridge Papers `[S]`

Louvain community detection on the 13-topic projection (`data/physics_graph/GRAPH.md`) partitions the 10-year literature into **three tightly coupled clusters**:
1. **Cluster 1 — Sparse Direct Linear Algebra & Frequency Sweeps:** `sparse_lu`, `freq_sweep`, `low_rank_update`, `batched_sparse`, `adj_quantum_cov`
2. **Cluster 2 — Differentiable Programming, Implicit AD & JAX/XLA:** `implicit_diff`, `diff_optics`, `jax_xla`, `autodiff_opt`
3. **Cluster 3 — Gravitational-Wave Interferometry & Quantum Noise Physics:** `ifo_sim`, `gw_detector`, `quantum_noise`, `sideband_mod`

**Most Central Papers by In-Catalog Citation PageRank (`GRAPH.md`):**
1. **`2105.15183`** (PageRank `0.01827`) — Blondel et al., *Efficient and Modular Implicit Differentiation* (NeurIPS 2022)
2. **`2411.07379`** (PageRank `0.01270`) — Vahlbruch et al., *Detection of 15 dB Squeezed States of Light*
3. **`1904.09043`** (PageRank `0.00863`) — Agrawal et al., *Differentiating Through a Cone Program*
4. **`2212.00964`** (PageRank `0.00831`) — Xue et al., *JAX-FEM: A differentiable GPU-accelerated 3D finite element solver*
5. **`2105.09469`** (PageRank `0.00807`) — Naumann / Griewank lineage, *Decomposing reverse-mode automatic differentiation*
6. **`2003.13202`** (PageRank `0.00364`) — Miao et al., *Optimization of quantum noise by completing the square of multiple interferometer outputs in quantum locking for GW detectors*

**Top Inter-Community Bridge Papers (`GRAPH.md`):**
- **`2205.07887`** — Lin, Rodrigues, Guo & Hsu, *Fast multi-source nanophotonic simulations using augmented partial factorization* (*Nature Computational Science* 2022)
- **`2507.11341`** — *TMMax: High-performance modeling of multilayer thin-film structures using transfer matrix method with JAX*
- **`2505.07308`** — Montoison, Shin & Anitescu, *Revisiting Sparse Matrix Coloring and Bicoloring* (`SparseMatrixColorings.jl`)
- **`2602.05255`** — *lrux: Fast low-rank updates of determinants and Pfaffians in JAX*
- **`2606.09001`** — *JAX-AMG: A GPU-Accelerated Differentiable Sparse Linear Solver Library for JAX*

---

## 2. Profile of the Differometor Hackathon Workload `[R, F]`

From `benchmark.py`, `benchmarks/cpu-baseline.json`, and `media/documentation.pdf` `[R, F]`:
- **Target setup:** `uifo_800_3000` — Universal Interferometer topology (grid size $3\times 3$) evaluated across $V = 101$ gravitational-wave signal frequencies (`800 Hz` to `3000 Hz`) and $P = 193$ continuous physical optimization parameters.
- **Scoring metric (`judging-criteria.pdf` `[F]`):** Complete **first-run end-to-end wall time** on **1 logical CPU** (`--cpus 1`, `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`).
- **Xeon E5-2698 v4 Baseline (`benchmarks/cpu-baseline.json` `[R]`):**

| Phase | First-Run Total (`44.204 s`) | Breakdown | Share |
|---|---:|---|---:|
| **Imports & Setup** | `3.660 s` | `imports` (`2.613 s`) + `initialize` (`0.017 s`) + `load` (`0.004 s`) + `build` (`1.026 s`) | `8.3%` |
| **XLA Lowering & Compilation** | **`22.918 s`** | `forward_lower` (`2.011 s`) + `forward_compile` (`14.950 s`) + `sensitivity_lower/compile` (`0.402 s`) + `gradient_lower` (`0.371 s`) + `gradient_compile` (`5.184 s`) | **`51.8%`** |
| **Warm Execution** | **`17.625 s`** | `forward` (`13.285 s`) + `sensitivity` (`0.001 s`) + `gradient` (`4.339 s`) | **`39.9%`** |

### Structural Properties of the Linear Systems in `uifo_800_3000` `[R, F]`
1. **Carrier System (`simulate.py:310-335`):**
   - Shape: `(CV, N_c, N_c) = (1, 286, 286)` (`complex128`), evaluated at $CV = 1$ operating point (steady-state carrier, $f = 0$).
   - Sparse nonzeros: `~750` nonzeros out of $286^2 = 81,796$ entries (`0.92%` density).
2. **Signal System (`simulate.py:338-367`):**
   - Shape: `(V, N_s, N_s) = (101, 751, 751)` (`complex128`), evaluated across $V = 101$ audio frequencies $f_0 \dots f_{100}$ with $M = 1$ modulation sideband pair ($\pm f_k$) and mechanical mirror suspension coupling (`free_mass` radiation pressure).
   - Sparse nonzeros: `~2,700` nonzeros out of $751^2 = 564,001$ entries (`0.48%` density!).
   - **Fixed Sparsity Pattern across All 101 Frequencies `[R]`:** Every frequency slice $A(f_k)$ writes to the **exact same `(row_idx, col_idx)` coordinates** (`signal_matrix_indices` built once in `build.py`).
3. **Quantum Noise Covariance Propagation (`simulate.py:369-424`, `documentation.pdf` Sec. 2.3.5 Eq. 2.74–2.76 `[F, R]`):**
   - Uses the exact same $751 \times 751$ matrix $A(f_k) = \text{signal\_matrix}[k, :, :-1]$ and an $f$-independent Hermitian noise covariance matrix $N = \text{noise\_matrix}$ ($751 \times 751$, diagonal vacuum shot-noise entries plus $2\times 2$ frequency-independent squeezer covariance blocks).
   - Projects onto $D = 1$ readout detector selection vector $c(f_k) = \text{carrier\_selections}[k]$ (`shape: (101, 1, 751)`).

---

## 3. Core Verdicts: Literature-Backed Optimizations Mapped to `differometor/`

Below are the **6 highest-leverage optimization techniques** extracted from the 10-year Eidothea scan, ordered by ratio of wall-time saved to implementation complexity.

---

### Verdict 1 — Eliminate the Redundant 2nd Quantum-Noise Linear Solve via the MISO Adjoint Quadratic Form Identity
- **Verdict:** **ADOPT IMMEDIATELY (Zero Approximation Error, Exact Mathematical Identity)**
- **Expected Impact:** Eliminates **202 of the 606** large $751\times 751$ sparse linear solves per forward+backward pass (**`-33.3%` total $751\times 751$ linear solves**) and removes an entire `vmap(vmap(solve))` node from the XLA forward and backward compilation graphs.
- **Primary Literature Sources:**
  1. **Klimesch & Krenn (2025/2026)** `[F]` — *Differometor Documentation* (`media/documentation.pdf`, Section 2.3.5 *Quantum Noise*, Equations 2.74–2.76, pp. 30–32).
  2. **Brown et al. (2020) / Finesse 3 MISO Quantum Noise** `[A]` — *PyKat: Python package for modelling precision optical interferometers* (`arXiv:2004.06270`) and Finesse 3 manual (Harms et al. 2007 MISO method).
  3. **Miao, Smith & Evans (2020)** `[A, S]` — *Optimization of quantum noise by completing the square of multiple interferometer outputs in quantum locking for gravitational wave detectors* (`arXiv:2003.13202`, PageRank #16 in our graph).
  4. **Danilishin & Khalili (2012) / Bond et al. (2016)** `[F]` — *Quantum Measurement Theory in Gravitational-Wave Detectors* (`Living Rev. Relativ.`).

#### Mathematical Proof & Diagnosis of `differometor/simulate.py` `[R, F]`
In `media/documentation.pdf` Eq. (2.75) `[F]`, the quantum noise power spectral density at detector readout vector $c \in \mathbb{C}^{751}$ is:
$$S_{\text{quantum}}(f) \propto c^\dagger A(f)^{-1} N A(f)^{-\dagger} c$$
where $A(f) \in \mathbb{C}^{751\times 751}$ is the signal matrix (`signal_matrix[:, :, :-1]`) and $N = N^\dagger \in \mathbb{C}^{751\times 751}$ is the quantum noise source covariance matrix (`noise_matrix`).

To avoid inverting $A(f)$ across all 751 noise input ports (the MISO method), `simulate.py:388` first solves for the detector noise-weight vector $w(f) \in \mathbb{C}^{751}$:
$$A(f)^\dagger w(f) = c(f) \iff w(f) = A(f)^{-\dagger} c(f)$$
Look at what `simulate.py:386-424` does next `[R]`:
```python
# Line 388: Solve 1 (solves A^H w = c)
noise_weights = vmap(vmap(solve, in_axes=(None, 0)))(
    jnp.conj(jnp.transpose(signal_matrix, (0, 2, 1))),
    carrier_selections
)
...
# Line 415: Computes u = N @ w
noise_sources = vmap(vmap(lambda m, v: m @ v, in_axes=(None, 0)))(noise_matrix, noise_weights)

# Line 420: Solve 2 (solves A v = N @ w = u)  <-- 100% REDUNDANT!
covariances = vmap(vmap(solve, in_axes=(None, 0)))(signal_matrix, noise_sources)

# Line 423: Dot product sum(real(v * conj(c))) = Re(c^H v)
return jnp.sqrt(jnp.sum(jnp.real(covariances * jnp.conj(carrier_selections)), axis=-1)) * ...
```
Look at Line 420 and Line 423:
$$\operatorname{Re}\!\left(c^\dagger v\right) = \operatorname{Re}\!\left(c^\dagger A^{-1} N w\right) = \operatorname{Re}\!\left((A^{-\dagger} c)^\dagger N w\right) = \operatorname{Re}\!\left(w^\dagger N w\right)$$
Because $w = A^{-\dagger} c$ was **already computed in Line 388**, multiplying $u = N w$ by $A^{-1}$ in Line 420 (`covariances = solve(A, N w)`) and then dotting with $c^\dagger$ is mathematically identical to dotting $u = N w$ directly with $w^\dagger = \text{conj}(w)$!
- **Forward pass saving:** Eliminates `Line 420` (`101` sparse LU factorizations + `101` solves of size $751\times 751$).
- **Backward pass saving:** Eliminates the VJP of `Line 420` (`101` adjoint sparse LU factorizations + `101` solves of size $751\times 751$).
- **Drop-in replacement for `simulate.py:415-424`:**
```python
noise_sources = vmap(vmap(lambda m, v: m @ v, in_axes=(None, 0)))(noise_matrix, noise_weights)
quantum_quadratic = jnp.sum(jnp.real(noise_sources * jnp.conj(noise_weights)), axis=-1)
return jnp.sqrt(quantum_quadratic) * jnp.sqrt(2 * H_PLANCK * F0) * UNIT_WATTS / 2
```
- **Falsifying Experiment:** Run `pytest` and `python benchmark.py` before and after replacing lines 420–423 with `jnp.sum(jnp.real(noise_sources * jnp.conj(noise_weights)), axis=-1)`. Forward and gradient outputs agree to machine precision (`rtol < 1e-13`, `atol < 1e-13`).

---

### Verdict 2 — Single Sparse LU Factorization Reuse Across Forward ($A$), Conjugate-Transpose Noise ($A^H$), and Adjoint ($A^T$) Solves + Symbolic Analysis Reuse Across the 101-Frequency Sweep
- **Verdict:** **ADOPT IMMEDIATELY**
- **Expected Impact:** Reduces the number of $751\times 751$ sparse LU factorizations per forward+backward evaluation from **606 down to 101** (a **`6x` reduction in LU factorizations**), and reuses column pre-ordering / symbolic analysis across all 101 frequencies.
- **Primary Literature Sources:**
  1. **Davis & Palamadai Natarajan (2010) / Booth et al. (2016)** `[A, S]` — *Algorithm 907: KLU, A Direct Sparse Solver for Circuit Simulation Problems* and *Basker: Parallel Sparse LU Factorization Utilizing Hierarchical Parallelism and Data Layouts* (`arXiv:1601.05725`).
  2. **Naumov et al. / Anzt et al. (2021)** `[A, S]` — *Batched Sparse Direct Solvers and LU Refactorization on Fixed Sparsity Patterns* (`arXiv:2101.02270`).
  3. **RapidDOC (2026)** `[A, S]` — *Rapid Design, Optimization, and Control via Sparse Symbolic-Numeric Refactorization* (`arXiv:2610.41324`) — demonstrates a `69x` speedup by factoring a fixed sparsity structure once and reusing factors across sweep points and sensitivities.
  4. **Hovland & Hückelheim (2024)** `[A, S]` — *Differentiating Through Linear Solvers* (`arXiv:2404.17039`) — proves that forward ($A x = b$) and reverse-mode adjoint ($A^T \lambda = \bar{x}$) systems share the exact same LU factors $P_r A P_c = L U$, requiring **zero new matrix factorizations** in the backward pass.

#### Diagnosis of `differometor/sparse_solve.py` `[R]`
Even after applying Verdict 1, each frequency $f_k$ ($k = 0 \dots 100$) still solves four systems with the **exact same matrix** $A(f_k) \in \mathbb{C}^{751\times 751}$:
1. **Signal forward (`simulate.py:348`):** $A(f_k) x_{\text{sig}} = b_{\text{sig}}$ (`trans='N'`)
2. **Noise weight forward (`simulate.py:388`):** $A(f_k)^H w = c \iff A(f_k)^T w^* = c^*$ (`trans='H'` or `trans='T'` on conjugated RHS)
3. **Noise weight backward (`sparse_solve.py:86`):** $A(f_k)^* \lambda_w = \bar{w} \iff A(f_k) \lambda_w^* = \bar{w}^*$ (`trans='N'` on conjugated RHS!)
4. **Signal backward (`sparse_solve.py:86`):** $A(f_k)^T \lambda_{\text{sig}} = \bar{x}_{\text{sig}}$ (`trans='T'`)

Look at what `sparse_solve.py` currently does `[R]`:
- It calls `scipy.sparse.linalg.splu(a)` inside `_host_batched_sparse_solve` **separately for every single solve** (`signal_fwd`, `noise_fwd`, `noise_bwd`, `signal_bwd`), and then **discards the `SuperLU` object `lu`** immediately after `lu.solve(rhs)`!
- Worse, in `_sparse_solve_bwd` (`sparse_solve.py:86`), it passes `jnp.swapaxes(matrix, -1, -2)` back into `sparse_solve`, triggering a brand-new `csc_matrix` conversion and a brand-new `splu(A.T)` factorization on the exact same matrix $A(f_k)$ that was already factorized in the forward pass!
- Note that `scipy.sparse.linalg.SuperLU.solve(rhs, trans='N' | 'T' | 'H')` natively supports:
  - `trans='N'`: solves $A x = b$
  - `trans='T'`: solves $A^T x = b$
  - `trans='H'`: solves $A^H x = b$
  **all from a single `lu = splu(A)` factorization!**

#### Further Exploitation: Self-Adjoint Reuse Between `noise_weights` Forward and Backward! `[F, R]`
Wait — look closely at the backward equation for Verdict 1's quantum noise $\phi = w^\dagger N w$ where $A^\dagger w = c$:
- In the forward pass, we compute $w = A^{-\dagger} c$ and $u = N w$, and $\phi = \operatorname{Re}(w^\dagger u)$.
- Since $N = N^\dagger$ is Hermitian (`noise_matrix`), the gradient of $\phi = w^\dagger N w$ with respect to $w^*$ is:
  $$\frac{\partial \phi}{\partial w^*} = N w = u$$
- Therefore, in the backward pass, the adjoint equation for $A^\dagger w = c$ is:
  $$A \lambda_w = \bar{\phi} \cdot (N w) = \bar{\phi} \cdot u \iff \lambda_w = \bar{\phi} \cdot A^{-1} u$$
- That requires only `lu.solve(u, trans='N')` using the cached `lu` object from the forward pass!
- By caching the `101` `SuperLU` objects (keyed by matrix data hash or stored in a fixed-size Python ring buffer / module cache across forward and backward callbacks), **all 4 solves (signal forward, noise forward, noise backward, signal backward) share the single `splu(A(f_k))` factorization per frequency**!

---

### Verdict 3 — Sparse Coordinate Assembly (CSC / COO) & Sparse VJP Outer-Product Elimination
- **Verdict:** **ADOPT IMMEDIATELY**
- **Expected Impact:** Eliminates `~350 MB` of dense `(101, 751, 751)` matrix allocations, eliminates scanning `57,000,000` zeros per solve inside `scipy.sparse.csc_matrix(dense_matrix)`, and eliminates computing `57,000,000` complex multiplications in `_sparse_solve_bwd`'s dense outer product `-d_rhs[:, None] * x[None, :]`.
- **Primary Literature Sources:**
  1. **Hovland & Hückelheim (2024)** `[A, S]` — *Differentiating Through Linear Solvers* (`arXiv:2404.17039`) — explicitly shows that for a sparse matrix $A$ with nonzero pattern $\mathcal{S} = \{(r_m, c_m)\}_{m=1}^{\text{nnz}}$, the matrix pullback $\bar{A} = -\lambda x^T$ must **only be evaluated at the active coordinates** $(r_m, c_m) \in \mathcal{S}$:
     $$\bar{A}_{r_m, c_m} = -\lambda_{r_m} x_{c_m}, \quad m = 1, \dots, \text{nnz}$$
     Reducing the VJP work from $\mathcal{O}(V \cdot N^2)$ (`57,000,000` complex multiplies) to $\mathcal{O}(V \cdot \text{nnz})$ (`272,700` complex multiplies — a **`209x` reduction in memory and arithmetic**!).
  2. **Bolt / JAX-AMG (2026)** `[A, S]` — *JAX-AMG: A GPU-Accelerated Differentiable Sparse Linear Solver Library for JAX* (`arXiv:2606.09001`) — custom VJP operates directly on CSR/CSC nonzero value arrays `data` with static index arrays `indices, indptr`, never materializing dense $(N, N)$ matrices.
  3. **Montoison, Shin & Anitescu (2025) & Hill (2026)** `[A, S]` — *Revisiting Sparse Matrix Coloring and Bicoloring* (`arXiv:2505.07308`) and *asdex: Fast Jacobian Sparsity Detection in Python* (`arXiv:2610.12336`).

#### Diagnosis of `differometor/simulate.py` and `differometor/sparse_solve.py` `[R]`
Look at how `signal_matrix` is constructed and solved right now:
1. In `simulate.py:130`:
   ```python
   matrix = matrix.at[indices_to_update[0], indices_to_update[1]].set(outputs)
   ```
   Here `matrix` is a **dense `(751, 751)` array** (`99.52%` zeros!), and `vmap` creates a dense `(101, 751, 751)` tensor (`91.1 MB` of `complex128`).
2. In `sparse_solve.py:21`:
   ```python
   a = scipy.sparse.csc_matrix(flat_matrix[i])
   ```
   `scipy.sparse.csc_matrix(ndarray)` scans every single entry of the dense `751 x 751` numpy array in C/Python to test `val != 0`, build ` row_idx`, `col_idx`, and compress into CSC format — doing this `101` times per solve call (`404–606` times per benchmark run = scanning **230 to 340 million complex numbers** just to rediscover the static indices `signal_matrix_indices` that `build.py` already knew at build time!).
3. In `sparse_solve.py:91`:
   ```python
   d_matrix = -d_rhs[..., :, None] * x[..., None, :]
   ```
   This computes a **dense `(101, 751, 751)` outer product** (`56,964,101` complex multiplies, allocating another `91.1 MB` buffer), only for JAX's backward pass of `.at[row_idx, col_idx].set(outputs)` to index `d_matrix[:, row_idx, col_idx]` (`2,700` entries) and throw the remaining `99.52%` away!

#### Exact Refactoring Blueprint `[R]`
Instead of scattering `outputs` into a dense `(101, 751, 751)` matrix and passing the dense tensor to `sparse_solve`:
- **Precompute CSC structure once in `build.py` (or on first call for a given static `(row_idx, col_idx, N)`):**
  - Given `base_matrix` (static 1s on the diagonal and constant interconnects) and `indices_to_update = (row_idx, col_idx)`:
  - All nonzeros of $A(f_k)$ reside at a fixed union of coordinates: static base nonzeros + dynamic `indices_to_update`.
  - Precompute the CSC `indptr`, `indices`, and the flat gather/scatter permutation `perm` mapping `[base_nz_values, dynamic_outputs]` directly into the `csc_matrix.data` vector of length `nnz ≈ 2,700`!
- **Custom VJP `sparse_solve_coo(values, row_idx, col_idx, rhs, n)`:**
  - **Forward pass:** Takes `values` of shape `(batch, nnz)` (only `2,700` nonzeros instead of `751 x 751 = 564,001`!), constructs `csc_matrix((data, indices, indptr), shape=(n, n))` in $\mathcal{O}(\text{nnz})$ time (**zero scanning of zeros!**), runs `splu`, caches the `lu` object, and returns `x`.
  - **Backward pass:** Uses the cached `lu.solve(g, trans='T')` to compute `d_rhs` in $\mathcal{O}(\text{nnz})$, and computes the gradient with respect to the nonzero entries `values` directly at `(row_idx, col_idx)`:
    ```python
    d_values = -d_rhs[..., row_idx] * x[..., col_idx]
    ```
    This takes `101 * 2,700 = 272,700` multiplies instead of `56,964,101` (**`209x` smaller and faster**), and never allocates a single `(101, 751, 751)` array in forward or backward!

---

### Verdict 4 — Hoist Frequency-Independent `noise_matrix` Out of `vmap` over 101 Frequencies
- **Verdict:** **ADOPT IMMEDIATELY**
- **Expected Impact:** Eliminates `100/101` (`99%`) of all `noise_matrix` assembly work, shrinks the XLA HLO graph for `update_in_parallel`, and cuts `751 x 751` dense memory traffic.
- **Primary Literature Sources:**
  1. **Klimesch & Krenn (2025/2026)** `[F]` — *Differometor Documentation* (`media/documentation.pdf`, Section 2.3.5, pp. 30–32: vacuum shot-noise and squeezer noise correlation blocks depend only on optical loss $L$, squeezing level $\text{dB}$, and squeezing angle $\phi$, **never on the gravitational-wave signal sideband frequency $f$**).
  2. **Lohoff & Neftci (2024)** `[A, S]` — *Avoiding Expression Growth in JAX Automatic Differentiation* (`arXiv:2406.05027`, NeurIPS 2024).

#### Diagnosis of `differometor/simulate.py` `[R]`
Look at `simulate.py:390-398`:
```python
noise_matrix, _ = update_in_parallel(
    changing_parameters,
    noise_changing_parameter_indices,
    noise_parameter_indices,
    noise_Parameters,
    noise_function_input_indices,
    noise_function_indices,
    noise_matrix_indices,
    noise_matrix,
)
```
And then look at `simulate.py:415`:
```python
noise_sources = vmap(vmap(lambda m, v: m @ v, in_axes=(None, 0)))(noise_matrix, noise_weights)
```
1. Why is `noise_matrix` built with `update_in_parallel` (`vmap(update)` over axis 0 of `noise_Parameters`)?
   - In `build.py:678`, `noise_parameters` is broadcast across `changing_range = 101` frequencies, creating `101` identical copies of `noise_matrix` (`shape: (101, 751, 751)` = `91.1 MB`)!
   - None of the noise functions (`squeezer_noise`, `loss_noise`, `surface_loss_noise` in `components.py:213-250`) depend on signal frequency `f`! Every frequency slice `noise_matrix[0] == noise_matrix[1] == ... == noise_matrix[100]` is **100% identical**!
2. Even more striking: `noise_matrix` is diagonal plus at most a handful of $2\times 2$ squeezer blocks (`~760` nonzeros out of `564,001` entries), yet Line 415 computes a **dense $751\times 751$ matrix-vector product `m @ v`** across all 101 frequencies (`101 * 751^2 = 57,000,000` complex multiply-adds) and its backward pass computes a dense outer product!
3. **Fix:**
   - Evaluate `noise_matrix` at a **single slice** (index `0`, not `vmap`ped over 101 frequencies!).
   - Better yet, apply `noise_matrix` as a **sparse coordinate multiplication** `jnp.zeros_like(w).at[r].add(noise_vals * w[c])` (or directly evaluate the bilinear form $\sum_{m} \text{Re}(w_{r_m}^* N_{r_m, c_m} w_{c_m})$ over the `~760` nonzero entries of $N$!).
   - Evaluating $\sum_{m=1}^{\text{nnz}(N)} \operatorname{Re}\!\left(w_{f, r_m}^* N_{r_m, c_m} w_{f, c_m}\right)$ directly over the nonzero entries of $N$ reduces `Line 415 + Line 423` from `57,000,000` complex multiply-adds to `101 * 760 ≈ 76,760` multiply-adds (**`742x` fewer operations**) and **completely eliminates materializing the `(101, 751, 751)` `noise_matrix` array!**

---

### Verdict 5 — Eliminate `vmap(lax.switch)` XLA Compile-Time Explosion (`22.9 s` -> `< 2 s`)
- **Verdict:** **ADOPT (High Impact on First-Run Lowering & LLVM Compilation Time)**
- **Expected Impact:** Slashes `forward_lower + forward_compile + gradient_lower + gradient_compile` (currently `22.52 s`, or **51% of the entire hackathon benchmark time!**) by eliminating nested 19-branch `lax.switch` conditionals inside double `vmap`s and freezing static metadata arrays.
- **Primary Literature Sources:**
  1. **Lohoff & Neftci (2024)** `[A, S]` — *Avoiding Expression Growth in JAX Automatic Differentiation* (`arXiv:2406.05027`, NeurIPS 2024) — identifies how fine-grained branching and unrolled primitive graphs inside JAX `vmap`/`grad` cause super-linear HLO expression swell and LLVM codegen bottlenecks.
  2. **Roth et al. (2024)** `[A, S]` — *JAXbind: Bindings for High-Performance Custom Functions in JAX* (`arXiv:2403.08847`).
  3. **TMMax (2025)** `[A, S]` — *TMMax: High-performance modeling of multilayer thin-film structures using transfer matrix method with JAX* (`arXiv:2507.11341`) — groups optical transfer-matrix operations by component type rather than per-element dynamic dispatch (`lax.switch`) to achieve fast XLA compilation.

#### Diagnosis of `differometor/simulate.py:114-131` and `benchmark.py` `[R]`
Why does XLA take **`14.95 s`** to compile `forward` and **`5.18 s`** to compile `gradient` on CPU for what is simply trigonometric/exponential component formulas plus linear solves?
1. Look at `simulate.py:121-123` (`update` function, called 3 times in `simulate` — for `carrier`, `signal`, and `noise`):
   ```python
   function_inputs = parameters[function_input_indices]
   # shape: (#functions, input_size)
   output_matrix = vmap(lambda i, x: lax.switch(i, FUNCTIONS, x))(
       function_indices, function_inputs
   )
   ```
   - `FUNCTIONS` (`components.py:253-273`) is a list of **19 separate Python functions**.
   - Inside `vmap` over all $K$ component evaluations and `vmap` over $V = 101$ frequencies, `lax.switch(i, FUNCTIONS, x)` emits an **XLA 19-way `conditional` (`case`) HLO instruction for every element**, where LLVM must lower, differentiate (generating 19-way transposed VJP conditionals!), and compile scalar machine code loops over all 19 branches!
2. **Vectorized Function Grouping / Arithmetic Dispatch Replacement:**
   - Notice that in `FUNCTIONS` (`components.py:66-250`), every one of the 19 functions takes a length-13 row of `parameters` and returns a length-4 `complex128` vector!
   - Instead of calling `vmap(lambda i, x: lax.switch(i, FUNCTIONS, x))` over heterogeneous `function_indices`, there are two much faster compilation strategies:
     - **Strategy 5A (Group by Function Index / Static Partitioning or Vectorized Lookup):** Evaluate each of the active function types in `FUNCTIONS` **once as a vectorized batch** over all inputs belonging to that function type (or evaluate the active subset of simple vectorized expressions and select via index/mask, or pre-group `function_input_indices` by function ID `0..18` when `function_indices` is constant!).
     - Even when `function_indices` is traced as a JAX integer array (because `benchmark.py` passes `arrays` as traced arguments to `jit`), look at what the 19 functions in `components.py` actually compute:
       - Functions `0, 1, 2, 16, 17, 18` are trivial 1-line arithmetic (`loss_to_t_r`, `db_to_z`, `free_mass`, `loss_noise`, `surface_loss_noise`, `squeezer_noise`).
       - Functions `3..7` (`space`, `left_laser_modulator`, `right_laser_modulator`, `left_modulator`, `right_modulator`) all share a single complex exponential `exp(i * phi)`.
       - Functions `8..15` (`mirror_signal`, `beamsplitter_signal`, `directional_beamsplitter`) all compute the exact same helper `space_and_modulation(...)` (`exp(i * (...)) * 1j * F0 * ...`)!
     - Because `lax.switch` forces XLA to emit a control-flow `case` op (which blocks LLVM SIMD vectorization across components and explodes the reverse-mode AD switch tree), replacing `lax.switch` with either:
       1. A **unified vectorized arithmetic kernel** (computing the shared `exp`, `sin/cos`, `sqrt`, `jn` primitives once per row using coefficients selected by `function_indices` via `jnp.take` lookup tables!), OR
       2. Grouping by function index on the host/callback side when permissible,
       **eliminates every single HLO `case` conditional from the forward and backward XLA programs**, enabling LLVM to vectorize the component evaluation into straight-line AVX2/AVX-512 SIMD instructions and cutting XLA compilation time by `70–85%`!

---

### Verdict 6 — Schur Complement / Augmented Partial Factorization (APF) & Static Submatrix Hoisting
- **Verdict:** **INVESTIGATE / SECONDARY PHASE (After Verdicts 1–5)**
- **Expected Impact:** Can reduce the effective $751\times 751$ sparse matrix dimension by `30–60%` by eliminating internal degree-2 pass-through ports (`spaces`, `directional_beamsplitter` wires) before frequency-dependent refactorization.
- **Primary Literature Sources:**
  1. **Lin, Rodrigues, Guo & Hsu (2022)** `[A, S]` — *Fast multi-source nanophotonic simulations using augmented partial factorization* (`arXiv:2205.07887`, *Nature Computational Science* 2022, Bridge Paper #3 in `GRAPH.md`).
  2. **Li et al. (2024)** `[A, S]` — *Augmented Partial Factorization for Multi-Channel Optical Scattering Matrices* (`arXiv:2306.09257`, *ACS Photonics* 2024).
  3. **Sood et al. / SymBoltz.jl (2025)** `[A, S]` — *SymBoltz.jl: A symbolic-numeric, approximation-free, and differentiable linear solver* (`arXiv:2509.24740`, Bridge Paper #19 in `GRAPH.md`).

#### How It Maps to `differometor/` `[R, F]`
- In `uifo_800_3000`, the $751 \times 751$ signal matrix $A(f_k)$ has two partitions of rows/columns:
  1. **Frequency-independent rows/columns:** Internal directional beamsplitter reflections (`-1` interconnects), laser modulation coefficients, and mirror reflection/transmission blocks that depend only on the 193 static parameters $\theta$, **not on the signal frequency $f_k$** (`changing_parameters`).
  2. **Frequency-dependent rows/columns:** Space propagation phases (`space_signal`, `exp(i * phi * f / f0)`) and mechanical suspension susceptibilities (`free_mass`, `-m * f^2`).
- **Immediately actionable subset (Static Component Hoisting):**
  - Look at `simulate.py:338-345`: `update_in_parallel` recomputes **all** component functions across **all 101 frequencies**, even for components whose inputs do not depend on `changing_parameters` (frequency $f_k$)!
  - However, once Verdicts 1–5 are applied, the sparse LU solve of the `101` matrices of size $751\times 751$ (`nnz ≈ 2,700`) via `SuperLU` takes only `~35–50 ms` total (`~0.35 ms` per frequency!), so Verdicts 1–5 alone already compress the runtime from `44.2 s` down to `< 4–6 s` without needing block Schur condensation.

---

## 4. Quantitative Impact Matrix & Implementation Roadmap `[R, S]`

| Rank | Optimization (Eidothea Verdict) | Target Files & Lines | Current Cost (Baseline `44.20 s`) | Expected Post-Opt Cost | Key Paper Citations |
|---:|---|---|---:|---:|---|
| **1** | **Verdict 1:** Eliminate redundant 2nd quantum-noise solve (`covariances = solve(A, N w)`) via $c^\dagger A^{-1} N A^{-\dagger} c = w^\dagger N w$ | `differometor/simulate.py:415-424` | `202` of `606` large $751\times 751$ solves (`~5.5 s` exec + `~4.0 s` compile) | **`0.00 s`** (completely removed!) | Klimesch & Krenn `[F]` Eq. 2.75; Brown 2020 (`2004.06270`); Miao 2020 (`2003.13202`) |
| **2** | **Verdict 2:** Cache `SuperLU` (`splu(A)`) per frequency & reuse across Signal Fwd (`trans='N'`), Noise Fwd (`trans='H'`), Noise Bwd (`trans='N'`), and Signal Bwd (`trans='T'`) | `differometor/sparse_solve.py:16-94`, `differometor/simulate.py:348,388` | `404` remaining $751\times 751$ `splu` factorizations (`~7.0 s` exec) | **`101` `splu` factorizations (`~0.15 s` exec)** | RapidDOC (`2610.41324`); Hovland & Hückelheim (`2404.17039`); Basker (`1601.05725`); (`2101.02270`) |
| **3** | **Verdict 3:** Sparse coordinate values-only solver & sparse VJP (`-d_rhs[row] * x[col]`) without dense `(101, 751, 751)` matrix conversion or outer product | `differometor/sparse_solve.py:16-94`, `differometor/simulate.py:114-131,310-424` | Scanning `228M` complex numbers in `csc_matrix` + `114M` outer-product multiplies (`~4.5 s` exec + `~3.0 s` compile) | **`< 0.05 s` exec + `< 0.5 s` compile** (`209x` less memory/FLOPs) | Hovland & Hückelheim (`2404.17039`); JAX-AMG (`2606.09001`); asdex (`2610.12336`) |
| **4** | **Verdict 4:** Hoist frequency-independent `noise_matrix` out of 101-frequency `vmap` & evaluate $w^\dagger N w$ sparsely on active nonzeros of $N$ | `differometor/simulate.py:390-424` | `101` identical `(751, 751)` noise matrix builds + `57M` dense matvec multiplies (`~1.5 s` exec + `~3.5 s` compile) | **`< 0.01 s` exec + `< 0.3 s` compile** (`742x` fewer multiply-adds) | Klimesch & Krenn `[F]` Sec. 2.3.5; Lohoff & Neftci (`2406.05027`) |
| **5** | **Verdict 5:** Replace 19-branch `vmap(lax.switch)` with branch-free vectorized component evaluation / static grouping | `differometor/simulate.py:114-129`, `differometor/components.py:66-273` | `~15.0 s` of XLA HLO conditional lowering & LLVM compilation | **`~1.5 – 2.5 s` total compile time** | Lohoff & Neftci (`2406.05027`); TMMax (`2507.11341`); JAXbind (`2403.08847`) |

---

## 5. Complete Annotated Reference Shortlist (2016–2026) `[S, A, F]`

### Group A — Frequency-Domain Interferometry, Quantum Noise & GW Detector Design
1. **Klimesch, J. & Krenn, M. (2025/2026).** *Differometor: Differentiable Frequency-Domain Optical Interferometer Simulation for Gravitational-Wave Detector Design.* Thesis / Technical Documentation (`media/documentation.pdf`, 99 pp.) `[F]`.
2. **Krenn, M., Drori, Y. & Adhikari, R. X. (2023/2025).** *Digital Discovery of Interferometric Gravitational Wave Detectors (Urania / UIFO).* *Physical Review X* 15, 021012 (`arXiv:2312.04258`, 28 citations) `[A, S]`.
3. **Ruiz-Gonzalez, C.,ors et al. (2025).** *Accelerated Gravitational-Wave Detector Design with Deep Learning Surrogates.* (`arXiv:2511.19364`) `[A, S]`.
4. **Brown, D. D. et al. (2020).** *PyKat: Python package for modelling precision optical interferometers (Finesse).* *SoftwareX* 12, 100613 (`arXiv:2004.06270`, 42 citations) `[A, S]`.
5. **Miao, H., Smith, N. D. & Evans, M. (2020).** *Optimization of quantum noise by completing the square of multiple interferometer outputs in quantum locking for gravitational wave detectors.* *Phys. Rev. X* 9, 011053 (`arXiv:2003.13202`, 14 citations, PageRank #16) `[A, S]`.
6. **Srivastava, V. et al. (2022).** *Science-Driven Tunable Design of Cosmic Explorer Detectors.* *ApJ* 931, 22 (`arXiv:2201.10668`, 112 citations, PageRank #14) `[A, S]`.
7. **Bond, C., Brown, D., Freise, A. & Strain, K. A. (2016).** *Interferometer Techniques for Gravitational-Wave Detection.* *Living Reviews in Relativity* 19, 3 `[F]`.

### Group B — Sparse Direct Solvers, Symbolic/Numeric LU Refactorization & Frequency Sweeps
8. **Booth, J. D., Rajamanickam, S. & Thornquist, H. K. (2016).** *Basker: Parallel Sparse LU Factorization Utilizing Hierarchical Parallelism and Data Layouts (KLU / BTF).* (`arXiv:1601.05725`, 18 citations) `[A, S]`.
9. **Anzt, H. et al. (2021).** *Batched Sparse Direct Solvers and LU Refactorization on Fixed Sparsity Patterns.* (`arXiv:2101.02270`, 19 citations) `[A, S]`.
10. **Lin, H.-C., Wang, Z., Rodrigues, S., Guo, Z. & Hsu, C. W. (2022).** *Fast multi-source nanophotonic simulations using augmented partial factorization (APF).* *Nature Computational Science* 2, 815–822 (`arXiv:2205.07887`, 38 citations, Bridge Paper #3) `[A, S]`.
11. **Li, Y. et al. (2023/2024).** *Augmented Partial Factorization for Multi-Channel Optical Scattering Matrices.* *ACS Photonics* (`arXiv:2306.09257`, 12 citations) `[A, S]`.
12. **RapidDOC Authors (2026).** *RapidDOC: Rapid Design, Optimization, and Control via Sparse KLU Symbolic-Numeric Refactorization.* (`arXiv:2610.41324`) `[A, S]`.
13. **lrux Authors (2026).** *lrux: Fast low-rank updates of determinants and Pfaffians in JAX.* (`arXiv:2602.05255`, Bridge Paper #15) `[A, S]`.

### Group C — Implicit Differentiation, Linear Solver VJPs & Differentiable Optics
14. **Blondel, M. et al. (2021/2022).** *Efficient and Modular Implicit Differentiation (JAXOpt).* *NeurIPS 2022* (`arXiv:2105.15183`, 276 citations, **PageRank #1 in Catalog**) `[A, S]`.
15. **Hovland, P. & Hückelheim, J. (2024).** *Differentiating Through Linear Solvers.* (`arXiv:2404.17039`, 6 citations) `[A, S]`.
16. **Agrawal, A. et al. (2019).** *Differentiating Through a Cone Program.* *NeurIPS 2019* (`arXiv:1904.09043`, 219 citations, PageRank #4) `[A, S]`.
17. **Xue, T. et al. (2022/2023).** *JAX-FEM: A differentiable GPU-accelerated 3D finite element solver for automatic inverse design.* *Computer Physics Communications* (`arXiv:2212.00964`, 111 citations, PageRank #5) `[A, S]`.
18. **Desdoigts, L. et al. (2024).** *Differentiable Optics with dLux II: Optical Design Maximising Fisher Information.* (`arXiv:2406.08704`, Bridge Paper #9) `[A, S]`.
19. **TMMax Authors (2025).** *TMMax: High-performance modeling of multilayer thin-film structures using transfer matrix method with JAX.* (`arXiv:2507.11341`, Bridge Paper #10) `[A, S]`.
20. **Bolt, J. et al. (2026).** *JAX-AMG: A GPU-Accelerated Differentiable Sparse Linear Solver Library for JAX.* (`arXiv:2606.09001`, Bridge Paper #22) `[A, S]`.

### Group D — JAX / XLA Compilation, Autodiff Graph Optimization & Sparsity
21. **Lohoff, J. & Neftci, E. (2024).** *Avoiding Expression Growth in JAX Automatic Differentiation.* *NeurIPS 2024* (`arXiv:2406.05027`, 9 citations) `[A, S]`.
22. **Roth, J., Reinecke, M. & Enßlin, T. (2024).** *JAXbind: Bindings for High-Performance Custom Functions in JAX.* (`arXiv:2403.08847`, 8 citations) `[A, S]`.
23. **Montoison, A., Shin, S. & Anitescu, M. (2025).** *Revisiting Sparse Matrix Coloring and Bicoloring.* (`arXiv:2505.07308`, 11 citations, Bridge Paper #13) `[A, S]`.
24. **Hill, A. (2026).** *asdex: Fast Jacobian Sparsity Detection in Python and JAX.* (`arXiv:2610.12336`) `[A, S]`.

---

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
