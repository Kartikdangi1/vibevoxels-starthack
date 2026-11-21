# Papers and references for optimizing Differometor

This folder collects papers and primary technical references that inform optimization ideas for this repository. It is an annotated reading list with links to publisher, conference, arXiv, or author-hosted copies. Open-access PDFs are included where available; paywalled references remain linked rather than copied.

Open-access copies downloaded into this repository are listed under [`pdfs/`](pdfs/). Other entries link to their publication pages or online copies.

### PDFs included here

- [Efficient and Modular Implicit Differentiation](pdfs/efficient_implicit_differentiation.pdf)
- [OptNet: Differentiable Optimization as a Layer in Neural Networks](pdfs/optnet_implicit_differentiation.pdf)
- [Differentiate the Solver, Not the Equation](pdfs/reverse_sweep_adjoints_block_simulation.pdf)
- [Compiling Machine Learning Programs via High-Level Tracing (JAX)](pdfs/jax_high_level_tracing.pdf)
- [Operator Fusion in XLA](pdfs/operator_fusion_xla.pdf)

## Start here

| Priority | Read | Why it matters here |
|---|---|---|
| 1 | [Algorithm 907: KLU, A Direct Sparse Solver for Circuit Simulation Problems](https://doi.org/10.1145/1824801.1824814) | Closest solver analogue: sparse, unsymmetric systems from circuit-like models. Helps assess whether the dense JAX solves should be replaced by a sparse direct solver. |
| 2 | [Efficient and Modular Implicit Differentiation](https://proceedings.neurips.cc/paper_files/paper/2022/hash/228b9279ecf9bbafe582406850c57115-Abstract.html) | A practical framework for implicit derivatives. Relevant to replacing autodiff through dense factorization with an adjoint/custom derivative while retaining derivatives through component values. |
| 3 | [Generalized Nested Dissection](https://epubs.siam.org/doi/abs/10.1137/0716027) | Establishes why separator-based elimination can reduce fill and work for planar or well-separated graphs. Relevant only if the built optical matrix's graph has the assumed separators. |
| 4 | [Compiling Machine Learning Programs via High-Level Tracing](https://research.google/pubs/compiling-machine-learning-programs-via-high-level-tracing/) | Background on JAX tracing and XLA compilation. Helps frame why a large traced graph and dynamic control flow can dominate first-call compile time. |
| 5 | [JAX's slow tracing and compilation guide](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html) | Not a paper, but the most directly actionable source for collecting JAX compile-stage timings and equation-count profiles before rewriting the update path. |
| 6 | [Schwarz–Schur Involution: Lightspeed Differentiable Sparse Linear Solvers](https://proceedings.mlr.press/v267/wang25a.html) | Recent differentiable direct-solver work based on recursively condensing sparse systems into smaller dense Schur problems. Its GPU results do not transfer directly to this CPU benchmark, but its condensation strategy is relevant to a future verified block solver. |

## Linear systems and sparsity

### Algorithm 907: KLU, A Direct Sparse Solver for Circuit Simulation Problems

Timothy A. Davis and Ekanathan Palamadai Natarajan. *ACM Transactions on Mathematical Software*, 37(3), 2010. [DOI / paper](https://doi.org/10.1145/1824801.1824814)

KLU targets sparse unsymmetric systems arising in circuit simulation. This is a strong candidate to study because the simulator assembles a circuit-like optical network and currently sends dense matrices to `jnp.linalg.solve`. The paper motivates testing a sparse CPU direct solver with symbolic analysis reused across the 101 frequencies. It does not establish that KLU will win at this problem size, and JAX integration plus complex conjugate-transpose solves remain important constraints.

### Generalized Nested Dissection

Richard J. Lipton, Donald J. Rose, and Robert Endre Tarjan. *SIAM Journal on Numerical Analysis*, 16(2), 1979. [SIAM paper page](https://epubs.siam.org/doi/abs/10.1137/0716027)

The paper generalizes nested dissection to planar and almost-planar graphs. It is relevant to the suggested cell-wise elimination for the 3×3 optical grid. Before implementing it, construct the actual sparsity graph and verify that separators are small and that unsymmetric complex LU pivoting does not erase the expected benefit.

### Nested-Dissection Orderings for Sparse LU with Partial Pivoting

Igor Brainman and Sivan Toledo. *SIAM Journal on Matrix Analysis and Applications*, 23(4), 2002. [SIAM paper page](https://epubs.siam.org/doi/10.1137/S0895479801385037)

More directly relevant than the classical Cholesky setting when considering unsymmetric LU: it studies nested-dissection orderings with partial pivoting. Useful background if the structural solver retains general complex, unsymmetric matrices.

### Algorithm 930: FACTORIZE: An Object-Oriented Linear System Solver for MATLAB

Timothy A. Davis. *ACM Transactions on Mathematical Software*, 39(4), 2013. [DOI / paper](https://doi.org/10.1145/2491491.2491498)

Discusses reusing a matrix factorization for multiple right-hand sides. The implementation language and interface differ, but the numerical pattern directly matches this simulator: factor a frequency-specific matrix once, then use its factors for the signal, detector adjoint weights, and covariance propagation.

### Schwarz–Schur Involution: Lightspeed Differentiable Sparse Linear Solvers

Yu Wang, Mazdak Abulnaga, Yaël Balbastre, and Bruce Fischl. *Proceedings of the 42nd International Conference on Machine Learning*, PMLR 267, 2025. [PMLR paper page and PDF](https://proceedings.mlr.press/v267/wang25a.html)

The method recursively condenses a sparse system into smaller dense interface problems and batches those small operations. This supports the general Schur-complement direction in T2/T2c. Its experiments are GPU-oriented, so it is a design reference rather than evidence that the same implementation will beat the current CPU dense solve.

### Differentiable Preconditioned Krylov Solvers for Square Unsymmetric Linear Systems

Alexis Montoison, Michel Schanen, Jan Hückelheim, and Paul Hovland. *Proceedings of the 2024 International Conference on Algorithmic Differentiation*, pp. 234–245. [SIAM proceedings page](https://epubs.siam.org/doi/abs/10.1137/1.9781611979039.17)

Relevant if a sparse iterative solve is explored instead of direct LU. It discusses how to obtain accurate derivatives in the presence of preconditioning; blindly differentiating through a preconditioned solver may not give the desired derivative. The current benchmark's exact sensitivity tolerance makes convergence and gradient accuracy central.

## Adjoint and implicit differentiation

### Efficient and Modular Implicit Differentiation

Mathieu Blondel et al. *Advances in Neural Information Processing Systems 35*, 2022. [Conference paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/228b9279ecf9bbafe582406850c57115-Abstract.html) · [arXiv](https://arxiv.org/abs/2105.15183)

Local copy: [`pdfs/efficient_implicit_differentiation.pdf`](pdfs/efficient_implicit_differentiation.pdf).

Presents a modular approach to implicit differentiation using optimality or fixed-point conditions. The relevant idea is to differentiate the mathematical solve through an adjoint system, rather than retaining every operation in the forward solver's autodiff tape. Adapt the linear-system case to the carrier → signal → noise dependency chain.

### OptNet: Differentiable Optimization as a Layer in Neural Networks

Brandon Amos and J. Zico Kolter. *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 2017. [PMLR paper page](https://proceedings.mlr.press/v70/amos17a.html)

An accessible derivation of gradients from implicit optimality conditions and adjoint linear systems. Its subject is quadratic optimization rather than optical simulation, but the derivative strategy provides useful background for a custom VJP around the simulator's linear solves.

### Differentiate the Solver, Not the Equation: Reverse-Sweep Adjoints for Block Implicit Simulation

Lei Shu et al. arXiv preprint, 2026. [arXiv](https://arxiv.org/abs/2608.08559)

Recent exploratory work on reversing block implicit solvers while preserving their block structure. This may be useful if the signal system is refactored into independent upper/lower blocks and a Schur complement. It is a preprint, and its methods should be checked against this simulator's exact, finite direct solves before adoption.

## JAX, compilation, and memory

### Compiling Machine Learning Programs via High-Level Tracing

Roy Frostig, Matthew Johnson, and Chris Leary. SysML, 2018. [Google Research publication page](https://research.google/pubs/compiling-machine-learning-programs-via-high-level-tracing/) · [author-hosted PDF](https://www-cs.stanford.edu/~rfrostig/pubs/jax-mlsys2018.pdf)

The JAX system paper. Useful context for understanding how tracing exposes array operations to XLA and why reducing dynamic switching, repeated scatters, and redundant intermediate arrays can help compilation and execution.

### Operator Fusion in XLA: Analysis and Evaluation

2023. [arXiv](https://arxiv.org/abs/2301.13062)

Studies XLA fusion behavior. Useful when evaluating proposed copy-removal, scatter-combining, and elementwise-fusion changes: some apparent intermediates may already be fused away, so inspect optimized HLO and measurements before assuming a speedup.

### JAX-AMG: A GPU-Accelerated Differentiable Sparse Linear Solver Library for JAX

2026. *SoftwareX*, 35, 102966. [Publisher page](https://doi.org/10.1016/j.softx.2026.102966) · [arXiv](https://arxiv.org/abs/2606.09001)

An example of integrating sparse solvers and adjoint differentiation into JAX. Its NVIDIA GPU / algebraic-multigrid focus does not match this repository's CPU-only benchmark, so treat it as future-work context rather than a near-term dependency recommendation.

## Domain context

### Artificial Scientific Discovery of Interferometric Gravitational Wave Detectors

Paper describing the detector-design context and UIFO topology used by this repository. [arXiv:2312.04258](https://arxiv.org/abs/2312.04258)

Useful for understanding the physical layout and why the setup has a grid-like organization. It is not a numerical linear-solver paper; the matrix sparsity and block assumptions still need to be confirmed directly from `differometor/build.py` output.

## Suggested reading order for this repo

1. Read the JAX compilation guide and capture the current JAXPR/HLO profile.
2. Read KLU and the nested-dissection references while inspecting matrix sparsity and graph separators.
3. Read Blondel et al. and OptNet before implementing an adjoint/custom VJP.
4. Use the XLA fusion paper to sanity-check low-level memory and fusion hypotheses.
5. Consult the UIFO paper for physical context, then validate every structural assumption against the built matrices.

## Techniques applied in the current implementation

- **Factor reuse:** one LU factorization per frequency supplies the signal, conjugate-transpose detector, and covariance solves (T1).
- **Static specialization:** matrix-update calls are grouped by component function in a static plan; frequency-independent signal updates are built once (T3/T4).
- **Bounded memory:** frequencies are processed with `lax.map` and checkpointing rather than materializing a full 101-frequency matrix batch (T6).
- **Implicit/custom adjoint:** explored and reverted because the gradient failed the benchmark's finite-difference check. Native LU autodiff remains enabled.
