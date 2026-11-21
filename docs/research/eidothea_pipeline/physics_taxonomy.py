"""Taxonomy for the Differometor / Faster Physics project (2011-2026, 15 years).

Covers frequency-domain optical interferometer simulation, gravitational-wave detector
quantum noise, differentiable sparse linear solvers, symbolic/numeric LU reuse across
frequency sweeps (KLU, SuperLU, UMFPACK), implicit differentiation (VJP/JVP), and
JAX/XLA compilation and execution optimization.

Groups E and F were added after the hackathon implementation (host execution, JVP transposition, iterative
refinement for ill-conditioned solves, compile/cold-start cost) and cover cross-domain analogues: fields that
solve the same problem, a fixed-sparsity complex linear system swept over a parameter with adjoint gradients
(circuit simulation, power grids, FDFD/geophysics, structural dynamics, quantum transport, photonic cascades,
model order reduction).

Same contract as ``robotics_taxonomy.py`` (``SUBFIELDS``, ``tag``, ``tag_title``), so it plugs into
``eidothea analyze-robotics-sweep --taxonomy physics`` and ``eidothea graph build --taxonomy physics``.
Regexes are lowercase ASCII, as ``tests/test_taxonomy.py`` enforces.
"""

# ruff: noqa: E501  (regex tables read better on one line)

from __future__ import annotations

import re

SUBFIELDS: list[tuple[str, str, str, str]] = [
    # --- A. Gravitational-wave interferometry & optical simulation
    ("ifo_sim", "A interferometry", "Frequency-domain optical interferometer simulation (Finesse, Differometor)",
     r"finesse|differometor|interferometer simulat|optical simulat|frequency[- ]domain.{0,40}interferom|"
     r"quasi[- ]static.{0,40}interferom|plane[- ]wave.{0,40}interferom|optical cavity simulat|fabry[- ]perot.{0,40}simulat"),
    ("gw_detector", "A interferometry", "Gravitational-wave detector optical design and topology",
     r"gravitational[- ]wave detector|advanced ligo|\baligo\b|einstein telescope|cosmic explorer|ligo voyager|"
     r"interferometer topology|computational design.{0,40}(interferom|detector|optical)|optical layout|signal recycling|"
     r"power recycling|sagnac interferometer|michelson interferometer"),
    ("quantum_noise", "A interferometry", "Quantum noise, shot noise, radiation pressure and squeezing",
     r"quantum noise|radiation pressure noise|shot noise|squeezed (light|states|vacuum)|frequency[- ]dependent squeezing|"
     r"homodyne detect|quadrature.{0,30}noise|quantum radiation pressure|standard quantum limit|\bsql\b|"
     r"optomechanical (coupling|interaction|damping|instability|spring)|ponderomotive"),
    ("sideband_mod", "A interferometry", "Carrier, sideband modulation and transfer matrix formalisms",
     r"sideband|two[- ]photon formalism|audio[- ]band|upper and lower sideband|modulation[- ]demodulation|"
     r"scattering matrix|transfer matrix.{0,40}(optical|cavity|interferom)|coupling matrix|field amplitude"),
    # --- B. Sparse direct linear solvers & frequency sweeps
    ("sparse_lu", "B sparse solvers", "Sparse LU factorization and direct linear solvers (KLU, SuperLU, UMFPACK)",
     r"sparse (lu\b|direct|linear solver|matrix|factorization|factorisation|gaussian elimination|cholesky)|"
     r"\bklu\b|superlu|umfpack|pardiso|mumps|suitesparse|symbolic factori[sz]ation|numeric(al)? factori[sz]ation|"
     r"fill[- ]reducing|amd ordering|colamd|elimination tree"),
    ("freq_sweep", "B sparse solvers", "Multi-frequency sweep acceleration, reuse and model order reduction",
     r"frequency sweep|multi[- ]frequency|broadband.{0,30}(sweep|simulat|solver)|fast frequency sweep|"
     r"pade approxim|model order reduction|reduced[- ]order model|krylov subspace|rational interpolation|"
     r"loewner|vector fitting|shifted linear system"),
    ("low_rank_update", "B sparse solvers", "Low-rank updates, Sherman-Morrison-Woodbury and Schur complement",
     r"sherman[- ]morrison|woodbury|low[- ]rank (update|modification|perturbation)|rank[- ]1 update|"
     r"schur complement|block elimination|condensation|domain decomposition|nested dissection|kron reduction"),
    ("batched_sparse", "B sparse solvers", "Batched and vectorized sparse solvers (CPU/SIMD/GPU)",
     r"batched (sparse|lu\b|linear|solver|factori[sz]ation)|vectori[sz]ed (sparse|solver|lu\b)|"
     r"identical sparsity|fixed sparsity pattern|same sparsity pattern|refactori[sz]ation|block[- ]diagonal (system|batch|embedding)|many small (sparse|linear)"),
    # --- C. Differentiable physics, implicit differentiation & sensitivity
    ("implicit_diff", "C autodiff & adjoint", "Implicit differentiation and adjoint state methods for linear systems",
     r"implicit differentiation|adjoint (method|state|sensitivity|equation|variable|solve)|"
     r"differentiable (linear solver|sparse|simulation|physics|optics)|reverse[- ]mode.{0,40}(linear|sparse|solve)|"
     r"custom[_ ](vjp|jvp|gradient)|pullback|vector[- ]jacobian product|\bvjp\b"),
    ("adj_quantum_cov", "C autodiff & adjoint", "Adjoint elimination in covariance / quadratic forms and trace derivatives",
     r"covariance propagation|quadratic form.{0,40}(gradient|derivative|adjoint)|"
     r"lyapunov equation|sylvester equation|stochastic trace|bilinear form.{0,40}(gradient|adjoint)|"
     r"noise (covariance|spectral density|propagation).{0,50}(gradient|sensitivity|adjoint)"),
    ("diff_optics", "C autodiff & adjoint", "Differentiable optical and photonic circuit design",
     r"differentiable (optics|optical|photonics|photonic|interferom|ray tracing|wave optics)|"
     r"inverse design.{0,40}(optical|photonic|interferom|cavity|metamaterial)|"
     r"topology optimi[sz]ation.{0,40}(optical|photonic|interferom)"),
    # --- D. JAX, XLA & scientific compiler optimization
    ("jax_xla", "D jax & compiler", "JAX, XLA compilation, tracing overhead and custom primitives",
     r"\bjax\b|\bxla\b|just[- ]in[- ]time compil|\bjit\b.{0,30}compil|tracing overhead|compile time|"
     r"lowering|hlo\b|stablehlo|mlir|pure_callback|custom_call|ffi\b|dlpack"),
    ("autodiff_opt", "D jax & compiler", "Automatic differentiation graph optimization, checkpointing and sparsity",
     r"automatic differentiation|autodiff|algorithmi(c|cal) differentiation|jacobian sparsity|"
     r"sparse (jacobian|hessian|autodiff|ad\b)|graph coloring|activity analysis|dead code elimination|"
     r"remateriali[sz]ation|checkpointing|mixed[- ]mode"),
    # --- E. Host execution, numerics and compile cost (lessons from the hackathon implementation)
    ("host_exec", "E host & numerics", "Running traced programs outside XLA: host callbacks, jaxpr interpreters, staging",
     r"pure_callback|host callback|io_callback|jaxpr|partial evaluation|staging|trace[- ]and[- ]replay|"
     r"interpreter.{0,40}(autodiff|jax|trace|graph)|ahead[- ]of[- ]time|eager execution|tracing jit|wengert"),
    ("jvp_transpose", "E host & numerics", "Linearization and transposition: reverse mode derived from the forward (JVP) program",
     r"transpos.{0,30}(jvp|linear|tangent)|linearize|linearization.{0,40}(autodiff|reverse|jax)|forward[- ]over[- ]reverse|"
     r"reverse[- ]over[- ]forward|tangent.{0,20}transpos|\bjvp\b.{0,40}\bvjp\b|cheap(er)? gradient"),
    ("iter_refine", "E host & numerics", "Iterative refinement, mixed precision and accuracy of ill-conditioned sparse solves",
     r"iterative refinement|mixed[- ]precision.{0,40}(solver|linear|lu\b|factori)|ill[- ]conditioned|condition number|"
     r"backward error|componentwise|extended precision|double[- ]double|pivot growth|static pivoting|threshold pivoting|"
     r"rounding error"),
    ("compile_cold_start", "E host & numerics", "Compile latency, cold start, control-flow cost under vectorisation",
     r"compile time|compilation (time|latency|overhead|cost)|cold start|start[- ]?up (time|latency)|time to first|"
     r"persistent compilation cache|llvm|lax\.switch|control flow.{0,40}(vectori|compil|gpu|simd)|branch divergence|"
     r"lazy import|import time"),
    ("iter_recycling", "E host & numerics", "Krylov recycling and preconditioner reuse across sequences of related systems",
     r"krylov recycling|recycl(ing|ed) krylov|gcro|preconditioner (reuse|update|recycling)|sequence of (linear|related) systems|"
     r"parametri[sz]ed linear systems|\bilu\b preconditioner|gmres.{0,40}(sparse|frequency|sweep)|algebraic multigrid"),
    # --- F. Cross-domain analogues: same math in other industries
    ("xd_circuit", "F cross-domain", "Circuit simulation: SPICE, modified nodal analysis, AC/noise analysis, harmonic balance",
     r"ngspice|hspice|ltspice|spice (simulat|netlist|model|circuit)|modified nodal analysis|harmonic balance|(analog|electronic|vlsi|rf|microwave|integrated|power|digital)[- ]circuit.{0,15}simulat|\bac (small[- ]signal|analysis)\b|small[- ]signal analysis|"
     r"adjoint (network|circuit)|noise analysis.{0,40}(circuit|adjoint|rf)|periodic steady[- ]state|transistor[- ]level|"
     r"\beda\b|electronic design automation"),
    ("xd_power_grid", "F cross-domain", "Power-system simulation: power flow, contingency analysis, network reduction",
     r"power flow|load flow|contingency analysis|power system.{0,40}(simulat|solver|sparse)|admittance matrix|"
     r"network reduction|kron reduction|\bptdf\b|\blodf\b|\bopf\b|optimal power flow|grid simulat"),
    ("xd_fdfd_geo", "F cross-domain", "Frequency-domain wave solvers: FDFD, Helmholtz, RCWA, frequency-domain full-waveform inversion",
     r"\bfdfd\b|finite[- ]difference frequency[- ]domain|helmholtz|\brcwa\b|rigorous coupled[- ]wave|method of moments|"
     r"frequency[- ]domain (full[- ]waveform|seismic|electromagnetic|wave)|full[- ]waveform inversion|\bfwi\b|\bcsem\b|"
     r"maxwell.{0,40}(sparse|direct|factori)"),
    ("xd_struct_dyn", "F cross-domain", "Structural dynamics: frequency response functions, substructuring, component mode synthesis",
     r"frequency response function|\bfrf\b|dynamic substructuring|substructuring (method|technique|approach|analysis)|component mode synthesis|craig[- ]bampton|dynamic condensation|"
     r"guyan|modal (truncation|superposition)|structural dynamics.{0,40}(sparse|sensitivity|reduc)"),
    ("xd_quantum_transport", "F cross-domain", "Quantum transport and selected inversion: NEGF, recursive Green's function, Dyson",
     r"selected inversion|recursive green|\bnegf\b|non[- ]equilibrium green|green'?s function.{0,40}(sparse|block|recursive|algorithm)|"
     r"landauer|quantum transport|dyson equation|block[- ]tridiagonal"),
    ("xd_photonic_cascade", "F cross-domain", "Scattering-matrix cascading, Redheffer star product, signal-flow graphs, photonic circuit simulators",
     r"redheffer|star product|cascad.{0,30}scattering|coupled[- ]mode (theory|equation)|photonic (integrated )?circuit.{0,30}(simulat|solver|design)|"
     r"signal[- ]flow graph|mason'?s (gain )?rule|s-matrix|scattering network|microwave network"),
    ("xd_mor_control", "F cross-domain", "Large-scale LTI model reduction and frequency response from control theory",
     r"balanced truncation|iterative rational krylov|\birka\b|descriptor system|large[- ]scale (lti|linear time|dynamical)|"
     r"parametric model (order )?reduction|reduced basis|h2[- ]optimal|moment matching|transfer function.{0,40}(large|sparse)"),
    ("xd_ordering_graph", "F cross-domain", "Graph algorithms behind sparse elimination: treewidth, ordering, partitioning",
     r"treewidth|tree decomposition|nested dissection|elimination ordering|minimum degree|graph partition|supernod|multifrontal|"
     r"hypergraph partition|symbolic analysis"),
    ("surrogate_ml", "F cross-domain", "Learned surrogates and neural operators as an alternative to exact solvers",
     r"surrogate (model|network)|neural surrogate|deep learning surrogate|neural operator|operator learning|fourier neural operator|"
     r"physics[- ]informed neural|emulator"),
]

_COMPILED = [(i, g, lab, re.compile(rx)) for i, g, lab, rx in SUBFIELDS]


def tag(title: str, abstract: str) -> list[str]:
    text = f"{title} {abstract}".lower()
    return [i for i, _g, _l, rx in _COMPILED if rx.search(text)]


def tag_title(title: str) -> set[str]:
    """Subfields whose regex matches the title alone (a stronger signal than the abstract)."""
    return {i for i, _g, _l, rx in _COMPILED if rx.search(title.lower())}
