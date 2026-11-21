"""Taxonomy for the Differometor / Faster Physics project (2016-2026).

Covers frequency-domain optical interferometer simulation, gravitational-wave detector
quantum noise, differentiable sparse linear solvers, symbolic/numeric LU reuse across
frequency sweeps (KLU, SuperLU, UMFPACK), implicit differentiation (VJP/JVP), and
JAX/XLA compilation and execution optimization.

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
     r"identical sparsity|fixed sparsity pattern|same sparsity pattern|refactori[sz]ation"),
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
]

_COMPILED = [(i, g, lab, re.compile(rx)) for i, g, lab, rx in SUBFIELDS]


def tag(title: str, abstract: str) -> list[str]:
    text = f"{title} {abstract}".lower()
    return [i for i, _g, _l, rx in _COMPILED if rx.search(text)]


def tag_title(title: str) -> set[str]:
    """Subfields whose regex matches the title alone (a stronger signal than the abstract)."""
    return {i for i, _g, _l, rx in _COMPILED if rx.search(title.lower())}
