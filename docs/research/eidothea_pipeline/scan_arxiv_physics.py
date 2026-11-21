"""Probes for the Differometor / Faster Physics hackathon task across the last 15 years (2011-2026).

    eidothea scan-arxiv-physics --out data/physics_probes

Questions behind the probes (groups, in the order Differometor hits them):

* ``ifo_``  Frequency-domain optical interferometer simulation (Finesse 2/3, Differometor),
            gravitational-wave detector topology optimization, quantum noise (shot + radiation pressure),
            and two-photon / sideband matrix formulations;
* ``slu_``  Sparse direct linear solvers for fixed-sparsity complex systems across frequency sweeps:
            symbolic/numeric LU factorization split (KLU, SuperLU, UMFPACK), Schur complement /
            static condensation of degree-2 optical nodes, and low-rank / Sherman-Morrison updates;
* ``adj_``  Implicit differentiation and adjoint methods for complex sparse linear systems A(theta)x = b(theta),
            eliminating redundant adjoint solves in quantum noise covariance propagation (w = A^{-H} c, v = A^{-1} N w),
            and differentiable optical / photonic circuit solvers;
* ``jax_``  JAX / XLA compilation & runtime optimization: reducing trace/lowering/LLVM compile time,
            custom VJP primitives, FFI / host callback batching, and sparse automatic differentiation;
* ``num_``/``host_``/``jvp_``/``cold_``/``iter_``  lessons from the implementation: iterative refinement on
            ill-conditioned sparse solves, jaxpr interpreters / host execution, JVP transposition, compile and
            cold-start cost, Krylov recycling;
* ``xd_``   cross-domain analogues, other industries solving the same fixed-sparsity complex system swept over a
            parameter with adjoint gradients (circuits, power grids, FDFD/geophysics, structural dynamics,
            quantum transport, photonic cascades, model order reduction, sparse ordering).

Scans a window (default 15 years, 2011-01 to 2026-11; 10 years with --years 10) split into sub-windows plus
the top relevance-ranked papers. At least 3.5 s between requests, resumable.
"""

# ruff: noqa: E501  (query tables read better on one line)

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from pathlib import Path

from eidothea.probes.scan_arxiv_monthly import API, get, parse

WINDOWS_10Y = [
    ("2016-01 to 2018-06", "201601010000", "201806302359"),
    ("2018-07 to 2020-12", "201807010000", "202012312359"),
    ("2021-01 to 2022-12", "202101010000", "202212312359"),
    ("2023-01 to 2024-12", "202301010000", "202412312359"),
    ("2025-01 to 2026-11", "202501010000", "202611302359"),
]
ALL_10Y = ("201601010000", "202611302359")
WINDOWS_15Y = [
    ("2011-01 to 2013-06", "201101010000", "201306302359"),
    ("2013-07 to 2015-12", "201307010000", "201512312359"),
] + WINDOWS_10Y
ALL_15Y = ("201101010000", "202611302359")

PROBES: dict[str, str] = {
    # --- A. Interferometer simulation & gravitational-wave detector design
    "ifo_finesse_differometor": '(abs:Finesse OR abs:Differometor OR abs:"PyKat" OR abs:"frequency-domain interferometer" OR abs:"interferometer simulator" OR abs:"interferometer simulation") AND (abs:gravitational OR abs:interferometer OR abs:optical OR abs:LIGO)',
    "ifo_gw_computational_design": '(abs:"gravitational-wave detector" OR abs:"gravitational wave detector" OR abs:"interferometer topology" OR abs:"optical configuration") AND (abs:"computational design" OR abs:"topology optimization" OR abs:"gradient-based" OR abs:differentiable OR abs:optimization) AND (abs:quantum OR abs:sensitivity OR abs:noise)',
    "ifo_quantum_noise_matrix": '(abs:"quantum noise" OR abs:"radiation pressure noise" OR abs:"shot noise" OR abs:"squeezed light" OR abs:"ponderomotive") AND (abs:"transfer matrix" OR abs:"scattering matrix" OR abs:"sideband" OR abs:"two-photon" OR abs:"homodyne" OR abs:"linear system") AND (abs:interferometer OR abs:gravitational OR abs:cavity)',
    "ifo_optomechanics_freq": '(abs:"optomechanical" OR abs:"radiation pressure") AND (abs:"frequency domain" OR abs:"susceptibility" OR abs:"audio-band" OR abs:"sideband") AND (abs:interferometer OR abs:cavity OR abs:LIGO)',
    # --- B. Sparse linear solvers, symbolic/numeric LU reuse & frequency sweeps
    "slu_klu_superlu_refactor": '(abs:KLU OR abs:SuperLU OR abs:UMFPACK OR abs:SuiteSparse OR abs:"sparse LU" OR abs:"sparse direct solver") AND (abs:"symbolic factorization" OR abs:"refactorization" OR abs:"sparsity pattern" OR abs:"circuit simulation" OR abs:"frequency")',
    "slu_freq_sweep_fast": '(abs:"frequency sweep" OR abs:"multi-frequency" OR abs:"fast frequency sweep" OR abs:"shifted linear system") AND (abs:"sparse" OR abs:"direct solver" OR abs:"LU factorization" OR abs:"Schur complement" OR abs:"reduced order" OR abs:"interpolation")',
    "slu_schur_condensation": '(abs:"Schur complement" OR abs:"static condensation" OR abs:"block elimination" OR abs:"Kron reduction" OR abs:"graph reduction") AND (abs:"sparse linear" OR abs:"sparse matrix" OR abs:"frequency domain" OR abs:"helmholtz" OR abs:"scattering" OR abs:"optical")',
    "slu_low_rank_woodbury": '(abs:"Sherman-Morrison" OR abs:"Woodbury" OR abs:"low-rank update" OR abs:"rank-one update") AND (abs:"sparse" OR abs:"linear solver" OR abs:"frequency" OR abs:"adjoint" OR abs:"sensitivity")',
    "slu_batched_sparse_solver": '(abs:"batched sparse" OR abs:"batch sparse" OR abs:"identical sparsity" OR abs:"same sparsity pattern") AND (abs:LU OR abs:solver OR abs:factorization OR abs:linear)',
    # --- C. Differentiable sparse solvers, implicit differentiation & adjoint optics
    "adj_implicit_sparse_solve": '(abs:"implicit differentiation" OR abs:"adjoint method" OR abs:"reverse-mode" OR abs:"custom VJP" OR abs:"differentiable programming") AND (abs:"sparse linear" OR abs:"linear solver" OR abs:"sparse matrix" OR abs:"direct solver" OR abs:"LU factorization")',
    "adj_differentiable_optics": '(abs:"differentiable" OR abs:"automatic differentiation" OR abs:"autodiff" OR abs:"adjoint") AND (abs:"optical simulation" OR abs:"photonic circuit" OR abs:"scattering matrix" OR abs:"transfer matrix" OR abs:"interferometer" OR abs:"S-matrix")',
    "adj_covariance_quadratic": '(abs:"adjoint" OR abs:"automatic differentiation" OR abs:"sensitivity analysis") AND (abs:"noise covariance" OR abs:"power spectral density" OR abs:"quadratic form" OR abs:"bilinear form" OR abs:"resolvent" OR abs:"Green function") AND (abs:frequency OR abs:linear OR abs:sparse)',
    # --- D. JAX / XLA compilation & autodiff efficiency
    "jax_physics_simulation": '(abs:JAX OR abs:XLA) AND (abs:"differentiable simulation" OR abs:"differentiable physics" OR abs:"optics" OR abs:"photonic" OR abs:"interferometer" OR abs:"sparse" OR abs:"PDE" OR abs:"scientific computing")',
    "jax_compile_autodiff_opt": '(abs:JAX OR abs:XLA OR abs:"StableHLO" OR abs:"MLIR") AND (abs:"compilation time" OR abs:"compile time" OR abs:"tracing" OR abs:"custom VJP" OR abs:"automatic differentiation" OR abs:"sparse Jacobian")',
    "jax_sparse_autodiff": '(abs:"sparse automatic differentiation" OR abs:"sparse Jacobian" OR abs:"sparse autodiff" OR abs:"implicit differentiation") AND (abs:JAX OR abs:Julia OR abs:Python OR abs:compiler OR abs:"scientific computing")',
    # --- E. Targeted exact-system probes (Urania/Differometor, sparse VJP, Finesse/KLU)
    "ifo_urania_digital_discovery": '(abs:Urania OR abs:Differometor OR abs:"digital discovery" OR abs:"universal interferometer" OR abs:UIFO OR au:Krenn) AND (abs:gravitational OR abs:interferometer OR abs:LIGO OR abs:Finesse)',
    "slu_finesse_matrix_solver": '(abs:Finesse OR abs:"interferometer matrix" OR abs:"coupling matrix" OR abs:"homodyne" OR abs:"quantum noise") AND (abs:sparse OR abs:KLU OR abs:MISO OR abs:matrix OR abs:solver) AND (abs:gravitational OR abs:LIGO OR abs:interferometer)',
    "adj_linear_solve_vjp": '(abs:"linear solver" OR abs:"linear system" OR abs:"sparse solver" OR abs:"LU factorization") AND (abs:"automatic differentiation" OR abs:"reverse-mode" OR abs:"adjoint" OR abs:"VJP" OR abs:"implicit differentiation") AND (abs:JAX OR abs:PyTorch OR abs:Julia OR abs:SciPy OR abs:differentiable)',
    "slu_symbolic_numeric_lu": '(abs:"symbolic factorization" OR abs:"numeric factorization" OR abs:"numerical factorization" OR abs:"refactorization") AND (abs:"sparse LU" OR abs:"sparse direct" OR abs:SuperLU OR abs:KLU OR abs:UMFPACK OR abs:PARDISO)',
    # --- F. Lessons from the implementation: numerics, host execution, compile cost
    "num_iterative_refinement": '(abs:"iterative refinement" OR abs:"mixed precision") AND (abs:"sparse" OR abs:"LU factorization" OR abs:"direct solver" OR abs:"linear system") AND (abs:accuracy OR abs:"ill-conditioned" OR abs:"backward error" OR abs:"high precision")',
    "num_illcond_sparse_accuracy": '(abs:"ill-conditioned" OR abs:"backward error" OR abs:"componentwise" OR abs:"pivot growth" OR abs:"static pivoting") AND (abs:"sparse LU" OR abs:"sparse direct" OR abs:"sparse linear" OR abs:SuperLU OR abs:MUMPS)',
    "host_jaxpr_interpreter": '(abs:jaxpr OR abs:"host callback" OR abs:pure_callback OR abs:"partial evaluation" OR abs:"program transformation") AND (abs:interpreter OR abs:staging OR abs:"just-in-time" OR abs:"tracing") AND (abs:JAX OR abs:"automatic differentiation" OR abs:"differentiable programming")',
    "jvp_transposition_reverse": '(abs:linearization OR abs:transposition OR abs:"forward-mode") AND (abs:"reverse-mode" OR abs:JVP OR abs:VJP OR abs:"vector-Jacobian") AND (abs:"automatic differentiation" OR abs:JAX OR abs:"differentiable programming")',
    "cold_start_compile_latency": '(abs:"cold start" OR abs:"startup time" OR abs:"time to first" OR abs:"compilation latency" OR abs:"ahead-of-time" OR abs:"compilation cache") AND (abs:JAX OR abs:Julia OR abs:Python OR abs:"scientific computing" OR abs:"compiler" OR abs:"numerical")',
    "iter_krylov_recycling": '(abs:"Krylov recycling" OR abs:"recycling Krylov" OR abs:"preconditioner reuse" OR abs:"sequence of linear systems" OR abs:"parametrized linear systems") AND (abs:sparse OR abs:frequency OR abs:GMRES OR abs:preconditioner)',
    # --- G. Cross-domain: the same math in other industries
    "xd_circuit_adjoint_noise": '(abs:"circuit simulation" OR abs:SPICE OR abs:"modified nodal analysis" OR abs:"harmonic balance" OR abs:"noise analysis") AND (abs:adjoint OR abs:"sensitivity analysis" OR abs:"sparse LU" OR abs:KLU OR abs:refactorization OR abs:"noise")',
    "xd_circuit_ac_sweep": '(abs:"AC analysis" OR abs:"small-signal" OR abs:"frequency response" OR abs:"harmonic balance") AND (abs:sparse OR abs:LU OR abs:Krylov OR abs:reuse OR abs:recycling OR abs:"model order reduction")',
    "xd_power_grid_sparse": '(abs:"power flow" OR abs:"contingency analysis" OR abs:"power system simulation" OR abs:"optimal power flow") AND (abs:sparse OR abs:"LU factorization" OR abs:refactorization OR abs:batched OR abs:"low-rank" OR abs:Woodbury OR abs:differentiable)',
    "xd_fdfd_helmholtz_multi": '(abs:FDFD OR abs:"frequency-domain" OR abs:Helmholtz OR abs:Maxwell OR abs:RCWA) AND (abs:"multiple frequencies" OR abs:adjoint OR abs:"sparse direct" OR abs:"factorization reuse" OR abs:"inverse design" OR abs:"multiple right-hand")',
    "xd_geophysics_fwi_direct": '(abs:"full-waveform inversion" OR abs:"seismic" OR abs:"CSEM" OR abs:"electromagnetic modeling") AND abs:"frequency domain" AND (abs:"sparse direct" OR abs:"multiple sources" OR abs:factorization OR abs:MUMPS OR abs:"LU")',
    "xd_structural_frf_substructuring": '(abs:"frequency response function" OR abs:substructuring OR abs:"component mode synthesis" OR abs:"dynamic condensation") AND (abs:sparse OR abs:Schur OR abs:reuse OR abs:sensitivity OR abs:adjoint OR abs:"model reduction")',
    "xd_negf_selected_inversion": '(abs:"selected inversion" OR abs:"recursive Green" OR abs:NEGF OR abs:"non-equilibrium Green" OR abs:"block-tridiagonal") AND (abs:sparse OR abs:algorithm OR abs:LU OR abs:efficient OR abs:"linear scaling")',
    "xd_photonic_cascade_smatrix": '(abs:"scattering matrix" OR abs:"S-matrix" OR abs:Redheffer OR abs:"coupled-mode" OR abs:"signal flow graph") AND (abs:"photonic circuit" OR abs:"integrated photonics" OR abs:"circuit simulation" OR abs:cascading OR abs:network) AND (abs:simulation OR abs:solver OR abs:differentiable OR abs:fast)',
    "xd_model_order_reduction_lti": '(abs:"model order reduction" OR abs:"balanced truncation" OR abs:"rational Krylov" OR abs:IRKA OR abs:"moment matching") AND (abs:"descriptor system" OR abs:"large-scale" OR abs:"frequency response" OR abs:parametric)',
    "xd_parametric_mor_sweep": '(abs:"parametric model order reduction" OR abs:"reduced basis" OR abs:"interpolation") AND (abs:"frequency sweep" OR abs:"frequency domain" OR abs:"sparse linear" OR abs:"wideband")',
    "xd_ordering_treewidth": '(abs:treewidth OR abs:"nested dissection" OR abs:"elimination ordering" OR abs:"minimum degree" OR abs:multifrontal OR abs:supernodal) AND (abs:sparse OR abs:LU OR abs:Cholesky OR abs:"direct solver")',
    "xd_neural_surrogate_optics": '(abs:"neural surrogate" OR abs:"deep learning surrogate" OR abs:"operator learning" OR abs:"neural operator") AND (abs:interferometer OR abs:photonic OR abs:optics OR abs:gravitational OR abs:"circuit")',
}


def select(names: list[str] | None) -> list[str] | None:
    """Probe names matching any of ``names`` exactly or as a prefix (``ifo_`` = the whole group)."""
    if not names:
        return None
    return [n for n in PROBES if any(n == o or n.startswith(o) for o in names)]


def run_10y(probes: dict[str, str], out: Path, top: int = 100, only: list[str] | None = None, years: int = 15) -> None:
    """Run ``probes`` across the last ``years`` (15: 2011-2026, 10: 2016-2026) against the arXiv API, resumable."""
    windows, (lo, hi) = (WINDOWS_15Y, ALL_15Y) if years == 15 else (WINDOWS_10Y, ALL_10Y)
    out.mkdir(parents=True, exist_ok=True)
    counts_path = out / "counts.json"
    counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}
    for name, q in probes.items():
        if only and name not in only:
            continue
        if (out / f"{name}.jsonl").exists() and name in counts:
            continue
        per = []
        for label, a, b in windows:
            params = {"search_query": f"({q}) AND submittedDate:[{a} TO {b}]", "start": 0,
                      "max_results": 1}
            _, total = parse(get(f"{API}?{urllib.parse.urlencode(params)}"))
            per.append([label, total])
        params = {"search_query": f"({q}) AND submittedDate:[{lo} TO {hi}]", "start": 0,
                  "max_results": top, "sortBy": "relevance", "sortOrder": "descending"}
        rows, total = parse(get(f"{API}?{urllib.parse.urlencode(params)}"))
        (out / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
        counts[name] = {"windows": per, "total": total, "query": q}
        counts_path.write_text(json.dumps(counts, indent=1))
        print(f"{name}: total {total}; per window {[c for _, c in per]}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--years", type=int, choices=(10, 15), default=15)
    ap.add_argument("--only", nargs="*", default=None, help="probe names or group prefixes (ifo_, slu_, adj_, jax_, num_, host_, jvp_, cold_, iter_, xd_)")
    args = ap.parse_args()
    run_10y(PROBES, Path(args.out), args.top, select(args.only), args.years)
    return 0


if __name__ == "__main__":
    sys.exit(main())
