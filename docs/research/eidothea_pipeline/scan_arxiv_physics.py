"""Probes for the Differometor / Faster Physics hackathon task across the last 10 years (2016-2026).

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
            custom VJP primitives, FFI / host callback batching, and sparse automatic differentiation.

Scans across a 10-year window (2016-01 to 2026-11) with five 2-year windows plus top relevance-ranked
and recent papers. At least 3.5 s between requests, resumable.
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
}


def select(names: list[str] | None) -> list[str] | None:
    """Probe names matching any of ``names`` exactly or as a prefix (``ifo_`` = the whole group)."""
    if not names:
        return None
    return [n for n in PROBES if any(n == o or n.startswith(o) for o in names)]


def run_10y(probes: dict[str, str], out: Path, top: int = 100, only: list[str] | None = None) -> None:
    """Run ``probes`` across the 10-year window (2016-2026) against the arXiv API, resumable."""
    out.mkdir(parents=True, exist_ok=True)
    counts_path = out / "counts.json"
    counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}
    for name, q in probes.items():
        if only and name not in only:
            continue
        if (out / f"{name}.jsonl").exists() and name in counts:
            continue
        per = []
        for label, a, b in WINDOWS_10Y:
            params = {"search_query": f"({q}) AND submittedDate:[{a} TO {b}]", "start": 0,
                      "max_results": 1}
            _, total = parse(get(f"{API}?{urllib.parse.urlencode(params)}"))
            per.append([label, total])
        params = {"search_query": f"({q}) AND submittedDate:[{ALL_10Y[0]} TO {ALL_10Y[1]}]", "start": 0,
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
    ap.add_argument("--only", nargs="*", default=None, help="probe names or group prefixes (ifo_, slu_, adj_, jax_)")
    args = ap.parse_args()
    run_10y(PROBES, Path(args.out), args.top, select(args.only))
    return 0


if __name__ == "__main__":
    sys.exit(main())
