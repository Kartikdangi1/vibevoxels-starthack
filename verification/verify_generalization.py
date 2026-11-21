"""Does the engine stay correct on problems OTHER than the public benchmark?

The public UIFO is one point in a large family. A company test will use different topologies,
sizes, modulation modes, sweeps and parameter values, so this script fuzzes all of those and
checks the default host path against independent implementations:

  A. random UIFO family: sizes 2..N x seeds x modulation modes (space / amplitude / frequency)
     host vs in-graph JAX (values + gradients); host vs upstream dense for the small ones
  B. aLIGO and Voyager in all three modulation modes
  C. perturbed public fixture: random jitter of the 193 design variables around the public optimum
     (different operating points of the SAME topology), host vs JAX values + gradient
  D. different sweep types: laser power sweep, mirror tuning sweep, two simultaneous sweeps,
     a single frequency, and duplicated frequencies
  E. short optimizer runs (Adam on the public problem): host and JAX trajectories must agree and the
     loss must go down, i.e. the gradient is usable end-to-end, not just locally correct

    python verification/verify_generalization.py [--quick] [--seeds 6] [--max-size 4]
"""
import argparse
import sys

import differometor as df
from common import (Report, aligo, build_arrays, cavity, jax, jnp, load_public, np, public_frequencies,
                    run_backend, scalar_objective, set_backend, uifo, uifo_loss, voyager)

R = Report("generalization to other problems")
MODES = ("space_modulation", "amplitude_modulation", "frequency_modulation")
SWEEP = [("f", "frequency")]


def agree(name, a_outs, b_outs, tol=2e-4):
    """Outputs agree to `tol` of each array's own scale (two correct LU orderings differ ~1e-4 on ill-conditioned optics)."""
    worst, ok = 0.0, True
    for a, b in zip(a_outs, b_outs):
        a, b = np.asarray(a), np.asarray(b)
        if b.size == 0:
            continue
        if a.shape != b.shape or not np.isfinite(a).all():
            ok = False
            continue
        rel = float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))
        worst, ok = max(worst, rel), ok and rel <= tol
    return R.check(name, ok, f"worst rel diff {worst:.2e}")


def grad_of(backend, dense, sparse, x):
    return np.asarray(jax.grad(lambda v: scalar_objective(run_backend(backend, dense, sparse, v)))(x))


def dark_detector(carrier, meta):
    """True when the detector ports carry ~zero carrier power (a degenerate, dark-port setup).

    There the quantum-noise weights are built from carrier values that are pure round-off (~eps), amplified by an
    ill-conditioned signal system (cond ~1e13). No implementation, including the original dense solver, can be
    accurate for the noise there (verified: random UIFO size 2, seed 201), so only carrier/signal are compared.
    """
    power = np.asarray(df.power_detector(carrier))
    detector = power[np.asarray(meta[0])]
    return float(np.max(detector)) < 1e-20 * max(float(np.max(power)), 1e-300)


def compare_case(name, setup, pairs, values, opt, with_dense=False, with_grad=True):
    try:
        dense, sparse, meta = build_arrays(setup, pairs, values, opt)
    except Exception as exc:  # a build failure is itself a finding
        return R.check(f"{name}: builds", False, f"{type(exc).__name__}: {exc}")
    host, jx = run_backend("host", dense, sparse), run_backend("jax", dense, sparse)
    dark = dark_detector(host[0], meta)
    n_cmp = 2 if dark else 3  # carrier, signal (+ noise when the detector is lit)
    if dark:
        print(f"  note: {name}: dark detector port -> noise (round-off) not compared", flush=True)
    agree(f"{name}: host == jax values", host[:n_cmp], jx[:n_cmp])
    if with_dense:
        agree(f"{name}: host == dense values", host[:n_cmp], run_backend("dense", dense, sparse)[:n_cmp])
    if with_grad and not dark:
        x = jnp.asarray(dense["optimized_parameters"])
        gh, gj = grad_of("host", dense, sparse, x), grad_of("jax", dense, sparse, x)
        scale = max(float(np.max(np.abs(gj))), 1e-12)
        R.check(f"{name}: host gradient == jax gradient ({x.size} params)",
                np.isfinite(gh).all() and np.max(np.abs(gh - gj)) <= 1e-3 * scale,
                f"worst rel diff {np.max(np.abs(gh - gj)) / scale:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--max-size", type=int, default=4)
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.max_size = 2, 3
    rng = np.random.default_rng(2026)
    freqs8 = jnp.asarray(np.geomspace(5, 5000, 8))

    # ---- A. random UIFO family
    print("\n# A. random UIFO family")
    for size in range(2, args.max_size + 1):
        for seed in range(args.seeds):
            for mode in MODES[:1] if args.quick else MODES:
                S, opt = uifo(size, random=True, random_seed=100 * size + seed, mode=mode)
                compare_case(f"uifo{size} seed{seed} {mode.split('_')[0]}", S, SWEEP, freqs8, opt,
                             with_dense=(size <= 2))

    # ---- B. real instruments, all modulation modes
    print("\n# B. aLIGO / Voyager, all modulation modes")
    f6 = jnp.asarray(public_frequencies(6))
    for label, factory in (("aligo", aligo), ("voyager", voyager)):
        for mode in MODES:
            S, opt = factory(mode=mode)
            compare_case(f"{label} {mode.split('_')[0]}", S, SWEEP, f6, opt, with_grad=not args.quick)

    # ---- C. perturbed operating points of the public topology
    print("\n# C. perturbed public fixture")
    case = load_public(6)
    arrays, ports, x0, ref = case["arrays"], case["ports"], case["params"], jnp.asarray(case["ref"])
    for k, rel in enumerate((1e-4, 1e-3, 1e-2) if not args.quick else (1e-3,)):
        x = np.asarray(x0) + rel * rng.normal(size=x0.shape) * np.maximum(np.abs(np.asarray(x0)), 1e-3)
        pairs_flat = [p[0] if isinstance(p[0], list) else p for p in case["pairs"]]
        for i, (_, prop) in enumerate(pairs_flat):  # stay inside physical bounds
            if prop == "reflectivity":
                x[i] = float(np.clip(x[i], 1e-9, 1 - 1e-9))
            if prop == "loss":
                x[i] = float(np.clip(x[i], 0.0, 1.0))
        x = jnp.asarray(x)
        out = {}
        for backend in ("host", "jax"):
            set_backend(backend)
            (loss, sens), grad = jax.value_and_grad(lambda v: uifo_loss(v, arrays, ports, ref), has_aux=True)(x)
            out[backend] = (float(loss), np.asarray(sens), np.asarray(grad))
        (lh, sh, gh), (lj, sj, gj) = out["host"], out["jax"]
        R.check(f"perturbation {rel:g}: finite", np.isfinite(lh) and np.isfinite(sh).all() and np.isfinite(gh).all(),
                f"loss={lh:.5f}")
        R.close(f"perturbation {rel:g}: host sensitivity == jax", np.log(sh), np.log(sj), atol=1e-5)
        R.check(f"perturbation {rel:g}: host gradient == jax gradient",
                np.max(np.abs(gh - gj)) <= 1e-3 * max(np.max(np.abs(gj)), 1e-12),
                f"{np.max(np.abs(gh - gj)) / max(np.max(np.abs(gj)), 1e-12):.2e}")
    set_backend("host")

    # ---- D. sweep variety
    print("\n# D. sweep variety")
    S = cavity()
    compare_case("laser power sweep", S, [("l0", "power")], jnp.linspace(0.1, 10, 12), [("m0", "reflectivity")], with_dense=True)
    compare_case("tuning sweep", cavity(), [("m1", "tuning")], jnp.linspace(-90, 90, 16), [("m0", "reflectivity")], with_dense=True)
    two = jnp.stack([jnp.linspace(-30, 30, 10), jnp.linspace(0.5, 2, 10)])
    compare_case("two simultaneous sweeps", cavity(), [("m0", "tuning"), ("l0", "power")], two,
                 [("m1", "reflectivity")], with_dense=True)
    Sa, opt = aligo()
    compare_case("single frequency", Sa, SWEEP, jnp.asarray([100.0]), opt, with_grad=False)
    compare_case("duplicated frequencies", Sa, SWEEP, jnp.asarray([50.0, 50.0, 200.0, 200.0]), opt, with_grad=False)

    # ---- E. optimizer trajectories
    print("\n# E. short Adam runs on the public problem (host vs jax)")
    import optax
    steps = 3 if args.quick else 12
    traj = {}
    for backend in ("host", "jax"):
        set_backend(backend)
        is_r = jnp.asarray([p[1] == "reflectivity" for p in pairs_flat])
        lo = jnp.where(is_r, 1e-12, -jnp.inf)
        hi = jnp.where(is_r, 1 - 1e-12, jnp.inf)  # R = 1 or > 1 gives NaN (sqrt(1-R))
        opt_ = optax.adam(1e-10)
        x, state, losses = x0, opt_.init(x0), []
        vg = jax.value_and_grad(lambda v: uifo_loss(v, arrays, ports, ref)[0])
        for _ in range(steps):
            loss, g = vg(x)
            upd, state = opt_.update(g, state)
            x = jnp.clip(optax.apply_updates(x, upd), lo, hi)
            losses.append(float(loss))
        traj[backend] = np.asarray(losses)
    set_backend("host")
    R.check("optimizer: loss finite throughout", np.isfinite(traj["host"]).all())
    R.check("optimizer: loss decreases", traj["host"][-1] < traj["host"][0],
            f"{traj['host'][0]:.6e} -> {traj['host'][-1]:.6e}")
    R.close("optimizer: host trajectory == jax trajectory", traj["host"], traj["jax"],
            atol=1e-6 + 1e-3 * abs(traj["host"][0] - traj["host"][-1]))
    return R.finish()


if __name__ == "__main__":
    sys.exit(main())
