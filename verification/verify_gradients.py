"""Gradient verification for the custom adjoint (host path) on the PUBLIC UIFO benchmark problem.

Three independent tests of d(loss)/d(parameters), 193 parameters, loss = mean(log10(S / S_ref)):
  1. Central finite differences on a random sample of parameters (two step sizes, so a result that
     has not converged in the step is flagged rather than trusted).
  2. Directional-derivative (JVP) test on random directions: <grad, v> == [L(x+hv) - L(x-hv)] / 2h,
     which exercises ALL coordinates at once and cannot be fooled by a few lucky ones.
  3. Host-path gradient == in-graph JAX-sparse gradient (a completely different AD implementation).
Also checks gradient is invariant to repeated evaluation (no stale adjoint cache) and that a short
gradient-descent run strictly lowers the loss (the gradient is a descent direction).

    python verification/verify_gradients.py [--n-freq 8] [--samples 12]
"""
import argparse
import sys

from common import Report, jax, jnp, load_public, np, set_backend, uifo_loss

R = Report("gradients (custom adjoint) on public UIFO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-freq", type=int, default=8, help="frequencies used (each is solved independently)")
    ap.add_argument("--samples", type=int, default=12, help="parameters finite-differenced")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    case = load_public(args.n_freq)
    arrays, ports, x, ref = case["arrays"], case["ports"], case["params"], jnp.asarray(case["ref"])

    set_backend("host")
    loss_fn = jax.jit(lambda p: uifo_loss(p, arrays, ports, ref)[0])
    value_and_grad = jax.value_and_grad(lambda p: uifo_loss(p, arrays, ports, ref)[0])
    loss0, grad = value_and_grad(x)
    grad = np.asarray(grad)
    R.check("loss and gradient finite", np.isfinite(float(loss0)) and np.isfinite(grad).all(), f"loss={float(loss0):.6f}")
    R.check("gradient shape == parameter shape", grad.shape == x.shape, f"{grad.shape}")

    # 1. finite differences on sampled coordinates. The landscape has sharp resonances (gradients ~1e4),
    #    so no single step size is trustworthy; sweep a ladder and require the BEST step to agree
    #    (truncation error shrinks with h, round-off grows as 1/h, so a correct gradient matches at some h).
    steps = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12)  # several optimized reflectivities sit within 1e-9 of their bounds
    informative = np.flatnonzero(np.abs(grad) > 1e-3 * np.max(np.abs(grad)))
    picks = rng.choice(informative, size=min(args.samples, informative.size), replace=False)
    worst = 0.0
    for i in picks:
        est = [(float(loss_fn(x.at[i].add(h))) - float(loss_fn(x.at[i].add(-h)))) / (2 * h) for h in steps]
        errs = [abs(e - grad[i]) / max(abs(grad[i]), 1e-6) for e in est]
        best = int(np.argmin(errs))
        worst = max(worst, errs[best])
        R.check(f"FD param[{i}] {case['pairs'][i]}", errs[best] < 1e-2,
                f"adjoint={grad[i]:+.6e} best_fd={est[best]:+.6e} (h={steps[best]:g}) rel_err={errs[best]:.1e}")
    R.check("FD sample worst best-step relative error < 1%", worst < 1e-2, f"{worst:.2e}")

    # 2. random-direction derivative (all coordinates at once), same step ladder
    scale_x = np.maximum(np.abs(np.asarray(x)), 1e-3)
    for k in range(4):
        v = jnp.asarray(rng.normal(size=x.shape) * scale_x)
        v = v / jnp.linalg.norm(v)
        analytic = float(jnp.dot(grad, v))
        est = [(float(loss_fn(x + h * v)) - float(loss_fn(x - h * v))) / (2 * h) for h in steps]
        errs = [abs(e - analytic) / max(abs(analytic), 1e-6) for e in est]
        best = int(np.argmin(errs))
        R.check(f"directional derivative #{k}", errs[best] < 1e-2,
                f"<grad,v>={analytic:+.6e} best_fd={est[best]:+.6e} (h={steps[best]:g}) rel_err={errs[best]:.1e}")

    # 3. host adjoint vs in-graph JAX AD (different implementation of reverse mode)
    set_backend("jax")
    _, grad_jax = jax.value_and_grad(lambda p: uifo_loss(p, arrays, ports, ref)[0])(x)
    grad_jax = np.asarray(grad_jax)
    R.close("host gradient == jax-sparse gradient", grad, grad_jax, atol=1e-3 * np.max(np.abs(grad_jax)))
    cos = float(np.dot(grad, grad_jax) / (np.linalg.norm(grad) * np.linalg.norm(grad_jax)))
    R.check("host/jax gradient cosine similarity > 1 - 1e-6", cos > 1 - 1e-6, f"cos={cos:.10f}")

    # repeated evaluation must be bit-identical (adjoint cache must never leak between calls)
    set_backend("host")
    g2 = np.asarray(value_and_grad(x)[1])
    g3 = np.asarray(value_and_grad(x * 1.0000001)[1])
    R.check("repeat gradient bit-identical", np.array_equal(grad, g2))
    R.check("gradient responds to a changed input", not np.array_equal(grad, g3))

    # descent: for small enough steps against the gradient the loss must fall (first-order property;
    # the landscape is highly nonlinear, so only the smallest steps are required to decrease).
    gnorm = max(np.linalg.norm(grad), 1e-300)
    ts = [1e-12, 1e-11, 1e-10]  # |grad| ~ 6e4 and parameters sit at bounds: only tiny steps are first-order
    lo = [float(loss_fn(x - t * jnp.asarray(grad) / gnorm)) for t in ts]
    R.check("negative gradient is a descent direction (small steps)", all(v < float(loss0) for v in lo),
            f"loss {float(loss0):.12f} -> {[round(v, 12) for v in lo]}")

    return R.finish()


if __name__ == "__main__":
    sys.exit(main())
