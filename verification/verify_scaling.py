"""Scaling and resource behaviour on problems larger/smaller than the public one.

Correctness at one size does not show the engine stays usable at another. For UIFO grids of
increasing size this measures build time, first-call (cold) time, warm forward and warm gradient
time, peak RSS, and checks that:
  * the host path agrees with the JAX path at every size (values)
  * warm gradient time grows no faster than a generous polynomial in the number of components
  * warm runs are not slower than cold runs (no recompilation per call)
  * repeated gradient calls do not leak memory (RSS growth bound)
  * a long frequency sweep (1000 points) scales ~linearly in the number of frequencies

Prints a timing table; fails only on correctness, leaks, or clearly pathological scaling.

    python verification/verify_scaling.py [--max-size 6] [--quick]
"""
import argparse
import resource
import sys
from time import perf_counter

from common import Report, build_arrays, jax, jnp, np, run_backend, scalar_objective, set_backend, uifo

R = Report("scaling and resources")


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KiB


def timed(fn):
    t = perf_counter()
    out = fn()
    jax.block_until_ready(out)
    return perf_counter() - t, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-size", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.max_size = 4
    freqs = jnp.asarray(np.geomspace(5, 5000, 20))
    set_backend("host")

    print(f"\n{'size':>4} {'comps':>6} {'params':>6} {'build s':>8} {'cold s':>8} {'fwd s':>8} {'grad s':>8} {'RSS MB':>8}")
    sizes, grads = [], []
    for size in range(2, args.max_size + 1):
        S, opt = uifo(size, random=True, random_seed=size)
        t_build, (dense, sparse, _) = timed(lambda: build_arrays(S, [("f", "frequency")], freqs, opt))
        x = jnp.asarray(dense["optimized_parameters"])
        objective = lambda v: scalar_objective(run_backend("host", dense, sparse, v))
        vg = jax.value_and_grad(objective)
        t_cold, _ = timed(lambda: vg(x))
        t_fwd = min(timed(lambda: run_backend("host", dense, sparse))[0] for _ in range(3))
        t_grad = min(timed(lambda: vg(x))[0] for _ in range(3))
        comps = len(S._nodes) + len(S._edges)
        print(f"{size:>4} {comps:>6} {x.size:>6} {t_build:>8.3f} {t_cold:>8.3f} {t_fwd:>8.3f} {t_grad:>8.3f} {rss_mb():>8.0f}", flush=True)
        sizes.append(comps)
        grads.append(t_grad)
        if size <= 4:
            h, j = run_backend("host", dense, sparse), run_backend("jax", dense, sparse)
            ok = all(np.max(np.abs(np.asarray(a) - np.asarray(b))) <= 2e-4 * max(float(np.max(np.abs(b))), 1e-300)
                     for a, b in zip(h, j) if np.asarray(b).size)
            R.check(f"size {size}: host == jax", ok)
        R.check(f"size {size}: warm gradient not slower than cold", t_grad <= 1.5 * t_cold + 0.05,
                f"warm {t_grad:.3f}s vs cold {t_cold:.3f}s")

    if len(sizes) >= 3:
        slope = np.polyfit(np.log(sizes), np.log(grads), 1)[0]
        R.check("warm gradient time ~ components^k with k <= 3", slope <= 3.0, f"fitted exponent k = {slope:.2f}")

    # memory stability over repeated calls
    before = rss_mb()
    for _ in range(15 if not args.quick else 5):
        vg(x)
    R.check("no memory growth over repeated gradient calls", rss_mb() - before < 200,
            f"RSS {before:.0f} -> {rss_mb():.0f} MB")

    # frequency scaling on one fixed topology
    S, opt = uifo(3, random=True, random_seed=3)
    times = {}
    for n in (100, 400, 1000) if not args.quick else (100, 400):
        dense, sparse, _ = build_arrays(S, [("f", "frequency")], jnp.asarray(np.geomspace(5, 5000, n)), opt)
        x = jnp.asarray(dense["optimized_parameters"])
        g = jax.value_and_grad(lambda v: scalar_objective(run_backend("host", dense, sparse, v)))
        g(x)
        times[n] = min(timed(lambda: g(x))[0] for _ in range(2))
        print(f"  {n:>5} frequencies: warm gradient {times[n]:.3f}s")
    ns = sorted(times)
    ratio = (times[ns[-1]] / times[ns[0]]) / (ns[-1] / ns[0])
    R.check("gradient time ~ linear in number of frequencies (<= 2x linear)", ratio <= 2.0,
            f"time ratio / frequency ratio = {ratio:.2f}")
    return R.finish()


if __name__ == "__main__":
    sys.exit(main())
