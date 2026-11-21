"""Cross-check the optimized backends against the ORIGINAL dense upstream solver.

`simulate_dense` is the untouched upstream algorithm (dense batched LU, no sparsity tricks), so
agreement here shows the sparse compression, host SuperLU path and custom adjoint did not change
the physics. Compared on the cavity, aLIGO, Voyager and random UIFOs (several sizes/seeds):

  * carrier / signal / noise values: host vs dense, jax-sparse vs dense
  * reverse-mode gradients wrt optimized parameters: host vs dense, jax-sparse vs dense

    python verification/verify_vs_dense.py [--quick]
"""
import argparse
import sys

from common import (Report, aligo, build_arrays, cavity, jax, jnp, np, public_frequencies,
                    run_backend, scalar_objective, uifo, voyager)

R = Report("host / jax-sparse vs upstream dense")


def cases(quick):
    f = jnp.asarray(public_frequencies(8 if quick else 16))
    out = {
        "cavity(tuning sweep)": (cavity(), [("m0", "tuning")], jnp.linspace(-180, 180, 30),
                                 [("m0", "reflectivity"), ("m1", "reflectivity"), ("l0", "power")]),
    }
    for name, factory in (("aligo", aligo), ("voyager", voyager)):
        S, opt = factory()
        out[f"{name}(freq sweep)"] = (S, [("f", "frequency")], f, opt)
    for size, seed in ((2, 0), (3, 1)) if quick else ((2, 0), (3, 1), (3, 7), (4, 2)):
        S, opt = uifo(size, random=True, random_seed=seed)
        out[f"uifo{size}(seed {seed})"] = (S, [("f", "frequency")], jnp.asarray(np.geomspace(5, 5000, 6)), opt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    for name, (setup, pairs, values, opt) in cases(args.quick).items():
        print(f"\n# {name}")
        dense, sparse, _ = build_arrays(setup, pairs, values, opt)
        x = jnp.asarray(dense["optimized_parameters"])

        ref = run_backend("dense", dense, sparse)
        for backend in ("host", "jax"):
            got = run_backend(backend, dense, sparse)
            for label, g, r in zip(("carrier", "signal", "noise"), got, ref):
                if r.size == 0:
                    continue
                scale = float(jnp.max(jnp.abs(r)))
                # Ill-conditioned optics (Voyager): two correct LU orderings differ ~1e-4 of the scale.
                R.close(f"{backend}.{label} == dense.{label}", g, r, atol=2e-4 * scale,
                        detail=f"(scale {scale:.2e})")

        if x.size == 0:
            continue
        grads = {}
        for backend in ("dense", "host", "jax"):
            grads[backend] = np.asarray(jax.grad(
                lambda v, b=backend: scalar_objective(run_backend(b, dense, sparse, v)))(x))
        floor = max(1e-6, float(np.max(np.abs(grads["dense"]))))
        for backend in ("host", "jax"):
            R.close(f"{backend}.gradient == dense.gradient ({x.size} params)", grads[backend], grads["dense"],
                    atol=1e-3 * floor, detail=f"(|g|max {floor:.2e})")
        R.check(f"{name}: gradient not trivially zero", np.max(np.abs(grads["dense"])) > 0)

    return R.finish()


if __name__ == "__main__":
    sys.exit(main())
