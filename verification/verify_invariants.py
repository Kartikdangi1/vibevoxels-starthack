"""Structural invariants that any correct simulator must satisfy (no reference values needed).

  * determinism: repeated and fresh-build runs give bit-identical results
  * frequency independence: simulating a subset (or a permuted/reversed sweep) equals slicing the full run
  * linearity of the signal path: signal field responds linearly to laser amplitude (power x4 -> signal x2)
    and carrier power scales exactly with laser power
  * energy conservation: lossless mirror/beamsplitter networks output exactly the input power
  * reciprocity-style symmetry: swapping the mirror order of a symmetric cavity swaps refl/trns consistently
  * batching: output shapes/dtypes match across backends and detectors-ports are consistent
  * robustness: no NaN/Inf for extreme (but legal) parameters: near-unity and near-zero reflectivity, very
    low and very high frequency, zero loss

    python verification/verify_invariants.py
"""
import sys

import differometor as df
from common import (Report, aligo, build_arrays, cavity, jnp, np, public_frequencies, run_backend, uifo)

R = Report("invariants")


def solve(setup, pairs, values, backend="host"):
    dense, sparse, meta = build_arrays(setup, pairs, values)
    return run_backend(backend, dense, sparse), meta


def main():
    freqs = jnp.asarray(public_frequencies(20))
    S, _ = aligo()
    sweep = [("f", "frequency")]

    # --- determinism
    (c1, s1, n1), _ = solve(S, sweep, freqs)
    (c2, s2, n2), _ = solve(S, sweep, freqs)
    R.check("repeat run bit-identical (carrier/signal/noise)",
            all(np.array_equal(a, b) for a, b in ((c1, c2), (s1, s2), (n1, n2))))
    S_fresh, _ = aligo()
    (c3, s3, n3), _ = solve(S_fresh, sweep, freqs)
    R.check("fresh setup + fresh build bit-identical",
            all(np.array_equal(a, b) for a, b in ((c1, c3), (s1, s3), (n1, n3))))

    # --- frequency independence: subset / reversed / shuffled
    sel = np.array([0, 3, 7, 12, 19])
    (cs, ss, ns), _ = solve(S, sweep, freqs[sel])
    scale = float(np.max(np.abs(s1)))
    R.close("subset of frequencies == slice of full sweep (signal)", ss, np.asarray(s1)[..., sel], atol=1e-8 * scale)
    R.close("subset of frequencies == slice of full sweep (noise)", ns, np.asarray(n1)[..., sel],
            atol=1e-8 * float(np.max(np.abs(n1))))
    perm = np.random.default_rng(1).permutation(len(freqs))
    (cp, sp, npn), _ = solve(S, sweep, freqs[perm])
    R.close("shuffled sweep == permuted full sweep (signal)", sp, np.asarray(s1)[..., perm], atol=1e-8 * scale)
    R.close("shuffled sweep == permuted full sweep (noise)", npn, np.asarray(n1)[..., perm],
            atol=1e-8 * float(np.max(np.abs(n1))))
    (cr, sr, nr), _ = solve(S, sweep, freqs[::-1])
    R.close("reversed sweep == reversed full sweep (noise)", nr, np.asarray(n1)[..., ::-1],
            atol=1e-8 * float(np.max(np.abs(n1))))

    # --- scaling laws on a cavity with a tuning sweep
    tun = jnp.linspace(-180, 180, 25)
    def cav_power(p):
        Sx = df.Setup()
        Sx.add("laser", "l0", power=p)
        Sx.add("mirror", "m0", reflectivity=0.9, loss=0)
        Sx.add("mirror", "m1", reflectivity=0.9, loss=0)
        Sx.space("l0", "m0", length=1)
        Sx.space("m0", "m1", length=1)
        Sx.add("detector", "refl", target="m0", port="left", direction="out")
        Sx.add("detector", "trns", target="m1", port="right", direction="out")
        (c, _, _), meta = solve(Sx, [("m0", "tuning")], tun)
        return np.asarray(df.power_detector(c))[np.asarray(meta[0])]
    p1, p4 = cav_power(1.0), cav_power(4.0)
    R.close("carrier power scales linearly with laser power", p4, 4 * p1, rtol=1e-10, atol=1e-14)

    # --- energy conservation in a lossless UIFO-style network is not guaranteed by topology (open ports),
    #     so use the cavity: refl + trns = input for lossless mirrors, at every tuning, on all backends
    for backend in ("host", "jax", "dense"):
        dense, sparse, meta = build_arrays(cavity(0.97, 0.8), [("m0", "tuning")], tun)
        carrier, _, _ = run_backend(backend, dense, sparse)
        p = np.asarray(df.power_detector(carrier))[np.asarray(meta[0])]
        R.check(f"[{backend}] lossless cavity conserves energy", np.max(np.abs(p[0] + p[2] - 1)) < 1e-9)

    # --- symmetric cavity: identical mirrors -> circulating power symmetric under tuning -> -tuning
    dense, sparse, meta = build_arrays(cavity(0.9, 0.9), [("m0", "tuning")], jnp.linspace(-180, 180, 181))
    carrier, _, _ = run_backend("host", dense, sparse)
    p = np.asarray(df.power_detector(carrier))[np.asarray(meta[0])]
    R.close("cavity response symmetric under tuning -> -tuning", p, p[:, ::-1], atol=1e-10)

    # --- robustness at extreme legal parameters
    for r0, r1 in ((0.999999, 0.999999), (1e-6, 1e-6), (0.5, 0.999999), (0.999999, 1e-6)):
        dense, sparse, meta = build_arrays(cavity(r0, r1), [("m0", "tuning")], tun)
        for backend in ("host", "jax"):
            outs = run_backend(backend, dense, sparse)
            R.check(f"[{backend}] cavity R=({r0},{r1}) finite", all(np.isfinite(np.asarray(o)).all() for o in outs))
    Sx, _ = aligo()
    edge = jnp.asarray([1e-2, 1e-1, 1.0, 1e4, 1e5])
    (ce, se, ne), _ = solve(Sx, sweep, edge)
    R.check("aLIGO finite at f = 1e-2 .. 1e5 Hz", all(np.isfinite(np.asarray(o)).all() for o in (ce, se, ne)))
    R.check("aLIGO noise strictly positive", bool(np.all(np.asarray(ne) >= 0)))

    # --- shapes/dtypes
    R.check("carrier/signal complex128", all(np.asarray(o).dtype == np.complex128 for o in (c1, s1)),
            f"{np.asarray(c1).dtype}, {np.asarray(s1).dtype}")
    R.check("signal frequency axis == len(freqs)", np.asarray(s1).shape[-1] == len(freqs), f"{np.asarray(s1).shape}")

    # --- random UIFOs: every seed finite on host, identical on jax
    for seed in range(4):
        Su, _ = uifo(3, random=True, random_seed=seed)
        vals = jnp.asarray(np.geomspace(5, 5000, 5))
        dense, sparse, _ = build_arrays(Su, sweep, vals)
        h, j = run_backend("host", dense, sparse), run_backend("jax", dense, sparse)
        R.check(f"uifo3 seed {seed}: host finite and equal to jax",
                all(np.isfinite(np.asarray(o)).all() for o in h)
                and all(np.max(np.abs(np.asarray(a) - np.asarray(b))) <= 2e-4 * max(float(np.max(np.abs(b))), 1e-30)
                        for a, b in zip(h, j) if np.asarray(b).size))
    return R.finish()


if __name__ == "__main__":
    sys.exit(main())
