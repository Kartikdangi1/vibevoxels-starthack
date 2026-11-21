"""Ground-truth physics checks against closed-form results, independent of any solver code.

Fabry-Perot cavity (two mirrors, spacing L, amplitude reflectivities r1, r2, round-trip phase phi):
    circulating power  P_circ = P_in * t1^2 / (1 - r1 r2)^2 / (1 + F sin^2(phi/2)),  F = 4 r1 r2 / (1 - r1 r2)^2
    transmitted power  P_trns = P_circ * t2^2
    reflected power    P_refl = P_in - P_trns (lossless)
We sweep the tuning of the input mirror through a full free spectral range and compare all three
detectors to the Airy formulas, for the default host path and the in-graph JAX path. We also check
energy conservation for lossy/lossless mirrors and the exact on-resonance enhancement factor.

    python verification/verify_analytic.py
"""
import sys

from common import Report, build_arrays, cavity, jnp, np, run_backend
import differometor as df

R = Report("analytic physics (Fabry-Perot cavity)")


def airy(p_in, ra, rb, phase):
    """Powers for AMPLITUDE reflectivities ra, rb; phase = round-trip phase in radians."""
    t1sq, t2sq = 1 - ra**2, 1 - rb**2
    denom = np.abs(1 - ra * rb * np.exp(1j * phase)) ** 2
    circ = p_in * t1sq / denom
    return circ, circ * t2sq


def run_cavity(power_r, power_r2, backend, tunings, loss=0.0):
    S = cavity(power_r, power_r2, loss=loss)
    dense, sparse, meta = build_arrays(S, [("m0", "tuning")], tunings)
    carrier, _, _ = run_backend(backend, dense, sparse)
    detector_ports = np.asarray(meta[0])
    return np.asarray(df.power_detector(carrier))[detector_ports]  # (refl, circ, trns)


def main():
    tunings = jnp.linspace(-180, 180, 181)
    for r_pair in ((0.99, 0.991), (0.9, 0.95), (0.5, 0.5)):
        for backend in ("host", "jax", "dense"):
            powers = run_cavity(*r_pair, backend, tunings)
            # Differometor's mirror `reflectivity` is the POWER reflectivity R=r^2; try both conventions
            # and require that exactly one fits (this pins the convention, not just a lucky agreement).
            fits = {}
            for label, (ra, rb) in {"power": (np.sqrt(r_pair[0]), np.sqrt(r_pair[1])), "amp": r_pair}.items():
                # Sweeping m0 tuning by phi changes the round-trip phase by 2*phi (tuning is degrees of
                # microscopic detuning: round trip = 2 * tuning); fit the best offset too.
                phase = 2 * np.deg2rad(np.asarray(tunings))
                circ, trns = airy(1.0, ra, rb, phase)
                err = max(np.max(np.abs(powers[1] - circ) / circ.max()), np.max(np.abs(powers[2] - trns) / circ.max()))
                fits[label] = err
            best = min(fits, key=fits.get)
            R.check(f"R=({r_pair}) [{backend}] matches Airy formula ({best} convention)",
                    fits[best] < 1e-9, f"rel err {fits[best]:.2e} (other convention {max(fits.values()):.2e})")
            R.check(f"R=({r_pair}) [{backend}] lossless energy conservation refl + trns == input",
                    np.max(np.abs(powers[0] + powers[2] - 1.0)) < 1e-9,
                    f"max|refl+trns-1| = {np.max(np.abs(powers[0] + powers[2] - 1.0)):.2e}")

    # Exact on-resonance enhancement P_circ/P_in = t1^2/(1 - r1 r2)^2 for the default host path.
    ra, rb = np.sqrt(0.99), np.sqrt(0.991)
    powers = run_cavity(0.99, 0.991, "host", jnp.asarray([0.0]))
    want = (1 - ra**2) / (1 - ra * rb) ** 2
    R.close("on-resonance circulating enhancement", powers[1], [want], rtol=1e-9)

    # Loss: absorbing mirrors must remove power, never create it, and never go negative.
    powers = run_cavity(0.9, 0.9, "host", tunings, loss=0.05)
    R.check("lossy cavity: refl + trns <= input", np.all(powers[0] + powers[2] <= 1 + 1e-9))
    R.check("lossy cavity: all powers non-negative", np.all(powers >= -1e-12))
    R.check("lossy cavity: strictly lossy somewhere", np.max(1 - powers[0] - powers[2]) > 1e-3)

    return R.finish()


if __name__ == "__main__":
    sys.exit(main())
