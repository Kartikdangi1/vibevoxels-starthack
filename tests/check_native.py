"""Native kernels vs JAX: values and VJPs (JAX complex convention) for all 19 component functions.

Inputs: genuinely complex (exercises the conj/anti-holomorphic terms), real physical-range,
negative frequencies, and zero cotangents. Also a real-direction finite-difference check.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "true")
import jax, jax.numpy as jnp, numpy as np
from differometor.components import FUNCTIONS
from differometor.native import native_function

rng = np.random.default_rng(0)
B = 64


def inputs(kind, name):
    if kind == "complex":
        return rng.uniform(0.2, 2.0, (B, 7)) + 1j * rng.uniform(-0.5, 0.5, (B, 7))
    x = rng.uniform(0.05, 0.95, (B, 7))  # loss/reflectivity-like
    if kind == "physical":
        x[:, 0] = rng.choice([-1, 1], B) * 10 ** rng.uniform(0, 4, B)  # +/- frequency (Hz)
        x[:, 1] = rng.uniform(1, 4000, B) if "space" in name else rng.uniform(-90, 90, B)
        # refractive indices in [1, 1.2] and |alpha| <= 45 deg keep arcsin away from its branch point
        if name in ("surface",):
            x[:, 2] = rng.uniform(0, 90, B); x[:, 3] = rng.uniform(-5e3, 5e3, B)
            x[:, 4] = rng.uniform(1, 1.2, B); x[:, 5] = rng.uniform(1, 1.2, B); x[:, 6] = rng.uniform(-45, 45, B)
        if "optomechanical" in name:
            x[:, 1] = rng.uniform(0, 90, B); x[:, 4] = rng.uniform(1, 1.2, B)
            x[:, 5] = rng.uniform(1, 1.2, B); x[:, 6] = rng.uniform(-45, 45, B)
        if name.startswith("force_calculation"):
            x[:, 0] = rng.uniform(-60, 60, B); x[:, 1] = rng.uniform(1, 1.2, B); x[:, 2] = rng.uniform(1, 1.2, B)
        if name == "susceptibility":
            x[:, 1] = rng.uniform(0.01, 200, B)
        if name == "squeezer":
            x[:, 0] = rng.uniform(0.01, 20, B); x[:, 1] = rng.uniform(-180, 180, B)
    if kind == "zero_alpha":  # boundary-adjacent: alpha = 0, tuning = 0, unit refractive indices
        alpha = {"surface": 6, "corrected_optomechanical_phase_left": 5,
                 "corrected_optomechanical_phase_right": 6}.get(name, 0 if name.startswith("force") else None)
        if alpha is not None:
            x[:, alpha] = 0.0
        if name == "surface":
            x[:, 2] = 0.0; x[:, 4] = 1.0; x[:, 5] = 1.0
        if "optomechanical" in name:
            x[:, 1] = 0.0; x[:, 4] = 1.0; x[:, 5] = 1.0
        if name.startswith("force_calculation"):
            x[:, 1] = 1.0; x[:, 2] = 1.0
    return (x + 0j).astype(np.complex128)


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    scale = np.maximum(np.abs(b), 1e-300 + np.abs(b).max() * 1e-12)
    err = np.where(np.isnan(a) & np.isnan(b), 0, np.abs(a - b) / scale)
    return float(np.nanmax(err)) if np.all(np.isfinite(err) | np.isnan(b)) else np.inf


worst = 0.0
for fn in FUNCTIONS:
    nf = native_function(fn)
    assert nf is not None, fn.__name__
    for kind in ("complex", "physical", "zero_alpha"):
        x = inputs(kind, fn.__name__)
        yref, pb = jax.vjp(jax.vmap(fn), jnp.asarray(x))
        y = nf(x)
        for zero_ct in (False, True):
            ct = np.zeros(y.shape, complex) if zero_ct else rng.normal(size=y.shape) + 1j * rng.normal(size=y.shape)
            gref = np.asarray(pb(jnp.asarray(ct))[0])
            g = nf.vjp(x, ct)
            ey = rel(y, np.asarray(yref))
            # d/d(frequency) of space_modulation is (k f cos(k f) - sin(k f)) / f^2 with k f ~ 1e-8 for
            # f ~ 1 Hz: catastrophic cancellation in any evaluation order (JAX's included). Frequency is
            # a swept, never-optimised input; its derivative is checked separately at a looser bound.
            if fn.__name__.startswith("space_modulation"):
                eg = max(rel(g[:, 1:], gref[:, 1:]), rel(g[:, :1], gref[:, :1]) * 1e-4)
            else:
                eg = rel(g, gref)
            worst = max(worst, ey, eg)
            if ey > 1e-12 or eg > 1e-10:
                print(f"FAIL {fn.__name__:40s} {kind:10s} zero_ct={zero_ct} fwd {ey:.1e} vjp {eg:.1e}")
        # real-direction central finite difference of Re<ct, y> (independent of JAX)
        if kind == "physical":
            ct = rng.normal(size=y.shape) + 1j * rng.normal(size=y.shape)
            g = nf.vjp(x, ct)
            for i in range(7):
                h = 1e-6 * np.maximum(np.abs(x[:, i].real), 1e-3)
                e = np.zeros((B, 7)); e[:, i] = h
                fd = (np.real(np.sum(ct * nf(x + e), 1)) - np.real(np.sum(ct * nf(x - e), 1))) / (2 * h)
                # FD roundoff floor: eps * |Re<ct, y>| / h
                floor = 1e-15 * np.abs(np.sum(np.abs(ct * nf(x)), 1)) / h
                efd = (np.abs(fd - np.real(g[:, i])) / (np.abs(fd) + np.abs(np.real(g[:, i])) + 1e3 * floor + 1e-300)).max()
                if efd > 1e-3:  # FD truncation on phases up to ~800 rad reaches ~1e-4
                    print(f"FD FAIL {fn.__name__} input {i}: {efd:.1e}")
    print(f"{fn.__name__:40s} ok")
print("WORST rel (fwd/vjp vs JAX):", worst)
