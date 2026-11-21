"""The default host solver must agree with the in-graph sparse JAX path (DIFFEROMETOR_BACKEND=jax),
values and reverse-mode gradients, on topologies beyond the benchmark's UIFO."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import differometor as df
from differometor.setups import aligo, voyager

DATA = Path(__file__).resolve().parent.parent / "examples" / "data"


def _cavity():
    S = df.Setup()
    S.add("laser", "l0", power=1)
    S.add("mirror", "m0", reflectivity=0.99, loss=0)
    S.add("mirror", "m1", reflectivity=0.991, loss=0)
    S.space("l0", "m0", length=1)
    S.space("m0", "m1", length=1)
    S.add("detector", "refl", target="m0", port="left", direction="out")
    S.add("detector", "circ", target="m1", port="left", direction="in")
    S.add("detector", "trns", target="m1", port="right", direction="out")
    return S, [("m0", "tuning")], jnp.linspace(-180, 180, 40)


def _frequency_sweep(factory):
    S, _ = factory()
    freqs = jnp.asarray(json.loads((DATA / "aligo_test_data.json").read_text())["frequencies"])[::10]
    return S, [("f", "frequency")], freqs


CASES = {"aligo": lambda: _frequency_sweep(aligo), "voyager": lambda: _frequency_sweep(voyager), "cavity": _cavity}


@pytest.mark.parametrize("name", list(CASES))
def test_host_matches_jax(name, monkeypatch):
    S, pairs, values = CASES[name]()
    arrays, *_ = df.run_build_step(S, pairs, values)

    def scalar(x, backend):
        monkeypatch.setenv("DIFFEROMETOR_BACKEND", backend)
        outs = df.simulate(**{**arrays, "optimized_parameters": x})
        # Position-weighted so the gradient is not ~0 (a plain sum over a full periodic sweep cancels).
        return sum(jnp.sum(jnp.real(o) * jnp.linspace(0.5, 1.5, o.size).reshape(o.shape)) for o in outs[:3] if o.size)

    x = arrays["optimized_parameters"]
    monkeypatch.setenv("DIFFEROMETOR_BACKEND", "host")
    host = df.simulate(**arrays)
    monkeypatch.setenv("DIFFEROMETOR_BACKEND", "jax")
    ref = df.simulate(**arrays)
    for new, old in zip(host, ref):
        assert new.shape == old.shape
        if old.size:
            scale = float(jnp.max(jnp.abs(old)))
            # Voyager is ill-conditioned: two correct sparse LUs differ by ~1e-4 of the scale.
            np.testing.assert_allclose(new, old, rtol=0, atol=2e-4 * scale)

    g_host = np.asarray(jax.grad(lambda v: scalar(v, "host"))(x))
    g_ref = np.asarray(jax.grad(lambda v: scalar(v, "jax"))(x))
    assert np.all(np.isfinite(g_host))
    # The cavity's lone optimized parameter has a ~1e-11 (round-off) gradient in both paths, so the floor
    # keeps that case a finiteness/agreement check; aLIGO and Voyager have O(1) gradients compared tightly.
    np.testing.assert_allclose(np.real(g_host), np.real(g_ref), rtol=0, atol=1e-3 * max(1e-6, np.max(np.abs(g_ref))))
