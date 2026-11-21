"""Other topologies (aLIGO, Voyager, carrier-swept cavity) must give the same results through the
default fast path as through the original path (dense assembly + `jnp.linalg.solve` + the full
19-way `lax.switch`)."""
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import differometor as df
from differometor.setups import aligo, voyager

sim = sys.modules["differometor.simulate"]
DATA = Path(__file__).resolve().parent.parent / "examples" / "data"


def _original_path(monkeypatch):
    monkeypatch.setattr(sim, "_DENSE", True)
    monkeypatch.setattr(sim, "solve", sim._dense_solve)
    monkeypatch.setattr(sim, "_DISPATCH_MODE", "full")


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
    frequencies = jnp.asarray(json.loads((DATA / "aligo_test_data.json").read_text())["frequencies"])[::10]
    return S, [("f", "frequency")], frequencies


@pytest.mark.parametrize("name", ["aligo", "voyager", "cavity"])
def test_matches_original_path(name, monkeypatch):
    S, pairs, values = {"aligo": lambda: _frequency_sweep(aligo), "voyager": lambda: _frequency_sweep(voyager), "cavity": _cavity}[name]()
    arrays, *_ = df.run_build_step(S, pairs, values)
    fast = df.simulate(**arrays)
    _original_path(monkeypatch)
    slow = df.simulate(**arrays)
    for new, old in zip(fast, slow):
        assert new.shape == old.shape
        if old.size:
            scale = float(jnp.max(jnp.abs(old)))
            # Voyager's system is ill-conditioned: dense vs sparse LU roundoff alone differs by ~6e-5.
            np.testing.assert_allclose(new, old, rtol=0, atol=2e-4 * scale)
