"""The sparse signal/noise pipeline (shared LU, values-only assembly, w^H N w identity, per-site
dispatch) must reproduce the original dense implementation: values and gradients.

Run: python -m pytest tests -q   (needs the examples/data fixtures; uses 6 of the 101 frequencies)
"""
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces

sim = sys.modules["differometor.simulate"]  # `differometor.simulate` the name is the function
DATA = Path(__file__).resolve().parent.parent / "examples" / "data"
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]


@pytest.fixture(scope="module")
def problem():
    fixture = json.loads((DATA / "uifo_800_3000.json").read_text())
    reference = json.loads((DATA / "uifo_800_3000_test_data.json").read_text())
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    values = []
    for pair in pairs:
        name, prop = pair[0] if isinstance(pair[0], list) else pair
        values.append((fixture["nodes"] | fixture["edges"])[name]["properties"][prop])
    frequencies = jnp.asarray(reference["frequencies"])[::17][:6]
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    assert not np.any(np.asarray(arrays["noise_matrix"])), "sparse noise path assumes an empty base noise matrix"
    return jnp.asarray(values, dtype=jnp.float64), arrays, jnp.asarray(ports)


def _loss_and_grad(parameters, arrays, ports):
    def loss(p):
        carrier, signal, noise = df.simulate(**dict(arrays, optimized_parameters=p))
        detected = df.signal_detector(carrier, signal)[ports].squeeze()
        return jnp.mean(jnp.log10(noise / jnp.abs(detected))), (carrier, signal, noise)

    (value, outputs), gradient = jax.value_and_grad(loss, has_aux=True)(parameters)
    return value, outputs, gradient


def test_sparse_matches_dense_reference(problem, monkeypatch):
    parameters, arrays, ports = problem
    sparse = _loss_and_grad(parameters, arrays, ports)

    monkeypatch.setattr(sim, "_DENSE", True)
    monkeypatch.setattr(sim, "solve", sim._dense_solve)
    monkeypatch.setattr(sim, "_CARRIER_DISPATCH", (tuple(sim.FUNCTIONS), np.arange(len(sim.FUNCTIONS))))
    monkeypatch.setattr(sim, "_NOISE_DISPATCH", sim._CARRIER_DISPATCH)
    monkeypatch.setattr(sim, "_SIGNAL_DISPATCH", sim._CARRIER_DISPATCH)  # full 19-way switch, as originally
    dense = _loss_and_grad(parameters, arrays, ports)

    np.testing.assert_allclose(sparse[0], dense[0], rtol=1e-9)
    for new, old in zip(sparse[1], dense[1]):
        np.testing.assert_allclose(new, old, rtol=1e-6, atol=1e-12 * float(jnp.max(jnp.abs(old))))
    np.testing.assert_allclose(sparse[2], dense[2], rtol=1e-5, atol=1e-5 * float(jnp.max(jnp.abs(dense[2]))))


def test_sparse_solvers_self_check():
    from differometor.sparse_solve import _self_check

    _self_check()
