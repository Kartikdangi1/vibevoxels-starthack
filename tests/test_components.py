"""`evaluate_components` (branch-free) must equal `FUNCTIONS[i](row)` for every function: values and
gradients, on random rows and on the real rows of the uifo fixture."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax, vmap

import differometor as df
from differometor.components import FUNCTIONS, evaluate_components
from differometor.setups import constrain_inter_grid_cell_spaces

DATA = Path(__file__).resolve().parent.parent / "examples" / "data"


def reference(function_indices, inputs):
    return vmap(lambda i, x: lax.switch(i, FUNCTIONS, x))(function_indices, inputs.astype(jnp.complex128))


def random_rows(rng, n):
    cols = [rng.uniform(0.1, 0.9, n) for _ in range(4)] + [rng.uniform(1, 1.5, n), rng.uniform(1, 1.5, n), rng.uniform(0, 30, n)]
    return jnp.asarray(np.stack(cols, axis=1))


@pytest.mark.parametrize("function_index", range(len(FUNCTIONS)))
def test_each_function_random_rows(function_index):
    rng = np.random.default_rng(function_index)
    rows = random_rows(rng, 16)
    ids = jnp.full((16,), function_index)
    np.testing.assert_allclose(evaluate_components(ids, rows), reference(ids, rows), rtol=1e-12, atol=1e-300)

    weights = jnp.asarray(rng.normal(size=(16, 4)) + 1j * rng.normal(size=(16, 4)))
    loss = lambda f: lambda x: jnp.sum(jnp.abs(f(ids, x) * weights) ** 2)
    g_new, g_ref = jax.grad(loss(evaluate_components))(rows), jax.grad(loss(reference))(rows)
    assert np.all(np.isfinite(g_new))
    np.testing.assert_allclose(g_new, jnp.real(g_ref), rtol=1e-9, atol=1e-9 * float(jnp.max(jnp.abs(g_ref))) + 1e-300)


def test_mixed_rows_on_fixture_have_finite_matching_gradients():
    fixture = json.loads((DATA / "uifo_800_3000.json").read_text())
    reference_data = json.loads((DATA / "uifo_800_3000_test_data.json").read_text())
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"])
    frequencies = jnp.asarray(reference_data["frequencies"])[::25]
    arrays, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    parameters = jnp.real(arrays["parameters"])[0]
    for prefix in ("carrier", "signal", "noise"):
        ids = arrays[f"{prefix}_function_indices"]
        idx = arrays[f"{prefix}_function_input_indices"]
        rows = parameters[idx]
        np.testing.assert_allclose(evaluate_components(ids, rows), reference(ids, rows), rtol=1e-11, atol=1e-14)
        weights = jnp.cos(jnp.arange(rows.shape[0] * 4).reshape(-1, 4) * 0.37)
        f = lambda fn: lambda p: jnp.sum(jnp.real(fn(ids, p[idx])) * weights)
        g_new, g_ref = jax.grad(f(evaluate_components))(parameters), jax.grad(f(reference))(parameters)
        assert np.all(np.isfinite(g_new)), prefix
        np.testing.assert_allclose(g_new, g_ref, rtol=1e-8, atol=1e-8 * float(jnp.max(jnp.abs(g_ref))))
