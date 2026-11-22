"""Native kernels: the forward with cached partials must give exactly the plain values, and its VJP
exactly the recomputed (pruned) VJP. Exact equality is intended: every dual-number operation computes
its value with the same NumPy expression, in the same operand order, as the plain evaluation."""
import itertools

import numpy as np
import pytest

from differometor.native_kernels import KERNELS, NativeFunction

rng = np.random.default_rng(7)
NAMES = sorted(KERNELS, key=lambda f: f.__name__)


def _inputs(batch, real):
    x = rng.uniform(0.2, 1.8, (batch, 7)) * rng.choice([-1, 1], (batch, 7))
    if not real:
        x = x + 1j * rng.uniform(-0.5, 0.5, (batch, 7))
    return x.astype(np.complex128)


@pytest.mark.parametrize("function", NAMES, ids=lambda f: f.__name__)
@pytest.mark.parametrize("real", [True, False])
def test_cached_partials_are_exact(function, real):
    nf = NativeFunction(KERNELS[function])
    x = _inputs(37, real)
    ct = rng.normal(size=(37, 4)) + 1j * rng.normal(size=(37, 4))
    plain = nf(x)
    masks = [np.ones(7, bool)] + [np.array(m, bool) for m in itertools.islice(itertools.product([0, 1], repeat=7), 1, None, 9)]
    for columns in masks:
        y, cache = nf.forward_with_cache(x, columns)
        assert np.array_equal(y, plain, equal_nan=True), "dual-number forward values differ from the plain ones"
        g_cached = nf.vjp_from_cache(cache, ct, columns)
        g_recomputed = nf.vjp(x, ct, columns)
        assert np.array_equal(g_cached, g_recomputed, equal_nan=True)
        assert not np.any(g_cached[:, ~columns]), "unselected columns must get zero cotangent"
