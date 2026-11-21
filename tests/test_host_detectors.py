"""Host (NumPy, custom_vjp) detectors must match the original JAX detectors: values and reverse-mode gradients.

Comparisons are relative to each array's scale (atol = rtol * max|expected|) in addition to the elementwise
rtol: signal_detector sums two products that can nearly cancel, and NumPy and XLA may round such sums
differently in the last bit (e.g. XLA fusing a multiply-add), which an elementwise-only relative check would
flag as a large relative error. Every test uses its own fixed seed, so results do not depend on test order.
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "true")
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from differometor.components import (_power_detector_host, _power_detector_jax,
                                     _signal_detector_host, _signal_detector_jax)

jax.config.update("jax_enable_x64", True)


def crandn(rng, *shape):
    return jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def assert_close(actual, expected, rtol):
    expected = np.asarray(expected)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol * float(np.max(np.abs(expected))))


# (carrier shape, signal shape): benchmark layout (N,1) x (S,F), a plain vector pair, and a carrier sweep.
SHAPES = [((286, 1), (751, 101)), ((7,), (14,)), ((5, 3), (12, 3))]


@pytest.mark.parametrize("cs,ss", SHAPES)
def test_signal_detector(cs, ss):
    rng = np.random.default_rng(1)
    c, s = crandn(rng, *cs), crandn(rng, *ss)
    assert_close(_signal_detector_host(c, s), _signal_detector_jax(c, s), rtol=1e-14)
    out = _signal_detector_jax(c, s)
    w = jnp.asarray(rng.standard_normal(out.shape))

    def loss(fn):  # real scalar using both the real and imaginary parts and |.|, like the benchmark readout
        return lambda c, s: jnp.sum(w * jnp.real(fn(c, s))) + jnp.sum(jnp.imag(fn(c, s)) ** 2) + jnp.sum(jnp.log(jnp.abs(fn(c, s))))

    g_host = jax.jit(jax.grad(loss(_signal_detector_host), argnums=(0, 1)))(c, s)
    g_jax = jax.grad(loss(_signal_detector_jax), argnums=(0, 1))(c, s)
    for a, b in zip(g_host, g_jax):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert_close(a, b, rtol=1e-12)


@pytest.mark.parametrize("shape", [(286, 1), (9,)])
def test_power_detector(shape):
    rng = np.random.default_rng(2)
    z = crandn(rng, *shape)
    assert_close(_power_detector_host(z), _power_detector_jax(z), rtol=1e-14)
    w = jnp.asarray(rng.standard_normal(shape))
    g_host = jax.jit(jax.grad(lambda z: jnp.sum(w * jnp.log(_power_detector_host(z)))))(z)
    g_jax = jax.grad(lambda z: jnp.sum(w * jnp.log(_power_detector_jax(z))))(z)
    assert g_host.dtype == g_jax.dtype
    assert_close(g_host, g_jax, rtol=1e-12)


def test_vmap():
    rng = np.random.default_rng(3)
    c, s = crandn(rng, 4, 6, 1), crandn(rng, 4, 13, 2)
    assert_close(jax.vmap(_signal_detector_host)(c, s), jax.vmap(_signal_detector_jax)(c, s), rtol=1e-14)
