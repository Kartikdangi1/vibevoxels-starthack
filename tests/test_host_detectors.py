"""Host (NumPy, custom_vjp) detectors must match the original JAX detectors: values and reverse-mode gradients."""
import os

os.environ.setdefault("JAX_ENABLE_X64", "true")
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from differometor.components import (_power_detector_host, _power_detector_jax,
                                     _signal_detector_host, _signal_detector_jax)

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(0)


def crandn(*shape):
    return jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


# (carrier shape, signal shape): benchmark layout (N,1) x (S,F), a plain vector pair, and a carrier sweep.
SHAPES = [((286, 1), (751, 101)), ((7,), (14,)), ((5, 3), (12, 3))]


@pytest.mark.parametrize("cs,ss", SHAPES)
def test_signal_detector(cs, ss):
    c, s = crandn(*cs), crandn(*ss)
    np.testing.assert_allclose(_signal_detector_host(c, s), _signal_detector_jax(c, s), rtol=1e-14, atol=0)
    out = _signal_detector_jax(c, s)
    w = jnp.asarray(rng.standard_normal(out.shape))

    def loss(fn):  # real scalar using both the real and imaginary parts and |.|, like the benchmark readout
        return lambda c, s: jnp.sum(w * jnp.real(fn(c, s))) + jnp.sum(jnp.imag(fn(c, s)) ** 2) + jnp.sum(jnp.log(jnp.abs(fn(c, s))))

    g_host = jax.jit(jax.grad(loss(_signal_detector_host), argnums=(0, 1)))(c, s)
    g_jax = jax.grad(loss(_signal_detector_jax), argnums=(0, 1))(c, s)
    for a, b in zip(g_host, g_jax):
        assert a.shape == b.shape and a.dtype == b.dtype
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * float(jnp.max(jnp.abs(b))))


@pytest.mark.parametrize("shape", [(286, 1), (9,)])
def test_power_detector(shape):
    z = crandn(*shape)
    np.testing.assert_allclose(_power_detector_host(z), _power_detector_jax(z), rtol=1e-14, atol=0)
    w = jnp.asarray(rng.standard_normal(shape))
    g_host = jax.jit(jax.grad(lambda z: jnp.sum(w * jnp.log(_power_detector_host(z)))))(z)
    g_jax = jax.grad(lambda z: jnp.sum(w * jnp.log(_power_detector_jax(z))))(z)
    assert g_host.dtype == g_jax.dtype
    np.testing.assert_allclose(g_host, g_jax, rtol=1e-12, atol=0)


def test_vmap():
    c, s = crandn(4, 6, 1), crandn(4, 13, 2)
    np.testing.assert_allclose(jax.vmap(_signal_detector_host)(c, s), jax.vmap(_signal_detector_jax)(c, s), rtol=1e-14)
