"""Finite-difference check for complex normal and adjoint factored solves."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.test_util import check_grads
from jax.scipy.linalg import lu_factor

from differometor.simulate import solve_factored


def main():
    key_a, key_b = jax.random.split(jax.random.PRNGKey(17))
    raw = jax.random.normal(key_a, (4, 4), dtype=jnp.complex128)
    matrix = raw + 5 * jnp.eye(4, dtype=jnp.complex128)
    rhs = jax.random.normal(key_b, (4,), dtype=jnp.complex128)

    for trans in (0, 2):
        def solve(a, b):
            factors = jax.tree_util.tree_map(
                jax.lax.stop_gradient, lu_factor(a)
            )
            return solve_factored(a, factors, b, trans=trans)

        direct_matrix = matrix if trans == 0 else jnp.conjugate(matrix.T)
        direct = jnp.linalg.solve(direct_matrix, rhs)
        actual = solve(matrix, rhs)
        if not jnp.allclose(actual, direct, rtol=1e-12, atol=1e-12):
            raise AssertionError(f"trans={trans}: solution disagrees with jnp.linalg.solve")
        check_grads(
            solve,
            (matrix, rhs),
            order=1,
            modes=("fwd", "rev"),
            eps=1e-5,
            atol=2e-5,
            rtol=2e-4,
        )
        print(f"PASS trans={trans}: solution and first derivatives")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
