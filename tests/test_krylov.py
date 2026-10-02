"""Residual certification and flexible preconditioning regression tests."""

import os
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from jaxhps._krylov import fgmres

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "examples")
    ),
)
from wave_scattering_utils_3D import _gmres_python_loop  # noqa: E402


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_gmres_certifies_unpreconditioned_residual(scale):
    A = jnp.diag(jnp.array([1.0, 2.0, 3.0], dtype=jnp.complex128))
    b = jnp.ones(3, dtype=jnp.complex128)
    stats = {}
    x, info = _gmres_python_loop(
        lambda v: A @ v,
        b,
        M=lambda v: scale * v,
        tol=1e-10,
        restart=3,
        maxiter=6,
        stats=stats,
    )
    rel_res = float(jnp.linalg.norm(b - A @ x) / jnp.linalg.norm(b))
    assert info == 0
    assert rel_res <= 1e-10
    assert stats["final_rel_res"] == pytest.approx(rel_res, abs=1e-14)
    assert stats["true_res_history"][0] == 1.0


def test_gmres_rejects_singular_preconditioner():
    b = jnp.ones(3, dtype=jnp.complex128)
    _, info = _gmres_python_loop(
        lambda v: v,
        b,
        M=jnp.zeros_like,
        restart=3,
        maxiter=3,
    )
    assert info != 0


@pytest.mark.parametrize("n_rhs", [1, 2])
def test_fgmres_variable_preconditioner(n_rhs):
    rng = np.random.default_rng(4)
    A = jnp.asarray(
        rng.normal(size=(10, 10))
        + 1j * rng.normal(size=(10, 10))
        + 10 * np.eye(10)
    )
    b = jnp.asarray(
        rng.normal(size=(10, n_rhs)) + 1j * rng.normal(size=(10, n_rhs))
    )
    if n_rhs == 1:
        b = b[:, 0]
    calls = 0

    def variable(v):
        nonlocal calls
        calls += 1
        return v * (1.0 if calls % 2 else 0.02)

    stats = {}
    x, info = fgmres(
        lambda v: A @ v,
        b,
        precond=variable,
        restart=10,
        maxiter=30,
        tol=1e-11,
        stats=stats,
    )
    assert info == 0
    np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-9, atol=1e-11)
    assert stats["n_precond"] == calls == stats["n_iter"]
    assert stats["final_rel_res"] <= 1e-11


def test_fgmres_honors_partial_restart_budget():
    A = jnp.diag(jnp.linspace(1, 30, 30).astype(jnp.complex128))
    b = jnp.ones(30, dtype=jnp.complex128)
    stats = {}
    x, info = fgmres(
        lambda v: A @ v, b, restart=3, maxiter=5, tol=1e-14, stats=stats
    )
    assert info != 0
    assert stats["n_iter"] == 5
    assert stats["n_cycles"] == 2
    assert stats["n_matvec"] == 8
    assert stats["final_rel_res"] == pytest.approx(
        float(jnp.linalg.norm(b - A @ x) / jnp.linalg.norm(b))
    )


@pytest.mark.parametrize("zero_rhs", [True, False])
def test_fgmres_zero_rhs_and_breakdown(zero_rhs):
    b = jnp.zeros(4) if zero_rhs else jnp.ones(4)
    x, info = fgmres(lambda v: v, b, precond=jnp.zeros_like)
    assert (info == 0) == zero_rhs
    np.testing.assert_array_equal(x, np.zeros(4))
