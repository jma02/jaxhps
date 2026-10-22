import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from fft_volume_3d import (
    evaluate_volume_field,
    prepare_volume_solver,
    solve_volume,
)
from phantom_inverse_3d import (
    BackgroundBorn,
    crop,
    make_background_born,
    make_forward,
    objective,
    sphere,
)

jax.config.update("jax_enable_x64", True)


def test_matches_existing_forward_solver():
    n, a, kappa = 8, 0.2, 10.0

    def coefficient(p):
        return -0.04 * np.exp(-np.sum(p**2, -1) / 0.01)

    points, b, _, operator = prepare_volume_solver(n, a, kappa, coefficient)
    direction = sphere(1)
    receivers = 0.36 * sphere(5)
    forward = make_forward(points, kappa, direction, receivers, tol=1e-10)
    observed, residual = forward(b)
    total, stats = solve_volume(
        operator, points, kappa, direction[0], tol=1e-10
    )
    reference = evaluate_volume_field(
        jnp.asarray(receivers),
        jnp.asarray(points),
        b * total,
        kappa,
        2 * a / n,
    )
    assert stats["info"] == 0
    assert float(residual.max()) < 1e-9
    np.testing.assert_allclose(observed[0], reference, rtol=1e-8, atol=1e-11)


def test_implicit_gradient_matches_finite_difference():
    points, window, truth = crop(8)
    forward = make_forward(
        points, 15.0, sphere(2), 0.36 * sphere(7), tol=1e-11
    )
    observed, _ = forward(jnp.asarray(truth))
    rng = np.random.default_rng(2)
    theta = jnp.asarray(rng.normal(0, 0.1, truth.size))
    direction = jnp.asarray(rng.normal(size=truth.size))
    direction /= jnp.linalg.norm(direction)

    def loss(x):
        return objective(x, forward, observed, jnp.asarray(window), 0.001)[0]

    analytic = float(jnp.vdot(jax.grad(loss)(theta), direction))
    scaled_gradient = jax.grad(lambda x: 1e-10 * loss(x))(theta) / 1e-10
    np.testing.assert_allclose(
        scaled_gradient, jax.grad(loss)(theta), rtol=1e-5, atol=1e-9
    )
    epsilon = 1e-4
    numerical = float(
        (loss(theta + epsilon * direction) - loss(theta - epsilon * direction))
        / (2 * epsilon)
    )
    np.testing.assert_allclose(analytic, numerical, rtol=1e-4, atol=1e-7)


def test_born_operator_matches_forward_linearization():
    from phantom_dataset_learn_3d import make_born

    n = 6
    axis = -0.2 + (np.arange(n) + 0.5) * 0.4 / n
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)
    directions = sphere(3)
    receivers = 0.36 * sphere(7, 0.3)
    forward = make_forward(points, 9.0, directions, receivers, 1e-10)
    linear, _ = make_born(points, 9.0, directions, receivers, 1, 0.0)
    b = jnp.asarray(np.random.default_rng(1).random((n, n, n)))
    _, tangent = jax.jvp(
        lambda x: forward(x)[0], (jnp.zeros((n, n, n)),), (b,)
    )
    assert jnp.linalg.norm(tangent - linear(b)) < 1e-6 * jnp.linalg.norm(
        tangent
    )


@pytest.mark.parametrize("double", [False, True])
def test_cached_background_born_matches_jvp_and_adjoint(double):
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", double)
    try:
        points, _, _ = crop(8)
        directions, receivers = sphere(3), 0.36 * sphere(7, 0.3)
        background = jnp.asarray(-0.06 * np.exp(-np.sum(points**2, -1) / 0.02))
        tol = 1e-10 if double else 1e-6
        forward = make_forward(points, 15.0, directions, receivers, tol)
        linear = make_background_born(
            points, 15.0, directions, receivers, background, tol
        )
        rng = np.random.default_rng(7)
        delta = jnp.asarray(rng.normal(size=background.shape))
        _, tangent = jax.jvp(lambda b: forward(b)[0], (background,), (delta,))
        error = jnp.linalg.norm(linear(delta) - tangent) / jnp.linalg.norm(
            tangent
        )
        assert float(error) < 20 * tol
        data = jnp.asarray(
            rng.normal(size=tangent.shape)
            + 1j * rng.normal(size=tangent.shape)
        )
        reference_adjoint = jax.grad(
            lambda b: jnp.real(jnp.vdot(data, forward(b)[0]))
        )(background)
        error = jnp.linalg.norm(
            linear.adjoint(data) - reference_adjoint
        ) / jnp.linalg.norm(reference_adjoint)
        assert float(error) < 20 * tol
        np.testing.assert_allclose(
            jnp.vdot(delta, linear.adjoint(data)),
            jnp.real(jnp.vdot(linear(delta), data)),
            rtol=20 * tol,
            atol=tol,
        )
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_receiver_batches_match_independent_source_solves(batch_size):
    n, a, kappa = 8, 0.2, 10.0

    def coefficient(p):
        return -0.04 * np.exp(-np.sum(p**2, -1) / 0.01)

    points, b, _, operator = prepare_volume_solver(n, a, kappa, coefficient)
    directions, receivers = sphere(3), 0.36 * sphere(7)
    forward = make_forward(
        points, kappa, directions, receivers, 1e-10, batch_size
    )
    observed, residuals = forward(b)
    assert float(residuals.max()) < 1e-9
    for index, direction in enumerate(directions):
        total, stats = solve_volume(
            operator, points, kappa, direction, tol=1e-10
        )
        assert stats["info"] == 0
        expected = evaluate_volume_field(
            jnp.asarray(receivers),
            jnp.asarray(points),
            b * total,
            kappa,
            2 * a / n,
        )
        np.testing.assert_allclose(
            observed[index], expected, rtol=1e-8, atol=1e-11
        )


def test_born_cg_uses_dynamic_fields_and_matches_dense_normal_equations():
    from breast_spheres_learn_3d import reconstruct_born

    rng = np.random.default_rng(12)
    shape = (3, 3, 3)
    sources = jnp.asarray(
        rng.normal(size=(3, 27)) + 1j * rng.normal(size=(3, 27))
    )
    receivers = jnp.asarray(
        rng.normal(size=(7, 27)) + 1j * rng.normal(size=(7, 27))
    )
    data = jnp.asarray(rng.normal(size=(3, 7)) + 1j * rng.normal(size=(3, 7)))
    window = jnp.asarray(rng.uniform(0.1, 1.0, shape))

    def reconstruct(operator, data, window):
        return reconstruct_born(operator, data, window, 8, 0.1)

    for factor in (1.0, 0.7):
        operator = BackgroundBorn(
            sources * factor, receivers, shape, np.dtype("float64")
        )
        matrix = np.asarray(
            operator.sources[:, None, :] * receivers[None, :, :]
        )
        matrix = matrix.reshape(-1, 27) * np.asarray(window).ravel()
        normal = np.real(matrix.conj().T @ matrix)
        normal += 0.1 * float(jnp.sum(jnp.abs(data) ** 2)) * np.eye(27)
        rhs = np.real(matrix.conj().T @ np.asarray(data).ravel())
        x = np.zeros_like(rhs)
        r, p = rhs.copy(), rhs.copy()
        rr = r @ r
        for _ in range(8):
            q = normal @ p
            alpha = rr / (p @ q)
            x += alpha * p
            r -= alpha * q
            rr_new = r @ r
            p = r + (rr_new / rr) * p
            rr = rr_new
        result = reconstruct(operator, data, window)
        np.testing.assert_allclose(
            result, window * x.reshape(shape), rtol=1e-9, atol=1e-11
        )
        traced = jax.make_jaxpr(reconstruct)(operator, data, window)
        assert not traced.consts
