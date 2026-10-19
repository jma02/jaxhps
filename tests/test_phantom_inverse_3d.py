import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from fft_volume_3d import (
    evaluate_volume_field,
    prepare_volume_solver,
    solve_volume,
)
from phantom_inverse_3d import crop, make_forward, objective, sphere

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
