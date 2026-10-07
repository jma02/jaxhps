"""Independent quadrature and radial-reference checks for FFT scattering."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad_vec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from fft_volume_3d import (  # noqa: E402
    evaluate_volume_field,
    prepare_volume_solver,
    solve_volume,
    truncated_green_transform,
)
from radial_reference_3d import radial_reference  # noqa: E402

jax.config.update("jax_enable_x64", True)


def test_multiplier_removable_singularities():
    kappa, radius = 4.0, 4.5
    s = np.array([0.0, 0.2, kappa, 7.0])
    expected, _ = quad_vec(
        lambda r: r * np.exp(1j * kappa * r) * np.sinc(s * r / np.pi),
        0,
        radius,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    np.testing.assert_allclose(
        truncated_green_transform(s, kappa, radius), expected, atol=1e-12
    )


def test_zero_contrast():
    points, _, _, operator = prepare_volume_solver(
        8, 1.25, 4.0, lambda x: np.zeros(x.shape[:-1])
    )
    u, stats = solve_volume(operator, points, 4, np.array([0, 0, 1]))
    np.testing.assert_allclose(u, np.exp(4j * points[..., 2]), atol=1e-12)
    assert stats["info"] == 0


def test_radial_refinement():
    def radial(r):
        return -0.4 * np.maximum(1 - (np.asarray(r) / 0.6) ** 2, 0) ** 4

    direction = np.array([0, 0, 1])
    targets = np.array([[2.5, 0, 0], [0, 0, 2.5], [0, 0, -2.5]])
    reference = radial_reference(targets, direction, 4, radial, 0.6)
    refined = radial_reference(
        targets,
        direction,
        4,
        radial,
        0.6,
        ell_max=40,
        rtol=1e-12,
        atol=1e-14,
        origin_fraction=5e-6,
        step_fraction=0.0125,
    )
    np.testing.assert_allclose(reference, refined, rtol=1e-8, atol=1e-12)
    errors = []
    for n in (16, 32):
        points, b, _, operator = prepare_volume_solver(
            n, 1.25, 4, lambda pts: radial(np.linalg.norm(pts, axis=-1))
        )
        u, stats = solve_volume(operator, points, 4, direction, tol=1e-10)
        assert stats["final_rel_res"] < 1e-10
        field = np.asarray(
            evaluate_volume_field(
                jnp.asarray(targets), jnp.asarray(points), b * u, 4, 2.5 / n
            )
        )
        errors.append(
            np.linalg.norm(field - reference) / np.linalg.norm(reference)
        )
    assert errors[1] < errors[0] / 4
    assert errors[1] < 1e-4
