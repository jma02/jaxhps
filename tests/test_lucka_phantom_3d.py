"""Regularity, support, and tissue-bound tests for the benchmark phantom."""

import sys
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from lucka_phantom_3d import bump, step_down, tissue_phantom  # noqa: E402


def test_c4_transition_jets():
    poly = Polynomial([0, 0, 0, 0, 0, 126, -420, 540, -315, 70])
    t = np.linspace(0, 1, 101)
    np.testing.assert_allclose(step_down(1 - t, 1, 1), poly(t), atol=2e-13)
    for order in range(1, 5):
        np.testing.assert_allclose(poly.deriv(order)([0, 1]), 0, atol=1e-10)
    compact = Polynomial([1, 0, -1]) ** 5
    np.testing.assert_allclose(bump(t), compact(t), atol=2e-15)
    for order in range(5):
        np.testing.assert_allclose(
            compact.deriv(order)([-1, 1]), 0, atol=1e-10
        )


@pytest.mark.parametrize("geometry", ["sphere", "hemisphere"])
def test_support_bounds_shape_and_scale(geometry):
    rng = np.random.default_rng(5)
    points = rng.uniform(-1.25, 1.25, (8, 1000, 3))
    b = tissue_phantom(points, 1.25, geometry)
    assert b.shape == points.shape[:-1]
    assert np.all(np.isfinite(b))
    assert b.min() >= -0.041
    assert b.max() <= 0.174
    np.testing.assert_allclose(b, tissue_phantom(points / 1.25, 1, geometry))
    for axis in range(3):
        for side in (-1, 1):
            face = points.copy()
            face[..., axis] = side * 1.25
            np.testing.assert_array_equal(
                tissue_phantom(face, 1.25, geometry), 0
            )
    if geometry == "hemisphere":
        above = points.copy()
        above[..., 2] = 0.55 * 1.25
        np.testing.assert_array_equal(tissue_phantom(above), 0)


def test_old_fibro_threshold_is_continuous():
    radius, z0 = 1.0, 0.6875
    # The old fourth-power mask crossed 0.5 on this radial cut.
    r = 0.35 * radius * (0.75 + 0.25 * np.sqrt(1 - 0.5**0.25))
    centre = np.array([0.0, 0.0, z0 - 0.45 * radius])
    samples = centre + np.array([[r - 1e-8, 0, 0], [r + 1e-8, 0, 0]])
    assert abs(np.diff(tissue_phantom(samples))[0]) < 1e-7


def test_chest_wall_rolloff():
    z0 = 0.6875
    eps = np.array([1e-3, 5e-4, 2.5e-4])
    points = np.column_stack([np.zeros(3), np.zeros(3), z0 - eps])
    vals = np.abs(tissue_phantom(points))
    assert np.all(vals[1:] < vals[:-1] / 25)
    assert np.all(vals > 0)
