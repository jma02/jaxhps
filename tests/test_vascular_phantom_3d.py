"""Check the refined phantom's geometry, support, and reproducibility."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from lucka_phantom_3d import vascular_geometry, vascular_phantom  # noqa: E402


@pytest.mark.parametrize("variant,count", [("vascular", 45), ("dense", 75)])
def test_branch_connectivity_and_taper(variant, count):
    branches, _ = vascular_geometry(variant)
    assert len(branches) == count
    for start, _, end, radius in branches:
        assert end[2] < start[2]
        if start[2] < -0.13:
            parents = [b for b in branches if np.array_equal(b[2], start)]
            assert len(parents) == 1
            assert radius < parents[0][3]


@pytest.mark.parametrize("variant", ["vascular", "dense"])
def test_support_bounds_scale_and_seed(variant):
    points = np.random.default_rng(2).uniform(-1.25, 1.25, (10, 100, 3))
    b = vascular_phantom(points, variant=variant)
    assert b.shape == (10, 100)
    assert np.all(np.isfinite(b))
    assert b.min() >= -0.047
    assert b.max() <= 0.174
    np.testing.assert_allclose(
        b, vascular_phantom(points / 1.25, a=1, variant=variant), atol=1e-14
    )
    assert (
        np.linalg.norm(b - vascular_phantom(points, variant=variant, seed=8))
        > 0.01
    )
    for axis in range(3):
        for side in (-1, 1):
            face = points.copy()
            face[..., axis] = side * 1.25
            np.testing.assert_array_equal(
                vascular_phantom(face, variant=variant), 0
            )


def test_branch_is_visible_in_coefficient():
    start, middle, end, _ = vascular_geometry()[0][0]
    centre = (start + 2 * middle + end) / 4
    world = centre + [0, 0, 0.6875]
    assert vascular_phantom(world) > 0.05
