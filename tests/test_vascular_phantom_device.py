"""Compare tiled device sampling with the independent CPU evaluator."""

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from lucka_phantom_3d import vascular_geometry, vascular_phantom  # noqa: E402
from vascular_phantom_device import vascular_phantom_device  # noqa: E402

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "variant,seed,a,batch", [("vascular", 7, 1.25, 64), ("dense", 8, 1.0, 117)]
)
def test_device_sampling_matches_cpu_geometry_and_support(
    variant, seed, a, batch
):
    branches, _ = vascular_geometry(variant, seed)
    centres = np.stack([(s + 2 * m + e) / 4 for s, m, e, _ in branches])
    centres = centres * (0.8 * a) + [0, 0, 0.55 * a]
    random = np.random.default_rng(2).uniform(-a, a, (500, 3))
    points = np.concatenate(
        [
            random,
            centres,
            np.array([[a, 0, 0], [-a, 0, 0], [0, a, 0], [0, 0, a]]),
        ]
    )
    expected = vascular_phantom(points, a, variant, seed)
    actual = vascular_phantom_device(points, a, variant, seed, batch)
    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-13)
    np.testing.assert_array_equal(actual[-4:], 0)


def test_device_sampling_preserves_grid_shape():
    axis = np.linspace(-1.25, 1.25, 12)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)
    actual = vascular_phantom_device(points, variant="dense", batch_size=128)
    np.testing.assert_allclose(
        actual, vascular_phantom(points, variant="dense"), atol=2e-14
    )
    assert actual.shape == (12, 12, 12)
