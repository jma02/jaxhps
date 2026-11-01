import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from born_acquisition_3d import (
    acquisition_indices,
    dense_adjoint,
    dense_signals,
    subset_indices,
)
from phantom_inverse_3d import BackgroundBorn


@pytest.mark.parametrize("count,held", [(64, 8), (512, 64), (8, 2), (3, 7)])
def test_acquisition_partitions_and_nested_subsets(count, held):
    order = acquisition_indices(count, held)
    np.testing.assert_array_equal(np.sort(order), np.arange(count + held))
    assert len(np.intersect1d(order[:count], order[count:])) == 0
    subset = order[subset_indices(count, max(1, count // 2))]
    assert set(subset) <= set(order[:count])
    assert len(set(subset)) == max(1, count // 2)


def test_batched_dense_reference_matches_full_operator():
    rng = np.random.default_rng(41)
    sources, receivers = [
        jnp.asarray(
            rng.normal(size=(count, 60)) + 1j * rng.normal(size=(count, 60)),
            dtype=jnp.complex64,
        )
        for count in (7, 11)
    ]
    images = jnp.asarray(rng.normal(size=(3, 60)), dtype=jnp.float32)
    data = jnp.asarray(
        rng.normal(size=(7, 11)) + 1j * rng.normal(size=(7, 11)),
        dtype=jnp.complex64,
    )
    dense = BackgroundBorn(sources, receivers, (3, 4, 5), np.dtype("float32"))
    blocks = [
        dense_signals(sources, receivers[first : first + 4], images)
        for first in range(0, 11, 4)
    ]
    signals = jnp.concatenate(blocks, axis=2)
    for i in range(len(images)):
        np.testing.assert_allclose(
            signals[i], dense(images[i].reshape(3, 4, 5)), rtol=1e-5, atol=2e-5
        )
    adjoint = sum(
        dense_adjoint(
            sources, receivers[first : first + 4], data[:, first : first + 4]
        )
        for first in range(0, 11, 4)
    )
    normal = sum(
        dense_adjoint(
            sources,
            receivers[first : first + 4],
            signals[0, :, first : first + 4],
        )
        for first in range(0, 11, 4)
    )
    np.testing.assert_allclose(
        adjoint, dense.adjoint(data).ravel(), rtol=1e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        normal,
        dense.adjoint(dense(images[0].reshape(3, 4, 5))).ravel(),
        rtol=1e-5,
        atol=2e-4,
    )
    assert (
        float(
            jnp.abs(
                jnp.real(jnp.vdot(signals[0], data))
                - jnp.vdot(images[0], adjoint)
            )
        )
        < 1e-3
    )
    jax.block_until_ready(normal)
