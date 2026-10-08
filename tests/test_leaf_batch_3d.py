"""Local batching preserves complex, multiple-source impedance maps."""

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from jaxhps import DiscretizationNode3D, Domain, PDEProblem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from wave_scattering_utils_3D import _local_impedance_maps  # noqa: E402

jax.config.update("jax_enable_x64", True)


def test_uneven_leaf_batches_preserve_maps_and_sources():
    root = DiscretizationNode3D(-1, 1, -1, 1, -1, 1)
    domain = Domain(p=4, q=2, root=root, L=1)
    points = np.asarray(domain.interior_points)
    ones = np.ones(points.shape[:-1])
    source = np.stack(
        (np.exp(1j * points[..., 0]), points[..., 1] ** 2), axis=-1
    )
    problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=ones,
        D_yy_coefficients=ones,
        D_zz_coefficients=ones,
        I_coefficients=(2 + 0.3j) * ones + points[..., 2],
        source=source,
        use_ItI=True,
        eta=2,
    )
    expected = _local_impedance_maps(problem, None)
    actual = _local_impedance_maps(problem, 3)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="positive"):
        _local_impedance_maps(problem, 0)
