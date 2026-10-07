"""Reuse the physical operator for a different illumination."""

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from lucka_phantom_3d import tissue_phantom  # noqa: E402
from wave_scattering_utils_3D import (  # noqa: E402
    load_SD_matrices_3D,
    solve_scattering_bie_3D_matfree,
)

jax.config.update("jax_enable_x64", True)
FIXTURE = ROOT / "data/examples/SD_3D/SD_k4_q4_L1_a1.25.npz"


@pytest.mark.skipif(not FIXTURE.exists(), reason="exterior fixture required")
@pytest.mark.parametrize("precond", ["jacobi", "shift-coarse:0.1"])
def test_reuse_cartesian_new_direction(precond):
    sd = load_SD_matrices_3D(str(FIXTURE))
    kwargs = dict(
        b_cartesian=tissue_phantom,
        method="fgmres",
        precond=precond,
        tol=1e-9,
        maxiter=600,
        stats={},
    )
    first = solve_scattering_bie_3D_matfree(
        sd, None, np.array([[0, 0, 1]]), return_solver=True, **kwargs
    )
    direction = np.array([[0.6, 0, 0.8]])
    field, normal, info = first["resolve"](direction)
    fresh = solve_scattering_bie_3D_matfree(sd, None, direction, **kwargs)
    assert info["converged"]
    assert info["gmres_stats"]["final_rel_res"] <= 1e-9
    np.testing.assert_allclose(field, fresh["uscat_b"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        normal, fresh["uscat_dn_b"], rtol=1e-7, atol=1e-9
    )
    assert np.linalg.norm(field - first["uscat_b"]) > 1e-3
