r"""End-to-end 3D Helmholtz scattering tests using the HPS+BIE coupling.

These tests exercise the full pipeline from the Gillman-Barnett-Martinsson
formulation in 3D via :func:`wave_scattering_utils_3D.solve_scattering_bie_3D`:

* Build the variable-coefficient ItI solver on the cube ``[-0.5, 0.5]^3``.
* Take the top-level ItI matrix, convert to DtN by Cayley.
* Couple with fmm3dbie-generated single/double-layer matrices via the
  exterior BIE ``(1/2 I - D + S T_int) u^s = S (u^inc_n - T_int u^inc)``.
* Evaluate the scattered field at points outside the cube via the exterior
  representation ``u^s = D u^s - S u^s_n`` and compare against analytic
  references.

Three cases:
* ``test_transparent_b_zero`` -- with ``b == 0``, the cube is "transparent" and
  the scattered field must be zero up to discretization error.  This isolates
  the coupling code: any bug here surfaces immediately without confounding
  it with interior accuracy.
* ``test_radial_bump_vs_mie`` -- with a smooth compactly supported radial
  potential ``b(r) = -A (1 - (r/R)^2)^4 1_{r<R}``, compare the off-surface
  scattered field against a Mie-style reference (see ``examples/mie_3d.py``).
* ``test_radial_bump_vs_mie_convergence`` -- the same comparison over sweeps
  of fixtures that refine one discretization knob at a time (boundary order
  q, or octree depth L), asserting the error against the Mie reference
  decreases at every refinement step.

The first two tests are gated on the env var ``JAXHPS_SD_3D_NPZ`` (a single
fixture file) and the sweeps on ``JAXHPS_SD_3D_NPZ_SWEEP_Q`` /
``JAXHPS_SD_3D_NPZ_SWEEP_L`` (comma-separated lists); when the files are
absent, the tests are skipped.  See ``examples/gen_SD_3D.py`` and
``scripts/build_fmm3dbie.sh`` for how to generate the matrices.
"""

import os
import sys
import logging

import numpy as np
import pytest
import jax
import jax.numpy as jnp

# Make the example modules importable.
_EX_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "examples")
sys.path.insert(0, os.path.abspath(_EX_DIR))

from wave_scattering_utils_3D import (  # noqa: E402
    eval_uscat_offsurface_3D,
    load_SD_matrices_3D,
    solve_scattering_bie_3D,
)
from mie_3d import mie_scattered_field  # noqa: E402


SD_NPZ = os.environ.get("JAXHPS_SD_3D_NPZ", "/tmp/SD_k4_q8_L1.npz")

# Comma-separated fixture paths for the convergence sweeps.  Each sweep
# varies one discretization knob while everything else stays fixed:
# "q" raises the boundary order (and with it the interior order p = q + 4)
# at fixed tree depth; "L" deepens the octree at fixed boundary order.
SD_NPZ_SWEEPS = {
    "q": os.environ.get(
        "JAXHPS_SD_3D_NPZ_SWEEP_Q",
        "/tmp/SD_k4_q4_L1.npz,/tmp/SD_k4_q6_L1.npz,/tmp/SD_k4_q8_L1.npz",
    ),
    "L": os.environ.get(
        "JAXHPS_SD_3D_NPZ_SWEEP_L",
        "/tmp/SD_k4_q4_L1.npz,/tmp/SD_k4_q4_L2.npz,/tmp/SD_k4_q4_L3.npz",
    ),
}

# Ceiling on the finest-discretization error for each sweep.  Observed with
# the driver default p = q + 4: 5.2e-4 at (q=8, L=1) for the q sweep,
# 2.9e-5 at (q=4, L=3) for the L sweep.
FINEST_REL_ERR = {"q": 2e-3, "L": 2e-4}

# Radial scattering potential shared by the Mie comparison tests:
# a smooth bump b(r) = A (1 - (r/R)^2)^4 supported on r < R.
A_BUMP = -0.4
R_BUMP = 0.3


def _bump_radial(r):
    rho = np.where(r < R_BUMP, r / R_BUMP, 1.0)
    return np.where(r < R_BUMP, A_BUMP * (1.0 - rho * rho) ** 4, 0.0)


def _exterior_targets():
    """Ring of 8 points at r = 1.0 in the z=0 plane (well outside both the
    cube and supp(b)), plus two off-equator points."""
    phi = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    return np.concatenate(
        [
            np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1),
            np.array([[0.0, 0.0, 1.0], [0.7, 0.0, 0.7]]),
        ]
    )


@pytest.fixture(scope="module")
def sd_fixture():
    if not os.path.exists(SD_NPZ):
        pytest.skip(
            f"SD matrices not available at {SD_NPZ}; "
            f"set JAXHPS_SD_3D_NPZ to a file produced by examples/gen_SD_3D.py."
        )
    sd = load_SD_matrices_3D(SD_NPZ)
    return sd


def _eval_offsurface(out, targets, kappa, src_idx=None):
    """Evaluate the scattered field at ``targets`` from a driver result."""
    uscat_b = out["uscat_b"]
    uscat_dn_b = out["uscat_dn_b"]
    if src_idx is not None:
        uscat_b = uscat_b[:, src_idx]
        uscat_dn_b = uscat_dn_b[:, src_idx]
    return np.asarray(
        eval_uscat_offsurface_3D(
            target_pts=jnp.asarray(targets),
            src_pts=jnp.asarray(out["boundary_points"]),
            src_normals=jnp.asarray(out["normals"]),
            src_weights=jnp.asarray(out["sdp"]["wts"]),
            uscat_b=jnp.asarray(uscat_b),
            uscat_dn_b=jnp.asarray(uscat_dn_b),
            k=kappa,
        )
    )


def test_transparent_b_zero(sd_fixture, caplog) -> None:
    """``b == 0`` should give zero scattered field everywhere."""
    caplog.set_level(logging.INFO)
    kappa = sd_fixture["kappa"]

    source_dirs = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    source_dirs /= np.linalg.norm(source_dirs, axis=-1, keepdims=True)

    out = solve_scattering_bie_3D(
        sd_fixture, b_radial=np.zeros_like, source_dirs=source_dirs
    )

    err_b = float(np.max(np.abs(out["uscat_b"])))
    err_n = float(np.max(np.abs(out["uscat_dn_b"])))
    logging.info(
        "transparent: max|u^s| on bdry = %.3e, max|u^s_n| = %.3e",
        err_b,
        err_n,
    )
    # The BIE residual is bounded by the accuracy of the (S, D) matrices
    # plus the interior ItI accuracy; both are ~1e-6 at q=10, L=0.  Keep
    # the threshold loose.
    assert err_b < 1e-4, f"u^s on boundary should be ~0; got {err_b:.3e}"
    assert err_n < 1e-3, f"u^s_n on boundary should be ~0; got {err_n:.3e}"

    # Off-surface evaluation at a handful of exterior points.
    targets = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.7, 0.7, 0.7],
        ]
    )
    u_off = _eval_offsurface(out, targets, kappa)
    err_off = float(np.max(np.abs(u_off)))
    logging.info("transparent: max|u^s(target)| = %.3e", err_off)
    assert err_off < 1e-4, f"off-surface u^s should be ~0; got {err_off:.3e}"
    jax.clear_caches()


def test_radial_bump_vs_mie(sd_fixture, caplog) -> None:
    """Smooth radial bump scattering: compare to Mie reference at exterior targets."""
    caplog.set_level(logging.INFO)
    kappa = sd_fixture["kappa"]

    source_dirs = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    out = solve_scattering_bie_3D(
        sd_fixture, b_radial=_bump_radial, source_dirs=source_dirs
    )
    logging.info(
        "radial bump: max|u^s| on bdry = %.3e",
        float(np.max(np.abs(out["uscat_b"]))),
    )

    targets = _exterior_targets()
    u_off = _eval_offsurface(out, targets, kappa, src_idx=0)
    u_mie = mie_scattered_field(
        target_pts=targets,
        source_direction=source_dirs[0],
        kappa=kappa,
        b_radial=_bump_radial,
        R_supp=R_BUMP,
        ell_max=20,
    )

    rel_err = float(np.linalg.norm(u_off - u_mie) / np.linalg.norm(u_mie))
    max_abs = float(np.max(np.abs(u_off - u_mie)))
    logging.info(
        "Mie vs BIE: rel_err = %.3e, max abs err = %.3e, |u_mie|_max = %.3e",
        rel_err,
        max_abs,
        float(np.max(np.abs(u_mie))),
    )
    for i, t in enumerate(targets):
        logging.info(
            "  target=%s  u_off=%s  u_mie=%s",
            np.round(t, 3),
            u_off[i],
            u_mie[i],
        )
    assert rel_err < 5e-2, f"Mie/BIE rel error too large: {rel_err:.3e}"
    jax.clear_caches()


@pytest.mark.parametrize("vary", ["q", "L"])
def test_radial_bump_vs_mie_convergence(vary, caplog) -> None:
    """Error against the Mie reference must fall as the discretization is refined.

    Runs the radial-bump problem on a sweep of (S, D) fixtures that refine
    one discretization knob (parametrized as ``vary``) at the same kappa and
    cube size:

    * ``vary="q"``: increasing boundary order q at fixed tree depth.  The
      interior Chebyshev order follows the driver default p = q + 4, so this
      sweep raises the order of the interior HPS solve and the boundary
      quadrature together (p-refinement).
    * ``vary="L"``: increasing octree depth L at fixed boundary order
      (h-refinement).

    Asserts that the relative error at exterior targets decreases at every
    step of the sweep.
    """
    caplog.set_level(logging.INFO)
    env_var = f"JAXHPS_SD_3D_NPZ_SWEEP_{vary.upper()}"
    paths = [s.strip() for s in SD_NPZ_SWEEPS[vary].split(",") if s.strip()]
    missing = [path for path in paths if not os.path.exists(path)]
    if len(paths) < 2 or missing:
        pytest.skip(
            f"Convergence sweep needs >= 2 SD fixture files; missing "
            f"{missing}. Set {env_var} to a comma-separated list of files "
            f"produced by examples/gen_SD_3D.py at increasing {vary}."
        )

    sds = [load_SD_matrices_3D(path) for path in paths]
    kappa = sds[0]["kappa"]
    fixed = "L" if vary == "q" else "q"
    knobs = [int(sd[vary]) for sd in sds]
    for sd in sds[1:]:
        assert sd["kappa"] == kappa, "sweep fixtures must share kappa"
        assert sd["a"] == sds[0]["a"], "sweep fixtures must share cube size"
        assert sd[fixed] == sds[0][fixed], f"sweep fixtures must share {fixed}"
    assert knobs == sorted(knobs) and len(set(knobs)) == len(knobs), (
        f"sweep fixtures must have strictly increasing {vary}; got {knobs}"
    )

    source_dirs = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    targets = _exterior_targets()
    u_mie = mie_scattered_field(
        target_pts=targets,
        source_direction=source_dirs[0],
        kappa=kappa,
        b_radial=_bump_radial,
        R_supp=R_BUMP,
        ell_max=20,
    )

    rel_errs = []
    for sd in sds:
        out = solve_scattering_bie_3D(
            sd, b_radial=_bump_radial, source_dirs=source_dirs
        )
        u_off = _eval_offsurface(out, targets, kappa, src_idx=0)
        rel_err = float(np.linalg.norm(u_off - u_mie) / np.linalg.norm(u_mie))
        rel_errs.append(rel_err)
        logging.info(
            "q=%d L=%d (p=%d): rel_err vs Mie = %.3e",
            sd["q"],
            sd["L"],
            sd["q"] + 4,
            rel_err,
        )
        jax.clear_caches()

    pairs = list(zip(knobs, rel_errs))
    for (k_coarse, e_coarse), (k_fine, e_fine) in zip(pairs, pairs[1:]):
        assert e_fine < e_coarse, (
            f"no error reduction from {vary}={k_coarse} to "
            f"{vary}={k_fine}: {pairs}"
        )
    assert rel_errs[-1] < FINEST_REL_ERR[vary], (
        f"finest discretization too inaccurate: {pairs}"
    )
    assert rel_errs[0] > 5 * rel_errs[-1], (
        f"sweep shows < 5x total error reduction: {pairs}"
    )
