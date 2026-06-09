r"""End-to-end 3D Helmholtz scattering tests using the HPS+BIE coupling.

These tests exercise the full pipeline from the Gillman-Barnett-Martinsson
formulation in 3D:

* Build the variable-coefficient ItI solver on the cube ``[-0.5, 0.5]^3``.
* Take the top-level ItI matrix, convert to DtN by Cayley.
* Couple with fmm3dbie-generated single/double-layer matrices via the
  exterior BIE ``(1/2 I - D + S T_int) u^s = S (u^inc_n - T_int u^inc)``.
* Recover the scattered impedance trace and reuse the HPS solver to obtain
  the scattered field in the interior of the cube.
* Evaluate the scattered field at points outside the cube via the exterior
  representation ``u^s = D u^s - S u^s_n`` and compare against analytic
  references.

Two cases:
* ``test_transparent_b_zero`` -- with ``b == 0``, the cube is "transparent" and
  the scattered field must be zero up to discretization error.  This isolates
  the coupling code: any bug here surfaces immediately without confounding
  it with interior accuracy.
* ``test_radial_bump_vs_mie`` -- with a smooth compactly supported radial
  potential ``b(r) = -A (1 - (r/R)^2)^4 1_{r<R}``, compare the off-surface
  scattered field against a Mie-style reference (see ``examples/mie_3d.py``).

Gated on the env var ``JAXHPS_SD_3D_NPZ``: if not set, the tests are
skipped.  See ``examples/gen_SD_3D.py`` and ``scripts/build_fmm3dbie.sh``
for how to generate the matrices.
"""

import os
import sys
import logging

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from jaxhps._build_solver import build_solver
from jaxhps._discretization_tree import DiscretizationNode3D
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem

# Make the example modules importable.
_EX_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "examples")
sys.path.insert(0, os.path.abspath(_EX_DIR))

from wave_scattering_utils_3D import (  # noqa: E402
    eval_uscat_offsurface_3D,
    get_DtN_from_ItI_3D,
    get_scattering_uscat_impedance_3D,
    load_SD_matrices_3D,
    permute_to_domain,
)
from mie_3d import mie_scattered_field  # noqa: E402


SD_NPZ = os.environ.get("JAXHPS_SD_3D_NPZ", "/tmp/SD_k4_q8_L1.npz")


def _outward_normals_for_boundary(
    boundary_points: np.ndarray, root
) -> np.ndarray:
    n = np.zeros_like(boundary_points)
    eps = 1e-9
    n[np.abs(boundary_points[:, 0] - root.xmin) < eps] = [-1, 0, 0]
    n[np.abs(boundary_points[:, 0] - root.xmax) < eps] = [1, 0, 0]
    n[np.abs(boundary_points[:, 1] - root.ymin) < eps] = [0, -1, 0]
    n[np.abs(boundary_points[:, 1] - root.ymax) < eps] = [0, 1, 0]
    n[np.abs(boundary_points[:, 2] - root.zmin) < eps] = [0, 0, -1]
    n[np.abs(boundary_points[:, 2] - root.zmax) < eps] = [0, 0, 1]
    return n


@pytest.fixture(scope="module")
def sd_fixture():
    if not os.path.exists(SD_NPZ):
        pytest.skip(
            f"SD matrices not available at {SD_NPZ}; "
            f"set JAXHPS_SD_3D_NPZ to a file produced by examples/gen_SD_3D.py."
        )
    sd = load_SD_matrices_3D(SD_NPZ)
    return sd


def _build_problem(sd, b_callable, source_dirs, eta_override=None):
    """Build a PDEProblem for the scattered-field interior solve.

    The total field satisfies  Lap u + kappa^2 (1 - b) u = 0,
    so u^s = u - u^inc satisfies
        Lap u^s + kappa^2 (1 - b) u^s = kappa^2 b u^inc.
    """
    a = sd["a"]
    q = sd["q"]
    L = sd["L"]
    kappa = sd["kappa"]
    eta = float(eta_override if eta_override is not None else kappa)

    root = DiscretizationNode3D(
        xmin=-a, xmax=a, ymin=-a, ymax=a, zmin=-a, zmax=a
    )
    # p is only relevant to the interior; we use p = q + 2 (matches existing
    # tests; the boundary node layout depends on q, not p).
    p = q + 2
    domain = Domain(p=p, q=q, root=root, L=L)

    int_pts = np.asarray(domain.interior_points)  # (n_leaves, p^3, 3)
    r_int = np.linalg.norm(int_pts, axis=-1)
    b_int = b_callable(r_int)
    I_coeffs = (kappa**2 * (1.0 - b_int)).astype(np.complex128)

    # Source per source direction: kappa^2 b(x) u^inc(x).
    # int_pts has shape (n_leaves, p^3, 3); source_dirs has shape (n_src, 3).
    phases = np.einsum("lpd,sd->lps", int_pts, source_dirs)
    uin_int = np.exp(1j * kappa * phases)  # (n_leaves, p^3, n_src)
    src = (kappa**2 * b_int[..., None] * uin_int).astype(np.complex128)

    ones = np.ones_like(I_coeffs)
    problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=ones,
        D_yy_coefficients=ones,
        D_zz_coefficients=ones,
        I_coefficients=I_coeffs,
        source=src,
        use_ItI=True,
        eta=eta,
    )
    return problem, root, kappa, eta


def _align_sd_to_domain(sd, domain, root):
    """Permute SD data so its boundary ordering matches ``domain.boundary_points``.

    Cross-checks the permuted normals against the HPS-derived outward normals.
    """
    bp_domain = np.asarray(domain.boundary_points).reshape(-1, 3)
    P, sdp = permute_to_domain(sd, bp_domain)
    nrm_hps = _outward_normals_for_boundary(bp_domain, root)
    if not np.allclose(sdp["normals"], nrm_hps):
        bad = float(np.linalg.norm(sdp["normals"] - nrm_hps, axis=-1).max())
        raise AssertionError(
            f"normals don't agree after permutation: max diff {bad:.2e}"
        )
    return sdp, bp_domain, nrm_hps


def test_transparent_b_zero(sd_fixture, caplog) -> None:
    """``b == 0`` should give zero scattered field everywhere."""
    caplog.set_level(logging.INFO)
    sd = sd_fixture

    source_dirs = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    source_dirs /= np.linalg.norm(source_dirs, axis=-1, keepdims=True)

    problem, root, kappa, eta = _build_problem(
        sd, b_callable=lambda r: np.zeros_like(r), source_dirs=source_dirs
    )
    sdp, bp_domain, nrm_hps = _align_sd_to_domain(sd, problem.domain, root)

    T_ItI = build_solver(problem, return_top_T=True)
    T_DtN = get_DtN_from_ItI_3D(jnp.asarray(T_ItI), eta)

    bp_j = jnp.asarray(bp_domain)
    n_j = jnp.asarray(nrm_hps)
    S_j = jnp.asarray(sdp["S"])
    D_j = jnp.asarray(sdp["D"])

    imp, uscat_b, uscat_dn_b = get_scattering_uscat_impedance_3D(
        S=S_j,
        D=D_j,
        T_DtN=T_DtN,
        bdry_pts=bp_j,
        normals=n_j,
        k=kappa,
        eta=eta,
        source_dirs=jnp.asarray(source_dirs),
    )
    uscat_b = np.asarray(uscat_b)
    uscat_dn_b = np.asarray(uscat_dn_b)
    err_b = float(np.max(np.abs(uscat_b)))
    err_n = float(np.max(np.abs(uscat_dn_b)))
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
    u_off = np.asarray(
        eval_uscat_offsurface_3D(
            target_pts=jnp.asarray(targets),
            src_pts=bp_j,
            src_normals=n_j,
            src_weights=jnp.asarray(sdp["wts"]),
            uscat_b=jnp.asarray(uscat_b),
            uscat_dn_b=jnp.asarray(uscat_dn_b),
            k=kappa,
        )
    )
    err_off = float(np.max(np.abs(u_off)))
    logging.info("transparent: max|u^s(target)| = %.3e", err_off)
    assert err_off < 1e-4, f"off-surface u^s should be ~0; got {err_off:.3e}"
    jax.clear_caches()


def test_radial_bump_vs_mie(sd_fixture, caplog) -> None:
    """Smooth radial bump scattering: compare to Mie reference at exterior targets."""
    caplog.set_level(logging.INFO)
    sd = sd_fixture

    A_bump = -0.4
    R_bump = 0.3

    def b_radial(r):
        rho = np.where(r < R_bump, r / R_bump, 1.0)
        return np.where(r < R_bump, A_bump * (1.0 - rho * rho) ** 4, 0.0)

    source_dirs = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    problem, root, kappa, eta = _build_problem(
        sd, b_callable=b_radial, source_dirs=source_dirs
    )
    sdp, bp_domain, nrm_hps = _align_sd_to_domain(sd, problem.domain, root)

    T_ItI = build_solver(problem, return_top_T=True)
    T_DtN = get_DtN_from_ItI_3D(jnp.asarray(T_ItI), eta)

    bp_j = jnp.asarray(bp_domain)
    n_j = jnp.asarray(nrm_hps)
    S_j = jnp.asarray(sdp["S"])
    D_j = jnp.asarray(sdp["D"])

    imp, uscat_b, uscat_dn_b = get_scattering_uscat_impedance_3D(
        S=S_j,
        D=D_j,
        T_DtN=T_DtN,
        bdry_pts=bp_j,
        normals=n_j,
        k=kappa,
        eta=eta,
        source_dirs=jnp.asarray(source_dirs),
    )
    logging.info(
        "radial bump: max|u^s| on bdry = %.3e",
        float(np.max(np.abs(uscat_b))),
    )

    # Targets: ring of points at r = 1.0 (well outside both cube and supp(b)).
    n_tgt = 8
    phi = np.linspace(0, 2 * np.pi, n_tgt, endpoint=False)
    targets = np.stack(
        [
            np.cos(phi),
            np.sin(phi),
            np.zeros_like(phi),
        ],
        axis=-1,
    )
    # Add a couple of off-equator points.
    targets = np.concatenate(
        [
            targets,
            np.array([[0.0, 0.0, 1.0], [0.7, 0.0, 0.7]]),
        ]
    )

    u_off = np.asarray(
        eval_uscat_offsurface_3D(
            target_pts=jnp.asarray(targets),
            src_pts=bp_j,
            src_normals=n_j,
            src_weights=jnp.asarray(sdp["wts"]),
            uscat_b=jnp.asarray(uscat_b[:, 0]),
            uscat_dn_b=jnp.asarray(uscat_dn_b[:, 0]),
            k=kappa,
        )
    )
    u_mie = mie_scattered_field(
        target_pts=targets,
        source_direction=source_dirs[0],
        kappa=kappa,
        b_radial=b_radial,
        R_supp=R_bump,
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
