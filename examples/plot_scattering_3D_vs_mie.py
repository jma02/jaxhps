"""Visualize the 3D HPS+BIE scattered field vs. Mie reference.

Solves the same radial-bump scattering problem as
``tests/test_scattering_3D_BIE.py::test_radial_bump_vs_mie`` and plots the
scattered field on a slice through the ``y = 0`` plane.

Usage:
    python examples/plot_scattering_3D_vs_mie.py --npz /tmp/SD_k4_q8_L1.npz
"""

import argparse
import os
import sys

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from jaxhps._build_solver import build_solver
from jaxhps._discretization_tree import DiscretizationNode3D
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem
from jaxhps._solve import solve

sys.path.insert(0, os.path.dirname(__file__))
from wave_scattering_utils_3D import (  # noqa: E402
    eval_uscat_offsurface_3D,
    get_DtN_from_ItI_3D,
    get_scattering_uscat_impedance_3D,
    load_SD_matrices_3D,
    permute_to_domain,
)
from mie_3d import (  # noqa: E402
    mie_scattered_field,
    mie_scattered_field_interior,
)


def outward_normals(bp, root):
    n = np.zeros_like(bp)
    eps = 1e-9
    n[np.abs(bp[:, 0] - root.xmin) < eps] = [-1, 0, 0]
    n[np.abs(bp[:, 0] - root.xmax) < eps] = [1, 0, 0]
    n[np.abs(bp[:, 1] - root.ymin) < eps] = [0, -1, 0]
    n[np.abs(bp[:, 1] - root.ymax) < eps] = [0, 1, 0]
    n[np.abs(bp[:, 2] - root.zmin) < eps] = [0, 0, -1]
    n[np.abs(bp[:, 2] - root.zmax) < eps] = [0, 0, 1]
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default="/tmp/SD_k4_q8_L1.npz")
    p.add_argument("--out", default="scattering_3d_vs_mie.png")
    p.add_argument(
        "--n", type=int, default=160, help="grid resolution per dim"
    )
    p.add_argument("--R_bump", type=float, default=0.3)
    p.add_argument("--A_bump", type=float, default=-0.4)
    args = p.parse_args()

    sd = load_SD_matrices_3D(args.npz)
    a, q, L, kappa = sd["a"], sd["q"], sd["L"], sd["kappa"]
    R_bump, A_bump = args.R_bump, args.A_bump

    def b_radial(r):
        rho = np.where(r < R_bump, r / R_bump, 1.0)
        return np.where(r < R_bump, A_bump * (1.0 - rho * rho) ** 4, 0.0)

    source_dir = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    # Build interior PDE problem.
    root = DiscretizationNode3D(
        xmin=-a, xmax=a, ymin=-a, ymax=a, zmin=-a, zmax=a
    )
    domain = Domain(p=q + 2, q=q, root=root, L=L)
    int_pts = np.asarray(domain.interior_points)
    r_int = np.linalg.norm(int_pts, axis=-1)
    b_int = b_radial(r_int)
    I_coeffs = (kappa**2 * (1.0 - b_int)).astype(np.complex128)
    phases = np.einsum("lpd,sd->lps", int_pts, source_dir)
    uin_int = np.exp(1j * kappa * phases)
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
        eta=float(kappa),
    )

    # HPS solve -> top ItI -> DtN.
    T_ItI = build_solver(problem, return_top_T=True)
    T_DtN = get_DtN_from_ItI_3D(jnp.asarray(T_ItI), float(kappa))

    # Permute SD to domain ordering.
    bp_dom = np.asarray(domain.boundary_points).reshape(-1, 3)
    _, sdp = permute_to_domain(sd, bp_dom)
    nrm = outward_normals(bp_dom, root)

    # BIE solve.
    imp, uscat_b, uscat_dn_b = get_scattering_uscat_impedance_3D(
        S=jnp.asarray(sdp["S"]),
        D=jnp.asarray(sdp["D"]),
        T_DtN=T_DtN,
        bdry_pts=jnp.asarray(bp_dom),
        normals=jnp.asarray(nrm),
        k=float(kappa),
        eta=float(kappa),
        source_dirs=jnp.asarray(source_dir),
    )
    uscat_b = np.asarray(uscat_b)[:, 0]
    uscat_dn_b = np.asarray(uscat_dn_b)[:, 0]

    # HPS down-pass with the scattered-field incoming impedance -> u^s on the
    # interior Chebyshev grid.  Keep the n_src dim and slice it off after.
    us_interior_full = np.asarray(solve(problem, imp))
    # solve returns (n_leaves, p^3, n_src); take the single source we have.
    us_interior = us_interior_full[..., 0]

    # Slice grid in y = 0 plane.
    L_plot = 1.5
    xs = np.linspace(-L_plot, L_plot, args.n)
    zs = np.linspace(-L_plot, L_plot, args.n)
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    targets = np.stack(
        [X.ravel(), np.zeros_like(X.ravel()), Z.ravel()], axis=-1
    )
    rr = np.linalg.norm(targets, axis=-1)
    # Masks.  "inside_cube" -> use HPS interior solution (down-pass + interp).
    # "outside_cube"        -> use BIE exterior representation.
    inside_cube = (
        (np.abs(targets[:, 0]) <= a)
        & (np.abs(targets[:, 1]) <= a)
        & (np.abs(targets[:, 2]) <= a)
    )
    outside_cube = ~inside_cube

    # Exterior: BIE off-surface evaluation.
    u_bie = np.full(targets.shape[0], np.nan, dtype=np.complex128)
    u_bie[outside_cube] = np.asarray(
        eval_uscat_offsurface_3D(
            target_pts=jnp.asarray(targets[outside_cube]),
            src_pts=jnp.asarray(bp_dom),
            src_normals=jnp.asarray(nrm),
            src_weights=jnp.asarray(sdp["wts"]),
            uscat_b=jnp.asarray(uscat_b),
            uscat_dn_b=jnp.asarray(uscat_dn_b),
            k=float(kappa),
        )
    )

    # Interior: HPS down-pass solution interpolated onto the slice.
    # interp_from_interior_points takes coords spanning a full 3D rectilinear
    # grid; we just use a degenerate y-axis at y = 0.
    interp_y = np.array([0.0])
    u_interior_3d, _ = domain.interp_from_interior_points(
        samples=jnp.asarray(us_interior),
        eval_points_x=jnp.asarray(xs),
        eval_points_y=jnp.asarray(interp_y),
        eval_points_z=jnp.asarray(zs),
    )
    u_interior_2d = np.asarray(u_interior_3d).squeeze(axis=1)  # (n_x, n_z)
    # The slice flatten order is X, Z over meshgrid(xy), giving shape (n_z, n_x).
    # interp_from_interior_points returns (n_x, n_z); reorder to match.
    u_interior_slice = u_interior_2d.T  # (n_z, n_x)
    sel = inside_cube.reshape(args.n, args.n)
    u_bie_grid = u_bie.reshape(args.n, args.n)
    u_bie_grid[sel] = u_interior_slice[sel]
    u_bie = u_bie_grid.ravel()

    # Mie reference: exterior via outgoing-Hankel sum, interior via the
    # numerically-integrated radial ODE.
    valid_mie = np.ones(targets.shape[0], dtype=bool)
    u_mie = np.full(targets.shape[0], np.nan, dtype=np.complex128)
    out_supp = rr > R_bump
    u_mie[out_supp] = mie_scattered_field(
        target_pts=targets[out_supp],
        source_direction=source_dir[0],
        kappa=float(kappa),
        b_radial=b_radial,
        R_supp=R_bump,
        ell_max=20,
    )
    in_supp = ~out_supp
    u_mie[in_supp] = mie_scattered_field_interior(
        target_pts=targets[in_supp],
        source_direction=source_dir[0],
        kappa=float(kappa),
        b_radial=b_radial,
        R_supp=R_bump,
        ell_max=20,
    )

    re_bie = np.real(u_bie).reshape(args.n, args.n)
    re_mie = np.real(u_mie).reshape(args.n, args.n)
    err = np.full_like(re_bie, np.nan)
    # Compare everywhere both are defined.
    overlap = valid_mie
    err.ravel()[overlap] = np.abs(
        u_bie.ravel()[overlap] - u_mie.ravel()[overlap]
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    extent = [-L_plot, L_plot, -L_plot, L_plot]
    vmax = float(max(np.nanmax(np.abs(re_mie)), np.nanmax(np.abs(re_bie))))
    common = dict(
        extent=extent, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )

    axes[0].imshow(re_mie, **common)
    axes[0].set_title(rf"Mie ref: $\Re\,u^s$  (kappa={kappa})")

    axes[1].imshow(re_bie, **common)
    axes[1].set_title(r"HPS+BIE: $\Re\,u^s$")

    im = axes[2].imshow(err, extent=extent, origin="lower", cmap="viridis")
    axes[2].set_title(r"$|u^s_{\rm BIE} - u^s_{\rm Mie}|$")
    fig.colorbar(im, ax=axes[2], shrink=0.85)

    for ax in axes:
        ax.add_patch(
            Rectangle(
                (-a, -a),
                2 * a,
                2 * a,
                fill=False,
                edgecolor="black",
                linewidth=1.2,
                linestyle="--",
            )
        )
        ax.add_patch(
            Circle(
                (0, 0),
                R_bump,
                fill=False,
                edgecolor="gray",
                linewidth=0.8,
                linestyle=":",
            )
        )
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal")

    fig.suptitle(
        f"Scattered field on y=0 slice; bump  b(r) = {A_bump}(1-(r/{R_bump})^2)^4"
        f" 1_{{r<{R_bump}}};   plane wave  exp(i kappa x)",
        fontsize=11,
    )
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
