"""jaxhps 3D ItI driver with 2nd-order Engquist-Majda ABC.

Iteration approach: solve with current bdry_data, then update bdry_data using
2nd-order ABC residual:
    bdry_data <- (i / (2 kappa)) * Delta_tau u_outer

where Delta_tau is the surface (tangential) Laplacian on each cube face.
The driver iterates until bdry_data converges.

Notes on sign convention:
  - eta = -KAPPA in PDEProblem (outgoing Sommerfeld with η=-κ).
  - With this eta, G u = u_n + i*eta*u = u_n - i*kappa*u (the OUTGOING trace)
    is what bdry_data ("g_in") encodes.
  - 1st-order Sommerfeld outgoing: u_n - i*kappa*u = 0 -> bdry_data=0.
  - 2nd-order EM outgoing: u_n - i*kappa*u = (i/(2*kappa)) * Delta_tau u
    -> bdry_data = (i/(2*kappa)) * Delta_tau u_outer.
"""

# ruff: noqa: E402
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp
import numpy as np
from jax import config as jcfg

jcfg.update("jax_enable_x64", True)

from jaxhps import (
    Domain,
    DiscretizationNode3D,
    PDEProblem,
    build_solver,
    solve,
)
from jaxhps._grid_creation_3D import get_all_uniform_leaves_3D
from jaxhps._interpolation_methods import vmapped_interp_to_point_3D
from jaxhps._precompute_operators_3D import (
    get_face_1_idxes,
    get_face_2_idxes,
    get_face_3_idxes,
    get_face_4_idxes,
    get_face_5_idxes,
    get_face_6_idxes,
    precompute_diff_operators_3D,
)
from jaxhps.quadrature import (
    barycentric_lagrange_interpolation_matrix_2D,
    chebyshev_points,
    gauss_points,
)
from mie_setup import EPS_SPHERE, KAPPA, SPHERE_RADIUS, W_RAMP, get_tx

# --------------- params ---------------
L_PHYS = float(os.environ.get("ITI_L_PHYS", "2.0"))
P = int(os.environ.get("P", "12"))
Q = int(os.environ.get("Q", "10"))
L = int(os.environ.get("L", "2"))
N_ITER = int(os.environ.get("N_ITER", "8"))
DAMP = float(os.environ.get("DAMP", "0.7"))  # bdry <- DAMP*new + (1-DAMP)*old

print(
    f"jaxhps-ItI 2nd-order: L_PHYS={L_PHYS} p={P} q={Q} L={L} eta=-{KAPPA} "
    f"N_ITER={N_ITER}"
)

tx = jnp.array(get_tx())
N_TX = tx.shape[0]
print(f"  N_TX={N_TX}")

# --------------- domain & PDE ---------------
root = DiscretizationNode3D(
    xmin=-L_PHYS,
    xmax=L_PHYS,
    ymin=-L_PHYS,
    ymax=L_PHYS,
    zmin=-L_PHYS,
    zmax=L_PHYS,
)
domain = Domain(p=P, q=Q, root=root, L=L)
pts = domain.interior_points
xg = pts[..., 0]
yg = pts[..., 1]
zg = pts[..., 2]

r = jnp.sqrt(xg * xg + yg * yg + zg * zg)
mask = 0.5 * (1 - jnp.tanh((r - SPHERE_RADIUS) / W_RAMP))
n_field = 1.0 + (EPS_SPHERE - 1.0) * mask

ones = jnp.ones_like(xg)
I_coeffs = (KAPPA**2) * n_field


def source_for_tx(t):
    rr = jnp.sqrt(
        (xg - t[0]) ** 2 + (yg - t[1]) ** 2 + (zg - t[2]) ** 2 + 1e-8
    )
    ui = jnp.exp(1j * KAPPA * rr) / (4 * jnp.pi * rr)
    return (KAPPA**2) * (1.0 - n_field) * ui


sources = jax.vmap(source_for_tx, in_axes=0, out_axes=-1)(tx)

problem = PDEProblem(
    domain=domain,
    D_xx_coefficients=ones,
    D_yy_coefficients=ones,
    D_zz_coefficients=ones,
    I_coefficients=I_coeffs,
    source=sources,
    use_ItI=True,
    eta=-float(KAPPA),
)

t0 = time.perf_counter()
print("  build_solver ...")
build_solver(problem)
print(f"    built in {time.perf_counter() - t0:.0f}s")

# --------------- 2nd-order ABC machinery ---------------
# All leaves are uniform-cube of half-side h_leaf = L_PHYS / 2^L.
h_leaf = L_PHYS / (2**L)
diff_ops = precompute_diff_operators_3D(P, h_leaf)
du_dx, du_dy, du_dz, d_xx, d_yy, d_zz, _, _, _ = diff_ops
# Surface Laplacians per cube-face direction (acting on full volume Cheby grid).
# face1, face2 perpendicular to x: Delta_tau = D_yy + D_zz
# face3, face4 perpendicular to y: Delta_tau = D_xx + D_zz
# face5, face6 perpendicular to z: Delta_tau = D_xx + D_yy
Dt_x = d_yy + d_zz  # for x-normal faces
Dt_y = d_xx + d_zz  # for y-normal faces
Dt_z = d_xx + d_yy  # for z-normal faces

# Cheby->Gauss face interpolation
cheby_1d = chebyshev_points(P)
gauss_1d = gauss_points(Q)
Q_2D = barycentric_lagrange_interpolation_matrix_2D(
    cheby_1d, cheby_1d, gauss_1d, gauss_1d
)  # shape (Q^2, P^2)

face_idxes_list = [
    np.asarray(get_face_1_idxes(P)),
    np.asarray(get_face_2_idxes(P)),
    np.asarray(get_face_3_idxes(P)),
    np.asarray(get_face_4_idxes(P)),
    np.asarray(get_face_5_idxes(P)),
    np.asarray(get_face_6_idxes(P)),
]
Dt_per_face = [Dt_x, Dt_x, Dt_y, Dt_y, Dt_z, Dt_z]


def per_face_lap_to_gauss(u_volume_leaf, face_idx):
    """u_volume_leaf shape (p^3, n_RHS).
    Returns Δ_τ u on Gauss face: shape (q^2, n_RHS).
    """
    Dt = Dt_per_face[face_idx]
    fi = face_idxes_list[face_idx]
    lap_volume = Dt @ u_volume_leaf  # (p^3, n_RHS)
    lap_face_cheby = lap_volume[fi]  # (p^2, n_RHS)
    return Q_2D @ lap_face_cheby  # (q^2, n_RHS)


# --------------- map outer-leaves & gauss-block positions ---------------
# bdry_data is laid out [face1 | face2 | face3 | face4 | face5 | face6],
# each block has (2^L)^2 leaf-faces × q^2 Gauss points = (2^L)^2 * q^2 entries.
# For each cube face, we need to know:
#   (a) which volume leaves have one of their faces on this cube face
#   (b) which q^2 chunk within the cube-face block corresponds to that leaf
# Strategy: match by physical centroid of (volume-leaf face) vs (cube-face Gauss subblock centroid).
leaves = get_all_uniform_leaves_3D(domain.root, domain.L)
leaf_bounds = np.array(
    [[n.xmin, n.xmax, n.ymin, n.ymax, n.zmin, n.zmax] for n in leaves]
)  # (n_leaves, 6)
n_leaves = leaf_bounds.shape[0]
n_per_face = (2**L) ** 2 * Q**2  # number of bdry_data entries per cube face
n_per_leaf_face = Q**2

bdry_pts = np.asarray(domain.boundary_points)
print(f"  bdry total = {bdry_pts.shape[0]}, per-face = {n_per_face}")

# Pre-compute mapping (cube_face_idx, leaf_face_idx_within_cube_face) -> (volume_leaf_idx, slice_start)
# For each cube face f and each leaf-face-block within that face, find the volume leaf.

cube_faces = [
    (
        "xmin",
        0,
        lambda b: np.isclose(b[0], -L_PHYS),
        1,
        2,
    ),  # axis=0; tang axes y(1), z(2)
    ("xmax", 1, lambda b: np.isclose(b[1], L_PHYS), 1, 2),
    ("ymin", 2, lambda b: np.isclose(b[2], -L_PHYS), 0, 2),
    ("ymax", 3, lambda b: np.isclose(b[3], L_PHYS), 0, 2),
    ("zmin", 4, lambda b: np.isclose(b[4], -L_PHYS), 0, 1),
    ("zmax", 5, lambda b: np.isclose(b[5], L_PHYS), 0, 1),
]

# For each cube face, list (slice_start_in_bdry, volume_leaf_idx)
mapping = []  # list of (cube_face_idx_0to5, leaf_idx_in_volume, slice_start_in_bdry)
for name, cf_idx, leaf_filter, t1, t2 in cube_faces:
    block_start = cf_idx * n_per_face
    pts_face = bdry_pts[block_start : block_start + n_per_face]
    # split into n_per_face / Q^2 chunks of Q^2
    n_leafblocks = n_per_face // n_per_leaf_face
    for blk in range(n_leafblocks):
        chunk = pts_face[blk * n_per_leaf_face : (blk + 1) * n_per_leaf_face]
        c_t1 = chunk[:, t1].mean()
        c_t2 = chunk[:, t2].mean()
        # find leaf with matching face on this cube face and centroid close to (c_t1, c_t2)
        best = None
        for li in range(n_leaves):
            if not leaf_filter(leaf_bounds[li]):
                continue
            # leaf tangential mid
            m_t1 = (leaf_bounds[li, 2 * t1] + leaf_bounds[li, 2 * t1 + 1]) / 2
            m_t2 = (leaf_bounds[li, 2 * t2] + leaf_bounds[li, 2 * t2 + 1]) / 2
            d = abs(m_t1 - c_t1) + abs(m_t2 - c_t2)
            if best is None or d < best[1]:
                best = (li, d)
        assert best is not None
        mapping.append((cf_idx, best[0], block_start + blk * n_per_leaf_face))

print(f"  mapping len = {len(mapping)} (expected {6 * (2**L) ** 2})")

# Convert to arrays for jit-friendly dispatch
mapping_cf = np.array([m[0] for m in mapping])
mapping_li = np.array([m[1] for m in mapping])
mapping_st = np.array([m[2] for m in mapping])

# Sanity check: print one example
for face_idx_check in [0, 1, 2]:
    sel = np.where(mapping_cf == face_idx_check)[0][:3]
    for s in sel:
        cf, li, st = mapping[s]
        bp = bdry_pts[st : st + n_per_leaf_face]
        lb = leaf_bounds[li]
        print(
            f"    cube_face={cf} -> volume_leaf={li} "
            f"(xmin..xmax={lb[0]:.2f},{lb[1]:.2f}, "
            f"ymin..ymax={lb[2]:.2f},{lb[3]:.2f}, "
            f"zmin..zmax={lb[4]:.2f},{lb[5]:.2f}); "
            f"bdry chunk centroid = ({bp[:, 0].mean():.2f}, {bp[:, 1].mean():.2f}, {bp[:, 2].mean():.2f})"
        )

# --------------- iteration ---------------
n_bdry = domain.boundary_points.shape[0]
bdry = jnp.zeros((n_bdry, N_TX), dtype=jnp.complex128)


def update_bdry(u_volume):
    """Compute new bdry_data = (i/(2*kappa)) * Delta_tau u_outer."""
    new_bdry = np.zeros((n_bdry, N_TX), dtype=np.complex128)
    factor = 1j / (2.0 * KAPPA)
    for cf, li, st in mapping:
        u_leaf = u_volume[li]  # (p^3, N_TX)
        lap_face_gauss = np.asarray(
            per_face_lap_to_gauss(u_leaf, cf)
        )  # (q^2, N_TX)
        new_bdry[st : st + n_per_leaf_face, :] = factor * lap_face_gauss
    return jnp.array(new_bdry)


def measure_M(u_volume):
    """Sample u_volume at sensor TX positions (24x24 measurement matrix)."""
    pts_arr = np.asarray(tx)
    leaf_corners = np.stack(
        [
            np.array([[n.xmin, n.ymin, n.zmin], [n.xmax, n.ymax, n.zmax]])
            for n in leaves
        ]
    )
    bx = (
        (pts_arr[:, 0, None] >= leaf_corners[None, :, 0, 0])
        & (pts_arr[:, 0, None] <= leaf_corners[None, :, 1, 0])
        & (pts_arr[:, 1, None] >= leaf_corners[None, :, 0, 1])
        & (pts_arr[:, 1, None] <= leaf_corners[None, :, 1, 1])
        & (pts_arr[:, 2, None] >= leaf_corners[None, :, 0, 2])
        & (pts_arr[:, 2, None] <= leaf_corners[None, :, 1, 2])
    )
    patch = np.argmax(bx, axis=1)
    corn = jnp.array(leaf_corners[patch])
    M_cols = []
    for i in range(u_volume.shape[-1]):
        f = u_volume[patch, :, i]
        vals = vmapped_interp_to_point_3D(
            jnp.array(pts_arr[:, 0]),
            jnp.array(pts_arr[:, 1]),
            jnp.array(pts_arr[:, 2]),
            corn,
            f,
            P,
        )
        M_cols.append(vals)
    return np.asarray(jnp.squeeze(jnp.stack(M_cols, axis=-1)))


# Reference for tracking
mie_npz = os.path.join(os.path.dirname(__file__), "mie.npz")
M_mie = np.load(mie_npz)["umeas"] if os.path.exists(mie_npz) else None

print("\n  iteration:")
print("  iter | ||bdry||  | ||M||_F   | err vs Mie | recip")
print("  -----+-----------+-----------+------------+-------")

best_M = None
best_err = float("inf")
best_iter = -1
for it in range(N_ITER):
    t0 = time.perf_counter()
    u_scat = solve(problem, bdry)
    u_scat.block_until_ready()
    sec = time.perf_counter() - t0

    M_now = measure_M(u_scat)
    norm_M = np.linalg.norm(M_now)
    rec = np.linalg.norm(M_now - M_now.T) / norm_M
    err_str = "       —"
    err_val = None
    if M_mie is not None:
        err_val = np.linalg.norm(M_now - M_mie) / np.linalg.norm(M_mie)
        err_str = f"{err_val * 100:7.3f}%"
        if err_val < best_err:
            best_err = err_val
            best_M = M_now.copy()
            best_iter = it
    print(
        f"  {it:4d} | {float(jnp.linalg.norm(bdry)):.3e} | "
        f"{norm_M:.4e} | {err_str} | {rec:.2e}  ({sec:.0f}s)"
    )

    new_bdry_raw = update_bdry(u_scat)
    new_bdry = DAMP * new_bdry_raw + (1.0 - DAMP) * bdry
    diff = float(
        jnp.linalg.norm(new_bdry - bdry) / (jnp.linalg.norm(new_bdry) + 1e-30)
    )
    print(f"         new bdry: ||delta||/||new|| = {diff:.3e}")
    bdry = new_bdry
    if diff < 1e-5:
        print("  converged")
        break

# Final: save BEST iter's M
out = os.path.join(os.path.dirname(__file__), "jaxhps_iti_2nd.npz")
M_save = best_M if best_M is not None else M_now
np.savez(out, umeas=M_save, tx=np.asarray(tx))
print(f"\n  best iter = {best_iter}, err = {best_err * 100:.3f}%")
print(f"  saved -> {out}")
