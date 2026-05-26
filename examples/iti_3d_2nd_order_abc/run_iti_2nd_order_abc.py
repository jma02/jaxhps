"""jaxhps 3D ItI driver: 2nd-order Engquist-Majda ABC, MMS-forced plane wave.

The 2nd-order ABC equation,

    (partial_n - i kappa - (i / (2 kappa)) Delta_tau) u = 0,

is *not* satisfied by a plane wave at oblique incidence -- a plane wave
is what gets reflected by the ABC at non-normal angles. To turn the
2nd-order ABC into something a plane wave can exactly satisfy, we add
the analytic residual

    g_base(x) = -(i kappa / 2) (1 - k_hat . n_hat)^2 u*(x)

as a manufactured boundary forcing. The forced ABC equation,

    (partial_n - i kappa - (i / (2 kappa)) Delta_tau) u = g_base,

is then satisfied exactly by u*(x) = exp(i k . x).

In jaxhps's eta = -kappa convention the impedance trace passed in via
``bdry_data`` is g(x) = u_n(x) - i kappa u(x). Moving the Delta_tau term
to the right-hand side gives

    g = (i / (2 kappa)) Delta_tau u  +  g_base.       (*)

The analytic fixed point of (*) for u = u* is the 1st-order Sommerfeld
trace of the plane wave,

    g_star(x) = i kappa (k_hat . n_hat - 1) u*(x),

because (i/(2 kappa)) Delta_tau u* + g_base reduces to g_star (verify
algebraically). This driver does a one-shot validation:

  step 1.  set bdry_data = g_star, solve, verify u_computed = u* to
           machine precision (tests the 3D ItI solver path with a
           non-trivial analytic trace);
  step 2.  apply ONE iteration update, new_bdry = (i / (2 kappa)) *
           Delta_tau u_computed + g_base, and verify
           ||new_bdry - g_star|| / ||g_star|| is at machine precision
           (tests the 2nd-order ABC update operator: the per-face
           Delta_tau on the 3D Cheby grid, Cheby-to-Gauss interpolation,
           and leaf-to-cube-face routing).

We do not run a Picard iteration to *find* the fixed point. The 2nd-order
EM ABC iteration is well known to be unstable at large kappa h for any
relaxation parameter -- modes with k_tau^2 / (2 kappa) > 1 are amplified
by Delta_tau / (2 kappa) faster than they can be damped. For the
constant-coefficient plane-wave problem these high-tangential modes are
populated everywhere on the boundary, so the iteration diverges from
zero. A more robust scheme (e.g. solving (*) directly with GMRES, or
applying a tangential low-pass filter to bdry_data) would be needed to
*find* the fixed point starting from a poor initial guess; the
fixed-point verification above tests the same Delta_tau / Gauss / face
machinery without depending on Picard stability.
"""

# ruff: noqa: E402
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

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
from planewave_setup import (
    K_HAT,
    KAPPA,
    g1_sommerfeld_trace,
    g_base_residual,
    u_exact,
)

# --------------- params ---------------
L_PHYS = float(os.environ.get("ITI_L_PHYS", "1.0"))
P = int(os.environ.get("P", "12"))
Q = int(os.environ.get("Q", "10"))
L = int(os.environ.get("L", "2"))

print(
    f"jaxhps-ItI 2nd-order plane-wave: L_PHYS={L_PHYS} p={P} q={Q} L={L} "
    f"eta=-{KAPPA}"
)
print(f"  k_hat = {K_HAT}")

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

ones = jnp.ones_like(xg)
I_coeffs = (KAPPA**2) * ones  # constant epsilon = 1 => I_coeff = kappa^2

# Homogeneous PDE (no scatterer, no incident-field source).
sources = jnp.zeros((*xg.shape, 1), dtype=jnp.complex128)

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
print("build_solver ...")
build_solver(problem)
print(f"  built in {time.perf_counter() - t0:.0f}s")

# --------------- 2nd-order ABC surface-Laplacian machinery ---------------
h_leaf = L_PHYS / (2**L)
diff_ops = precompute_diff_operators_3D(P, h_leaf)
_, _, _, d_xx, d_yy, d_zz, _, _, _ = diff_ops
Dt_x = d_yy + d_zz  # Delta_tau on x-normal faces
Dt_y = d_xx + d_zz  # Delta_tau on y-normal faces
Dt_z = d_xx + d_yy  # Delta_tau on z-normal faces

cheby_1d = chebyshev_points(P)
gauss_1d = gauss_points(Q)
Q_2D = barycentric_lagrange_interpolation_matrix_2D(
    cheby_1d, cheby_1d, gauss_1d, gauss_1d
)  # (q^2, p^2)

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
    """Δ_τ u on a Gauss face: shape (q^2, n_rhs)."""
    Dt = Dt_per_face[face_idx]
    fi = face_idxes_list[face_idx]
    lap_volume = Dt @ u_volume_leaf  # (p^3, n_rhs)
    lap_face_cheby = lap_volume[fi]  # (p^2, n_rhs)
    return Q_2D @ lap_face_cheby  # (q^2, n_rhs)


# --------------- map outer leaf-faces <-> bdry_data block positions ---------------
# bdry_data is laid out face-by-face: [face1 | face2 | ... | face6], each
# block contains (2^L)^2 leaf-faces * q^2 Gauss points.
leaves = get_all_uniform_leaves_3D(domain.root, domain.L)
leaf_bounds = np.array(
    [[n.xmin, n.xmax, n.ymin, n.ymax, n.zmin, n.zmax] for n in leaves]
)  # (n_leaves, 6)
n_leaves = leaf_bounds.shape[0]
n_per_face = (2**L) ** 2 * Q**2
n_per_leaf_face = Q**2

bdry_pts = np.asarray(domain.boundary_points)
print(f"  bdry total = {bdry_pts.shape[0]}, per-face = {n_per_face}")

cube_faces = [
    ("xmin", 0, lambda b: np.isclose(b[0], -L_PHYS), 1, 2),
    ("xmax", 1, lambda b: np.isclose(b[1], L_PHYS), 1, 2),
    ("ymin", 2, lambda b: np.isclose(b[2], -L_PHYS), 0, 2),
    ("ymax", 3, lambda b: np.isclose(b[3], L_PHYS), 0, 2),
    ("zmin", 4, lambda b: np.isclose(b[4], -L_PHYS), 0, 1),
    ("zmax", 5, lambda b: np.isclose(b[5], L_PHYS), 0, 1),
]

mapping = []
for name, cf_idx, leaf_filter, t1, t2 in cube_faces:
    block_start = cf_idx * n_per_face
    pts_face = bdry_pts[block_start : block_start + n_per_face]
    n_leafblocks = n_per_face // n_per_leaf_face
    for blk in range(n_leafblocks):
        chunk = pts_face[blk * n_per_leaf_face : (blk + 1) * n_per_leaf_face]
        c_t1 = chunk[:, t1].mean()
        c_t2 = chunk[:, t2].mean()
        best = None
        for li in range(n_leaves):
            if not leaf_filter(leaf_bounds[li]):
                continue
            m_t1 = (leaf_bounds[li, 2 * t1] + leaf_bounds[li, 2 * t1 + 1]) / 2
            m_t2 = (leaf_bounds[li, 2 * t2] + leaf_bounds[li, 2 * t2 + 1]) / 2
            d = abs(m_t1 - c_t1) + abs(m_t2 - c_t2)
            if best is None or d < best[1]:
                best = (li, d)
        assert best is not None
        mapping.append((cf_idx, best[0], block_start + blk * n_per_leaf_face))

print(f"  mapping len = {len(mapping)} (expected {6 * (2**L) ** 2})")

# --------------- analytic fixed point and MMS forcing ---------------
g_star = (
    g1_sommerfeld_trace(bdry_pts, L_PHYS).astype(np.complex128).reshape(-1, 1)
)
g_base = g_base_residual(bdry_pts, L_PHYS).astype(np.complex128).reshape(-1, 1)
print(f"  ||g_star||_2 = {np.linalg.norm(g_star):.3e}")
print(f"  ||g_base||_2 = {np.linalg.norm(g_base):.3e}")

# --------------- step 1: solve at the analytic fixed point ---------------
bdry = jnp.array(g_star)
t0 = time.perf_counter()
print("\nstep 1: solve(problem, bdry = g_star) ...")
u_comp = solve(problem, bdry)
u_comp.block_until_ready()
print(f"  solved in {time.perf_counter() - t0:.1f}s")

u_ref = u_exact(np.asarray(xg), np.asarray(yg), np.asarray(zg))[..., None]
u_c = np.asarray(u_comp)
err_u_rel = np.linalg.norm(u_c - u_ref) / np.linalg.norm(u_ref)
err_u_max = np.max(np.abs(u_c - u_ref)) / np.max(np.abs(u_ref))
print(f"  rel L2 err   ||u - u*|| / ||u*||   = {err_u_rel:.3e}")
print(f"  rel Linf err |u - u*|_inf / |u*|_inf = {err_u_max:.3e}")


# --------------- step 2: apply ONE iteration update from u_computed ---------------
def update_bdry(u_volume):
    """Compute (i / (2 kappa)) * Delta_tau u_outer + g_base on each face."""
    new = np.zeros(g_star.shape, dtype=np.complex128)
    factor = 1j / (2.0 * KAPPA)
    for cf, li, st in mapping:
        u_leaf = u_volume[li]  # (p^3, 1)
        lap_face_gauss = np.asarray(per_face_lap_to_gauss(u_leaf, cf))
        new[st : st + n_per_leaf_face, :] = factor * lap_face_gauss
    return new + g_base


print("\nstep 2: new_bdry = (i / (2 kappa)) Delta_tau u  +  g_base")
t0 = time.perf_counter()
new_bdry = update_bdry(u_comp)
print(f"  built in {time.perf_counter() - t0:.1f}s")
err_g_rel = np.linalg.norm(new_bdry - g_star) / np.linalg.norm(g_star)
err_g_max = np.max(np.abs(new_bdry - g_star)) / np.max(np.abs(g_star))
print(f"  rel L2 err   ||new_bdry - g_star|| / ||g_star||   = {err_g_rel:.3e}")
print(
    f"  rel Linf err |new_bdry - g_star|_inf / |g_star|_inf = {err_g_max:.3e}"
)

out = os.path.join(os.path.dirname(__file__), "jaxhps_iti_2nd.npz")
np.savez(
    out,
    u_computed=u_c,
    u_exact=u_ref,
    g_star=g_star,
    new_bdry=new_bdry,
    points=np.asarray(pts),
)
print(f"\n  saved -> {out}")
