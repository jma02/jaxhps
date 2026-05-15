"""jaxhps 3D ItI driver for the strong-scattering sphere problem.

Uses the impedance-to-impedance merge path (devin/1777254507-iti-3d branch).
First-order Sommerfeld ABC on outer cube faces (eta=-kappa, g_in=0).
No PML, no Dirichlet outer wall.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import jax
import jax.numpy as jnp
from jax import config as jcfg
jcfg.update('jax_enable_x64', True)

from jaxhps import Domain, DiscretizationNode3D, PDEProblem, build_solver, solve
from jaxhps._interpolation_methods import vmapped_interp_to_point_3D
from jaxhps._grid_creation_3D import get_all_uniform_leaves_3D

from mie_setup import KAPPA, SPHERE_RADIUS, EPS_SPHERE, W_RAMP, get_tx

# ItI uses Robin ABC on outer faces; keep a buffer between r=1 sensors and cube boundary
L_PHYS = float(os.environ.get('ITI_L_PHYS', '2.0'))
P = int(os.environ.get('P', '12'))
Q = int(os.environ.get('Q', '10'))
L = int(os.environ.get('L', '2'))
print(f"jaxhps-ItI: L_PHYS={L_PHYS} p={P} q={Q} L={L} eta=-{KAPPA}")

tx = jnp.array(get_tx())
N_TX = tx.shape[0]
print(f"N_TX = {N_TX}")

root = DiscretizationNode3D(xmin=-L_PHYS, xmax=L_PHYS,
                            ymin=-L_PHYS, ymax=L_PHYS,
                            zmin=-L_PHYS, zmax=L_PHYS)
domain = Domain(p=P, q=Q, root=root, L=L)
pts = domain.interior_points
xg = pts[..., 0]; yg = pts[..., 1]; zg = pts[..., 2]

# Smoothed sphere
r = jnp.sqrt(xg * xg + yg * yg + zg * zg)
mask = 0.5 * (1 - jnp.tanh((r - SPHERE_RADIUS) / W_RAMP))
n_field = 1.0 + (EPS_SPHERE - 1.0) * mask  # = epsilon

ones = jnp.ones_like(xg)
I_coeffs = (KAPPA ** 2) * n_field

def source_for_tx(t):
    rr = jnp.sqrt((xg - t[0])**2 + (yg - t[1])**2 + (zg - t[2])**2 + 1e-8)
    ui = jnp.exp(1j * KAPPA * rr) / (4 * jnp.pi * rr)
    return (KAPPA ** 2) * (1.0 - n_field) * ui

sources = jax.vmap(source_for_tx, in_axes=0, out_axes=-1)(tx)
print(f"sources shape: {sources.shape}")

problem = PDEProblem(
    domain=domain,
    D_xx_coefficients=ones, D_yy_coefficients=ones, D_zz_coefficients=ones,
    I_coefficients=I_coeffs,
    source=sources,
    use_ItI=True,
    eta=-float(KAPPA),
)

t0 = time.perf_counter()
print("build ...")
build_solver(problem)
print(f"  built in {time.perf_counter() - t0:.1f}s")

n_bdry = domain.boundary_points.shape[0]
bdry = jnp.zeros((n_bdry, N_TX), dtype=jnp.complex128)

print("solve ...")
t0 = time.perf_counter()
u_scat = solve(problem, bdry); u_scat.block_until_ready()
print(f"  {N_TX} RHS solved in {time.perf_counter() - t0:.1f}s")

def interp_point_cloud(domain, samples, pts):
    leaves = get_all_uniform_leaves_3D(domain.root, domain.L)
    corners = jnp.stack([jnp.array([[n.xmin, n.ymin, n.zmin], [n.xmax, n.ymax, n.zmax]])
                         for n in leaves])
    b = ((pts[:, 0, None] >= corners[None, :, 0, 0]) & (pts[:, 0, None] <= corners[None, :, 1, 0]) &
         (pts[:, 1, None] >= corners[None, :, 0, 1]) & (pts[:, 1, None] <= corners[None, :, 1, 1]) &
         (pts[:, 2, None] >= corners[None, :, 0, 2]) & (pts[:, 2, None] <= corners[None, :, 1, 2]))
    patch = jnp.argmax(b, axis=1)
    corn = corners[patch]
    M = []
    for i in range(samples.shape[-1]):
        f = samples[patch, :, i]
        vals = vmapped_interp_to_point_3D(pts[:, 0], pts[:, 1], pts[:, 2], corn, f, domain.p)
        M.append(vals)
    return jnp.stack(M, axis=-1)

M = jnp.squeeze(interp_point_cloud(domain, u_scat, tx))
M_np = np.asarray(M)
print(f"||M||_F = {np.linalg.norm(M_np):.4e}")
print(f"reciprocity ||M-M^T||/||M|| = {np.linalg.norm(M_np - M_np.T) / np.linalg.norm(M_np):.3e}")
out = os.path.join(os.path.dirname(__file__), 'jaxhps_iti.npz')
np.savez(out, umeas=M_np, tx=np.asarray(tx))
print(f"saved -> {out}")
