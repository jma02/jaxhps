"""jaxhps 3D ItI driver: 1st-order Sommerfeld with analytic plane-wave trace.

Solves the homogeneous constant-coefficient Helmholtz equation

    (Delta + kappa^2) u = 0   on   [-L_PHYS, L_PHYS]^3

with boundary data set to the exact 1st-order Sommerfeld trace of the
plane wave u*(x) = exp(i k . x), |k| = kappa. With the analytic trace
prescribed, the 3D ItI solver should recover u* to machine precision.

This is the clean validation companion to ``run_iti_2nd_order_abc.py``:
constant coefficients, analytic reference, no smoothing, no scatterer.
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
from planewave_setup import (
    K_HAT,
    KAPPA,
    g1_sommerfeld_trace,
    u_exact,
)

# --------------- params ---------------
L_PHYS = float(os.environ.get("ITI_L_PHYS", "1.0"))
P = int(os.environ.get("P", "12"))
Q = int(os.environ.get("Q", "10"))
L = int(os.environ.get("L", "2"))

print(
    f"jaxhps-ItI 1st-order plane-wave: L_PHYS={L_PHYS} p={P} q={Q} L={L} "
    f"kappa={KAPPA}"
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
print(f"  built in {time.perf_counter() - t0:.1f}s")

# --------------- analytic 1st-order Sommerfeld trace ---------------
bp = np.asarray(domain.boundary_points)  # (n_bdry, 3)
g1 = g1_sommerfeld_trace(bp, L_PHYS).astype(np.complex128)
bdry = jnp.array(g1.reshape(-1, 1))
print(f"  bdry n = {bp.shape[0]}, ||g1||_2 = {np.linalg.norm(g1):.3e}")

# --------------- solve ---------------
t0 = time.perf_counter()
print("solving ...")
u_comp = solve(problem, bdry)
u_comp.block_until_ready()
print(f"  solved in {time.perf_counter() - t0:.1f}s")

# --------------- compare to analytic ---------------
u_ref = u_exact(np.asarray(xg), np.asarray(yg), np.asarray(zg))[..., None]
u_c = np.asarray(u_comp)

err_rel = np.linalg.norm(u_c - u_ref) / np.linalg.norm(u_ref)
err_max = np.max(np.abs(u_c - u_ref)) / np.max(np.abs(u_ref))
print(f"\n  rel L2 err   = {err_rel:.3e}")
print(f"  rel Linf err = {err_max:.3e}")

out = os.path.join(os.path.dirname(__file__), "jaxhps_iti_1st.npz")
np.savez(out, u_computed=u_c, u_exact=u_ref, points=np.asarray(pts))
print(f"  saved -> {out}")
