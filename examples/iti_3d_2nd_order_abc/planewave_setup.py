"""Plane-wave manufactured-solution setup for the 3D ItI ABC examples.

The two drivers in this folder validate the 3D ItI solver path against an
analytic solution. The PDE is the homogeneous constant-coefficient Helmholtz
equation

    (Delta + kappa^2) u = 0   on   Omega = [-L_PHYS, L_PHYS]^3,

and the manufactured solution is a plane wave

    u*(x) = exp(i k . x),   |k| = kappa.

Constant coefficients are used deliberately to remove any
discretization-of-coefficient confound: any error in the recovered
solution is attributable to the discretization of u or to the boundary
condition itself, not to a non-smooth coefficient field.

Conventions used by the drivers
-------------------------------
jaxhps's 3D ItI machinery uses ``eta = -kappa`` so that the impedance
trace passed in via ``bdry_data`` is

    g(x) = (partial u / partial n)(x) - i kappa u(x).

For a plane wave evaluated on a cube face with outward normal n_hat and
c = k_hat . n_hat, this trace is

    g_1*(x) = i kappa (c - 1) u*(x).

The 2nd-order Engquist-Majda ABC equation
``(partial_n - i kappa - (i / (2 kappa)) Delta_tau) u = 0`` is *not*
satisfied by a plane wave at angles c != 1, so to make u* the exact
solution of an inhomogeneous version of that ABC we add the analytic
residual

    g_base(x) = g_1*(x) - (i / (2 kappa)) Delta_tau u*(x)
              = -(i kappa / 2) (1 - c)^2 u*(x).

The 2nd-order driver iterates
``g <- (i / (2 kappa)) Delta_tau u_outer + g_base``. Its unique fixed
point is g = g_1*, which forces u_computed = u*.
"""

import numpy as np

# ---- problem parameters ----
KAPPA = 4.0

# Plane-wave direction. We pick a generic non-axis-aligned direction so
# every cube face sees a non-trivial (c != +-1) angle. Normalise to unit
# length so |K_VEC| = KAPPA exactly.
_K_DIR = np.array([1.0, 2.0, 2.0])
K_HAT = _K_DIR / np.linalg.norm(_K_DIR)
K_VEC = KAPPA * K_HAT


def u_exact(x, y, z):
    """Exact plane-wave solution evaluated at points (x, y, z)."""
    return np.exp(1j * (K_VEC[0] * x + K_VEC[1] * y + K_VEC[2] * z))


def outward_normals_for_boundary(bp, L_phys, tol=1e-9):
    """Return the outward unit normal at each cube-boundary Gauss point.

    bp: (n_bdry, 3) array of boundary point coordinates.
    L_phys: cube half-side.
    """
    n = np.zeros_like(bp)
    n[np.abs(bp[:, 0] + L_phys) < tol] = [-1.0, 0.0, 0.0]
    n[np.abs(bp[:, 0] - L_phys) < tol] = [1.0, 0.0, 0.0]
    n[np.abs(bp[:, 1] + L_phys) < tol] = [0.0, -1.0, 0.0]
    n[np.abs(bp[:, 1] - L_phys) < tol] = [0.0, 1.0, 0.0]
    n[np.abs(bp[:, 2] + L_phys) < tol] = [0.0, 0.0, -1.0]
    n[np.abs(bp[:, 2] - L_phys) < tol] = [0.0, 0.0, 1.0]
    return n


def g1_sommerfeld_trace(bp, L_phys):
    """Analytic 1st-order Sommerfeld trace g_1*(x) = i kappa (c - 1) u*(x)."""
    n_hat = outward_normals_for_boundary(bp, L_phys)
    assert np.all(np.linalg.norm(n_hat, axis=1) > 0.5), (
        "Some boundary points were not assigned a normal"
    )
    u_b = u_exact(bp[:, 0], bp[:, 1], bp[:, 2])
    c = n_hat @ K_HAT  # k_hat . n_hat
    return 1j * KAPPA * (c - 1.0) * u_b


def g_base_residual(bp, L_phys):
    """MMS residual for the 2nd-order ABC iteration:
    g_base(x) = -(i kappa / 2) (1 - c)^2 u*(x)."""
    n_hat = outward_normals_for_boundary(bp, L_phys)
    assert np.all(np.linalg.norm(n_hat, axis=1) > 0.5), (
        "Some boundary points were not assigned a normal"
    )
    u_b = u_exact(bp[:, 0], bp[:, 1], bp[:, 2])
    c = n_hat @ K_HAT
    return -0.5j * KAPPA * (1.0 - c) ** 2 * u_b
