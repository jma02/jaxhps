"""Partial-wave reference with origin-scaled radial solutions.

Writing R_l(r)=r^l F_l(r) removes the very small r^l initial amplitudes.
The matching coefficient uses F and F'+l F/R, so r^l cancels entirely.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import eval_legendre, spherical_jn, spherical_yn


def radial_reference(
    targets,
    direction,
    kappa,
    coefficient,
    radius,
    *,
    ell_max=32,
    rtol=1e-10,
    atol=1e-12,
    origin_fraction=1e-5,
    step_fraction=0.025,
):
    """Exterior scattered field for a radial coefficient supported in R."""
    targets = np.asarray(targets)
    distances = np.linalg.norm(targets, axis=-1)
    if np.any(distances <= radius):
        raise ValueError("reference targets must lie outside the support")
    direction = np.asarray(direction) / np.linalg.norm(direction)
    cosine = np.clip(targets @ direction / distances, -1, 1)
    field = np.zeros(len(targets), dtype=complex)
    kR = kappa * radius
    r0 = origin_fraction * radius
    origin_k2 = kappa**2 * (1 - coefficient(0.0))
    for ell in range(ell_max + 1):

        def rhs(r, f):
            return [
                f[1],
                -2 * (ell + 1) * f[1] / r
                - kappa**2 * (1 - coefficient(r)) * f[0],
            ]

        curvature = origin_k2 / (2 * ell + 3)
        sol = solve_ivp(
            rhs,
            (r0, radius),
            [1 - curvature * r0**2 / 2, -curvature * r0],
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=step_fraction * radius,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        f, df = sol.y[:, -1]
        derivative = (df + ell * f / radius) / kappa
        j = spherical_jn(ell, kR)
        dj = spherical_jn(ell, kR, derivative=True)
        hankel = j + 1j * spherical_yn(ell, kR)
        dh = dj + 1j * spherical_yn(ell, kR, derivative=True)
        outgoing = (
            (2 * ell + 1)
            * 1j**ell
            * (f * dj - derivative * j)
            / (derivative * hankel - f * dh)
        )
        kr = kappa * distances
        field += (
            outgoing
            * (spherical_jn(ell, kr) + 1j * spherical_yn(ell, kr))
            * eval_legendre(ell, cosine)
        )
    return field
