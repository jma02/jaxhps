"""Mie-like reference solver for 3D Helmholtz scattering by a radial medium.

Solves the 3D total-field Helmholtz equation

    Lap u + kappa^2 (1 - b(r)) u = 0

with an incident plane wave ``u^inc(x) = exp(i kappa w . x)`` (``w`` a unit
direction) and an outgoing-radiating scattered field ``u^s = u - u^inc``.
``b`` is assumed radial -- ``b = b(r)`` -- and compactly supported with
``b(r) = 0`` for ``r >= R_supp``.  Outside ``r = R_supp``, both the incident
and scattered fields are free-space solutions of the Helmholtz equation and
admit spherical-harmonic expansions in closed form.

For a plane wave propagating along ``+z``:

    u^inc(r, theta) = sum_{ell >= 0} (2 ell + 1) i^ell j_ell(kappa r) P_ell(cos theta).

For an arbitrary direction ``w``, we rotate the target points so that ``w``
aligns with ``+z``, evaluate the field, and rotate back; the scattered field
is rotation-covariant (radial medium), so no other change is needed.

For each ``ell``, the radial part ``R_ell(r)`` satisfies

    R'' + (2/r) R' + [kappa^2 (1 - b(r)) - ell(ell+1)/r^2] R = 0,  r in (0, R_supp),

which we solve numerically from ``r = eps`` (with the regular asymptotic
``R_ell(r) ~ r^ell``) up to ``r = R_supp``.  Outside, ``R`` must equal

    R_ell^{ext}(r) = a_ell j_ell(kappa r) + b_ell h_ell^{(1)}(kappa r),

with ``a_ell = (2 ell + 1) i^ell`` (so that the incident component is
reproduced) and ``b_ell`` the outgoing scattering coefficient we solve for.
Matching ``R_ell`` and ``R_ell'`` at ``r = R_supp`` gives a 2x2 system in the
unknowns ``(c_ell, b_ell)`` where ``c_ell`` is the normalization of the
interior solution.

The scattered field anywhere in ``r > R_supp`` is then

    u^s(r, theta) = sum_ell b_ell h_ell^{(1)}(kappa r) P_ell(cos theta).

Convergence in ``ell`` requires ``ell_max >~ kappa * r_max + buffer``, where
``r_max`` is the largest target radius of interest.  We pick a default of
``2 * (kappa * r_max + 6)`` which is comfortably converged for the regimes
we test.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import lpmv, spherical_jn, spherical_yn


def _h1(ell: int, x: np.ndarray) -> np.ndarray:
    """Spherical Hankel function of the first kind, ``h_ell^{(1)}(x)``."""
    return spherical_jn(ell, x) + 1j * spherical_yn(ell, x)


def _h1_prime(ell: int, x: np.ndarray) -> np.ndarray:
    return spherical_jn(ell, x, derivative=True) + 1j * spherical_yn(
        ell, x, derivative=True
    )


def _rotation_aligning_with_z(w: np.ndarray) -> np.ndarray:
    """Return rotation ``R`` such that ``R @ w = [0, 0, 1]``."""
    w = np.asarray(w, dtype=np.float64)
    w = w / np.linalg.norm(w)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(w, z))
    if c > 1.0 - 1e-15:
        return np.eye(3)
    if c < -1.0 + 1e-15:
        # 180 deg rotation about x-axis.
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(w, z)
    s = np.linalg.norm(v)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + (K @ K) * ((1.0 - c) / (s * s))


def _solve_radial_interior(
    ell: int, kappa: float, b_radial, R_supp: float, eps: float = 1e-4
):
    """Solve the radial Helmholtz ODE on ``[eps * R_supp, R_supp]``.

    Returns ``(R_end, Rp_end)``: value and derivative at ``r = R_supp`` of the
    regular interior solution (normalized so ``R_ell(r) ~ r^ell`` near 0).
    """
    r0 = eps * R_supp

    if ell == 0:
        y0 = np.array([1.0, 0.0])
    else:
        y0 = np.array([r0**ell, ell * r0 ** (ell - 1)])

    ll = ell * (ell + 1)

    def rhs(rr, y):
        R, Rp = y
        Rpp = (
            -2.0 / rr * Rp
            + (ll / (rr * rr) - kappa * kappa * (1.0 - b_radial(rr))) * R
        )
        return np.array([Rp, Rpp])

    sol = solve_ivp(
        rhs,
        (r0, R_supp),
        y0,
        method="RK45",
        rtol=1e-10,
        atol=1e-13,
        dense_output=False,
        max_step=0.05 * R_supp,
    )
    if not sol.success:
        raise RuntimeError(f"radial ODE failed at ell={ell}: {sol.message}")
    return float(sol.y[0, -1]), float(sol.y[1, -1])


def mie_scattering_coefficients(
    kappa: float,
    b_radial,
    R_supp: float,
    ell_max: int,
) -> np.ndarray:
    """Compute the outgoing-Hankel coefficients ``b_ell`` for ell=0..ell_max.

    Assumes a plane wave incident along ``+z`` with amplitude 1.
    """
    kR = kappa * R_supp
    coeffs = np.zeros(ell_max + 1, dtype=np.complex128)
    for ell in range(ell_max + 1):
        R_end, Rp_end = _solve_radial_interior(ell, kappa, b_radial, R_supp)
        jl = spherical_jn(ell, kR)
        jlp = spherical_jn(ell, kR, derivative=True)
        hl = _h1(ell, kR)
        hlp = _h1_prime(ell, kR)
        a_ell = (2 * ell + 1) * (1j**ell)
        #   c R_end           = a_ell jl  + b hl
        #   c (Rp_end / kappa) = a_ell jlp + b hlp
        # Solve the 2x2 for (c, b).
        M = np.array(
            [[R_end, -hl], [Rp_end / kappa, -hlp]], dtype=np.complex128
        )
        rhs_vec = a_ell * np.array([jl, jlp], dtype=np.complex128)
        c_val, b_val = np.linalg.solve(M, rhs_vec)
        coeffs[ell] = b_val
    return coeffs


def mie_scattered_field(
    target_pts: np.ndarray,
    source_direction: np.ndarray,
    kappa: float,
    b_radial,
    R_supp: float,
    ell_max: int | None = None,
) -> np.ndarray:
    """Scattered field ``u^s(x)`` at ``target_pts`` (each strictly outside ``R_supp``).

    Args:
        target_pts:        ``(n_t, 3)`` target locations.
        source_direction:  ``(3,)`` unit vector ``w``.
        kappa:             wavenumber.
        b_radial:          callable; ``b_radial(r)`` for scalar or array r in ``[0, R_supp]``.
        R_supp:            radius beyond which ``b(r) = 0``.
        ell_max:           truncation; if ``None``, picks ``2*(kappa*r_max + 6)``
                           rounded up.
    """
    target_pts = np.asarray(target_pts, dtype=np.float64)
    rot = _rotation_aligning_with_z(
        np.asarray(source_direction, dtype=np.float64)
    )
    pts_rot = target_pts @ rot.T  # apply rotation so w is now along +z

    r = np.linalg.norm(pts_rot, axis=-1)
    if np.any(r <= R_supp + 1e-12):
        raise ValueError(
            f"all target_pts must satisfy r > R_supp ({R_supp}); "
            f"found min(r) = {r.min():.6e}."
        )
    cos_theta = pts_rot[:, 2] / r
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    if ell_max is None:
        ell_max = int(np.ceil(2 * (kappa * r.max() + 6)))

    coeffs = mie_scattering_coefficients(kappa, b_radial, R_supp, ell_max)

    kr = kappa * r
    u_s = np.zeros_like(r, dtype=np.complex128)
    for ell in range(ell_max + 1):
        hl = _h1(ell, kr)
        P = lpmv(0, ell, cos_theta)
        u_s += coeffs[ell] * hl * P
    return u_s
