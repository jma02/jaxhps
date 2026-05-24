"""Exact Mie series reference for the strong-scattering single sphere.

Computes the 24x24 measurement matrix M[irx, itx] where:
  source at x_tx is a point source u^i(y) = exp(i k |y - x_tx|) / (4 pi |y - x_tx|)
  receiver evaluates u^s = u^total - u^i at x_rx
  M[irx, itx] = u^s(x_rx; x_tx)

For a hard sphere (radius R, refractive index N inside, 1 outside, center c)
in homogeneous water:

  u^s(y) = (i k / 4 pi) sum_l (2l+1) s_l h_l(k r_y) h_l(k r_tx) P_l(cos gamma)
                                                              for r_y > R

where r_tx = |x_tx - c|, r_y = |y - c|, cos gamma = (x_tx-c).(y-c)/(r_tx r_y).
The scattering coefficients s_l come from matching u and du/dr at r = R.

Note: the HPS solver uses a SMOOTH sphere (tanh ramp w=0.02) to keep the
coefficient field C^infty. The hard-sphere Mie series is therefore an
approximate reference for the smoothed solve at order O(w/R). With
w=0.02 and R=0.5 the resulting Mie-vs-truth gap is small relative to the
ABC-induced error we are studying here.
"""

import os
import sys
import time
import numpy as np
from scipy.special import spherical_jn, spherical_yn, eval_legendre

sys.path.insert(0, os.path.dirname(__file__))
from mie_setup import KAPPA, SPHERE_CENTER, SPHERE_RADIUS, EPS_SPHERE, get_tx


def hankel1(l, x):
    return spherical_jn(l, x) + 1j * spherical_yn(l, x)


def hankel1_prime(l, x):
    return spherical_jn(l, x, derivative=True) + 1j * spherical_yn(
        l, x, derivative=True
    )


def mie_s_l(l, k, R, eps):
    """Mie scattering coefficient for a penetrable sphere.
    eps = epsilon = n^2 (relative permittivity inside the sphere)."""
    k_in = k * np.sqrt(eps)
    jl_kR = spherical_jn(l, k * R)
    jlp_kR = spherical_jn(l, k * R, derivative=True)
    hl_kR = hankel1(l, k * R)
    hlp_kR = hankel1_prime(l, k * R)
    jl_kinR = spherical_jn(l, k_in * R)
    jlp_kinR = spherical_jn(l, k_in * R, derivative=True)
    # Avoid 0/0
    if abs(jl_kinR) < 1e-300:
        return 0.0 + 0.0j
    beta = k_in * jlp_kinR / jl_kinR
    num = beta * jl_kR - k * jlp_kR
    den = k * hlp_kR - beta * hl_kR
    return num / den


def mie_M(tx_coords, rx_coords, kappa, c, R, eps, lmax):
    """Build M[irx, itx] = u^s at rx_coords[irx] with source at tx_coords[itx]."""
    Ntx = len(tx_coords)
    Nrx = len(rx_coords)
    s = np.array([mie_s_l(l, kappa, R, eps) for l in range(lmax + 1)])
    r_tx = np.linalg.norm(tx_coords - c, axis=1)
    r_rx = np.linalg.norm(rx_coords - c, axis=1)
    hl_tx = np.array(
        [
            [hankel1(l, kappa * r_tx[i]) for l in range(lmax + 1)]
            for i in range(Ntx)
        ]
    )
    hl_rx = np.array(
        [
            [hankel1(l, kappa * r_rx[i]) for l in range(lmax + 1)]
            for i in range(Nrx)
        ]
    )
    ls = np.arange(lmax + 1)
    M = np.zeros((Nrx, Ntx), dtype=complex)
    for irx in range(Nrx):
        u_rx = (rx_coords[irx] - c) / r_rx[irx]
        for itx in range(Ntx):
            u_tx = (tx_coords[itx] - c) / r_tx[itx]
            cos_g = float(np.clip(np.dot(u_rx, u_tx), -1.0, 1.0))
            Pls = np.array([eval_legendre(l, cos_g) for l in range(lmax + 1)])
            acc = np.sum((2 * ls + 1) * s * hl_tx[itx] * hl_rx[irx] * Pls)
            M[irx, itx] = 1j * kappa / (4 * np.pi) * acc
    return M


def main():
    tx = get_tx()
    rx = tx
    c = np.asarray(SPHERE_CENTER)
    print(
        f"Mie reference: kappa={KAPPA} R={SPHERE_RADIUS} eps={EPS_SPHERE} kR={KAPPA * SPHERE_RADIUS:.2f}"
    )
    print(
        f"  N_TX = {len(tx)} on sphere of radius {np.linalg.norm(tx[0]):.2f}"
    )

    # Truncation: lmax >> kR_in to capture interior modes
    lmax = max(
        int(np.ceil(KAPPA * np.sqrt(EPS_SPHERE) * SPHERE_RADIUS + 30)),
        int(np.ceil(KAPPA * SPHERE_RADIUS + 30)),
    )
    print(f"  lmax = {lmax}")

    t0 = time.perf_counter()
    M = mie_M(tx, rx, KAPPA, c, SPHERE_RADIUS, EPS_SPHERE, lmax)
    print(f"  built in {time.perf_counter() - t0:.2f}s")

    # Truncation convergence
    M_lo = mie_M(
        tx[:2], tx[:2], KAPPA, c, SPHERE_RADIUS, EPS_SPHERE, lmax - 10
    )
    err_trunc = np.linalg.norm(M_lo - M[:2, :2]) / np.linalg.norm(M[:2, :2])
    print(f"  truncation rel err (lmax vs lmax-10) = {err_trunc:.2e}")

    # Reciprocity
    rec = np.linalg.norm(M - M.T) / np.linalg.norm(M)
    print(f"  reciprocity ||M - M^T|| / ||M|| = {rec:.3e}")

    print(f"  ||M||_F = {np.linalg.norm(M):.4e}")

    out = os.path.join(os.path.dirname(__file__), "mie.npz")
    np.savez(
        out,
        umeas=M,
        tx=tx,
        kappa=KAPPA,
        sphere_center=c,
        sphere_radius=SPHERE_RADIUS,
        eps_sphere=EPS_SPHERE,
    )
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
