r"""Free-space scattering by a spherical scatterer: radial source/receiver sweep.

A sanity test of the 3D HPS+BIE solver on the ``z = 0`` slice.  A radial
(spherically symmetric) scatterer sits at the origin in free space.  In the
``z = 0`` plane we place ``N`` plane-wave transmitters whose directions point
radially, ``w_i = (cos a_i, sin a_i, 0)`` with ``a_i = 2 pi i / N``, and ``N``
receivers on a circle of radius ``rho`` at the same angles,
``r_j = rho (cos a_j, sin a_j, 0)``.  We measure the scattered field
``u^s(w_i, r_j)`` for every transmitter/receiver pair.

Because the medium is rotationally invariant about the ``z`` axis, the
scattered field depends only on the angular separation
``gamma = a_j - a_i`` (at fixed ``rho``):

    u^s(w_i, r_j) = g(a_j - a_i).

Hence, once the transmitters and receivers are sorted by angle, the
measurement matrix is circulant (constant along diagonals) and every row is
the same smooth, sinusoidal function ``g`` of the angular separation -- the
"sine waves" expected from the plane-wave excitation.  We verify this
collapse quantitatively and, when available, against the Mie reference.

Usage:
    python examples/sphere_radial_scattering_3d.py \
        --npz data/examples/SD_3D/SD_k4_q8_L2_a1.25.npz \
        --N 64 --rho 2.5 --out sphere_radial
"""

import argparse
import os
import sys

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
from wave_scattering_utils_3D import (  # noqa: E402
    eval_uscat_offsurface_3D,
    load_SD_matrices_3D,
    solve_scattering_bie_3D,
)
from mie_3d import mie_scattered_field  # noqa: E402


def make_b_radial(R_bump, A_bump, shape):
    """Smooth, compactly supported radial scattering potential ``b(r)``."""
    if shape == "poly":

        def b_radial(r):
            rho = np.where(r < R_bump, r / R_bump, 1.0)
            return np.where(r < R_bump, A_bump * (1.0 - rho * rho) ** 4, 0.0)

        text = rf"$b(r) = {A_bump}\,(1-(r/{R_bump})^2)^4\,1_{{r<{R_bump}}}$"
    else:

        def b_radial(r):
            r = np.asarray(r, dtype=float)
            out = np.zeros_like(r)
            inside = r < R_bump
            t = 1.0 - (r[inside] / R_bump) ** 2
            out[inside] = A_bump * np.exp(1.0 - 1.0 / t)
            return out

        text = (
            rf"$b(r) = {A_bump}\,e^{{1 - 1/(1-(r/{R_bump})^2)}}"
            rf"\,1_{{r<{R_bump}}}$"
        )
    return b_radial, text


def measurement_matrix(sd, b_radial, angles, rho, p):
    """Scattered field ``u^s[j, i]`` at receiver ``j`` for transmitter ``i``.

    Transmitters are plane waves with in-plane radial directions; receivers
    sit on the radius-``rho`` circle at the same angles.  Returns the
    ``(N, N)`` complex matrix.
    """
    kappa = sd["kappa"]
    dirs = np.stack(
        [np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=-1
    )
    rx = rho * dirs

    out = solve_scattering_bie_3D(sd, b_radial, dirs, p=p)
    M = np.asarray(
        eval_uscat_offsurface_3D(
            target_pts=jnp.asarray(rx),
            src_pts=jnp.asarray(out["boundary_points"]),
            src_normals=jnp.asarray(out["normals"]),
            src_weights=jnp.asarray(out["sdp"]["wts"]),
            uscat_b=jnp.asarray(out["uscat_b"]),
            uscat_dn_b=jnp.asarray(out["uscat_dn_b"]),
            k=float(kappa),
        )
    )  # (n_rx, n_tx)
    return M, rx, dirs


def circulant_collapse(M):
    """Roll each transmitter column to a common origin and average.

    With ``M[j, i] = g(a_j - a_i)`` on a uniform angular grid, ``g`` is
    recovered by aligning column ``i`` to separation index ``j - i (mod N)``.
    Returns ``(g, residual)`` where ``residual`` is the max deviation of any
    column from the mean (a measure of the rotational symmetry).
    """
    N = M.shape[0]
    aligned = np.stack(
        [np.roll(M[:, i], -i) for i in range(N)], axis=0
    )  # (n_tx, sep)
    g = aligned.mean(axis=0)
    residual = float(np.max(np.abs(aligned - g[None, :])))
    return g, residual


def mie_curve(b_radial, kappa, rho, R_bump, seps):
    """Mie ``g(gamma)``: scattered field at separation angle ``gamma``.

    Transmitter along ``+x``; receiver at ``rho (cos gamma, sin gamma, 0)``.
    """
    rx = rho * np.stack(
        [np.cos(seps), np.sin(seps), np.zeros_like(seps)], axis=-1
    )
    return mie_scattered_field(
        target_pts=rx,
        source_direction=np.array([1.0, 0.0, 0.0]),
        kappa=float(kappa),
        b_radial=b_radial,
        R_supp=R_bump,
        ell_max=int(np.ceil(2 * (kappa * rho + 6))),
    )


def plot_geometry(rx, R_bump, a, fname):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 6.4), constrained_layout=True)
    ax.add_patch(plt.Circle((0, 0), R_bump, color="0.6", alpha=0.5))
    ax.add_patch(
        plt.Rectangle(
            (-a, -a), 2 * a, 2 * a, fill=False, ls="--", ec="k", lw=1.0
        )
    )
    ax.scatter(rx[:, 0], rx[:, 1], s=22, c="C0", zorder=3, label="tx/rx")
    for r in rx:
        ax.plot([0, r[0]], [0, r[1]], color="C0", lw=0.3, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        rf"$z=0$ slice: scatterer (supp $r<{R_bump}$), HPS cube,"
        "\n"
        rf"radial tx/rx ($\rho={np.linalg.norm(rx[0]):.2f}$)"
    )
    ax.legend(loc="upper right")
    fig.savefig(fname, dpi=130)
    print(f"wrote {fname}")


def plot_matrix(M, fname):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, m, title, kw in zip(
        axes,
        [np.real(M), np.abs(M)],
        [r"$\Re\,u^s$", r"$|u^s|$"],
        [dict(cmap="RdBu_r"), dict(cmap="viridis", vmin=0)],
    ):
        if "RdBu" in str(kw.get("cmap")):
            vm = float(np.abs(np.real(M)).max())
            kw.update(vmin=-vm, vmax=vm)
        im = ax.imshow(m, origin="lower", **kw)
        ax.set_title(f"{title}: measurement matrix")
        ax.set_xlabel("transmitter index (angle)")
        ax.set_ylabel("receiver index (angle)")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(fname, dpi=130)
    print(f"wrote {fname}")


def plot_sorted(M, g, seps, mie, residual, rel_mie, fname):
    import matplotlib.pyplot as plt

    N = M.shape[0]
    deg = np.degrees(seps)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    # Left: every transmitter's receiver trace, aligned to angular separation.
    ax = axes[0]
    for i in range(N):
        ax.plot(deg, np.real(np.roll(M[:, i], -i)), color="0.7", lw=0.6)
    ax.plot(deg, np.real(g), "C3", lw=2.0, label=r"mean $\Re\,g(\gamma)$")
    ax.set_title(
        rf"Aligned receiver traces (all {N} tx collapse), "
        rf"residual ${residual:.1e}$"
    )
    ax.set_xlabel(r"angular separation $\gamma = \beta - \alpha$ (deg)")
    ax.set_ylabel(r"$\Re\,u^s$")
    ax.legend(loc="upper right")

    # Right: sorted pattern g(gamma) vs Mie.
    ax = axes[1]
    ax.plot(deg, np.real(g), "C0", lw=2.0, label=r"HPS $\Re\,g$")
    ax.plot(deg, np.imag(g), "C1", lw=2.0, label=r"HPS $\Im\,g$")
    if mie is not None:
        ax.plot(deg, np.real(mie), "k--", lw=1.0, label=r"Mie $\Re\,g$")
        ax.plot(deg, np.imag(mie), "k:", lw=1.0, label=r"Mie $\Im\,g$")
    ttl = r"Sorted scattered pattern $g(\gamma)$"
    if rel_mie is not None:
        ttl += rf"  (rel. err vs Mie ${rel_mie:.1e}$)"
    ax.set_title(ttl)
    ax.set_xlabel(r"angular separation $\gamma$ (deg)")
    ax.set_ylabel(r"$u^s$")
    ax.legend(loc="upper right", ncol=2)
    fig.savefig(fname, dpi=130)
    print(f"wrote {fname}")


def plot_traces(M, angles, fname, n_show=6):
    """Scattered field at a few fixed receivers vs transmitter angle.

    Each receiver gives ``M[j, :] = g(a_j - alpha)``: the same sinusoid in the
    transmitter angle ``alpha``, phase-shifted by the receiver angle ``a_j``.
    """
    import matplotlib.pyplot as plt

    N = M.shape[0]
    deg = np.degrees(angles)
    sel = np.linspace(0, N, n_show, endpoint=False).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, part, lbl in zip(
        axes, [np.real, np.imag], [r"$\Re\,u^s$", r"$\Im\,u^s$"]
    ):
        for j in sel:
            ax.plot(
                deg,
                part(M[j, :]),
                lw=1.6,
                label=rf"rx @ {np.degrees(angles[j]):.0f}$^\circ$",
            )
        ax.set_title(f"{lbl} at fixed receivers vs transmitter angle")
        ax.set_xlabel(r"transmitter angle $\alpha$ (deg)")
        ax.set_ylabel(lbl)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.savefig(fname, dpi=130)
    print(f"wrote {fname}")


def plot_R_sweep(sd, A_bump, shape, radii, angles, rho, p, fname):
    """Overlay the sorted pattern ``g(gamma)`` for several scatterer radii.

    The angular content (number of oscillations of ``g``) grows with
    ``kappa * R``, so larger spheres produce higher-frequency sine waves.
    """
    import matplotlib.pyplot as plt

    kappa = sd["kappa"]
    deg = np.degrees(angles)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for idx, R in enumerate(sorted(radii)):
        b_radial, _ = make_b_radial(R, A_bump, shape)
        M, _, _ = measurement_matrix(sd, b_radial, angles, rho, p)
        g, _ = circulant_collapse(M)
        gn = np.real(g) / np.max(np.abs(np.real(g)))
        ax.plot(
            deg,
            gn,
            color=plt.cm.viridis(idx / max(len(radii) - 1, 1)),
            lw=1.8,
            label=rf"$R={R}$ ($\kappa R={kappa * R:.1f}$)",
        )
    ax.set_title(
        r"Sorted pattern $\Re\,g(\gamma)$ (normalized) vs scatterer size"
    )
    ax.set_xlabel(r"angular separation $\gamma$ (deg)")
    ax.set_ylabel(r"$\Re\,g / \max|\Re\,g|$")
    ax.legend(loc="upper right")
    fig.savefig(fname, dpi=130)
    print(f"wrote {fname}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--npz", default="data/examples/SD_3D/SD_k4_q8_L2_a1.25.npz"
    )
    ap.add_argument("--N", type=int, default=64, help="number of tx = rx")
    ap.add_argument(
        "--rho", type=float, default=2.5, help="receiver-circle radius"
    )
    ap.add_argument("--R_bump", type=float, default=0.3)
    ap.add_argument("--A_bump", type=float, default=-0.4)
    ap.add_argument("--bump", choices=["poly", "smooth"], default="poly")
    ap.add_argument("--p", type=int, default=None)
    ap.add_argument("--out", default="sphere_radial")
    ap.add_argument(
        "--npz_out",
        default=None,
        help="optional .npz dump of the measurement matrix and geometry",
    )
    ap.add_argument(
        "--R_sweep",
        type=float,
        nargs="*",
        default=None,
        help="if given, overlay the sorted pattern g(gamma) for each of these "
        "scatterer radii (oscillation count grows with kappa*R)",
    )
    args = ap.parse_args()

    sd = load_SD_matrices_3D(args.npz)
    a, kappa = sd["a"], sd["kappa"]
    if args.rho <= a * np.sqrt(3.0):
        print(
            f"warning: rho={args.rho} is close to the cube "
            f"(a*sqrt(3)={a * np.sqrt(3.0):.3f}); off-surface accuracy may "
            f"degrade near the cube corners."
        )

    b_radial, b_text = make_b_radial(args.R_bump, args.A_bump, args.bump)
    angles = 2.0 * np.pi * np.arange(args.N) / args.N

    M, rx, dirs = measurement_matrix(sd, b_radial, angles, args.rho, args.p)
    g, residual = circulant_collapse(M)

    # Mie reference for the sorted pattern (separation = grid angles).
    mie = mie_curve(b_radial, kappa, args.rho, args.R_bump, angles)
    rel_mie = float(np.linalg.norm(g - mie) / np.linalg.norm(mie))

    print(f"kappa={kappa}  N={args.N}  rho={args.rho}  cube a={a}")
    print(f"scatterer: {b_text}")
    print(f"rotational-symmetry residual (max col deviation): {residual:.3e}")
    print(f"  relative to |u^s|_max = {np.max(np.abs(M)):.3e}")
    print(f"sorted pattern g vs Mie, relative L2 error      : {rel_mie:.3e}")

    plot_geometry(rx, args.R_bump, a, f"{args.out}_geometry.png")
    plot_matrix(M, f"{args.out}_matrix.png")
    plot_sorted(M, g, angles, mie, residual, rel_mie, f"{args.out}_sorted.png")
    plot_traces(M, angles, f"{args.out}_traces.png")

    if args.R_sweep:
        plot_R_sweep(
            sd,
            args.A_bump,
            args.bump,
            args.R_sweep,
            angles,
            args.rho,
            args.p,
            f"{args.out}_sweep.png",
        )

    if args.npz_out is not None:
        np.savez(
            args.npz_out,
            M=M,
            angles=angles,
            rx=rx,
            dirs=dirs,
            kappa=kappa,
            rho=args.rho,
            R_bump=args.R_bump,
            A_bump=args.A_bump,
            g=g,
            mie=mie,
        )
        print(f"wrote {args.npz_out}")


if __name__ == "__main__":
    main()
