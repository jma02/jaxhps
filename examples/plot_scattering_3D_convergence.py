"""Plot convergence of the 3D HPS+BIE scattered field against the Mie reference.

Runs the same radial-bump scattering problem as
``tests/test_scattering_3D_BIE.py::test_radial_bump_vs_mie_convergence`` on
two fixture sweeps and plots the relative error at exterior targets:

* boundary order q = 4, 6, 8 at fixed tree depth L=1 (interior order
  follows the driver default p = q + 4), and
* octree depth L = 1, 2, 3 at fixed q = 4.

Usage:
    python examples/plot_scattering_3D_convergence.py \
        --sweep_q /tmp/SD_k4_q4_L1.npz,/tmp/SD_k4_q6_L1.npz,/tmp/SD_k4_q8_L1.npz \
        --sweep_L /tmp/SD_k4_q4_L1.npz,/tmp/SD_k4_q4_L2.npz,/tmp/SD_k4_q4_L3.npz
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from wave_scattering_utils_3D import (  # noqa: E402
    eval_scattering_solution,
    load_SD_matrices_3D,
    solve_scattering_bie_3D,
)
from mie_3d import mie_scattered_field, radial_bump  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sweep_q",
        default="/tmp/SD_k4_q4_L1.npz,/tmp/SD_k4_q6_L1.npz,/tmp/SD_k4_q8_L1.npz",
        help="comma-separated fixtures with increasing q at fixed L",
    )
    p.add_argument(
        "--sweep_L",
        default="/tmp/SD_k4_q4_L1.npz,/tmp/SD_k4_q4_L2.npz,/tmp/SD_k4_q4_L3.npz",
        help="comma-separated fixtures with increasing L at fixed q",
    )
    p.add_argument("--out", default="scattering_3d_convergence.png")
    p.add_argument("--R_bump", type=float, default=0.3)
    p.add_argument("--A_bump", type=float, default=-0.4)
    p.add_argument(
        "--bump",
        choices=["poly", "smooth"],
        default="poly",
        help="bump shape: poly = A (1-(r/R)^2)^4 (C^3 at r=R), "
        "smooth = A exp(1 - 1/(1-(r/R)^2)) (C^inf)",
    )
    args = p.parse_args()

    R_bump, A_bump = args.R_bump, args.A_bump

    def b_radial(r):
        return radial_bump(r, R_bump, A_bump, args.bump)

    if args.bump == "poly":
        bump_label = (
            rf"$b(r) = {A_bump}\,(1 - (r/{R_bump})^2)^4\,1_{{r<{R_bump}}}$"
        )
    else:
        bump_label = (
            rf"$b(r) = {A_bump}\,e^{{1 - 1/(1-(r/{R_bump})^2)}}"
            rf"\,1_{{r<{R_bump}}}$"
        )

    source_dirs = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    # Same ring and off-equator points as the Mie comparison tests.
    phi = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    targets = np.concatenate(
        [
            np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1),
            np.array([[0.0, 0.0, 1.0], [0.7, 0.0, 0.7]]),
        ]
    )

    paths_q = [s for s in args.sweep_q.split(",") if s]
    paths_L = [s for s in args.sweep_L.split(",") if s]

    kappa = load_SD_matrices_3D(paths_q[0])["kappa"]
    u_mie = mie_scattered_field(
        target_pts=targets,
        source_direction=source_dirs[0],
        kappa=float(kappa),
        b_radial=b_radial,
        R_supp=R_bump,
        ell_max=20,
    )

    # The two sweeps share their coarsest fixture; cache errors by path.
    err_cache = {}

    for path in dict.fromkeys(paths_q + paths_L):
        sd = load_SD_matrices_3D(path)
        assert sd["kappa"] == kappa, "all fixtures must share kappa"
        out = solve_scattering_bie_3D(sd, b_radial, source_dirs)
        u_off = eval_scattering_solution(out, targets, float(kappa))[:, 0]
        rel = float(np.linalg.norm(u_off - u_mie) / np.linalg.norm(u_mie))
        err_cache[path] = (int(sd["q"]), int(sd["L"]), rel)
        print(f"q={sd['q']} L={sd['L']}: rel_err = {rel:.3e}  ({path})")

    sweep_q = [err_cache[path] for path in paths_q]
    sweep_L = [err_cache[path] for path in paths_L]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    qs = [q for q, _, _ in sweep_q]
    errs_q = [e for _, _, e in sweep_q]
    axes[0].semilogy(qs, errs_q, "o-", color="tab:blue")
    axes[0].set_xlabel("boundary order $q$  (interior order $p = q + 4$)")
    axes[0].set_title(f"order refinement, $L = {sweep_q[0][1]}$")
    axes[0].set_xticks(qs)

    Ls = [ell for _, ell, _ in sweep_L]
    errs_L = [e for _, _, e in sweep_L]
    q_L = sweep_L[0][0]
    axes[1].semilogy(Ls, errs_L, "o-", color="tab:red")
    # Reference decay at order q+1: with the driver default p = q + 4, the
    # interior rate h^(p-3) equals h^(q+1), and the measured fmm3dbie
    # boundary-quadrature rate at q=4 is also ~h^5.  Leaf width halves per
    # level; anchored at the finest point.
    ref_order = q_L + 1
    ref = [errs_L[-1] * (2.0**ref_order) ** (Ls[-1] - ell) for ell in Ls]
    axes[1].semilogy(
        Ls, ref, "k--", linewidth=0.9, label=rf"$O(h^{{{ref_order}}})$"
    )
    axes[1].set_xlabel("octree depth $L$  (leaf width $2^{-L}$)")
    axes[1].set_title(f"mesh refinement, $q = {sweep_L[0][0]}$")
    axes[1].set_xticks(Ls)
    axes[1].legend()

    for ax in axes:
        ax.set_ylabel(r"rel. $\ell^2$ error vs Mie at exterior targets")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        rf"HPS+BIE vs Mie, $\kappa = {kappa}$;  {bump_label}",
        fontsize=11,
    )
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
