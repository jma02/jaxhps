"""Smoke test: load SD matrices and verify Green's identity on a HPS Domain.

Builds a ``Domain`` matching the (a, q, L) the .npz was generated for, permutes
the loaded matrices to the Domain's boundary-point ordering via
``permute_to_domain``, and checks the interior-trace Green's identity

    (1/2 I + D) u|_dOm = S u_n|_dOm

against an analytic point-source field with the source placed outside the
cube.  Confirms that (i) the loader works, (ii) the permutation correctly
aligns fmm3dbie's node order with the HPS Domain's order, and (iii) the
matrices retain their accuracy after permutation.

Usage
-----
    python examples/check_SD_3D_against_domain.py --npz /path/to/SD_k4_q10_L0.npz
"""

import argparse
import numpy as np

from jaxhps._discretization_tree import DiscretizationNode3D
from jaxhps._domain import Domain

from wave_scattering_utils_3D import load_SD_matrices_3D, permute_to_domain


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npz", required=True, help=".npz produced by gen_SD_3D.py"
    )
    p.add_argument(
        "--p",
        type=int,
        default=8,
        help="Interior Chebyshev order for the Domain (does not "
        "affect boundary nodes; just needed to construct Domain).",
    )
    args = p.parse_args()

    sd = load_SD_matrices_3D(args.npz)
    a, q, L, kappa = sd["a"], sd["q"], sd["L"], sd["kappa"]
    print(
        f"npz: a={a}, q={q}, L={L}, kappa={kappa}, npts={sd['boundary_points'].shape[0]}"
    )

    root = DiscretizationNode3D(
        xmin=-a,
        xmax=a,
        ymin=-a,
        ymax=a,
        zmin=-a,
        zmax=a,
    )
    domain = Domain(p=args.p, q=q, root=root, L=L)
    bp_domain = np.asarray(domain.boundary_points).reshape(-1, 3)

    P, sdp = permute_to_domain(sd, bp_domain)
    print(f"permutation matched all {len(P)} nodes")

    # Outward normal from the HPS box: we reconstruct it the same way as in
    # tests/test_pipeline_uniform_3D_ItI.py.
    nrm = np.zeros_like(bp_domain)
    eps = 1e-9
    nrm[np.abs(bp_domain[:, 0] - root.xmin) < eps] = [-1, 0, 0]
    nrm[np.abs(bp_domain[:, 0] - root.xmax) < eps] = [1, 0, 0]
    nrm[np.abs(bp_domain[:, 1] - root.ymin) < eps] = [0, -1, 0]
    nrm[np.abs(bp_domain[:, 1] - root.ymax) < eps] = [0, 1, 0]
    nrm[np.abs(bp_domain[:, 2] - root.zmin) < eps] = [0, 0, -1]
    nrm[np.abs(bp_domain[:, 2] - root.zmax) < eps] = [0, 0, 1]

    # Sanity: permuted normals from the npz should agree with HPS-derived ones.
    if not np.allclose(sdp["normals"], nrm):
        bad = np.linalg.norm(sdp["normals"] - nrm, axis=-1).max()
        raise RuntimeError(
            f"normals don't agree after permutation: max diff {bad:.2e}"
        )

    # Point source outside the cube: u(x) = exp(i k r) / (4 pi r), r = |x - x_src|.
    x_src = np.array([2.0 * a + 0.3, -0.4, 0.7])
    rel = bp_domain - x_src
    r = np.linalg.norm(rel, axis=-1)
    u_b = np.exp(1j * kappa * r) / (4 * np.pi * r)
    grad_u = u_b[:, None] * (1j * kappa - 1.0 / r)[:, None] * rel / r[:, None]
    u_n = np.einsum("ij,ij->i", grad_u, nrm)

    lhs = 0.5 * u_b + sdp["D"] @ u_b
    rhs = sdp["S"] @ u_n
    res = float(np.max(np.abs(lhs - rhs)) / np.max(np.abs(rhs)))
    print(f"Green's identity residual: {res:.3e}")
    print(
        f"  max|lhs| = {np.max(np.abs(lhs)):.3e}, max|rhs| = {np.max(np.abs(rhs)):.3e}"
    )


if __name__ == "__main__":
    main()
