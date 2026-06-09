"""Generate 3D Helmholtz single- and double-layer matrices on a cube boundary.

This is the 3D analog of ``examples/driver_gen_SD_matrices.m`` (which uses
chunkIE in 2D).  It uses fmm3dbie to assemble the on-boundary single- and
double-layer matrices ``S`` and ``D`` for the 3D Helmholtz kernel

    G(x, y) = exp(i kappa |x - y|) / (4 pi |x - y|)

on the cube ``[-a, a]^3``, discretized to match the HPS box's outer boundary
trace: 6 cube faces, optionally each split into ``2^L x 2^L`` flat sub-patches
of tensor-product Gauss-Legendre nodes of order ``q`` (so ``q`` GL nodes per
direction per sub-patch).  Saved to ``.npz`` for later consumption by the JAX
side (analogous to how ``load_SD_matrices`` reads the 2D ``.mat`` files
shipped via Zenodo).

This script must be run inside an env that has ``fmm3dbie`` built (see
``scripts/build_fmm3dbie.sh`` for the build recipe).  It does not depend on
the rest of jaxhps and never imports JAX.

Usage
-----
    python examples/gen_SD_3D.py --q 8 --L 0 --kappa 4.0 --a 0.5 \
        --out data/examples/SD_3D/SD_k4_q8_L0_a0.5.npz
"""

import argparse
import os
import time
from itertools import product

import numpy as np


# ---------------------------------------------------------------------------
# Geometry: build cube as 6 * (2^L)^2 flat quad patches
# ---------------------------------------------------------------------------

# Per-face parametrization (u, v) in [-1, 1]^2 -> R^3 chosen so that the
# induced surface normal n = (d/du x d/dv) / |...|  points OUTWARD.  We fix
# each face by giving its origin, the world-frame u-axis, and v-axis; the
# outward normal then comes out as u_axis x v_axis (we verified this in
# scripts/build_fmm3dbie.sh probe).
FACES = [
    # (face_name, origin, u_axis, v_axis)  -- on the unit cube [-1, 1]^3
    (
        "xmin",
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
    ),
    (
        "xmax",
        np.array([+1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ),
    (
        "ymin",
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ),
    (
        "ymax",
        np.array([0.0, +1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
    ),
    (
        "zmin",
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ),
    (
        "zmax",
        np.array([0.0, 0.0, +1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ),
]


def build_cube_srcvals(a: float, q: int, L: int):
    """Assemble fmm3dbie srcvals/iptype/ixyzs arrays for the cube ``[-a, a]^3``.

    Each of 6 faces is split into ``2^L x 2^L`` flat sub-patches; each sub-patch
    is an ``iptype = 11`` quad with ``q x q`` tensor-product GL nodes.

    Returns
    -------
    norders, ixyzs, iptype, srcvals : ndarray
        fmm3dbie standard data layout (1-based ixyzs).
    face_idx : ndarray of shape (npts,)
        Which face index (0..5) each point belongs to.  Useful downstream
        when matching to the HPS box's boundary point ordering.
    """
    n_sub = 2**L
    sub_h = 2.0 / n_sub  # sub-patch width in cube-local coords [-1, 1]
    npatches = 6 * n_sub * n_sub
    norder = q - 1
    npols = q * q
    npts = npatches * npols

    # 1D GL nodes on [-1, 1]
    u1d, _ = np.polynomial.legendre.leggauss(q)
    U, V = np.meshgrid(u1d, u1d, indexing="xy")  # both shape (q, q)
    u_ref = U.ravel()  # (q^2,)
    v_ref = V.ravel()

    srcvals = np.zeros((12, npts), dtype=np.float64)
    iptype = np.full(npatches, 11, dtype=np.int64)
    norders = np.full(npatches, norder, dtype=np.int64)
    ixyzs = np.arange(npatches + 1, dtype=np.int64) * npols + 1
    face_idx_per_point = np.zeros(npts, dtype=np.int64)

    ip = 0
    for fi, (_, origin, u_ax, v_ax) in enumerate(FACES):
        for su, sv in product(range(n_sub), repeat=2):
            # Sub-patch (u,v) in [-1, 1]^2 maps into the parent face's
            # [-1, 1]^2 via (u', v') = (u_lo + sub_h*(u+1)/2, v_lo + sub_h*(v+1)/2).
            u_lo = -1.0 + su * sub_h
            v_lo = -1.0 + sv * sub_h
            u_parent = u_lo + sub_h * (u_ref + 1.0) / 2.0
            v_parent = v_lo + sub_h * (v_ref + 1.0) / 2.0

            # World position on the cube of half-side ``a``: origin + a * (u_ax * u' + v_ax * v')
            xyz = (
                a * origin[None, :]
                + a * u_parent[:, None] * u_ax[None, :]
                + a * v_parent[:, None] * v_ax[None, :]
            )  # (q^2, 3)

            # Tangents:   d xyz / d u_ref  =  a * (sub_h / 2) * u_ax,   similarly for v.
            du_world = a * (sub_h / 2.0) * u_ax
            dv_world = a * (sub_h / 2.0) * v_ax
            n_world = np.cross(
                u_ax, v_ax
            )  # already unit and outward for these faces

            i0 = ip * npols
            i1 = i0 + npols
            srcvals[0:3, i0:i1] = xyz.T
            srcvals[3:6, i0:i1] = du_world[:, None]
            srcvals[6:9, i0:i1] = dv_world[:, None]
            srcvals[9:12, i0:i1] = n_world[:, None]
            face_idx_per_point[i0:i1] = fi
            ip += 1

    return norders, ixyzs, iptype, srcvals, face_idx_per_point


# ---------------------------------------------------------------------------
# Matrix assembly via fmm3dbie
# ---------------------------------------------------------------------------


def build_S_D_matrices(
    norders, ixyzs, iptype, srcvals, kappa: float, eps: float
):
    """Call ``helm_comb_dir_fds_block_matgen`` twice to extract S and D.

    The block matgen routine returns the dense matrix of  ``alpha * S + beta * D``
    discretized on the same surface nodes (target == source == every boundary
    node).  By choosing (alpha=1, beta=0) we get S; by choosing (alpha=0, beta=1)
    we get the pure principal-value D (no I/2 jump baked in -- verified by
    checking ``D @ 1 = -1/2`` at small kappa on the closed cube).  Callers are
    responsible for adding +/- I/2 wherever the jump relation calls for it.
    """
    import fmm3dbie as h3

    srccoefs = h3.surf_vals_to_coefs(norders, ixyzs, iptype, srcvals[0:9, :])
    npts = srcvals.shape[1]
    row_ind = np.arange(npts, dtype=np.int64) + 1
    col_ind = np.arange(npts, dtype=np.int64) + 1
    wts = h3.get_qwts(norders, ixyzs, iptype, srcvals)
    ifwrite = 0  # don't write near-field cache to disk

    def _matgen(alpha: complex, beta: complex):
        zpars = np.array([kappa + 0j, alpha, beta], dtype=np.complex128)
        nifds, nrfds, nzfds = h3.helm_comb_dir_fds_block_mem(
            norders,
            ixyzs,
            iptype,
            srccoefs,
            srcvals,
            eps,
            zpars,
            ifwrite,
        )
        ifds, zfds = h3.helm_comb_dir_fds_block_init(
            norders,
            ixyzs,
            iptype,
            srccoefs,
            srcvals,
            eps,
            zpars,
            nifds,
            nzfds,
        )
        return h3.helm_comb_dir_fds_block_matgen(
            norders,
            ixyzs,
            iptype,
            srccoefs,
            srcvals,
            wts,
            eps,
            zpars,
            ifds,
            zfds,
            row_ind,
            col_ind,
            ifwrite,
        )

    t0 = time.time()
    S = _matgen(1.0 + 0j, 0.0 + 0j)
    t_S = time.time() - t0

    t0 = time.time()
    D = _matgen(0.0 + 0j, 1.0 + 0j)
    t_D = time.time() - t0

    return S, D, wts, dict(npts=npts, time_S=t_S, time_D=t_D)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--a",
        type=float,
        default=0.5,
        help="Cube half-side (cube is [-a, a]^3).",
    )
    p.add_argument(
        "--q",
        type=int,
        default=8,
        help="GL nodes per direction per sub-patch.",
    )
    p.add_argument(
        "--L",
        type=int,
        default=0,
        help="Refinement level: 2^L x 2^L sub-patches per face.",
    )
    p.add_argument(
        "--kappa", type=float, default=4.0, help="Helmholtz wavenumber."
    )
    p.add_argument(
        "--eps",
        type=float,
        default=1e-9,
        help="Quadrature tolerance handed to fmm3dbie.",
    )
    p.add_argument("--out", type=str, required=True, help="Output .npz path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    norders, ixyzs, iptype, srcvals, face_idx = build_cube_srcvals(
        args.a,
        args.q,
        args.L,
    )
    print(
        f"npatches = {iptype.size}, npts = {srcvals.shape[1]}, kappa = {args.kappa}"
    )
    print("Calling fmm3dbie matgen for S ...")
    S, D, wts, info = build_S_D_matrices(
        norders,
        ixyzs,
        iptype,
        srcvals,
        args.kappa,
        args.eps,
    )
    print(
        f"  S: {S.shape}, ||S||_F = {np.linalg.norm(S):.3e}, time = {info['time_S']:.1f}s"
    )
    print(
        f"  D: {D.shape}, ||D||_F = {np.linalg.norm(D):.3e}, time = {info['time_D']:.1f}s"
    )
    # surface area sanity (sum of weights ~ 6*(2a)^2 for cube)
    area = float(wts.sum())
    area_exact = 6.0 * (2.0 * args.a) ** 2
    print(
        f"  surface area: {area:.10f}  (exact {area_exact:.10f}, "
        f"rel err {abs(area - area_exact) / area_exact:.2e})"
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out,
        S=S,
        D=D,
        wts=wts,
        boundary_points=srcvals[0:3, :].T,
        normals=srcvals[9:12, :].T,
        face_idx=face_idx,
        a=args.a,
        q=args.q,
        L=args.L,
        kappa=args.kappa,
        eps=args.eps,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
