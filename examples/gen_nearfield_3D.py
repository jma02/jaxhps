"""Generate near-field correction matrices for FMM-accelerated BIE solves.

For small L (where the full dense S, D fit in memory), the corrections can be
extracted from the dense matrices via
:func:`wave_scattering_utils_3D.build_nearfield_correction`.

For large L (L >= 3), the dense matrices are too big.  This script uses
fmm3dbie to generate only the near-field blocks (patch-pair submatrices)
and stores them as sparse CSR corrections in a .npz file.

The resulting .npz is consumed by the GMRES+FMM solver in
:func:`wave_scattering_utils_3D.solve_bie_gmres_fmm`.

Usage
-----
    # Run inside the fmm3dbie env (see scripts/build_fmm3dbie.sh)
    python examples/gen_nearfield_3D.py \\
        --q 8 --L 3 --kappa 10.0 --a 1.25 \\
        --out data/examples/NF_3D/NF_k10_q8_L3_a1.25.npz

Requires fmm3dbie (Fortran) but NOT jaxhps or JAX.
"""

import argparse
import os
import time

import numpy as np
from scipy.sparse import csr_matrix

# Reuse the cube geometry builder from gen_SD_3D.py
from gen_SD_3D import build_cube_srcvals


def identify_near_patches(
    patch_centers: np.ndarray,
    patch_widths: np.ndarray,
    near_ratio: float = 4.0,
) -> list:
    """Return list of (i, j) patch-index pairs within ``near_ratio * max_width``."""
    from scipy.spatial import cKDTree

    threshold = near_ratio * patch_widths.max()
    tree = cKDTree(patch_centers)
    pairs = []
    for ip in range(len(patch_centers)):
        neighbors = tree.query_ball_point(patch_centers[ip], threshold)
        for jp in neighbors:
            pairs.append((ip, jp))
    return pairs


def smooth_kernel_blocks(
    bdry_pts: np.ndarray,
    normals: np.ndarray,
    wts: np.ndarray,
    kappa: float,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
):
    """Evaluate the smooth (uncorrected) kernel blocks for S and D.

    These are what the FMM evaluates for nearby source-target pairs (minus the
    self-interaction, which the FMM skips).  The near-field correction is
    ``exact_block - smooth_block``.
    """
    xi = bdry_pts[row_idx][:, None, :]
    yj = bdry_pts[col_idx][None, :, :]
    diff = xi - yj
    r = np.linalg.norm(diff, axis=-1)
    wj = wts[col_idx]

    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.exp(1j * kappa * r) / (4.0 * np.pi * r)
    S_smooth = G * wj[None, :]
    S_smooth[r == 0] = 0.0

    nj = normals[col_idx]
    n_dot_diff = np.einsum("ijk,jk->ij", diff, nj)
    with np.errstate(divide="ignore", invalid="ignore"):
        dGdny = (1.0 / r - 1j * kappa) / r * G * n_dot_diff
    D_smooth = dGdny * wj[None, :]
    D_smooth[r == 0] = 0.0

    return S_smooth, D_smooth


def build_exact_block(
    norders,
    ixyzs,
    iptype,
    srccoefs,
    srcvals,
    wts,
    kappa: float,
    eps: float,
    row_ind: np.ndarray,
    col_ind: np.ndarray,
    ifwrite: int = 0,
):
    """Use fmm3dbie to build exact S and D blocks for given row/col indices."""
    import fmm3dbie as h3

    def _matgen(alpha, beta):
        zpars = np.array([kappa + 0j, alpha, beta], dtype=np.complex128)
        nifds, _, nzfds = h3.helm_comb_dir_fds_block_mem(
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

    S_block = _matgen(1.0 + 0j, 0.0 + 0j)
    D_block = _matgen(0.0 + 0j, 1.0 + 0j)
    return S_block, D_block


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=float, default=1.25)
    p.add_argument("--q", type=int, default=8)
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--kappa", type=float, default=10.0)
    p.add_argument("--eps", type=float, default=1e-9)
    p.add_argument("--near_ratio", type=float, default=4.0)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    print(
        f"Generating near-field corrections: a={args.a}, q={args.q}, "
        f"L={args.L}, kappa={args.kappa}"
    )

    # Build geometry
    norders, ixyzs, iptype, srcvals, face_idx = build_cube_srcvals(
        args.a,
        args.q,
        args.L,
    )
    import fmm3dbie as h3

    srccoefs = h3.surf_vals_to_coefs(norders, ixyzs, iptype, srcvals[0:9, :])
    wts = h3.get_qwts(norders, ixyzs, iptype, srcvals)

    bdry_pts = srcvals[0:3, :].T
    normals = srcvals[9:12, :].T
    npts = bdry_pts.shape[0]
    q2 = args.q**2
    n_patches = npts // q2
    print(f"  npts={npts}, n_patches={n_patches}")

    # Patch metadata
    patches = np.arange(npts).reshape(n_patches, q2)
    patch_centers = np.array(
        [bdry_pts[patches[i]].mean(axis=0) for i in range(n_patches)]
    )
    patch_widths = np.array(
        [
            np.max(np.ptp(bdry_pts[patches[i]], axis=0))
            for i in range(n_patches)
        ]
    )

    near_pairs = identify_near_patches(
        patch_centers, patch_widths, args.near_ratio
    )
    print(
        f"  near pairs: {len(near_pairs)} "
        f"(avg {len(near_pairs) / n_patches:.1f} per patch)"
    )

    # Initialize fmm3dbie FDS data structures (once for all blocks)
    # We must generate the full near-field data, then extract per-block
    # Actually, matgen with restricted row/col indices handles this.

    # Build correction blocks
    n_pairs = len(near_pairs)
    total_entries = n_pairs * q2 * q2
    rows_arr = np.empty(total_entries, dtype=np.int64)
    cols_arr = np.empty(total_entries, dtype=np.int64)
    S_vals = np.empty(total_entries, dtype=np.complex128)
    D_vals = np.empty(total_entries, dtype=np.complex128)

    t0 = time.perf_counter()
    idx = 0
    for pi, (ip, jp) in enumerate(near_pairs):
        ri = patches[ip]
        ci = patches[jp]
        # 1-based indices for fmm3dbie
        row_ind_f = ri + 1
        col_ind_f = ci + 1

        S_exact, D_exact = build_exact_block(
            norders,
            ixyzs,
            iptype,
            srccoefs,
            srcvals,
            wts,
            args.kappa,
            args.eps,
            row_ind_f,
            col_ind_f,
        )
        S_smooth, D_smooth = smooth_kernel_blocks(
            bdry_pts,
            normals,
            wts,
            args.kappa,
            ri,
            ci,
        )

        block_size = q2 * q2
        rr, cc = np.meshgrid(ri, ci, indexing="ij")
        rows_arr[idx : idx + block_size] = rr.ravel()
        cols_arr[idx : idx + block_size] = cc.ravel()
        S_vals[idx : idx + block_size] = (S_exact - S_smooth).ravel()
        D_vals[idx : idx + block_size] = (D_exact - D_smooth).ravel()
        idx += block_size

        if (pi + 1) % 500 == 0 or pi == n_pairs - 1:
            elapsed = time.perf_counter() - t0
            print(
                f"  [{pi + 1}/{n_pairs}] {elapsed:.1f}s "
                f"({elapsed / (pi + 1) * 1000:.1f} ms/pair)"
            )

    S_corr = csr_matrix(
        (S_vals[:idx], (rows_arr[:idx], cols_arr[:idx])),
        shape=(npts, npts),
    )
    D_corr = csr_matrix(
        (D_vals[:idx], (rows_arr[:idx], cols_arr[:idx])),
        shape=(npts, npts),
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # Save sparse matrices and metadata
    np.savez(
        args.out,
        S_corr_data=S_corr.data,
        S_corr_indices=S_corr.indices,
        S_corr_indptr=S_corr.indptr,
        D_corr_data=D_corr.data,
        D_corr_indices=D_corr.indices,
        D_corr_indptr=D_corr.indptr,
        shape=np.array(S_corr.shape),
        boundary_points=bdry_pts,
        normals=normals,
        wts=wts,
        face_idx=face_idx,
        a=args.a,
        q=args.q,
        L=args.L,
        kappa=args.kappa,
        eps=args.eps,
        near_ratio=args.near_ratio,
        n_near_pairs=len(near_pairs),
    )
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s. Wrote {args.out}")
    print(f"  S_corr nnz={S_corr.nnz}, D_corr nnz={D_corr.nnz}")
    mem_mb = (S_corr.data.nbytes + D_corr.data.nbytes) / 1e6
    print(f"  Total correction memory: {mem_mb:.1f} MB")


if __name__ == "__main__":
    main()
