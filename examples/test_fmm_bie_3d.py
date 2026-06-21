"""Validate FMM-accelerated BIE solve against the dense solver.

Uses the existing L=2 SD matrices (q=8, kappa=4, a=1.25) to:
1. Build near-field corrections from the dense S, D
2. Verify FMM+correction matvec matches dense S@v and D@v
3. Solve the BIE with GMRES+FMM and compare u^s against dense LU solve
"""

import os
import sys
import time

import numpy as np

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("JAXHPS_TIMING", "1")

import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
from wave_scattering_utils_3D import (
    build_nearfield_correction,
    fmm_matvec_S,
    fmm_matvec_D,
    load_SD_matrices_3D,
    permute_to_domain,
    outward_normals_for_cube_boundary,
    get_DtN_from_ItI_3D,
    get_uin_and_dn_3D,
    get_scattering_uscat_impedance_3D,
    solve_bie_gmres_fmm,
)

from jaxhps import DiscretizationNode3D, Domain, PDEProblem, build_solver

SD_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "examples",
    "SD_3D",
    "SD_k4_q8_L2_a1.25.npz",
)


def make_bumps(pts):
    """Simple centered radial bump for testing."""
    r = np.linalg.norm(pts, axis=-1)
    R = 0.3
    A = -0.4
    mask = (r < R).astype(float)
    return A * (1 - (r / R) ** 2) ** 4 * mask


def main():
    sd = load_SD_matrices_3D(SD_PATH)
    a, q, L, kappa = sd["a"], sd["q"], sd["L"], sd["kappa"]
    eta = float(kappa)
    p = q + 4
    print(f"Loaded SD: a={a}, q={q}, L={L}, kappa={kappa}")
    print(f"  n_bdry = {sd['S'].shape[0]}")

    root = DiscretizationNode3D(
        xmin=-a, xmax=a, ymin=-a, ymax=a, zmin=-a, zmax=a
    )
    domain = Domain(p=p, q=q, root=root, L=L)
    bp = np.asarray(domain.boundary_points).reshape(-1, 3)
    _, sdp = permute_to_domain(sd, bp)
    nrm = outward_normals_for_cube_boundary(bp, root)
    wts = np.asarray(sdp["wts"])

    S_dense = np.asarray(sdp["S"])
    D_dense = np.asarray(sdp["D"])
    n_bdry = bp.shape[0]

    # --- Step 1: Build near-field correction ---
    print("\n=== Step 1: Building near-field correction ===")
    t0 = time.perf_counter()
    nf = build_nearfield_correction(sdp, bp, nrm, wts, q, L, kappa)
    print(f"  Near-field correction built in {time.perf_counter() - t0:.2f}s")
    print(f"  n_patches={nf['n_patches']}, n_near_pairs={nf['n_near_pairs']}")
    print(f"  S_corr nnz={nf['nnz_S']}, D_corr nnz={nf['nnz_D']}")
    print(f"  Sparsity: {nf['nnz_S'] / n_bdry**2 * 100:.1f}% of dense")

    # --- Step 2: Validate FMM+correction matvec ---
    print("\n=== Step 2: Validate FMM+correction matvec ===")
    rng = np.random.default_rng(42)
    v = rng.standard_normal(n_bdry) + 1j * rng.standard_normal(n_bdry)

    Sv_dense = S_dense @ v
    Sv_fmm = fmm_matvec_S(v, bp, wts, kappa, nf["S_corr"], nf["fmm_eps"])
    err_S = np.linalg.norm(Sv_fmm - Sv_dense) / np.linalg.norm(Sv_dense)
    print(f"  S matvec: ||FMM - dense|| / ||dense|| = {err_S:.2e}")

    Dv_dense = D_dense @ v
    Dv_fmm = fmm_matvec_D(v, bp, nrm, wts, kappa, nf["D_corr"], nf["fmm_eps"])
    err_D = np.linalg.norm(Dv_fmm - Dv_dense) / np.linalg.norm(Dv_dense)
    print(f"  D matvec: ||FMM - dense|| / ||dense|| = {err_D:.2e}")

    # --- Step 3: HPS interior solve (same for both methods) ---
    print("\n=== Step 3: HPS interior solve ===")
    n_src = 4  # small batch
    phi = np.linspace(0, 2 * np.pi, n_src, endpoint=False)
    source_dirs = np.column_stack([np.cos(phi), np.sin(phi), np.zeros(n_src)])

    int_pts = np.asarray(domain.interior_points)
    b_int = make_bumps(int_pts)
    I_coeffs = (kappa**2 * (1.0 - b_int)).astype(np.complex128)
    phases = np.einsum("lpd,sd->lps", int_pts, source_dirs)
    uin_int = np.exp(1j * kappa * phases)
    src = (kappa**2 * b_int[..., None] * uin_int).astype(np.complex128)
    ones = np.ones_like(I_coeffs)
    problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=ones,
        D_yy_coefficients=ones,
        D_zz_coefficients=ones,
        I_coefficients=I_coeffs,
        source=src,
        use_ItI=True,
        eta=eta,
    )
    T_ItI = build_solver(problem, return_top_T=True)
    T_DtN = get_DtN_from_ItI_3D(jnp.asarray(T_ItI), eta)
    T_DtN_np = np.asarray(T_DtN)

    # --- Step 4: Dense BIE solve (reference) ---
    print("\n=== Step 4: Dense BIE solve (reference) ===")
    t0 = time.perf_counter()
    _, uscat_b_dense, uscat_dn_dense = get_scattering_uscat_impedance_3D(
        S=jnp.asarray(sdp["S"]),
        D=jnp.asarray(sdp["D"]),
        T_DtN=T_DtN,
        bdry_pts=jnp.asarray(bp),
        normals=jnp.asarray(nrm),
        k=float(kappa),
        eta=eta,
        source_dirs=jnp.asarray(source_dirs),
    )
    uscat_b_dense = np.asarray(uscat_b_dense)
    print(f"  Dense solve: {time.perf_counter() - t0:.2f}s")
    print(f"  ||u^s||_max = {np.abs(uscat_b_dense).max():.4e}")

    # --- Step 5: GMRES+FMM solve ---
    print("\n=== Step 5: GMRES+FMM solve ===")
    uin, uin_dn = get_uin_and_dn_3D(
        float(kappa),
        jnp.asarray(bp),
        jnp.asarray(nrm),
        jnp.asarray(source_dirs),
    )
    t0 = time.perf_counter()
    _, uscat_b_fmm, _, info = solve_bie_gmres_fmm(
        T_DtN_np,
        bp,
        nrm,
        wts,
        kappa,
        eta,
        np.asarray(uin),
        np.asarray(uin_dn),
        nf,
        tol=1e-8,
        maxiter=200,
        restart=50,
    )
    dt_fmm = time.perf_counter() - t0
    print(f"  GMRES+FMM solve: {dt_fmm:.2f}s")
    print(f"  Converged: {info['converged']}")
    print(f"  GMRES info: {info['gmres_info']}")

    # --- Step 6: Compare ---
    print("\n=== Step 6: Comparison ===")
    diff = uscat_b_fmm - uscat_b_dense
    rel_err = np.linalg.norm(diff) / np.linalg.norm(uscat_b_dense)
    max_err = np.abs(diff).max()
    print(f"  Relative L2 error: {rel_err:.2e}")
    print(f"  Max absolute error: {max_err:.2e}")
    print(f"  ||u^s_dense||_max = {np.abs(uscat_b_dense).max():.4e}")
    print(f"  ||u^s_fmm||_max = {np.abs(uscat_b_fmm).max():.4e}")

    # Summary
    print("\n" + "=" * 60)
    if rel_err < 1e-4:
        print("PASS: FMM+GMRES matches dense solve to high precision")
    elif rel_err < 1e-2:
        print("OK: FMM+GMRES matches dense solve to moderate precision")
    else:
        print("FAIL: FMM+GMRES disagrees with dense solve")
    print(f"  S matvec rel err: {err_S:.2e}")
    print(f"  D matvec rel err: {err_D:.2e}")
    print(f"  BIE solve rel err: {rel_err:.2e}")


if __name__ == "__main__":
    main()
