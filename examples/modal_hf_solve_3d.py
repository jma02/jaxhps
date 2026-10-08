"""Modal app for high-frequency forward scattering with FMM-accelerated BIE.

Runs on an A100 GPU:
1. Builds fmm3dbie in the container (for near-field corrections)
2. Generates near-field correction matrices for the target (q, L, kappa, a)
3. Runs HPS interior solve (JAX on GPU) + GMRES+FMM exterior BIE
4. Prints results and saves output

Usage
-----
    # Smoke test (L=2, matches existing dense SD)
    modal run examples/modal_hf_solve_3d.py --mode smoke

    # High-frequency solve (L=3, kappa from frequency)
    modal run examples/modal_hf_solve_3d.py --mode solve --freq-khz 150 --a 0.055
"""

import modal

from modal_fmm_image import fmm_image

app = modal.App("jaxhps-hf-solve")

vol = modal.Volume.from_name("jaxhps-data", create_if_missing=True)


@app.function(
    image=fmm_image,
    gpu="H100",
    timeout=7200,
    volumes={"/data": vol},
)
def run_hf_solve(
    mode: str = "smoke",
    freq_khz: float = 150.0,
    a: float = 0.055,
    q: int = 8,
    L: int = None,
    p: int = None,
    near_ratio: float = 4.0,
    n_src: int = 4,
    fmm_eps: float = 1e-7,
    gmres_tol: float = 1e-6,
    solver_mode: str = "dense_block",
):
    import sys
    import os
    import time
    import numpy as np

    sys.path.insert(0, "/root/jaxhps/examples")
    os.environ["JAXHPS_TIMING"] = "1"

    import jax

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX version: {jax.__version__}")
    import jax.numpy as jnp

    # Physical parameters
    c_bg = 1500.0  # m/s
    if mode == "smoke":
        kappa = 4.0
        a = 1.25
        q = 8
        L = 2
        freq_khz = kappa * c_bg / (2 * np.pi * a) / 1e3
        print(
            f"Smoke test: kappa={kappa}, a={a}, q={q}, L={L} "
            f"(f={freq_khz:.1f} kHz)"
        )
    else:
        kappa = 2 * np.pi * freq_khz * 1e3 * a / c_bg
        if L is None:
            L = max(1, int(np.ceil(np.log2(2 * kappa * a / (q + 4)))))
        print(
            f"Solve: f={freq_khz} kHz, a={a} m, kappa={kappa:.2f}, "
            f"kappa*a={kappa * a:.1f}, q={q}, L={L}"
        )

    p = p or q + 4

    # Step 1: Generate near-field corrections
    print("\n=== Step 1: Near-field corrections ===")
    from gen_nearfield_3D import generate_nearfield
    from wave_scattering_utils_3D import load_nearfield_correction

    nf_path = f"/data/NF_k{kappa:.2f}_q{q}_L{L}_a{a}.npz"
    regenerate = os.environ.get("REGEN_NF", "0") == "1"
    if not os.path.exists(nf_path) or regenerate:
        generate_nearfield(
            nf_path, a, q, L, kappa, near_ratio=near_ratio, bulk=True
        )
        vol.commit()
    nf = load_nearfield_correction(nf_path, fmm_eps=fmm_eps)

    # Step 2: HPS interior solve
    print("\n=== Step 2: HPS interior solve ===")
    from jaxhps import DiscretizationNode3D, Domain, PDEProblem, build_solver
    from wave_scattering_utils_3D import (
        get_DtN_from_ItI_3D,
        get_uin_and_dn_3D,
        solve_bie_gpu_advanced,
        outward_normals_for_cube_boundary,
    )

    root = DiscretizationNode3D(
        xmin=-a, xmax=a, ymin=-a, ymax=a, zmin=-a, zmax=a
    )
    domain = Domain(p=p, q=q, root=root, L=L)
    int_pts = np.asarray(domain.interior_points)
    bp = np.asarray(domain.boundary_points).reshape(-1, 3)
    nrm = outward_normals_for_cube_boundary(bp, root)

    # Simple centered bump
    r_int = np.linalg.norm(int_pts, axis=-1)
    R_bump = min(0.3, 0.8 * a)
    A_bump = -0.4
    mask = (r_int < R_bump).astype(float)
    b_int = A_bump * (1 - (r_int / R_bump) ** 2) ** 4 * mask
    I_coeffs = (kappa**2 * (1.0 - b_int)).astype(np.complex128)

    phi = np.linspace(0, 2 * np.pi, n_src, endpoint=False)
    source_dirs = np.column_stack([np.cos(phi), np.sin(phi), np.zeros(n_src)])
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
        eta=float(kappa),
    )

    t0 = time.perf_counter()
    T_ItI = build_solver(problem, return_top_T=True)
    dev = jax.devices()[0]
    R = jax.device_put(jnp.asarray(T_ItI), dev)
    jax.block_until_ready(R)
    T_DtN = get_DtN_from_ItI_3D(R, float(kappa))
    jax.block_until_ready(T_DtN)
    T_DtN_np = np.asarray(T_DtN)  # CPU copy for permutation / fallback
    # Free HPS intermediates to make GPU memory available for BIE kernels
    del T_ItI, R
    import gc

    gc.collect()
    dt_hps = time.perf_counter() - t0
    print(f"  HPS build+Cayley: {dt_hps:.2f}s")
    print(
        f"  T_DtN shape: {T_DtN_np.shape}, memory: {T_DtN_np.nbytes / 1e9:.2f} GB"
    )

    # Need bdry_pts in correct order for FMM
    # For NF correction, bdry_pts must match the fmm3dbie ordering
    # The HPS domain has its own ordering; we need to reconcile
    bdry_pts_nf = nf.get("boundary_points", bp)
    normals_nf = nf.get("normals", nrm)
    wts_nf = nf.get("wts", np.ones(bp.shape[0]))

    # For the HPS domain ordering, check if we need a permutation
    # If NF was generated from fmm3dbie (gen_nearfield_3D.py), the ordering
    # differs from Domain.boundary_points. We permute T_DtN to match.
    if bdry_pts_nf.shape == bp.shape and not np.allclose(
        bdry_pts_nf, bp, atol=1e-10
    ):
        from scipy.spatial import cKDTree

        tree_nf = cKDTree(bdry_pts_nf)
        _, perm = tree_nf.query(bp)
        # perm[hps_i] = nf_i.  We need T_DtN in NF ordering:
        # T_DtN_nf[nf_i, nf_j] = T_DtN[hps_i, hps_j] where hps_i = perm_inv[nf_i]
        perm_inv = np.argsort(perm)
        T_DtN_np = T_DtN_np[np.ix_(perm_inv, perm_inv)]
        # Re-upload permuted T_DtN to GPU
        del T_DtN
        T_DtN_gpu = jax.device_put(jnp.asarray(T_DtN_np), dev)
        print("  Permuted T_DtN to fmm3dbie ordering (re-uploaded to GPU)")
    else:
        T_DtN_gpu = T_DtN  # already on GPU, no permutation needed
        wts_nf = nf.get("wts", np.ones(bp.shape[0]))
        bdry_pts_nf = bp
        normals_nf = nrm
    # Free remaining HPS arrays
    del problem, domain
    gc.collect()

    # Step 3: GMRES+GPU solve
    print(f"\n=== Step 3: GMRES+GPU solve (solver_mode={solver_mode}) ===")
    uin, uin_dn = get_uin_and_dn_3D(
        float(kappa),
        jnp.asarray(bdry_pts_nf),
        jnp.asarray(normals_nf),
        jnp.asarray(source_dirs),
    )
    t0 = time.perf_counter()
    if solver_mode in ("dense_block", "matfree"):
        imp, uscat_b, uscat_dn_b, info = solve_bie_gpu_advanced(
            T_DtN_gpu,
            bdry_pts_nf,
            normals_nf,
            wts_nf,
            float(kappa),
            float(kappa),
            np.asarray(uin),
            np.asarray(uin_dn),
            nf,
            tol=gmres_tol,
            maxiter=200,
            restart=50,
            matrix_free=(solver_mode != "dense_block"),
            use_preconditioner=True,
            block_rhs=(solver_mode != "matfree"),
        )
    else:
        # Default: dense sequential (baseline, same as solve_bie_gmres_gpu)
        from wave_scattering_utils_3D import solve_bie_gmres_gpu

        imp, uscat_b, uscat_dn_b, info = solve_bie_gmres_gpu(
            T_DtN_gpu,
            bdry_pts_nf,
            normals_nf,
            wts_nf,
            float(kappa),
            float(kappa),
            np.asarray(uin),
            np.asarray(uin_dn),
            nf,
            tol=gmres_tol,
            maxiter=200,
            restart=50,
        )
    dt_gmres = time.perf_counter() - t0
    print(f"  GMRES+GPU: {dt_gmres:.2f}s")
    print(f"  Converged: {info['converged']}")
    print(f"  ||u^s||_max = {np.abs(uscat_b).max():.6e}")
    print(f"  GMRES info: {info['gmres_info']}")

    print("\n=== Summary ===")
    print(f"  kappa={kappa:.4f}, a={a}, L={L}, q={q}, p={p}")
    print(
        f"  n_bdry={bdry_pts_nf.shape[0]}, n_interior={int_pts.shape[0] * int_pts.shape[1]}"
    )
    print(f"  HPS time: {dt_hps:.2f}s")
    print(f"  GMRES+FMM time: {dt_gmres:.2f}s")
    print(f"  Total: {dt_hps + dt_gmres:.2f}s")

    return dict(
        kappa=float(kappa),
        a=a,
        L=L,
        q=q,
        p=p,
        n_src=n_src,
        uscat_max=float(np.abs(uscat_b).max()),
        hps_time=dt_hps,
        gmres_time=dt_gmres,
        converged=info["converged"],
    )


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    freq_khz: float = 150.0,
    a: float = 0.055,
    q: int = 8,
    n_src: int = 4,
    solver_mode: str = "dense_block",
):
    result = run_hf_solve.remote(
        mode=mode,
        freq_khz=freq_khz,
        a=a,
        q=q,
        n_src=n_src,
        solver_mode=solver_mode,
    )
    print(f"\nResult: {result}")
