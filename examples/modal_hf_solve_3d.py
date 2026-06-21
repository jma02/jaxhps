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

import os
import modal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = modal.App("jaxhps-hf-solve")

# Build image with all dependencies
fmm_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "gfortran",
        "libopenblas-dev",
        "make",
        "git",
        "curl",
    )
    .pip_install(
        "numpy<2",
        "setuptools<60",
        "fmm3dpy",
        "scipy",
        "jax[cuda12]",
        "charset_normalizer",
    )
    .run_commands(
        # Clone and build fmm3dbie
        "git clone --recurse-submodules https://github.com/fastalgorithms/fmm3dbie.git /opt/fmm3dbie",
        "cd /opt/fmm3dbie && git checkout ddc93f53e60181b79928fb896a678b49865810aa && git submodule update --recursive",
        # Patch setup.py typo
        "sed -i \"s|'../src/stok_wrappers/stok_comb_vel.f'|'../src/stok_wrappers/stok_comb_vel.f90'|\" /opt/fmm3dbie/python/setup.py",
        # Fix non-ASCII chars in Fortran sources (f2py encoding issue)
        "find /opt/fmm3dbie/src -name '*.f90' -exec sed -i 's/[^[:print:]\\t]//g' {} +",
        # Build static lib
        "cd /opt/fmm3dbie && cp make.inc.linux.gnu.openblas make.inc && make -j$(nproc) lib",
        # Build and install Python wrapper
        "cd /opt/fmm3dbie/python && FMMBIE_LIBS='-fopenmp -lopenblas' "
        "FFLAGS='-fallow-argument-mismatch -fPIC -O3 -funroll-loops -std=legacy -w' "
        "python setup.py install",
        # Verify
        "python -c 'import fmm3dbie; print(\"fmm3dbie OK\")'",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/root/jaxhps",
        ignore=["data/**", ".git/**", "**/__pycache__/**", "**/*.npz"],
        copy=True,
    )
    .run_commands("pip install --no-deps /root/jaxhps")
)

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
    nf_path = f"/data/NF_k{kappa:.2f}_q{q}_L{L}_a{a}.npz"
    regen = os.environ.get("REGEN_NF", "0") == "1"
    if os.path.exists(nf_path) and not regen:
        print(f"  Loading cached: {nf_path}")
        from wave_scattering_utils_3D import load_nearfield_correction

        nf = load_nearfield_correction(nf_path, fmm_eps=fmm_eps)
    else:
        print("  Generating with fmm3dbie...")
        from gen_SD_3D import build_cube_srcvals
        import fmm3dbie as h3
        from scipy.sparse import csr_matrix

        norders, ixyzs, iptype, srcvals, face_idx = build_cube_srcvals(a, q, L)
        srccoefs = h3.surf_vals_to_coefs(
            norders, ixyzs, iptype, srcvals[0:9, :]
        )
        wts = h3.get_qwts(norders, ixyzs, iptype, srcvals)
        bdry_pts = srcvals[0:3, :].T
        normals = srcvals[9:12, :].T
        npts = bdry_pts.shape[0]
        q2 = q**2
        n_patches = npts // q2
        patches = np.arange(npts).reshape(n_patches, q2)

        print(f"  npts={npts}, n_patches={n_patches}")

        # Identify near patches
        from scipy.spatial import cKDTree

        patch_centers = np.array(
            [bdry_pts[patches[i]].mean(0) for i in range(n_patches)]
        )
        patch_widths = np.array(
            [
                np.max(np.ptp(bdry_pts[patches[i]], axis=0))
                for i in range(n_patches)
            ]
        )
        threshold = near_ratio * patch_widths.max()
        tree = cKDTree(patch_centers)
        near_pairs = []
        for ip in range(n_patches):
            for jp in tree.query_ball_point(patch_centers[ip], threshold):
                near_pairs.append((ip, jp))
        print(
            f"  {len(near_pairs)} near pairs ({len(near_pairs) / n_patches:.1f} per patch)"
        )

        # Generate blocks
        n_pairs = len(near_pairs)
        total = n_pairs * q2 * q2
        rows_arr = np.empty(total, dtype=np.int64)
        cols_arr = np.empty(total, dtype=np.int64)
        S_vals = np.empty(total, dtype=np.complex128)
        D_vals = np.empty(total, dtype=np.complex128)

        eps_quad = 1e-9
        ifwrite = 0

        # Collect unique near-field indices for bulk generation
        near_row_set = set()
        near_col_set = set()
        for ip, jp in near_pairs:
            near_row_set.update(patches[ip].tolist())
            near_col_set.update(patches[jp].tolist())
        all_near_rows = np.array(sorted(near_row_set), dtype=np.int64)
        all_near_cols = np.array(sorted(near_col_set), dtype=np.int64)
        print(
            f"  Near-field index set: {len(all_near_rows)} rows, "
            f"{len(all_near_cols)} cols"
        )

        def _matgen_bulk(alpha, beta, row_f, col_f):
            zpars = np.array([kappa + 0j, alpha, beta], dtype=np.complex128)
            nifds, _, nzfds = h3.helm_comb_dir_fds_block_mem(
                norders,
                ixyzs,
                iptype,
                srccoefs,
                srcvals,
                eps_quad,
                zpars,
                ifwrite,
            )
            ifds, zfds = h3.helm_comb_dir_fds_block_init(
                norders,
                ixyzs,
                iptype,
                srccoefs,
                srcvals,
                eps_quad,
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
                eps_quad,
                zpars,
                ifds,
                zfds,
                row_f,
                col_f,
                ifwrite,
            )

        def _smooth_block(ri, ci):
            xi = bdry_pts[ri][:, None, :]
            yj = bdry_pts[ci][None, :, :]
            diff = xi - yj
            r = np.linalg.norm(diff, axis=-1)
            wj = wts[ci]
            with np.errstate(divide="ignore", invalid="ignore"):
                G = np.exp(1j * kappa * r) / (4.0 * np.pi * r)
            S_s = G * wj[None, :]
            S_s[r == 0] = 0.0
            nj = normals[ci]
            nd = np.einsum("ijk,jk->ij", diff, nj)
            with np.errstate(divide="ignore", invalid="ignore"):
                dG = (1.0 / r - 1j * kappa) / r * G * nd
            D_s = dG * wj[None, :]
            D_s[r == 0] = 0.0
            return S_s, D_s

        # Bulk generation: one fmm3dbie call per layer (S, D)
        t0 = time.perf_counter()
        print("  Generating bulk S near-field matrix...")
        S_bulk = _matgen_bulk(
            1 + 0j, 0 + 0j, all_near_rows + 1, all_near_cols + 1
        )
        print("  Generating bulk D near-field matrix...")
        D_bulk = _matgen_bulk(
            0 + 0j, 1 + 0j, all_near_rows + 1, all_near_cols + 1
        )
        dt_bulk = time.perf_counter() - t0
        print(f"  Bulk matgen: {dt_bulk:.1f}s, S shape={S_bulk.shape}")

        # Build index maps for fast block extraction
        row_map = {v: i for i, v in enumerate(all_near_rows)}
        col_map = {v: i for i, v in enumerate(all_near_cols)}

        idx = 0
        for pi, (ip, jp) in enumerate(near_pairs):
            ri = patches[ip]
            ci = patches[jp]
            ri_b = [row_map[r] for r in ri]
            ci_b = [col_map[c] for c in ci]
            S_ex = S_bulk[np.ix_(ri_b, ci_b)]
            D_ex = D_bulk[np.ix_(ri_b, ci_b)]
            S_sm, D_sm = _smooth_block(ri, ci)
            bs = q2 * q2
            rr, cc = np.meshgrid(ri, ci, indexing="ij")
            rows_arr[idx : idx + bs] = rr.ravel()
            cols_arr[idx : idx + bs] = cc.ravel()
            S_vals[idx : idx + bs] = (S_ex - S_sm).ravel()
            D_vals[idx : idx + bs] = (D_ex - D_sm).ravel()
            idx += bs

        S_corr = csr_matrix(
            (S_vals[:idx], (rows_arr[:idx], cols_arr[:idx])),
            shape=(npts, npts),
        )
        D_corr = csr_matrix(
            (D_vals[:idx], (rows_arr[:idx], cols_arr[:idx])),
            shape=(npts, npts),
        )

        # Save
        np.savez(
            nf_path,
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
            a=a,
            q=q,
            L=L,
            kappa=kappa,
            eps=eps_quad,
            near_ratio=near_ratio,
            n_near_pairs=len(near_pairs),
        )
        vol.commit()
        print(f"  Saved to {nf_path}")

        nf = dict(
            S_corr=S_corr,
            D_corr=D_corr,
            n_near_pairs=len(near_pairs),
            nnz_S=S_corr.nnz,
            nnz_D=D_corr.nnz,
            kappa=kappa,
            fmm_eps=fmm_eps,
            boundary_points=bdry_pts,
            normals=normals,
            wts=wts,
            a=a,
            q=q,
            L=L,
        )

    # Step 2: HPS interior solve
    print("\n=== Step 2: HPS interior solve ===")
    from jaxhps import DiscretizationNode3D, Domain, PDEProblem, build_solver
    from wave_scattering_utils_3D import (
        get_DtN_from_ItI_3D,
        get_uin_and_dn_3D,
        solve_bie_gmres_gpu,
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

    # Step 3: GMRES+GPU solve (direct summation, no CPU FMM)
    print("\n=== Step 3: GMRES+GPU solve ===")
    uin, uin_dn = get_uin_and_dn_3D(
        float(kappa),
        jnp.asarray(bdry_pts_nf),
        jnp.asarray(normals_nf),
        jnp.asarray(source_dirs),
    )
    t0 = time.perf_counter()
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
):
    result = run_hf_solve.remote(
        mode=mode,
        freq_khz=freq_khz,
        a=a,
        q=q,
        n_src=n_src,
    )
    print(f"\nResult: {result}")
