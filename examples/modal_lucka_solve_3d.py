"""Modal app: forward solve with Lucka et al. breast tissue coefficients.

Solves the time-harmonic Helmholtz equation with a multi-tissue phantom
(fat, fibroglandular, blood vessels, skin) at kappa=30 on H100.

Reference: Lucka et al., arXiv:2102.00755

Usage:
    modal run examples/modal_lucka_solve_3d.py
    modal run examples/modal_lucka_solve_3d.py --kappa 30 --solver-mode dense_block
"""

import os

import modal

from lucka_phantom_3d import (
    build_lucka_phantom,
    build_lucka_phantom_hemisphere,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = modal.App("jaxhps-lucka-breast")

fmm_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("gfortran", "libopenblas-dev", "make", "git", "curl")
    .pip_install(
        "numpy<2",
        "setuptools<60",
        "fmm3dpy",
        "scipy",
        "jax[cuda12]",
        "charset_normalizer",
    )
    .run_commands(
        "git clone --recurse-submodules https://github.com/fastalgorithms/fmm3dbie.git /opt/fmm3dbie",
        "cd /opt/fmm3dbie && git checkout ddc93f53e60181b79928fb896a678b49865810aa && git submodule update --recursive",
        "sed -i \"s|'../src/stok_wrappers/stok_comb_vel.f'|'../src/stok_wrappers/stok_comb_vel.f90'|\" /opt/fmm3dbie/python/setup.py",
        "find /opt/fmm3dbie/src -name '*.f90' -exec sed -i 's/[^[:print:]\\t]//g' {} +",
        "cd /opt/fmm3dbie && cp make.inc.linux.gnu.openblas make.inc && make -j$(nproc) lib",
        "cd /opt/fmm3dbie/python && FMMBIE_LIBS='-fopenmp -lopenblas' "
        "FFLAGS='-fallow-argument-mismatch -fPIC -O3 -funroll-loops -std=legacy -w' "
        "python setup.py install",
        "python -c 'import fmm3dbie; print(\"fmm3dbie OK\")'",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/root/jaxhps",
        ignore=["data/**", ".git/**", "**/__pycache__/**", "**/*.npz"],
        copy=True,
    )
    .run_commands("pip install --no-deps /root/jaxhps")
    .env({"PYTHONPATH": "/root/jaxhps/examples"})
)

vol = modal.Volume.from_name("jaxhps-data", create_if_missing=True)


@app.function(
    image=fmm_image,
    gpu="H100",
    timeout=7200,
    volumes={"/data": vol},
)
def run_lucka_solve(
    kappa: float = 30.0,
    a: float = 1.25,
    q: int = 8,
    L: int = 3,
    p: int = 12,
    n_src: int = 4,
    near_ratio: float = 4.0,
    gmres_tol: float = 1e-5,
    maxiter: int = 500,
    solver_mode: str = "matfree",
    geometry: str = "hemisphere",
):
    import sys
    import time

    import numpy as np

    sys.path.insert(0, "/root/jaxhps/examples")
    os.environ["JAXHPS_TIMING"] = "1"

    import jax
    import jax.numpy as jnp

    print(f"JAX devices: {jax.devices()}")
    print("=== Lucka breast forward problem ===")
    print(f"  kappa={kappa}, a={a}, kappa*a={kappa * a:.1f}")
    print(f"  L={L}, q={q}, p={p}, n_src={n_src}")
    print(f"  solver_mode={solver_mode}, geometry={geometry}")

    n_bdry = 6 * (4**L) * q**2
    T_mem_gb = n_bdry**2 * 16 / 1e9
    print(f"  n_bdry={n_bdry:,}, T_DtN memory={T_mem_gb:.2f} GB")

    # Step 1: Near-field corrections
    print("\n=== Step 1: Near-field corrections ===")
    nf_path = f"/data/NF_k{kappa:.2f}_q{q}_L{L}_a{a}.npz"
    if os.path.exists(nf_path):
        print(f"  Loading cached: {nf_path}")
        from wave_scattering_utils_3D import load_nearfield_correction

        nf = load_nearfield_correction(nf_path, fmm_eps=1e-7)
    else:
        print("  Generating with fmm3dbie...")
        from gen_SD_3D import build_cube_srcvals

        import fmm3dbie as h3
        from scipy.sparse import csr_matrix
        from scipy.spatial import cKDTree

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
        print(f"  {len(near_pairs)} near pairs")

        # Generate NF blocks
        near_row_set = set()
        near_col_set = set()
        for ip, jp in near_pairs:
            near_row_set.update(patches[ip].tolist())
            near_col_set.update(patches[jp].tolist())
        all_near_rows = np.array(sorted(near_row_set), dtype=np.int64)
        all_near_cols = np.array(sorted(near_col_set), dtype=np.int64)

        def _matgen_bulk(alpha, beta, row_f, col_f):
            zpars = np.array([kappa + 0j, alpha, beta], dtype=np.complex128)
            nifds, _, nzfds = h3.helm_comb_dir_fds_block_mem(
                norders,
                ixyzs,
                iptype,
                srccoefs,
                srcvals,
                1e-9,
                zpars,
                0,
            )
            ifds, zfds = h3.helm_comb_dir_fds_block_init(
                norders,
                ixyzs,
                iptype,
                srccoefs,
                srcvals,
                1e-9,
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
                1e-9,
                zpars,
                ifds,
                zfds,
                row_f,
                col_f,
                0,
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

        t0 = time.perf_counter()
        print("  Generating bulk S near-field...")
        S_bulk = _matgen_bulk(
            1 + 0j, 0 + 0j, all_near_rows + 1, all_near_cols + 1
        )
        print("  Generating bulk D near-field...")
        D_bulk = _matgen_bulk(
            0 + 0j, 1 + 0j, all_near_rows + 1, all_near_cols + 1
        )
        dt_nf = time.perf_counter() - t0
        print(f"  NF generation: {dt_nf:.1f}s")

        row_map = {v: i for i, v in enumerate(all_near_rows)}
        col_map = {v: i for i, v in enumerate(all_near_cols)}

        n_pairs = len(near_pairs)
        total = n_pairs * q2 * q2
        rows_arr = np.empty(total, dtype=np.int64)
        cols_arr = np.empty(total, dtype=np.int64)
        S_vals = np.empty(total, dtype=np.complex128)
        D_vals = np.empty(total, dtype=np.complex128)
        idx = 0
        for ip, jp in near_pairs:
            ri = patches[ip]
            ci = patches[jp]
            ri_b = [row_map[r_] for r_ in ri]
            ci_b = [col_map[c_] for c_ in ci]
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
            eps=1e-9,
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
            fmm_eps=1e-7,
            boundary_points=bdry_pts,
            normals=normals,
            wts=wts,
            a=a,
            q=q,
            L=L,
        )

    # Step 2: HPS interior solve with Lucka phantom
    print("\n=== Step 2: HPS interior solve (Lucka phantom) ===")
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

    # Build Lucka breast phantom
    if geometry == "hemisphere":
        b_int = build_lucka_phantom_hemisphere(int_pts, a)
    else:
        b_int = build_lucka_phantom(int_pts, a)
    print(f"  Phantom b(x): min={b_int.min():.4f}, max={b_int.max():.4f}")
    print(f"  Non-zero fraction: {(np.abs(b_int) > 1e-10).mean():.1%}")

    # PDE coefficients: Delta u + kappa^2 (1-b) u = kappa^2 b u^inc
    I_coeffs = (kappa**2 * (1.0 - b_int)).astype(np.complex128)

    # Source directions (plane waves on Fibonacci sphere for realism)
    phi_gold = (1 + np.sqrt(5)) / 2
    i_src = np.arange(n_src)
    theta_s = np.arccos(1 - 2 * (i_src + 0.5) / n_src)
    phi_s = 2 * np.pi * i_src / phi_gold
    source_dirs = np.column_stack(
        [
            np.sin(theta_s) * np.cos(phi_s),
            np.sin(theta_s) * np.sin(phi_s),
            np.cos(theta_s),
        ]
    )

    # Build incident fields on interior
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
    T_DtN_np = np.asarray(T_DtN)
    del T_ItI, R, T_DtN
    import gc

    gc.collect()
    # Free JAX compilation caches to reclaim fragmented GPU memory
    jax.clear_caches()
    dt_hps = time.perf_counter() - t0
    print(f"  HPS build+Cayley: {dt_hps:.2f}s")
    print(f"  T_DtN: {T_DtN_np.shape}, {T_DtN_np.nbytes / 1e9:.2f} GB")

    # Reconcile ordering (HPS vs fmm3dbie)
    bdry_pts_nf = nf.get("boundary_points", bp)
    normals_nf = nf.get("normals", nrm)
    wts_nf = nf.get("wts", np.ones(bp.shape[0]))

    if bdry_pts_nf.shape == bp.shape and not np.allclose(
        bdry_pts_nf, bp, atol=1e-10
    ):
        from scipy.spatial import cKDTree

        tree_nf = cKDTree(bdry_pts_nf)
        _, perm = tree_nf.query(bp)
        perm_inv = np.argsort(perm)
        T_DtN_np = T_DtN_np[np.ix_(perm_inv, perm_inv)]
        T_DtN_gpu = jax.device_put(jnp.asarray(T_DtN_np), dev)
        print("  Permuted T_DtN to fmm3dbie ordering")
    else:
        T_DtN_gpu = jax.device_put(jnp.asarray(T_DtN_np), dev)
        bdry_pts_nf = bp
        normals_nf = nrm

    del problem, domain
    gc.collect()

    # Step 3: GMRES solve
    print(f"\n=== Step 3: GMRES solve (mode={solver_mode}) ===")
    uin, uin_dn = get_uin_and_dn_3D(
        float(kappa),
        jnp.asarray(bdry_pts_nf),
        jnp.asarray(normals_nf),
        jnp.asarray(source_dirs),
    )
    t0 = time.perf_counter()
    if solver_mode == "dense_block":
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
            maxiter=maxiter,
            restart=200,
            matrix_free=False,
            use_preconditioner=True,
            block_rhs=True,
        )
    elif solver_mode in ("matfree", "matfree_block"):
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
            maxiter=maxiter,
            restart=200,
            matrix_free=True,
            use_preconditioner=True,
            block_rhs=(solver_mode == "matfree_block"),
        )
    else:
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
            maxiter=maxiter,
            restart=200,
        )
    dt_gmres = time.perf_counter() - t0

    print(f"  GMRES time: {dt_gmres:.2f}s")
    print(f"  Converged: {info['converged']}")
    print(f"  ||u^s||_max = {np.abs(uscat_b).max():.6e}")
    print(f"  GMRES info: {info['gmres_info']}")

    print("\n=== Summary ===")
    print(
        f"  Problem: Lucka breast phantom, kappa={kappa}, kappa*a={kappa * a:.1f}"
    )
    print(f"  Tissue contrast: b in [{b_int.min():.4f}, {b_int.max():.4f}]")
    print(f"  L={L}, q={q}, p={p}, n_bdry={bdry_pts_nf.shape[0]:,}")
    print(f"  HPS time: {dt_hps:.2f}s")
    print(f"  GMRES time: {dt_gmres:.2f}s")
    print(f"  Total: {dt_hps + dt_gmres:.2f}s")
    print(f"  Solver: {solver_mode}")
    print(f"  Memory: T_DtN={T_DtN_np.nbytes / 1e9:.2f} GB")

    return dict(
        kappa=float(kappa),
        kappa_a=float(kappa * a),
        a=a,
        L=L,
        q=q,
        p=p,
        n_src=n_src,
        n_bdry=int(bdry_pts_nf.shape[0]),
        b_min=float(b_int.min()),
        b_max=float(b_int.max()),
        uscat_max=float(np.abs(uscat_b).max()),
        hps_time=dt_hps,
        gmres_time=dt_gmres,
        total_time=dt_hps + dt_gmres,
        converged=info["converged"],
        gmres_info=info["gmres_info"],
        solver_mode=solver_mode,
        geometry=geometry,
        T_DtN_gb=float(T_DtN_np.nbytes / 1e9),
    )


@app.local_entrypoint()
def main(
    kappa: float = 30.0,
    a: float = 1.25,
    q: int = 8,
    levels: int = 3,
    p: int = 12,
    n_src: int = 4,
    solver_mode: str = "matfree",
    gmres_tol: float = 1e-5,
    maxiter: int = 500,
    geometry: str = "hemisphere",
):
    result = run_lucka_solve.remote(
        kappa=kappa,
        a=a,
        q=q,
        L=levels,
        p=p,
        n_src=n_src,
        solver_mode=solver_mode,
        gmres_tol=gmres_tol,
        maxiter=maxiter,
        geometry=geometry,
    )
    print(f"\nResult: {result}")
