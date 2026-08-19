r"""Spectral diagnostics for the flat (unmerged) 3D HPS interface system.

Unpreconditioned restarted GMRES stagnates on the flat interface system of
:mod:`jaxhps._matfree_iti_3D`.  This script measures the structural reason.

1. *Leaf ItI maps are unitary in the surface-quadrature inner product* when
   the medium is lossless (real wavenumber, real coefficient): the impedance
   map conserves energy, so :math:`W^{1/2} T W^{-1/2}` has all singular
   values 1.  Adding absorption (complex coefficient) makes it a strict
   contraction.

2. Therefore the interior gluing operator
   :math:`\mathcal{D} = \Pi\, T` (apply the leaf maps, then swap each node
   with its interface partner) is unitary-like as well, and the interface
   system :math:`I + \mathcal{D}` has spectrum on/near the circle
   :math:`|\lambda - 1| = 1`, which passes through the origin.  A spectrum
   that surrounds or touches 0 is exactly the configuration in which GMRES
   convergence is not governed by conditioning and in which preconditioners
   that only cluster eigenvalue magnitudes cannot help.

3. The same diagnostic for the coupled flat BIE operator, and measured GMRES
   iteration counts with a few candidate preconditioners, for comparison.

Everything here materializes the operators, so it is limited to small cases
(``L=1``, small ``q``); the production path never forms them.

Usage:
    python examples/spectrum_diagnostics_3D.py --q 4 --p 8 --kappa 4.0
    python examples/spectrum_diagnostics_3D.py --npz data/examples/SD_3D/SD_k4_q4_L1_a1.0.npz
"""

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial import ConvexHull, Delaunay

from jaxhps import DiscretizationNode3D, Domain, PDEProblem
from jaxhps.local_solve import local_solve_stage_uniform_3D_ItI
from jaxhps._matfree_iti_3D import (
    build_interface_maps,
    leaf_boundary_quad_weights,
    leaf_outgoing,
    make_flat_bie_operator,
    make_root_ItI_operator,
    materialize,
)

sys.path.insert(0, os.path.dirname(__file__))
from wave_scattering_utils_3D import (  # noqa: E402
    _gmres_python_loop,
    load_SD_matrices_3D,
    make_dense_SD_apply,
    permute_to_domain,
)

jax.config.update("jax_enable_x64", True)

OUT_DIR = "data/examples/spectrum_3D"


def build_leaf_maps(
    kappa: float,
    a: float,
    L: int,
    q: int,
    p: int,
    b_amp: float,
    absorption: float,
):
    """Leaf ItI maps for a smooth radial scatterer, optionally absorbing."""
    root = DiscretizationNode3D(
        xmin=-a, xmax=a, ymin=-a, ymax=a, zmin=-a, zmax=a
    )
    domain = Domain(p=p, q=q, root=root, L=L)
    pts = np.asarray(domain.interior_points)
    r = np.linalg.norm(pts, axis=-1)
    b = b_amp * np.exp(-4.0 * r**2)
    I_coeffs = (kappa**2 * (1.0 - b) + 1j * absorption * kappa**2).astype(
        np.complex128
    )
    ones = np.ones_like(I_coeffs)
    src = np.zeros_like(I_coeffs)
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
    _, T_leaves, _, h_leaves = local_solve_stage_uniform_3D_ItI(problem)
    maps = build_interface_maps(domain)
    return domain, T_leaves, h_leaves, maps


def unitarity_report(domain: Domain, T_leaves: jax.Array) -> dict:
    r"""Singular values of :math:`W^{1/2} T W^{-1/2}` over all leaves."""
    w = leaf_boundary_quad_weights(domain.root, domain.L, domain.q)
    s = np.sqrt(w)
    T = np.asarray(T_leaves)
    M = T * s[None, :, None] / s[None, None, :]
    sv = np.linalg.svd(M, compute_uv=False)
    return dict(
        sv_max=float(sv.max()),
        sv_min=float(sv.min()),
        max_dev_from_1=float(np.abs(sv - 1.0).max()),
        cond=float((sv.max() / sv.min())),
    )


def gluing_operator_dense(T_leaves: jax.Array, maps) -> np.ndarray:
    r"""Dense interior gluing operator :math:`\mathcal{D}` (boundary rows 0)."""
    n = maps.n_flat
    partner = np.asarray(maps.partner)
    is_int = np.asarray(maps.is_interior)
    E = jnp.eye(n, dtype=jnp.complex128)
    G = np.asarray(leaf_outgoing(T_leaves, E))
    p_safe = np.where(is_int, partner, 0)
    D = G[p_safe, :]
    D[~is_int, :] = 0.0
    return D


def origin_in_hull(evals: np.ndarray) -> dict:
    """Where the origin sits relative to the eigenvalue cloud."""
    pts = np.column_stack([evals.real, evals.imag])
    inside = bool(Delaunay(pts).find_simplex(np.zeros((1, 2)))[0] >= 0)
    hull = ConvexHull(pts)
    # signed distance of 0 to each hull facet; negative => interior side
    d = hull.equations[:, -1]
    return dict(
        origin_in_convex_hull=inside,
        min_abs_eig=float(np.abs(evals).min()),
        max_abs_eig=float(np.abs(evals).max()),
        origin_hull_margin=float(-d.max()),
    )


def gmres_run(A: np.ndarray, rhs: np.ndarray, M, tol, restart, maxiter):
    stats: dict = {}
    A_j = jnp.asarray(A)
    x, code = _gmres_python_loop(
        lambda v: A_j @ v,
        jnp.asarray(rhs),
        tol=tol,
        restart=restart,
        maxiter=maxiter,
        M=M,
        stats=stats,
    )
    res = float(np.linalg.norm(A @ np.asarray(x) - rhs) / np.linalg.norm(rhs))
    return dict(
        converged=bool(code == 0),
        n_matvec=int(stats["n_matvec"]),
        n_cycles=int(stats["n_cycles"]),
        rel_res=res,
        res_history=[float(v) for v in stats["res_history"]],
    )


def preconditioners(A: np.ndarray, maps, n_per_leaf: int) -> dict:
    """Candidate left preconditioners, all materialized (diagnostic only)."""
    out = {"none": None}

    diag = np.diag(A).copy()
    diag[diag == 0.0] = 1.0
    d_j = jnp.asarray(1.0 / diag)
    out["jacobi"] = lambda v: d_j * v

    n = A.shape[0]
    nb = n // n_per_leaf
    blocks = np.zeros((nb, n_per_leaf, n_per_leaf), dtype=A.dtype)
    for i in range(nb):
        sl = slice(i * n_per_leaf, (i + 1) * n_per_leaf)
        blocks[i] = A[sl, sl]
    binv = jnp.asarray(np.linalg.inv(blocks))

    def block_jacobi(v):
        vb = v.reshape(nb, n_per_leaf)
        return jnp.einsum("bij,bj->bi", binv, vb).reshape(v.shape)

    out["block_jacobi_leafwise"] = block_jacobi

    lu = lu_factor(A)
    out["exact_lu"] = lambda v: jnp.asarray(lu_solve(lu, np.asarray(v)))
    return out


def refine_study(args: argparse.Namespace) -> None:
    r"""Is the leaf ItI map unitary in the limit, or only approximately?

    Claim 1 above is a statement about the continuous operator.  The computed
    ItI matrix is only unitary up to the local discretization error, so the
    evidence has to be a refinement study: ``max|sigma - 1|`` and the spread
    of ``|lambda(D)|`` must go to zero as ``p, q`` grow at fixed geometry.
    """
    rows = []
    for q, p in args.refine_pairs:
        domain, T_leaves, _, maps = build_leaf_maps(
            args.kappa, args.a, args.L, q, p, args.b_amp, 0.0
        )
        rep = unitarity_report(domain, T_leaves)
        D = gluing_operator_dense(T_leaves, maps)
        ii = np.flatnonzero(np.asarray(maps.is_interior))
        ev_D = np.linalg.eigvals(D[np.ix_(ii, ii)])
        rows.append(
            dict(
                q=q,
                p=p,
                max_dev_sv=rep["max_dev_from_1"],
                abs_eig_min=float(np.abs(ev_D).min()),
                abs_eig_max=float(np.abs(ev_D).max()),
                min_abs_eig_I_plus_D=float(np.abs(1.0 + ev_D).min()),
            )
        )
        r = rows[-1]
        print(
            f"q={q:2d} p={p:2d}  max|sv-1| = {r['max_dev_sv']:.3e}  "
            f"|lambda(D)| in [{r['abs_eig_min']:.6f}, "
            f"{r['abs_eig_max']:.6f}]  min|lambda(I+D)| = "
            f"{r['min_abs_eig_I_plus_D']:.3e}"
        )

    absorb = []
    for sgn in (+1.0, -1.0):
        q, p = args.refine_pairs[-1]
        domain, T_leaves, _, _ = build_leaf_maps(
            args.kappa,
            args.a,
            args.L,
            q,
            p,
            args.b_amp,
            sgn * args.absorption,
        )
        rep = unitarity_report(domain, T_leaves)
        absorb.append(dict(sign=sgn, **rep))
        print(
            f"absorption sign {sgn:+.0f}: sv in "
            f"[{rep['sv_min']:.6f}, {rep['sv_max']:.6f}]"
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{args.out}_refine.json")
    with open(path, "w") as f:
        json.dump(
            dict(kappa=args.kappa, L=args.L, rows=rows, absorption=absorb),
            f,
            indent=2,
        )
    print(f"wrote {path}")


def main(args: argparse.Namespace) -> None:
    if args.mode == "refine":
        refine_study(args)
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    results: dict = dict(vars(args))

    # --- 1. leaf ItI unitarity, lossless vs absorbing -------------------
    unit = {}
    for absorption in (0.0, args.absorption):
        domain, T_leaves, _, maps = build_leaf_maps(
            args.kappa, args.a, args.L, args.q, args.p, args.b_amp, absorption
        )
        unit[f"absorption={absorption:g}"] = unitarity_report(domain, T_leaves)
    results["leaf_ItI_unitarity"] = unit
    print("\n== leaf ItI singular values (weighted) ==")
    for key, rep in unit.items():
        print(
            f"{key:>18}: sv in [{rep['sv_min']:.6f}, {rep['sv_max']:.6f}], "
            f"max|sv-1| = {rep['max_dev_from_1']:.3e}"
        )

    # --- 2. interface system spectrum (lossless case) -------------------
    domain, T_leaves, h_leaves, maps = build_leaf_maps(
        args.kappa, args.a, args.L, args.q, args.p, args.b_amp, 0.0
    )
    D = gluing_operator_dense(T_leaves, maps)
    is_int = np.asarray(maps.is_interior)
    ii = np.flatnonzero(is_int)
    D_ii = D[np.ix_(ii, ii)]
    ev_D = np.linalg.eigvals(D_ii)
    ev_A = 1.0 + ev_D
    results["gluing_operator"] = dict(
        n_interior=int(ii.size),
        abs_eig_min=float(np.abs(ev_D).min()),
        abs_eig_max=float(np.abs(ev_D).max()),
        **{f"interface_{k}": v for k, v in origin_in_hull(ev_A).items()},
    )
    print("\n== interior gluing operator D (interface rows) ==")
    print(
        f"|lambda(D)| in [{np.abs(ev_D).min():.6f}, "
        f"{np.abs(ev_D).max():.6f}]  (unitary => 1)"
    )
    r = results["gluing_operator"]
    print(
        f"I + D: min|lambda| = {r['interface_min_abs_eig']:.3e}, "
        f"origin in convex hull = {r['interface_origin_in_convex_hull']}"
    )

    A_root = np.asarray(
        materialize(make_root_ItI_operator(T_leaves, maps), maps.n_flat)
    )
    ev_root = np.linalg.eigvals(A_root)
    results["root_ItI_flat_operator"] = dict(
        n=int(maps.n_flat),
        cond=float(np.linalg.cond(A_root)),
        **origin_in_hull(ev_root),
    )

    # --- 3. coupled flat BIE operator ----------------------------------
    if args.npz is not None:
        sd = load_SD_matrices_3D(args.npz)
        bp = np.asarray(domain.boundary_points).reshape(-1, 3)
        _, sdp = permute_to_domain(sd, bp)
        apply_S, apply_D = make_dense_SD_apply(sdp["S"], sdp["D"])
        sd_label = os.path.basename(args.npz)
    else:
        rng = np.random.default_rng(7)
        n_b = int(np.asarray(maps.bdry_rows).size)
        scale = 1.0 / n_b
        S_m = jnp.asarray(
            scale
            * (
                rng.standard_normal((n_b, n_b))
                + 1j * rng.standard_normal((n_b, n_b))
            )
        )
        D_m = jnp.asarray(
            scale
            * (
                rng.standard_normal((n_b, n_b))
                + 1j * rng.standard_normal((n_b, n_b))
            )
        )
        apply_S, apply_D = make_dense_SD_apply(S_m, D_m)
        sd_label = "synthetic-random-SD"
    results["sd_source"] = sd_label

    A_bie = np.asarray(
        materialize(
            make_flat_bie_operator(
                T_leaves, maps, float(args.kappa), apply_S, apply_D
            ),
            maps.n_flat,
        )
    )
    ev_bie = np.linalg.eigvals(A_bie)
    results["flat_bie_operator"] = dict(
        n=int(maps.n_flat),
        cond=float(np.linalg.cond(A_bie)),
        **origin_in_hull(ev_bie),
    )
    print(f"\n== coupled flat BIE operator ({sd_label}) ==")
    rb = results["flat_bie_operator"]
    print(
        f"cond = {rb['cond']:.3e}, min|lambda| = {rb['min_abs_eig']:.3e}, "
        f"origin in convex hull = {rb['origin_in_convex_hull']}"
    )

    # --- 4. GMRES with candidate preconditioners -----------------------
    rng = np.random.default_rng(11)
    rhs = rng.standard_normal(maps.n_flat) + 1j * rng.standard_normal(
        maps.n_flat
    )
    rhs = rhs / np.linalg.norm(rhs)
    gm = {}
    for name, M in preconditioners(A_bie, maps, maps.n_per_leaf).items():
        gm[name] = gmres_run(
            A_bie, rhs, M, args.tol, args.restart, args.maxiter
        )
        g = gm[name]
        print(
            f"{name:>22}: matvecs = {g['n_matvec']:4d}  "
            f"rel_res = {g['rel_res']:.3e}  converged = {g['converged']}"
        )
    results["gmres"] = gm

    npz_path = os.path.join(OUT_DIR, f"{args.out}.npz")
    np.savez_compressed(
        npz_path,
        ev_gluing=ev_D,
        ev_interface=ev_A,
        ev_root=ev_root,
        ev_bie=ev_bie,
        **{
            f"res_{name}": np.asarray(g["res_history"])
            for name, g in gm.items()
        },
    )
    json_path = os.path.join(OUT_DIR, f"{args.out}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {npz_path}\nwrote {json_path}")

    if args.plot:
        make_plots(ev_D, ev_A, ev_bie, gm, args)


def make_plots(ev_D, ev_A, ev_bie, gm, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    th = np.linspace(0, 2 * np.pi, 400)

    ax[0].plot(np.cos(th), np.sin(th), "k--", lw=1, label="unit circle")
    ax[0].scatter(ev_D.real, ev_D.imag, s=6, alpha=0.6)
    ax[0].set_title(r"gluing operator $\mathcal{D}$")

    ax[1].plot(
        1 + np.cos(th), np.sin(th), "k--", lw=1, label=r"$|\lambda-1|=1$"
    )
    ax[1].scatter(ev_A.real, ev_A.imag, s=6, alpha=0.6)
    ax[1].plot([0], [0], "rx", ms=9, label="origin")
    ax[1].set_title(r"interface system $I + \mathcal{D}$")

    ax[2].scatter(ev_bie.real, ev_bie.imag, s=6, alpha=0.6)
    ax[2].plot([0], [0], "rx", ms=9, label="origin")
    ax[2].set_title("coupled flat BIE operator")

    for a_ in ax:
        a_.set_xlabel(r"$\Re\lambda$")
        a_.set_ylabel(r"$\Im\lambda$")
        a_.set_aspect("equal")
        a_.legend(loc="upper right", fontsize=8)
        a_.grid(alpha=0.3)
    fig.suptitle(
        rf"$\kappa = {args.kappa}$, $L={args.L}$, $q={args.q}$, $p={args.p}$"
    )
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, f"{args.out}_spectra.png")
    fig.savefig(p1, dpi=140)

    fig2, ax2 = plt.subplots(figsize=(6, 4.4))
    for name, g in gm.items():
        ax2.semilogy(g["res_history"], label=f"{name} ({g['n_matvec']} mv)")
    ax2.set_xlabel("GMRES iteration (within cycle)")
    ax2.set_ylabel("relative residual")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_title("coupled flat BIE: preconditioner comparison")
    fig2.tight_layout()
    p2 = os.path.join(OUT_DIR, f"{args.out}_gmres.png")
    fig2.savefig(p2, dpi=140)
    print(f"wrote {p1}\nwrote {p2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=str, default=None)
    parser.add_argument("--kappa", type=float, default=4.0)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--p", type=int, default=8)
    parser.add_argument("--b_amp", type=float, default=0.5)
    parser.add_argument("--absorption", type=float, default=0.2)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--restart", type=int, default=200)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--out", type=str, default="spectrum")
    parser.add_argument("--mode", choices=("full", "refine"), default="full")
    parser.add_argument("--plot", action="store_true", default=True)
    parsed = parser.parse_args()
    # p=20 with 8 leaves exceeds 30 GB of host memory in the local solve.
    parsed.refine_pairs = [(4, 8), (6, 12), (8, 16)]
    main(parsed)
