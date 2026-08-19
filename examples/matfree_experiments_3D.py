r"""Numerical study of the matrix-free-interior 3D HPS + BIE solver.

Four experiments, all driven through
:func:`wave_scattering_utils_3D.solve_scattering_bie_3D_matfree`, which keeps
the HPS hierarchy unmerged (no dense root DtN map) and solves the flat
interface/BIE system of :mod:`jaxhps._matfree_iti_3D`:

``mie``
    Convergence of the exterior scattered field against the analytic Mie
    series for a radial scatterer, refining the boundary order ``q`` at fixed
    level and refining the level ``L`` at fixed ``q``.  Also reports the
    dense-root-DtN driver on the same discretization, so accuracy and cost
    are compared at equal discretization.

``kappa``
    GMRES matvecs and accuracy versus wavenumber, at a fixed number of
    boundary points per wavelength.

``contrast``
    GMRES matvecs versus the contrast ``max b`` of the scattering potential,
    at fixed discretization and wavenumber.

``nsrc``
    Cost per right-hand side for ``n_src = 1, 4, 16, 64`` incident
    directions, solved one at a time versus all at once in a shared Krylov
    space, plus the memory used by the leaf blocks and the flat vectors.

The (S, D) boundary matrices come from ``examples/gen_SD_3D.py`` fixtures;
each experiment states which fixture it needs and skips cases whose fixture
is missing rather than silently changing the discretization.

Usage:
    python examples/matfree_experiments_3D.py mie
    python examples/matfree_experiments_3D.py kappa
    python examples/matfree_experiments_3D.py contrast
    python examples/matfree_experiments_3D.py nsrc
"""

import argparse
import json
import os
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from wave_scattering_utils_3D import (  # noqa: E402
    eval_uscat_offsurface_3D,
    load_SD_matrices_3D,
    solve_scattering_bie_3D,
    solve_scattering_bie_3D_matfree,
)
from mie_3d import mie_scattered_field  # noqa: E402

jax.config.update("jax_enable_x64", True)

SD_DIR = "data/examples/SD_3D"
OUT_DIR = "data/examples/matfree_experiments"
R_BUMP = 0.6
DEFAULT_DIR = np.array([[0.0, 0.0, 1.0]])


def b_poly(amp: float, R: float = R_BUMP):
    """Smooth compactly supported radial potential, ``max b = amp``."""

    def b_radial(r):
        rho = np.where(r < R, r / R, 1.0)
        return np.where(r < R, amp * (1.0 - rho * rho) ** 4, 0.0)

    return b_radial


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0**2


def sphere_targets(rho: float, n_theta: int = 12, n_phi: int = 24):
    th = np.arccos(np.linspace(-0.95, 0.95, n_theta))
    ph = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    return np.column_stack(
        [
            (rho * np.sin(TH) * np.cos(PH)).ravel(),
            (rho * np.sin(TH) * np.sin(PH)).ravel(),
            (rho * np.cos(TH)).ravel(),
        ]
    )


def mie_error(out: dict, sd: dict, b_radial, source_dirs, rho: float) -> float:
    """Relative L2 error of the exterior scattered field against Mie."""
    tgt = sphere_targets(rho)
    u = np.asarray(
        eval_uscat_offsurface_3D(
            jnp.asarray(tgt),
            jnp.asarray(out["boundary_points"]),
            jnp.asarray(out["normals"]),
            jnp.asarray(out["sdp"]["wts"]),
            jnp.asarray(out["uscat_b"]),
            jnp.asarray(out["uscat_dn_b"]),
            float(sd["kappa"]),
        )
    )
    if u.ndim == 1:
        u = u[:, None]
    errs = []
    for i, w in enumerate(np.asarray(source_dirs)):
        ref = mie_scattered_field(tgt, w, float(sd["kappa"]), b_radial, R_BUMP)
        errs.append(
            np.linalg.norm(u[:, i] - ref) / np.linalg.norm(ref),
        )
    return float(np.max(errs))


def sd_path(kappa: float, q: int, L: int, a: float) -> str:
    return os.path.join(SD_DIR, f"SD_k{kappa:g}_q{q}_L{L}_a{a:g}.npz")


def load_or_skip(path: str):
    if not os.path.exists(path):
        print(f"  [skip] missing fixture {path}")
        return None
    return load_SD_matrices_3D(path)


def run_matfree(
    sd,
    b_radial,
    source_dirs,
    method="gmres",
    tol=1e-10,
    maxiter=600,
    restart=200,
    p=None,
    precond="none",
):
    stats: dict = {}
    t0 = time.perf_counter()
    out = solve_scattering_bie_3D_matfree(
        sd,
        b_radial,
        source_dirs,
        method=method,
        tol=tol,
        maxiter=maxiter,
        restart=restart,
        stats=stats,
        p=p,
        precond=precond,
    )
    jax.block_until_ready(jnp.asarray(out["uscat_b"]))
    out["wall_time"] = time.perf_counter() - t0
    out["stats"] = stats
    return out


def leaf_vs_root_bytes(sd, out) -> dict:
    """Memory of the unmerged leaf blocks versus the dense root operator."""
    q, L = int(sd["q"]), int(sd["L"])
    n_leaf_bdry = 6 * q**2
    leaf_bytes = 8**L * n_leaf_bdry**2 * 16
    n_root = 6 * (2**L * q) ** 2
    root_bytes = n_root**2 * 16
    return dict(
        leaf_ItI_GB=leaf_bytes / 1024.0**3,
        root_DtN_GB=root_bytes / 1024.0**3,
        n_flat=int(out["maps"].n_flat),
        n_bdry=int(n_root),
    )


def exp_mie(args) -> None:
    """Convergence against Mie: refine q at L=1, then refine L at q=8."""
    b_radial = b_poly(args.b_amp)
    dirs = DEFAULT_DIR
    rows = []
    cases = (
        [tuple(int(v) for v in c.split(",")) for c in args.cases.split(";")]
        if args.cases
        else [(q, 1) for q in (4, 6, 8, 10, 12)] + [(8, 2)]
    )
    for q, L in cases:
        path = sd_path(4.0, q, L, 1.25)
        sd = load_or_skip(path)
        if sd is None:
            continue
        out = run_matfree(
            sd,
            b_radial,
            dirs,
            method="gmres",
            tol=args.tol,
            maxiter=args.maxiter,
            restart=args.restart,
            precond=args.precond,
        )
        err = mie_error(out, sd, b_radial, dirs, args.rho)
        st = out["stats"]
        row = dict(
            q=q,
            L=L,
            err_matfree=err,
            n_matvec=int(st["n_matvec"]),
            converged=bool(out["info"]["converged"]),
            rel_res=float(st["final_rel_res"]),
            time_matfree=out["wall_time"],
            **leaf_vs_root_bytes(sd, out),
        )
        if args.with_dense:
            t0 = time.perf_counter()
            outd = solve_scattering_bie_3D(sd, b_radial, dirs)
            jax.block_until_ready(jnp.asarray(outd["uscat_b"]))
            row["time_dense"] = time.perf_counter() - t0
            row["err_dense"] = mie_error(outd, sd, b_radial, dirs, args.rho)
            row["trace_diff"] = float(
                np.linalg.norm(out["uscat_b"] - outd["uscat_b"])
                / np.linalg.norm(outd["uscat_b"])
            )
        rows.append(row)
        dump("mie", rows, args)
        print(
            f"  q={q} L={L} n_bdry={row['n_bdry']:5d}  "
            f"err={err:.3e}  matvecs={row['n_matvec']:4d}  "
            f"t={row['time_matfree']:.1f}s"
            + (
                f"  err_dense={row['err_dense']:.3e}"
                f"  t_dense={row['time_dense']:.1f}s"
                f"  |diff|={row['trace_diff']:.2e}"
                if args.with_dense
                else ""
            )
        )
    dump("mie", rows, args)


def exp_kappa(args) -> None:
    """GMRES matvecs and accuracy versus wavenumber."""
    b_radial = b_poly(args.b_amp)
    rows = []
    for kappa, q, L in args.kappa_cases:
        path = sd_path(kappa, q, L, 1.25)
        sd = load_or_skip(path)
        if sd is None:
            continue
        out = run_matfree(
            sd,
            b_radial,
            DEFAULT_DIR,
            tol=args.tol,
            maxiter=args.maxiter,
            restart=args.restart,
            precond=args.precond,
        )
        err = mie_error(out, sd, b_radial, DEFAULT_DIR, args.rho)
        st = out["stats"]
        n_bdry = 6 * (2**L * q) ** 2
        ppw = (2**L * q) / (kappa * 2 * 1.25 / (2 * np.pi))
        rows.append(
            dict(
                precond=args.precond,
                kappa=kappa,
                q=q,
                L=L,
                n_bdry=n_bdry,
                pts_per_wavelength=float(ppw),
                err=err,
                n_matvec=int(st["n_matvec"]),
                rel_res=float(st["final_rel_res"]),
                converged=bool(out["info"]["converged"]),
                time=out["wall_time"],
                peak_rss_GB=peak_rss_gb(),
            )
        )
        dump("kappa", rows, args)
        r = rows[-1]
        print(
            f"  kappa={kappa:5g} q={q} L={L}  matvecs={r['n_matvec']:4d}  "
            f"err={err:.3e}  t={r['time']:.1f}s  "
            f"pts/wavelength={ppw:.1f}"
        )
    dump("kappa", rows, args)


def exp_contrast(args) -> None:
    """GMRES matvecs versus contrast of the scattering potential."""
    sd = load_or_skip(sd_path(4.0, 8, 1, 1.25))
    if sd is None:
        return
    rows = []
    for amp in args.contrasts:
        b_radial = b_poly(amp)
        out = run_matfree(
            sd,
            b_radial,
            DEFAULT_DIR,
            tol=args.tol,
            maxiter=args.maxiter,
            restart=args.restart,
            precond=args.precond,
        )
        err = mie_error(out, sd, b_radial, DEFAULT_DIR, args.rho)
        st = out["stats"]
        rows.append(
            dict(
                b_amp=amp,
                n_matvec=int(st["n_matvec"]),
                rel_res=float(st["final_rel_res"]),
                converged=bool(out["info"]["converged"]),
                err=err,
                time=out["wall_time"],
            )
        )
        print(
            f"  max b={amp:5g}  matvecs={rows[-1]['n_matvec']:4d}  "
            f"err={err:.3e}  t={rows[-1]['time']:.1f}s"
        )
    dump("contrast", rows, args)


def exp_nsrc(args) -> None:
    """Cost per RHS: shared Krylov space versus one solve per source."""
    sd = load_or_skip(sd_path(4.0, 8, 1, 1.25))
    if sd is None:
        return
    b_radial = b_poly(args.b_amp)
    rows = []
    for k in args.nsrc_list:
        dirs = fibonacci_dirs(k)
        out = run_matfree(
            sd,
            b_radial,
            dirs,
            tol=args.tol,
            maxiter=args.maxiter,
            restart=args.restart,
            precond=args.precond,
        )
        st = out["stats"]
        row = dict(
            n_src=k,
            shared_matvec=int(st["n_matvec"]),
            shared_time=out["wall_time"],
            shared_time_per_src=out["wall_time"] / k,
            converged=bool(out["info"]["converged"]),
            rel_res=float(st["final_rel_res"]),
            peak_rss_GB=peak_rss_gb(),
            **leaf_vs_root_bytes(sd, out),
        )
        if k <= args.seq_max:
            t0 = time.perf_counter()
            seq_mv = 0
            for i in range(k):
                o1 = run_matfree(
                    sd,
                    b_radial,
                    dirs[i : i + 1],
                    tol=args.tol,
                    maxiter=args.maxiter,
                    restart=args.restart,
                    precond=args.precond,
                )
                seq_mv += int(o1["stats"]["n_matvec"])
            row["seq_time"] = time.perf_counter() - t0
            row["seq_matvec"] = seq_mv
        rows.append(row)
        dump("nsrc", rows, args)
        print(
            f"  n_src={k:3d}  shared: matvecs={row['shared_matvec']:5d} "
            f"t={row['shared_time']:.1f}s "
            f"({row['shared_time_per_src']:.2f}s/src)"
            + (
                f"  sequential: matvecs={row['seq_matvec']:5d} "
                f"t={row['seq_time']:.1f}s"
                if "seq_time" in row
                else ""
            )
        )
    dump("nsrc", rows, args)


def fibonacci_dirs(n: int) -> np.ndarray:
    """``n`` roughly equidistributed unit directions."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.column_stack(
        [
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ]
    )


def dump(name: str, rows: list, args) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}{args.tag}.json")
    with open(path, "w") as f:
        json.dump(
            dict(experiment=name, args=vars(args), rows=rows), f, indent=2
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment", choices=("mie", "kappa", "contrast", "nsrc")
    )
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--rho", type=float, default=2.5)
    parser.add_argument("--b_amp", type=float, default=0.5)
    parser.add_argument("--with_dense", action="store_true")
    parser.add_argument("--seq_max", type=int, default=16)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--restart", type=int, default=200)
    parser.add_argument(
        "--precond", choices=("none", "jacobi"), default="none"
    )
    parser.add_argument("--cases", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    parsed = parser.parse_args()
    # (kappa, q, L): q grows with kappa to hold points-per-wavelength fixed.
    parsed.kappa_cases = [
        (2.0, 6, 1),
        (4.0, 8, 1),
        (8.0, 12, 1),
        (12.0, 14, 1),
        (16.0, 16, 1),
    ]
    parsed.contrasts = [0.1, 0.25, 0.5, 0.75, 1.0]
    parsed.nsrc_list = [1, 4, 16, 64]
    {
        "mie": exp_mie,
        "kappa": exp_kappa,
        "contrast": exp_contrast,
        "nsrc": exp_nsrc,
    }[parsed.experiment](parsed)
