"""Where the memory goes in a matrix-free coupled solve.

Reports resident memory after each stage of
``solve_scattering_bie_3D_matfree`` for one fixture, so the peak can be
attributed (leaf local solves, exterior S/D operators, Krylov basis) instead of
inferred.  Run as

    python examples/matfree_memory_profile_3D.py --npz data/examples/SD_3D/SD_k8_q12_L1_a1.25.npz
"""

import argparse
import json
import os
import resource
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = "data/examples/matfree_experiments"


def rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024**2
    return float("nan")


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npz", default="data/examples/SD_3D/SD_k8_q12_L1_a1.25.npz"
    )
    p.add_argument("--restart", type=int, default=400)
    args = p.parse_args()

    stages = []

    def mark(name: str) -> None:
        stages.append(dict(stage=name, rss_GB=rss_gb(), peak_GB=peak_gb()))
        print(f"  {name:28s} rss={stages[-1]['rss_GB']:6.2f} GB")

    mark("start")

    import jax.numpy as jnp  # noqa: E402

    from jaxhps import DiscretizationNode3D, Domain, PDEProblem  # noqa: E402
    from jaxhps._matfree_iti_3D import build_interface_maps  # noqa: E402
    from jaxhps.local_solve import (  # noqa: E402
        local_solve_stage_uniform_3D_ItI,
    )
    from wave_scattering_utils_3D import (  # noqa: E402
        load_SD_matrices_3D,
        permute_to_domain,
    )

    mark("imports")

    sd = load_SD_matrices_3D(args.npz)
    mark("load S, D fixture")

    a, q, L, kappa = sd["a"], sd["q"], sd["L"], sd["kappa"]
    eta = float(kappa)
    p_ord = q + 4
    root = DiscretizationNode3D(
        xmin=-a, xmax=a, ymin=-a, ymax=a, zmin=-a, zmax=a
    )
    domain = Domain(p=p_ord, q=q, root=root, L=L)
    mark("build domain")

    int_pts = np.asarray(domain.interior_points)
    r = np.linalg.norm(int_pts, axis=-1)
    R = 0.3
    b_int = np.where(r < R, -0.5 * (1.0 - (r / R) ** 2) ** 4, 0.0)
    ones = np.ones_like(b_int, dtype=np.complex128)
    d = np.array([[1.0, 0.0, 0.0]])
    uin_int = np.exp(1j * kappa * np.einsum("lpd,sd->lps", int_pts, d))
    problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=ones,
        D_yy_coefficients=ones,
        D_zz_coefficients=ones,
        I_coefficients=(kappa**2 * (1.0 - b_int)).astype(np.complex128),
        source=(kappa**2 * b_int[..., None] * uin_int).astype(np.complex128),
        use_ItI=True,
        eta=eta,
    )
    mark("assemble PDEProblem")

    _, T_leaves, _, h_leaves = local_solve_stage_uniform_3D_ItI(problem)
    T_leaves.block_until_ready()
    mark("leaf local solves")

    maps = build_interface_maps(domain)
    mark("interface maps")

    bp = np.asarray(domain.boundary_points).reshape(-1, 3)
    _, sdp = permute_to_domain(sd, bp)
    S = jnp.asarray(sdp["S"])
    S.block_until_ready()
    mark("S, D permuted to device")

    n_flat = maps.n_flat
    basis = jnp.zeros((args.restart + 1, n_flat), dtype=jnp.complex128)
    basis.block_until_ready()
    mark(f"Krylov basis (restart={args.restart})")

    sizes = dict(
        n_bdry=int(bp.shape[0]),
        n_flat=int(n_flat),
        leaf_ItI_GB=float(T_leaves.nbytes / 1024**3),
        h_leaves_GB=float(np.asarray(h_leaves).nbytes / 1024**3),
        S_plus_D_GB=float(2 * S.nbytes / 1024**3),
        krylov_basis_GB=float(basis.nbytes / 1024**3),
        dense_root_DtN_GB=float(bp.shape[0] ** 2 * 16 / 1024**3),
    )
    print(json.dumps(sizes, indent=2))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(
        OUT_DIR,
        f"memory_k{kappa:g}_q{q}_L{L}.json",
    )
    with open(out, "w") as f:
        json.dump(
            dict(
                npz=args.npz,
                q=int(q),
                L=int(L),
                kappa=float(kappa),
                stages=stages,
                sizes=sizes,
            ),
            f,
            indent=2,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
