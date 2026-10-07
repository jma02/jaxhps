"""One fresh-process accuracy/timing case. See accuracy_benchmarks.md."""

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from fft_volume_3d import (
    evaluate_volume_field,
    prepare_volume_solver,
    solve_volume,
)
from lucka_phantom_3d import tissue_phantom
from radial_reference_3d import radial_reference
from wave_scattering_utils_3D import (
    eval_uscat_offsurface_3D,
    load_SD_matrices_3D,
    solve_scattering_bie_3D_matfree,
)

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def targets():
    cosine, weights = np.polynomial.legendre.leggauss(12)
    phi = 2 * np.pi * np.arange(24) / 24
    c, p = np.meshgrid(cosine, phi, indexing="ij")
    xyz = 2.5 * np.stack(
        [np.sqrt(1 - c**2) * np.cos(p), np.sqrt(1 - c**2) * np.sin(p), c],
        axis=-1,
    ).reshape(-1, 3)
    return xyz, np.repeat(weights * (2 * np.pi / 24), 24)


def radial(r):
    return -0.4 * np.maximum(1 - (np.asarray(r) / 0.6) ** 2, 0) ** 4


def relative_error(field, reference, weights):
    return float(
        np.sqrt(
            np.sum(weights * abs(field - reference) ** 2)
            / np.sum(weights * abs(reference) ** 2)
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=["fft", "hps"], required=True)
    parser.add_argument(
        "--kind", choices=["radial", "phantom"], default="radial"
    )
    parser.add_argument("--kappa", type=float, default=4)
    parser.add_argument("--a", type=float, default=1.25)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--p", type=int)
    parser.add_argument("--precond", default="shift-coarse:0.1")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    xyz, weights = targets()
    direction = np.array([0.0, 0.0, 1.0])
    coefficient = (
        (lambda pts: radial(np.linalg.norm(pts, axis=-1)))
        if args.kind == "radial"
        else (lambda pts: tissue_phantom(pts, args.a))
    )
    device = jax.devices()[0]
    row = dict(
        parameters={**vars(args), "out": str(args.out)},
        environment=dict(
            python=platform.python_version(),
            jax=jax.__version__,
            numpy=np.__version__,
            scipy=scipy.__version__,
            device=str(device),
            device_kind=device.device_kind,
            dtype="complex128",
            platform=platform.platform(),
        ),
        source_hashes={
            str(p.relative_to(ROOT)): sha256(p)
            for folder in (ROOT / "src", ROOT / "examples")
            for p in sorted(folder.rglob("*.py"))
        },
        target_hash=hashlib.sha256(xyz.tobytes()).hexdigest(),
        repeats=[],
    )
    if (ROOT / ".git").exists():
        row["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        row["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    t0 = time.perf_counter()
    if args.solver == "fft":
        points, b, kernel, operator = prepare_volume_solver(
            args.n, args.a, args.kappa, coefficient
        )
        row["setup_seconds"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        u, info = solve_volume(operator, points, args.kappa, direction)
        row["first_solve_seconds"] = time.perf_counter() - t1
        row["first_info"] = info
        row["unknowns"] = args.n**3
        row["kernel_bytes"] = kernel.nbytes
        row["cold_setup_solve_seconds"] = time.perf_counter() - t0
        for _ in range(args.repeats):
            t1 = time.perf_counter()
            u, info = solve_volume(operator, points, args.kappa, direction)
            row["repeats"].append(
                dict(seconds=time.perf_counter() - t1, info=info)
            )
        t1 = time.perf_counter()
        field = np.asarray(
            evaluate_volume_field(
                jnp.asarray(xyz),
                jnp.asarray(points),
                b * u,
                args.kappa,
                2 * args.a / args.n,
            )
        )
        row["evaluation_seconds"] = time.perf_counter() - t1
    else:
        fixture = (
            ROOT
            / "data/examples/SD_3D"
            / (f"SD_k{args.kappa:g}_q{args.q}_L{args.L}_a{args.a:g}.npz")
        )
        row["fixture_sha256"] = sha256(fixture)
        sd = load_SD_matrices_3D(str(fixture))
        row["fixture_load_hash_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = solve_scattering_bie_3D_matfree(
            sd,
            None,
            direction[None],
            b_cartesian=coefficient,
            p=args.p,
            method="fgmres",
            precond=args.precond,
            tol=1e-8,
            maxiter=1200,
            restart=100,
            coarse_q=3 if args.L == 1 else 2,
            coarse_limit=2048,
            stats={},
            return_solver=True,
        )
        row["cold_setup_solve_seconds"] = time.perf_counter() - t0
        row["first_info"] = out["info"]
        row["unknowns"] = out["info"]["n_flat"]
        row["interior_samples"] = int(
            np.prod(out["problem"].I_coefficients.shape)
        )
        row["setup_seconds"] = out["info"]["setup_seconds"]
        row["first_solve_seconds"] = out["info"]["first_solve_seconds"]
        row["exterior_bytes"] = sd["S"].nbytes + sd["D"].nbytes
        ub, dn = out["uscat_b"], out["uscat_dn_b"]
        for _ in range(args.repeats):
            t1 = time.perf_counter()
            ub, dn, info = out["resolve"](direction[None])
            row["repeats"].append(
                dict(seconds=time.perf_counter() - t1, info=info)
            )
        t1 = time.perf_counter()
        field = np.asarray(
            eval_uscat_offsurface_3D(
                jnp.asarray(xyz),
                jnp.asarray(out["boundary_points"]),
                jnp.asarray(out["normals"]),
                jnp.asarray(out["sdp"]["wts"]),
                jnp.asarray(ub),
                jnp.asarray(dn),
                args.kappa,
            )
        ).ravel()
        row["evaluation_seconds"] = time.perf_counter() - t1
    row["gpu_memory"] = device.memory_stats()
    row["host_peak_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    row["field_real"] = field.real.tolist()
    row["field_imag"] = field.imag.tolist()
    row["weights"] = weights.tolist()
    row["targets"] = xyz.tolist()
    if args.kind == "radial":
        reference = radial_reference(xyz, direction, args.kappa, radial, 0.6)
        row["field_relative_error"] = relative_error(field, reference, weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in row.items()
                if k
                in (
                    "parameters",
                    "cold_setup_solve_seconds",
                    "field_relative_error",
                    "gpu_memory",
                    "host_peak_rss_bytes",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
