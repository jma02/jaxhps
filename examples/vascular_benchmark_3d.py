"""One FFT solve for a branching breast phantom at a specified box size in wavelengths."""

import argparse
import json
import platform
import resource
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
from lucka_phantom_3d import vascular_geometry, vascular_phantom


def receiver_points(count=512, radius=2.5):
    """Fixed equal-weight Fibonacci receiver samples, not a volume norm."""
    index = np.arange(count)
    z = 1 - 2 * (index + 0.5) / count
    phi = index * np.pi * (3 - np.sqrt(5))
    return radius * np.column_stack(
        [np.sqrt(1 - z**2) * np.cos(phi), np.sqrt(1 - z**2) * np.sin(phi), z]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=["vascular", "dense"], default="vascular"
    )
    parser.add_argument("--waves", type=float, default=10)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--restart", type=int, default=30)
    parser.add_argument("--receivers", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.waves <= 0 or args.n < 4 or args.receivers < 1:
        parser.error("positive waves, n>=4, and receivers>=1 required")
    jax.config.update("jax_enable_x64", True)
    a = 1.25
    kappa = np.pi * args.waves / a
    device = jax.devices()[0]
    branches, lobules = vascular_geometry(args.variant, args.seed)
    row = dict(
        parameters={
            **vars(args),
            "out": str(args.out),
            "a": a,
            "kappa": kappa,
        },
        environment=dict(
            python=platform.python_version(),
            jax=jax.__version__,
            numpy=np.__version__,
            scipy=scipy.__version__,
            device=device.device_kind,
            dtype="complex128",
        ),
        branches=len(branches),
        lobules=len(lobules),
        points_per_background_wavelength=args.n / args.waves,
        minimum_vessel_support_diameter=2
        * min(b[3] for b in branches)
        * 0.8
        * a,
        phase="setup",
    )

    def save():
        temporary = args.out.with_suffix(".tmp")
        temporary.write_text(json.dumps(row))
        temporary.replace(args.out)

    save()
    t0 = time.perf_counter()
    points, b, kernel, operator = prepare_volume_solver(
        args.n,
        a,
        kappa,
        lambda pts: vascular_phantom(pts, a, args.variant, args.seed),
    )
    row["setup_seconds"] = time.perf_counter() - t0
    row["coefficient_range"] = [float(b.min()), float(b.max())]
    row["phase"] = "solve"
    save()
    print(f"SETUP {row['setup_seconds']:.2f}s; starting solve", flush=True)
    t1 = time.perf_counter()
    u, stats = solve_volume(
        operator,
        points,
        kappa,
        np.array([0, 0, 1]),
        restart=args.restart,
        maxiter=args.maxiter,
    )
    row["solve_seconds"] = time.perf_counter() - t1
    row["solve"] = stats
    row["converged"] = stats["info"] == 0 and stats["final_rel_res"] <= 1e-8
    row["phase"] = "evaluation" if row["converged"] else "complete"
    save()
    print(
        f"SOLVE {row['solve_seconds']:.2f}s residual={stats['final_rel_res']:.3g}",
        flush=True,
    )
    if row["converged"]:
        xyz = receiver_points(args.receivers)
        t1 = time.perf_counter()
        field = np.asarray(
            evaluate_volume_field(
                jnp.asarray(xyz),
                jnp.asarray(points),
                b * u,
                kappa,
                2 * a / args.n,
                target_batch=4,
            )
        )
        row["evaluation_seconds"] = time.perf_counter() - t1
        row["field_real"] = field.real.tolist()
        row["field_imag"] = field.imag.tolist()
        row["receivers"] = xyz.tolist()
        row["field_norm"] = float(np.linalg.norm(field))
        row["slice_real"] = np.asarray(u[:, args.n // 2, :]).real.tolist()
        row["slice_abs"] = np.asarray(abs(u[:, args.n // 2, :])).tolist()
    row["phase"] = "complete"
    row["case_seconds"] = time.perf_counter() - t0
    row["kernel_bytes"] = kernel.nbytes
    row["device_memory"] = device.memory_stats()
    row["host_peak_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    save()


if __name__ == "__main__":
    main()
