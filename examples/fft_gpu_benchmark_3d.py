"""Fresh-process timing and field comparison for the FFT memory optimizations."""

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lucka_phantom_3d import vascular_phantom
from vascular_phantom_device import vascular_phantom_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--n", type=int, default=288)
    parser.add_argument("--waves", type=float, default=80)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.baseline:
        import baseline_fft as fft
    else:
        import fft_volume_3d as fft

    jax.config.update("jax_enable_x64", True)
    device = jax.devices()[0]
    n, a = args.n, 1.25
    kappa = np.pi * args.waves / a
    row = dict(
        baseline=args.baseline,
        n=n,
        waves=args.waves,
        a=a,
        kappa=kappa,
        variant="dense",
        seed=7,
        direction=[0, 0, 1],
        device=device.device_kind,
        jax=jax.__version__,
        numpy=np.__version__,
        python=platform.python_version(),
        precision="complex128",
        restart=30,
        maxiter=300,
        tolerance=1e-8,
    )
    original_kernel = fft.green_kernel

    def timed_kernel(n: int, a: float, kappa: float) -> jax.Array:
        start = time.perf_counter()
        result = original_kernel(n, a, kappa)
        result.block_until_ready()
        row["kernel_seconds"] = time.perf_counter() - start
        row["setup_memory"] = device.memory_stats()
        return result

    def coefficient(points: np.ndarray) -> np.ndarray | jax.Array:
        start = time.perf_counter()
        sample = vascular_phantom if args.baseline else vascular_phantom_device
        result = sample(points, a, "dense", 7)
        jax.block_until_ready(result)
        row["coefficient_seconds"] = time.perf_counter() - start
        return result

    fft.green_kernel = timed_kernel
    start = time.perf_counter()
    points, b, _, operator = fft.prepare_volume_solver(
        n, a, kappa, coefficient
    )
    row["setup_seconds"] = time.perf_counter() - start
    print("SETUP", row["setup_seconds"], flush=True)
    start = time.perf_counter()
    u, stats = fft.solve_volume(
        operator,
        points,
        kappa,
        np.array([0, 0, 1]),
        tol=1e-8,
        restart=30,
        maxiter=300,
    )
    row["solve_seconds"] = time.perf_counter() - start
    row["cold_seconds"] = row["setup_seconds"] + row["solve_seconds"]
    row["solve"] = stats
    row["cold_memory"] = device.memory_stats()
    row["converged"] = stats["info"] == 0 and stats["final_rel_res"] <= 1e-8
    print(
        "SOLVE",
        row["solve_seconds"],
        "RESIDUAL",
        stats["final_rel_res"],
        flush=True,
    )

    if args.baseline:
        np.save(args.reference, np.asarray(u))
        np.save(args.reference.with_suffix(".coefficient.npy"), np.asarray(b))
    else:
        reference = np.load(args.reference)
        row["volume_relative_difference"] = float(
            np.linalg.norm(np.asarray(u) - reference)
            / np.linalg.norm(reference)
        )
        del reference
        reference = np.load(args.reference.with_suffix(".coefficient.npy"))
        row["coefficient_relative_difference"] = float(
            np.linalg.norm(np.asarray(b) - reference)
            / np.linalg.norm(reference)
        )
        row["coefficient_max_difference"] = float(
            np.max(np.abs(np.asarray(b) - reference))
        )
        del reference
    index = np.arange(512)
    z = 1 - 2 * (index + 0.5) / len(index)
    phi = index * np.pi * (3 - np.sqrt(5))
    targets = 2.5 * np.column_stack(
        (np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z)
    )
    start = time.perf_counter()
    field = np.asarray(
        fft.evaluate_volume_field(
            jnp.asarray(targets),
            jnp.asarray(points),
            b * u,
            kappa,
            2 * a / n,
            target_batch=4,
        )
    )
    row["evaluation_seconds"] = time.perf_counter() - start
    row["field_real"], row["field_imag"] = (
        field.real.tolist(),
        field.imag.tolist(),
    )
    args.out.write_text(json.dumps(row, indent=2))
    if not row["converged"]:
        raise RuntimeError("true residual exceeds tolerance")


if __name__ == "__main__":
    main()
