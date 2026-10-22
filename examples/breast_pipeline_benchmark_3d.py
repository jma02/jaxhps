"""Measure repeated multi-source data generation and eight-step Born imaging.

The baseline keeps the original closed-over, unrolled CG calculation. Both
paths use identical hemispheres, fields, arithmetic precision and tolerances.
No network training or reconstruction-accuracy claim is part of this benchmark.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from breast_phantom_3d import breast_n_of_x, fibonacci_cap_points
from breast_spheres_learn_3d import A, KVAL, reconstruct_born, sample_spheres


def run(args: argparse.Namespace) -> None:
    if args.baseline:
        from baseline_inverse import make_background_born, make_forward
    else:
        from phantom_inverse_3d import make_background_born, make_forward

    jax.config.update("jax_enable_x64", False)
    start = time.perf_counter()
    n = args.n
    axis = -A + (np.arange(n) + 0.5) * (2 * A / n)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)
    radius = np.linalg.norm(points, axis=-1)
    window = jnp.asarray(
        0.25
        * (1 + np.tanh(KVAL * points[..., 1]))
        * (1 + np.tanh(KVAL * (1 - 1 / 30 - radius))),
        dtype=jnp.float32,
    )
    background = jnp.asarray(
        1
        - breast_n_of_x(
            points, np.zeros((0, 3)), np.zeros(0), np.zeros(0), kval=KVAL
        ),
        dtype=jnp.float32,
    )
    wavelength = args.wavelength_mm / 60.0
    kappa = 2 * np.pi / wavelength
    receivers = fibonacci_cap_points(256, 1 + wavelength, 0.1)
    emitters = fibonacci_cap_points(24, 1 + wavelength, 0.1)
    directions = -emitters / np.linalg.norm(emitters, axis=-1, keepdims=True)
    forward = make_forward(points, kappa, directions, receivers, 1e-6)
    reference, residuals = forward(background)
    residual = float(jnp.max(residuals))
    if not np.isfinite(residual) or residual > 5e-6:
        raise RuntimeError(f"Background residual {residual}")
    operator = make_background_born(
        points, kappa, directions, receivers, background, 1e-6
    )
    setup_seconds = time.perf_counter() - start
    print(f"SETUP {setup_seconds:.3f}s", flush=True)

    if args.baseline:
        linear, adjoint = operator

        @jax.jit
        def born(data: jax.Array) -> jax.Array:
            scale = jnp.sum(jnp.abs(data) ** 2)
            rhs = window * adjoint(data)

            def normal(x: jax.Array) -> jax.Array:
                return (
                    window * adjoint(linear((window * x).astype(jnp.float32)))
                    + 0.001 * scale * x
                )

            x = jnp.zeros_like(rhs)
            r, p = rhs, rhs
            rr = jnp.sum(r * r)
            for _ in range(8):
                q = normal(p)
                alpha = rr / jnp.sum(p * q)
                x = x + alpha * p
                r = r - alpha * q
                rr_new = jnp.sum(r * r)
                p = r + (rr_new / rr) * p
                rr = rr_new
            return window * x
    else:

        def born(data: jax.Array) -> jax.Array:
            return reconstruct_born(operator, data, window, 8, 0.001)

    rng = np.random.default_rng(0)
    rows, observations, images = [], [], []
    for index in range(args.count):
        centres, radii, mvals = sample_spheres(rng)
        truth = jnp.asarray(
            1 - breast_n_of_x(points, centres, radii, mvals, kval=KVAL),
            dtype=jnp.float32,
        )
        start = time.perf_counter()
        data, residuals = forward(truth)
        jax.block_until_ready((data, residuals))
        forward_seconds = time.perf_counter() - start
        start = time.perf_counter()
        reconstructed = born(data - reference).block_until_ready()
        born_seconds = time.perf_counter() - start
        residual = float(jnp.max(residuals))
        if not np.isfinite(residual) or residual > 5e-6:
            raise RuntimeError(f"Forward residual {residual}")
        if not bool(jnp.all(jnp.isfinite(reconstructed))):
            raise RuntimeError("Nonfinite Born reconstruction")
        rows.append(
            dict(
                index=index,
                forward_seconds=forward_seconds,
                born_seconds=born_seconds,
                residual=residual,
            )
        )
        observations.append(np.asarray(data))
        images.append(np.asarray(reconstructed))
        print(f"SAMPLE {json.dumps(rows[-1])}", flush=True)

    observed, images = np.stack(observations), np.stack(images)
    comparisons = {}
    if args.baseline:
        np.savez(
            args.reference,
            observed=observed,
            images=images,
            reference=np.asarray(reference),
        )
    else:
        with np.load(args.reference) as saved:
            for name, current, expected, tolerance in (
                ("measurements", observed, saved["observed"], 2e-5),
                (
                    "background_subtracted",
                    observed - np.asarray(reference),
                    saved["observed"] - saved["reference"],
                    2e-5,
                ),
                ("born", images, saved["images"], 1e-4),
            ):
                error = np.linalg.norm(
                    (current - expected).reshape(args.count, -1), axis=1
                ) / np.linalg.norm(expected.reshape(args.count, -1), axis=1)
                comparisons[name] = dict(
                    max_relative_difference=float(error.max()),
                    mean_relative_difference=float(error.mean()),
                    tolerance=tolerance,
                )
                if not np.isfinite(error).all() or error.max() > tolerance:
                    raise RuntimeError(
                        f"Numerical comparison failed: {name} {error.max()}"
                    )

    row = dict(
        baseline=args.baseline,
        n=n,
        wavelength_mm=args.wavelength_mm,
        sources=24,
        receivers=256,
        count=args.count,
        precision="complex64",
        tolerance=1e-6,
        born_iterations=8,
        damping=0.001,
        setup_seconds=setup_seconds,
        samples=rows,
        comparisons=comparisons,
        device=jax.devices()[0].device_kind,
        jax=jax.__version__,
        peak_memory_bytes=jax.devices()[0].memory_stats()["peak_bytes_in_use"],
    )
    args.out.write_text(json.dumps(row, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--wavelength-mm", type=float, default=5)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
