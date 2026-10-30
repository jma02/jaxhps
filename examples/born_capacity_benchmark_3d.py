"""Measure cached-field Born CG on the full 10 cm vascular phantom.

The vessel-free tissue is known exactly. Full-wave background subtraction
supplies the data; neither a learned correction nor a recovery study is run.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from breast_phantom_3d import fibonacci_cap_points
from breast_spheres_learn_3d import reconstruct_born
from lucka_phantom_3d import vascular_ellipsoids
from phantom_inverse_3d import (
    BackgroundBorn,
    make_background_born,
    make_forward,
)
from vascular_phantom_device import _sample


def run(
    args: argparse.Namespace,
) -> tuple[BackgroundBorn, jax.Array, jax.Array, jax.Array]:
    jax.config.update("jax_enable_x64", False)
    device = jax.devices()[0]
    started = time.perf_counter()
    row = dict(
        n=args.n,
        box_width_mm=100,
        wavelength_mm=args.wavelength_mm,
        sources=args.sources,
        receivers=args.receivers,
        geometry="lower hemispherical cap; radius 100 mm",
        phantom="dense vascular seed 7; exact vessel-free background",
        precision="complex64",
        damping=0.001,
        wave_tolerance=1e-6,
        cg_relative_residual_target=1e-4,
        device=device.device_kind,
        jax=jax.__version__,
        stages=[],
        cg=[],
        completed=False,
    )

    def checkpoint(stage: str) -> None:
        row["stages"].append(
            dict(
                stage=stage,
                elapsed_seconds=time.perf_counter() - started,
                memory=device.memory_stats(),
            )
        )
        args.out.write_text(json.dumps(row, indent=2))
        print(json.dumps(row["stages"][-1]), flush=True)

    checkpoint("starting")
    n, a = args.n, 1.25
    axis = -a + (np.arange(n) + 0.5) * (2 * a / n)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)
    families = vascular_ellipsoids("dense", 7)
    jax.config.update("jax_enable_x64", True)
    xyz = jnp.asarray(points)
    fibro, vessels = (
        tuple(
            jnp.asarray(np.stack(values))
            for values in zip(*(shape for group in family for shape in group))
        )
        for family in families
    )
    truth = _sample(xyz, a, fibro, vessels, 8192).astype(jnp.float32)
    background = _sample(
        xyz, a, fibro, tuple(value[:0] for value in vessels), 8192
    ).astype(jnp.float32)
    p = (xyz - jnp.array([0, 0, 0.55 * a])) / (0.8 * a)
    window = (
        (jnp.linalg.norm(p / jnp.array([0.92, 0.82, 1]), axis=-1) < 0.97)
        & (p[..., 2] < 0)
    ).astype(jnp.float32)
    jax.block_until_ready((truth, background, window))
    jax.config.update("jax_enable_x64", False)
    delta = truth - background
    if float(jnp.linalg.norm(delta)) == 0:
        raise RuntimeError("Grid does not sample the vessel contrast")
    del xyz, p, fibro, vessels
    gc.collect()
    checkpoint("phantom_sampled")

    kappa = 2 * np.pi / (args.wavelength_mm / 40)
    receivers = fibonacci_cap_points(args.receivers, 2.5, 0.1)[:, [0, 2, 1]]
    emitters = fibonacci_cap_points(args.sources, 2.5, 0.1)[:, [0, 2, 1]]
    receivers[:, 2] *= -1
    emitters[:, 2] *= -1
    directions = -emitters / np.linalg.norm(emitters, axis=-1, keepdims=True)
    checkpoint("forward_setup_started")
    start = time.perf_counter()
    forward = make_forward(points, kappa, directions, receivers, 1e-6)
    row["forward_setup_seconds"] = time.perf_counter() - start
    for name, coefficient in (("background", background), ("truth", truth)):
        start = time.perf_counter()
        measured, residuals = forward(coefficient)
        jax.block_until_ready((measured, residuals))
        residual = float(jnp.max(residuals))
        row[f"{name}_forward_seconds"] = time.perf_counter() - start
        row[f"{name}_forward_relative_residual"] = residual
        checkpoint(f"{name}_forward_finished")
        if not np.isfinite(residual) or residual > 5e-6:
            raise RuntimeError(f"{name} forward residual {residual}")
        if name == "background":
            reference = measured
        else:
            data = (measured - reference).block_until_ready()
    del forward, measured, residuals, reference, coefficient
    jax.clear_caches()
    gc.collect()

    checkpoint("field_preparation_started")
    start = time.perf_counter()
    operator = make_background_born(
        points, kappa, directions, receivers, background, 1e-6
    )
    jax.block_until_ready(operator)
    row["field_preparation_seconds"] = time.perf_counter() - start
    row["field_storage_bytes"] = (
        operator.sources.nbytes + operator.reciprocal_fields.nbytes
    )
    row["field_residual_gate_passed"] = True
    row["field_residual_gate"] = 5e-6
    checkpoint("fields_ready")
    del points, truth, background
    jax.clear_caches()
    gc.collect()

    start = time.perf_counter()
    image = reconstruct_born(
        operator, data, window, 8, 0.001
    ).block_until_ready()
    row["cold_eight_step_seconds"] = time.perf_counter() - start
    checkpoint("cold_cg_finished")
    del image
    row["warm_eight_step_seconds"] = []
    for _ in range(3):
        start = time.perf_counter()
        image = reconstruct_born(
            operator, data, window, 8, 0.001
        ).block_until_ready()
        row["warm_eight_step_seconds"].append(time.perf_counter() - start)
        del image
    checkpoint("warm_cg_finished")

    linear_data = operator(delta)
    row["born_model_relative_data_error"] = float(
        jnp.linalg.norm(linear_data - data) / jnp.linalg.norm(data)
    )
    probe = jax.random.normal(
        jax.random.PRNGKey(2), data.shape, dtype=data.dtype
    )
    lhs = float(jnp.real(jnp.vdot(linear_data, probe)))
    rhs = float(jnp.vdot(delta, operator.adjoint(probe)))
    row["adjoint_relative_error"] = abs(lhs - rhs) / max(
        abs(lhs), abs(rhs), 1e-30
    )
    if (
        not np.isfinite(row["adjoint_relative_error"])
        or row["adjoint_relative_error"] > 1e-4
    ):
        raise RuntimeError("Real-adjoint identity failed")
    del linear_data, probe
    rhs = window * operator.adjoint(data)
    penalty = 0.001 * jnp.sum(jnp.abs(data) ** 2)
    checkpoint("operator_checks_finished")

    for iterations in (8, 16, 32, 64, 128):
        start = time.perf_counter()
        image = reconstruct_born(
            operator, data, window, iterations, 0.001
        ).block_until_ready()
        elapsed = time.perf_counter() - start
        predicted = operator(image)
        normal_residual = (
            window * operator.adjoint(predicted) + penalty * image - rhs
        )
        result = dict(
            iterations=iterations,
            seconds_including_first_compile=elapsed,
            normal_relative_residual=float(
                jnp.linalg.norm(normal_residual) / jnp.linalg.norm(rhs)
            ),
            data_relative_residual=float(
                jnp.linalg.norm(predicted - data) / jnp.linalg.norm(data)
            ),
            contrast_relative_error=float(
                jnp.linalg.norm(image - delta) / jnp.linalg.norm(delta)
            ),
        )
        row["cg"].append(result)
        checkpoint(f"cg_{iterations}_checked")
        print(json.dumps(result), flush=True)
        if not all(np.isfinite(value) for value in result.values()):
            raise RuntimeError("Nonfinite CG result")
        if result["normal_relative_residual"] < 1e-4:
            break
        del image, predicted, normal_residual
    row["completed"] = True
    row["cg_converged"] = row["cg"][-1]["normal_relative_residual"] < 1e-4
    checkpoint("finished")
    return operator, data, window, delta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=288)
    parser.add_argument("--wavelength-mm", type=float, default=1.25)
    parser.add_argument("--sources", type=int, default=24)
    parser.add_argument("--receivers", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
