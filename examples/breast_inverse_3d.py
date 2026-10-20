"""Fit the whole synthetic breast from exterior full-wave measurements.

The entire pendant phantom, skin included, is reconstructed from plane-wave
measurements starting from water, continuing from a longer wavelength on a
coarse grid to a shorter wavelength on a finer grid.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import minimize

from lucka_phantom_3d import step_down, vascular_phantom
from phantom_inverse_3d import make_forward, objective, sphere

A = 1.25
CENTRE = np.array([0, 0, 0.55 * A])


def breast(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis = -A + (np.arange(n) + 0.5) * (2 * A / n)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)
    normalized = (points - CENTRE) / (0.8 * A)
    radius = np.linalg.norm(normalized / [0.92, 0.82, 1.0], axis=-1)
    window = step_down(radius, 1.08, 0.1) * step_down(
        normalized[..., 2], 0.1, 0.1
    )
    truth = vascular_phantom(points, A, variant="dense")
    return points, window, truth, radius


def scores(truth: np.ndarray, image: np.ndarray, core: np.ndarray) -> dict:
    target = (truth > 0.04) & core
    predicted = (image > 0.04) & core
    tp, fp, fn = (
        int(np.sum(target & predicted)),
        int(np.sum(~target & predicted)),
        int(np.sum(target & ~predicted)),
    )
    return dict(
        relative_image_error=float(
            np.linalg.norm(image - truth) / np.linalg.norm(truth)
        ),
        core_dice=2 * tp / max(1, 2 * tp + fp + fn),
        core_precision=tp / max(1, tp + fp),
        core_recall=tp / max(1, tp + fn),
        core_voxels=int(np.sum(target)),
    )


def run(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    directions = sphere(args.sources)
    receivers = CENTRE - [0, 0, 0.4] + 1.15 * sphere(args.receivers, 0.371)
    start = time.monotonic()
    print(f"DEVICE {jax.devices()[0]}", flush=True)
    theta, shape = None, (1,)
    stages = []
    for waves, n, seconds in zip(args.waves, args.grids, args.seconds):
        stage_start = time.monotonic()
        points, window, truth, radius = breast(n)
        core = (radius < 0.84) & (window > 0.99)
        theta = (
            np.zeros(truth.size)
            if theta is None
            else zoom(theta.reshape(shape), n / shape[0], order=1).ravel()
        )
        shape = truth.shape
        kappa = np.pi * waves / A
        forward = make_forward(
            points, kappa, directions, receivers, args.tolerance
        )
        observed, residuals = forward(jnp.asarray(truth))
        data_residual = float(jnp.max(residuals))
        if (
            not np.isfinite(data_residual)
            or data_residual > 5 * args.tolerance
        ):
            raise RuntimeError(
                f"Synthetic-data solve residual {data_residual}"
            )
        print(
            f"DATA waves={waves}, n={n}, residual={data_residual:.2g}, "
            f"seconds={time.monotonic() - stage_start:.0f}",
            flush=True,
        )
        value_grad = jax.jit(
            jax.value_and_grad(
                lambda x: objective(
                    x,
                    forward,
                    observed,
                    jnp.asarray(window),
                    args.regularization,
                ),
                has_aux=True,
            )
        )
        evaluations = 0
        history = []

        def evaluate(x: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal evaluations
            if time.monotonic() > stage_start + seconds:
                raise TimeoutError("stage budget reached")
            (value, (data_loss, max_residual)), gradient = value_grad(
                jnp.asarray(x)
            )
            loss, residual = float(value), float(max_residual)
            gradient = np.asarray(gradient, dtype=float)
            if (
                not np.isfinite(loss)
                or not np.all(np.isfinite(gradient))
                or residual > 5 * args.tolerance
            ):
                raise RuntimeError(
                    f"Unreliable objective: loss={loss}, residual={residual}"
                )
            evaluations += 1
            history.append(
                dict(
                    evaluation=evaluations,
                    data_loss=float(data_loss),
                    seconds=time.monotonic() - stage_start,
                )
            )
            print(
                f"FIT waves={waves}, eval={evaluations}, data_loss={float(data_loss):.4g}, "
                f"seconds={time.monotonic() - stage_start:.0f}",
                flush=True,
            )
            return loss, gradient

        def save(x: np.ndarray) -> None:
            np.savez_compressed(
                args.out / f"iterate-{waves:g}.npz",
                reconstruction=(0.1 * window * x.reshape(truth.shape)).astype(
                    np.float32
                ),
            )

        try:
            result = minimize(
                evaluate,
                theta,
                jac=True,
                method="L-BFGS-B",
                bounds=[(-1.0, 2.0)] * theta.size,
                callback=save,
                options=dict(
                    maxiter=args.iterations,
                    maxls=12,
                    ftol=1e-10,
                    gtol=1e-7,
                    maxcor=10,
                ),
            )
            theta, message = result.x, str(result.message)
        except TimeoutError as stop:
            theta = np.load(args.out / f"iterate-{waves:g}.npz")[
                "reconstruction"
            ]
            theta = (theta / (0.1 * np.where(window > 0, window, 1))).ravel()
            message = str(stop)
        image = 0.1 * window * theta.reshape(truth.shape)
        prediction, _ = forward(jnp.asarray(image))
        stage = dict(
            waves=waves,
            n=n,
            wavelength_mm=200 / waves,
            evaluations=evaluations,
            optimizer_message=message,
            seconds=time.monotonic() - stage_start,
            relative_data_error=float(
                jnp.linalg.norm(prediction - observed)
                / jnp.linalg.norm(observed)
            ),
            history=history,
            **scores(truth, image, core),
        )
        stages.append(stage)
        np.savez_compressed(
            args.out / f"stage-{waves:g}.npz",
            reconstruction=image.astype(np.float32),
            truth=truth.astype(np.float32),
            core=core,
            axis=points[:, 0, 0, 0],
        )
        (args.out / "results.json").write_text(
            json.dumps(
                dict(
                    parameters=vars(args) | dict(out=str(args.out)),
                    total_seconds=time.monotonic() - start,
                    stages=stages,
                ),
                indent=2,
            )
        )
        print(
            "STAGE "
            + json.dumps({k: v for k, v in stage.items() if k != "history"}),
            flush=True,
        )
        jax.clear_caches()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waves", type=float, nargs="+", default=[20, 40])
    parser.add_argument("--grids", type=int, nargs="+", default=[128, 192])
    parser.add_argument(
        "--seconds", type=float, nargs="+", default=[2400, 5400]
    )
    parser.add_argument("--sources", type=int, default=24)
    parser.add_argument("--receivers", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
