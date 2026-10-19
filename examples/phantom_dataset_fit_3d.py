"""Fit held-out vascular crops independently with the full-wave solver.

Each crop saved by ``phantom_dataset_learn_3d.py`` is reconstructed from
its own synthetic measurements, starting from water and continuing from
a longer to a shorter wavelength, so recovery statistics come from many
phantoms rather than one favourable example.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from phantom_dataset_learn_3d import core_scores
from phantom_inverse_3d import make_forward, objective, sphere


def run(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    saved = np.load(args.images)
    truth = saved["truth"][: args.count]
    points, window = saved["points"], saved["window"]
    interior = window > 0.99
    directions = sphere(args.sources)
    receivers = 0.36 * sphere(args.receivers, 0.371)
    forwards = {
        waves: make_forward(
            points, np.pi * waves / 0.2, directions, receivers, args.tolerance
        )
        for waves in args.waves
    }
    start = time.monotonic()
    print(
        f"DEVICE {jax.devices()[0]}; n={points.shape[0]}, crops={len(truth)}, "
        f"sources={args.sources}, receivers={args.receivers}",
        flush=True,
    )
    reconstructions = np.zeros((len(args.waves),) + truth.shape, np.float32)
    crops = []
    for index, target in enumerate(truth):
        theta = np.zeros(target.size)
        stages = []
        for level, waves in enumerate(args.waves):
            forward = forwards[waves]
            observed, residuals = forward(jnp.asarray(target))
            data_residual = float(jnp.max(residuals))
            if (
                not np.isfinite(data_residual)
                or data_residual > 5 * args.tolerance
            ):
                raise RuntimeError(
                    f"Synthetic-data solve residual {data_residual}"
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

            def evaluate(x: np.ndarray) -> tuple[float, np.ndarray]:
                if time.monotonic() > start + args.seconds:
                    raise TimeoutError("GPU work budget reached")
                (value, (_, max_residual)), gradient = value_grad(
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
                return loss, gradient

            result = minimize(
                evaluate,
                theta,
                jac=True,
                method="L-BFGS-B",
                bounds=[(-1.0, 2.0)] * theta.size,
                options=dict(
                    maxiter=args.iterations,
                    maxls=12,
                    ftol=1e-10,
                    gtol=1e-7,
                    maxcor=10,
                ),
            )
            theta = result.x
            image = 0.1 * window * theta.reshape(target.shape)
            reconstructions[level, index] = image
            prediction, _ = forward(jnp.asarray(image))
            stages.append(
                dict(
                    waves=waves,
                    iterations=int(result.nit),
                    relative_data_error=float(
                        jnp.linalg.norm(prediction - observed)
                        / jnp.linalg.norm(observed)
                    ),
                    **core_scores(target[None], image[None], interior),
                )
            )
        crops.append(dict(index=index, stages=stages))
        final = stages[-1]
        print(
            f"CROP {index}: image error {final['relative_image_error']:.3f}, "
            f"core Dice {final['core_dice']:.3f}, "
            f"seconds={time.monotonic() - start:.0f}",
            flush=True,
        )
        np.savez_compressed(
            args.out / "reconstructions.npz",
            reconstructions=reconstructions[:, : index + 1],
            truth=truth[: index + 1],
            window=window,
            points=points,
            waves=np.asarray(args.waves),
        )
    summary = {
        f"{waves:g}": core_scores(truth, reconstructions[level], interior)
        | dict(
            relative_data_error=float(
                np.mean(
                    [c["stages"][level]["relative_data_error"] for c in crops]
                )
            ),
            core_dice_min=float(
                min(c["stages"][level]["core_dice"] for c in crops)
            ),
            core_dice_median=float(
                np.median([c["stages"][level]["core_dice"] for c in crops])
            ),
        )
        for level, waves in enumerate(args.waves)
    }
    (args.out / "results.json").write_text(
        json.dumps(
            dict(
                parameters=vars(args)
                | dict(out=str(args.out), images=str(args.images)),
                total_seconds=time.monotonic() - start,
                summary=summary,
                crops=crops,
            ),
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--sources", type=int, default=24)
    parser.add_argument("--receivers", type=int, default=256)
    parser.add_argument("--waves", type=float, nargs="+", default=[5, 10])
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--seconds", type=float, default=3000)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
