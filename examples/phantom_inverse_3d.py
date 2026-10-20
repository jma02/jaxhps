"""Single-phantom full-wave inversion of a tapered 16 mm vascular crop.

Fit voxel contrasts to exterior pressure measurements, starting from water.
The image is used only to synthesize data and evaluate the reconstruction.
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres
from scipy.optimize import minimize

from fft_volume_3d import green_kernel
from lucka_phantom_3d import step_down, vascular_phantom


def sphere(count: int, phase: float = 0.0) -> np.ndarray:
    z = 1 - 2 * (np.arange(count) + 0.5) / count
    phi = np.arange(count) * np.pi * (3 - np.sqrt(5)) + phase
    return np.column_stack(
        [np.sqrt(1 - z**2) * np.cos(phi), np.sqrt(1 - z**2) * np.sin(phi), z]
    )


def crop(n: int, a: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = -a + (np.arange(n) + 0.5) * (2 * a / n)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)
    window = np.prod(step_down(np.abs(points), a, 0.2 * a), axis=-1)
    truth = vascular_phantom(points + [0.45, 0, 0.15], variant="dense")
    return points, window, window * truth


def make_forward(
    points: np.ndarray,
    kappa: float,
    directions: np.ndarray,
    receivers: np.ndarray,
    tol: float = 1e-6,
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    n = points.shape[0]
    h = float(points[1, 0, 0, 0] - points[0, 0, 0, 0])
    a = n * h / 2
    dtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
    kernel = green_kernel(n, a, kappa).astype(dtype)
    xyz = jnp.asarray(points.reshape(-1, 3))
    directions = jnp.asarray(directions)
    receivers = jnp.asarray(receivers)

    def receiver_row(target: jax.Array) -> jax.Array:
        distance = jnp.linalg.norm(target - xyz, axis=-1)
        return (
            -(kappa**2)
            * h**3
            * jnp.exp(1j * kappa * distance)
            / (4 * jnp.pi * distance)
        ).astype(dtype)

    @jax.jit
    def forward(b: jax.Array) -> tuple[jax.Array, jax.Array]:
        def operator(u: jax.Array) -> jax.Array:
            charge = b * u.reshape((n, n, n))
            padded = jnp.pad(charge, ((0, n),) * 3)
            potential = jnp.fft.ifftn(jnp.fft.fftn(padded) * kernel)[
                :n, :n, :n
            ]
            return u + kappa**2 * potential.ravel()

        def solve(direction: jax.Array) -> tuple[jax.Array, jax.Array]:
            inc = jnp.exp(1j * kappa * (xyz @ direction)).astype(dtype)

            def krylov(matvec, rhs):
                solution, _ = gmres(
                    matvec,
                    rhs,
                    tol=tol,
                    restart=20,
                    maxiter=10,
                    solve_method="incremental",
                )
                return solution

            total = jax.lax.custom_linear_solve(
                operator, inc, solve=krylov, transpose_solve=krylov
            )
            residual = jnp.linalg.norm(
                operator(total) - inc
            ) / jnp.linalg.norm(inc)
            charge = b.ravel() * total
            measure = jax.checkpoint(
                lambda target: jnp.sum(receiver_row(target) * charge)
            )
            return jax.lax.map(measure, receivers), residual

        return jax.lax.map(solve, directions)

    return forward


def objective(
    theta: jax.Array,
    forward: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    observed: jax.Array,
    window: jax.Array,
    regularization: float,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    b = 0.1 * window * theta.reshape(window.shape)
    predicted, residuals = forward(b)
    data_loss = jnp.sum(jnp.abs(predicted - observed) ** 2) / jnp.sum(
        jnp.abs(observed) ** 2
    )
    penalty = sum(jnp.mean(jnp.diff(b / 0.1, axis=i) ** 2) for i in range(3))
    return data_loss + regularization * penalty, (
        data_loss,
        jnp.max(residuals),
    )


def run(args: argparse.Namespace) -> None:
    jax.config.update("jax_enable_x64", args.double)
    args.out.mkdir(parents=True, exist_ok=True)
    points, window, truth = crop(args.n)
    directions = sphere(args.sources)
    receivers = 0.36 * sphere(args.receivers, 0.371)
    theta = np.zeros(truth.size)
    history = []
    stages = []
    start = time.monotonic()
    deadline = start + args.seconds
    device = jax.devices()[0]
    print(
        f"DEVICE {device}; n={args.n}, sources={args.sources}, receivers={args.receivers}",
        flush=True,
    )

    for waves in args.waves:
        stage_start = time.monotonic()
        kappa = np.pi * waves / 0.2
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
            f"DATA waves={waves}, max residual={data_residual:.3g}", flush=True
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

        def evaluate(x: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal evaluations
            if time.monotonic() > deadline:
                raise TimeoutError("GPU work budget reached")
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
                    waves=waves,
                    evaluation=evaluations,
                    loss=loss,
                    data_loss=float(data_loss),
                    residual=residual,
                    seconds=time.monotonic() - start,
                )
            )
            if evaluations == 1 or evaluations % 10 == 0:
                print(
                    f"FIT waves={waves}, eval={evaluations}, loss={loss:.6g}, seconds={time.monotonic() - start:.1f}",
                    flush=True,
                )
            return loss, gradient

        def save_iterate(x: np.ndarray) -> None:
            np.savez_compressed(
                args.out / "iterate.npz",
                reconstruction=0.1 * window * x.reshape(truth.shape),
            )

        result = minimize(
            evaluate,
            theta,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-1.0, 2.0)] * theta.size,
            callback=save_iterate,
            options=dict(
                maxiter=args.iterations,
                maxls=12,
                ftol=1e-10,
                gtol=1e-7,
                maxcor=10,
            ),
        )
        theta = result.x
        reconstruction = 0.1 * window * theta.reshape(truth.shape)
        prediction, residuals = forward(jnp.asarray(reconstruction))
        stage = dict(
            waves=waves,
            wavelength_mm=16 / waves,
            iterations=int(result.nit),
            evaluations=int(result.nfev),
            optimizer_message=str(result.message),
            seconds=time.monotonic() - stage_start,
            relative_data_error=float(
                jnp.linalg.norm(prediction - observed)
                / jnp.linalg.norm(observed)
            ),
            relative_image_error=float(
                np.linalg.norm(reconstruction - truth) / np.linalg.norm(truth)
            ),
            max_forward_residual=float(jnp.max(residuals)),
            max_data_residual=data_residual,
        )
        stages.append(stage)
        np.savez_compressed(
            args.out / f"stage-{waves:g}.npz",
            truth=truth,
            reconstruction=reconstruction,
            window=window,
            points=points,
            observed=np.asarray(observed),
            prediction=np.asarray(prediction),
            directions=directions,
            receivers=receivers,
        )
        (args.out / "results.json").write_text(
            json.dumps(
                dict(
                    parameters={**vars(args), "out": str(args.out)},
                    device=str(device),
                    stages=stages,
                    history=history,
                ),
                indent=2,
            )
        )
        print(f"STAGE {json.dumps(stage)}", flush=True)
        jax.clear_caches()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--sources", type=int, default=48)
    parser.add_argument("--receivers", type=int, default=512)
    parser.add_argument("--waves", type=float, nargs="+", default=[4, 8, 12])
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--regularization", type=float, default=0.001)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--seconds", type=float, default=3000)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
