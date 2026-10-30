"""Learn a background-linearized Born correction on a simple breast phantom.

A hemispherical breast with skin and tissue (the py-helm geometry) contains a
few random spherical inclusions per sample. Plane waves arrive from, and
scattered fields are recorded on, a spherical cap of sensors surrounding the
breast. The known sphere-free breast is the reference model: measurements are
linearized about it, a damped least-squares Born image of the inclusions is
formed, and a small 3D network learns to correct that image. Held-out samples
measure recovery; a full-wave fit from the same reference on a few held-out
samples gives the classical comparison.
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from breast_phantom_3d import breast_n_of_x, fibonacci_cap_points
from phantom_dataset_learn_3d import init_network, network
from phantom_inverse_3d import (
    BackgroundBorn,
    make_background_born,
    make_forward,
)
from tensor_train_born import SlabBorn
from compressed_born import BlockBorn
from spectral_born import SpectralBorn

A = 1.2
KVAL = 25.0
THRESHOLD = 0.02
Fields = tuple[jax.Array, jax.Array]


@partial(jax.jit, static_argnames=("iterations",))
def reconstruct_born(
    operator: BackgroundBorn | SlabBorn | BlockBorn | SpectralBorn,
    data: jax.Array,
    window: jax.Array,
    iterations: int,
    damping: float,
) -> jax.Array:
    scale = jnp.sum(jnp.abs(data) ** 2)
    rhs = window * operator.adjoint(data)

    def step(_, state):
        x, r, p, rr = state
        q = window * operator.adjoint(
            operator((window * p).astype(window.dtype))
        )
        q = q + damping * scale * p
        alpha = rr / jnp.sum(p * q)
        x = x + alpha * p
        r = r - alpha * q
        rr_new = jnp.sum(r * r)
        p = r + (rr_new / rr) * p
        return x, r, p, rr_new

    state = (jnp.zeros_like(rhs), rhs, rhs, jnp.sum(rhs * rhs))
    x, _, _, _ = jax.lax.fori_loop(0, iterations, step, state)
    return window * x


def sample_spheres(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(rng.integers(2, 5))
    radii = rng.uniform(0.08, 0.15, count)
    centres = []
    while len(centres) < count:
        centre = rng.uniform([-0.75, 0.15, -0.75], [0.75, 0.75, 0.75])
        if np.linalg.norm(centre) < 0.78:
            centres.append(centre)
    mvals = (1524.0 / 1485.0) ** 2 * rng.uniform(1.04, 1.16, count)
    return np.array(centres), radii, mvals


def run(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    n = args.n
    axis = -A + (np.arange(n) + 0.5) * (2 * A / n)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)
    radius = np.linalg.norm(points, axis=-1)
    window = (
        0.5
        * (1 + np.tanh(KVAL * points[..., 1]))
        * 0.5
        * (1 + np.tanh(KVAL * (1 - 1 / 30 - radius)))
    )
    interior = window > 0.99
    background = 1 - breast_n_of_x(
        points, np.zeros((0, 3)), np.zeros(0), np.zeros(0), kval=KVAL
    )
    rng = np.random.default_rng(args.seed)
    truth, descriptions = [], []
    for _ in range(args.train + args.test):
        centres, radii, mvals = sample_spheres(rng)
        truth.append(
            1 - breast_n_of_x(points, centres, radii, mvals, kval=KVAL)
        )
        descriptions.append(
            dict(
                centres=centres.tolist(),
                radii=radii.tolist(),
                mvals=mvals.tolist(),
            )
        )
    truth = np.stack(truth).astype(np.float32)
    delta_truth = truth - background.astype(np.float32)

    wavelength = args.wavelength_mm / 60.0
    kappa = 2 * np.pi / wavelength
    sensors = fibonacci_cap_points(args.receivers, 1 + wavelength, 0.1)
    emitters = fibonacci_cap_points(args.sources, 1 + wavelength, 0.1)
    directions = -emitters / np.linalg.norm(emitters, axis=-1, keepdims=True)
    forward = make_forward(points, kappa, directions, sensors, args.tolerance)
    print(
        f"DEVICE {jax.devices()[0]}; n={n}, samples={len(truth)}, "
        f"sources={args.sources}, receivers={args.receivers}, "
        f"wavelength={wavelength:.4f}, h={2 * A / n:.4f}",
        flush=True,
    )

    background_dev = jnp.asarray(background, dtype=jnp.float32)
    window_dev = jnp.asarray(window, dtype=jnp.float32)
    reference, residuals = forward(background_dev)
    residual = float(jnp.max(residuals))
    if not np.isfinite(residual) or residual > 5 * args.tolerance:
        raise RuntimeError(f"Background solve residual {residual}")
    print(
        f"BACKGROUND residual={residual:.2g}, "
        f"seconds={time.monotonic() - start:.0f}",
        flush=True,
    )
    operator = make_background_born(
        points, kappa, directions, sensors, background_dev, args.tolerance
    )
    print(
        f"BORN fields cached, seconds={time.monotonic() - start:.1f}",
        flush=True,
    )

    def born(data: jax.Array) -> jax.Array:
        return reconstruct_born(
            operator, data, window_dev, args.born_iterations, args.damping
        )

    observed, images = [], []
    for index, image in enumerate(truth):
        sample_start = time.monotonic()
        data, residuals = forward(jnp.asarray(image))
        residual = float(jnp.max(residuals))
        if not np.isfinite(residual) or residual > 5 * args.tolerance:
            raise RuntimeError(f"Synthetic-data solve residual {residual}")
        if args.noise > 0:
            key = jax.random.PRNGKey(args.seed + index)
            level = args.noise * jnp.sqrt(jnp.mean(jnp.abs(data) ** 2))
            data = data + level * jax.random.normal(
                key, data.shape, dtype=data.dtype
            )
        observed.append(data)
        forward_seconds = time.monotonic() - sample_start
        born_start = time.monotonic()
        images.append(born(data - reference).block_until_ready())
        born_seconds = time.monotonic() - born_start
        if (index + 1) % 20 == 0 or index + 1 == len(truth):
            np.savez_compressed(
                args.out / "dataset-partial.npz",
                truth=delta_truth[: index + 1],
                born=np.asarray(images),
                observed=np.asarray(observed),
                background=background.astype(np.float32),
                window=window.astype(np.float32),
            )
            (args.out / "dataset-partial.npz").replace(
                args.out / "dataset.npz"
            )
            print(f"DATA saved={index + 1}", flush=True)
        if index % 20 == 0:
            print(
                f"DATA sample={index}, residual={residual:.2g}, "
                f"forward={forward_seconds:.2f}s, born={born_seconds:.2f}s, "
                f"seconds={time.monotonic() - start:.1f}",
                flush=True,
            )
    observed = jnp.stack(observed)
    images = jnp.stack(images).astype(jnp.float32)
    if not np.isfinite(np.asarray(images)).all():
        raise RuntimeError("Born reconstructions are not finite")
    target = jnp.asarray(delta_truth)
    train = slice(0, args.train)
    test = slice(args.train, args.train + args.test)
    data_seconds = time.monotonic() - start
    print(f"DATA complete, seconds={data_seconds:.1f}", flush=True)

    params = jax.tree.map(
        lambda x: x.astype(jnp.float32),
        init_network(jax.random.PRNGKey(args.seed), args.width, args.depth),
    )
    m = [(jnp.zeros_like(k), jnp.zeros_like(b)) for k, b in params]
    v = [(jnp.zeros_like(k), jnp.zeros_like(b)) for k, b in params]

    def loss(p, x, y):
        return jnp.sum((network(p, x, window_dev) - y) ** 2) / jnp.sum(y**2)

    @jax.jit
    def step(p, m, v, x, y, t):
        value, grads = jax.value_and_grad(loss)(p, x, y)
        m = jax.tree.map(lambda a, g: 0.9 * a + 0.1 * g, m, grads)
        v = jax.tree.map(lambda a, g: 0.999 * a + 0.001 * g * g, v, grads)
        rate = args.learning_rate * jnp.sqrt(1 - 0.999**t) / (1 - 0.9**t)
        p = jax.tree.map(
            lambda a, mi, vi: a - rate * mi / (jnp.sqrt(vi) + 1e-8), p, m, v
        )
        return p, m, v, value

    order_rng = np.random.default_rng(args.seed)
    history = []
    deadline = start + args.seconds
    t = 0
    for epoch in range(args.epochs):
        if time.monotonic() > deadline:
            break
        order = order_rng.permutation(args.train)
        losses = []
        for batch in range(0, args.train, args.batch):
            if time.monotonic() > deadline:
                break
            idx = jnp.asarray(order[batch : batch + args.batch])
            t += 1
            params, m, v, value = step(
                params, m, v, images[idx], target[idx], t
            )
            losses.append(float(value))
            if not np.isfinite(losses[-1]):
                raise RuntimeError("Training loss is not finite")
        if not losses:
            break
        value = float(np.mean(losses))
        history.append(
            dict(
                epoch=epoch,
                loss=value,
                batches=len(losses),
                seconds=time.monotonic() - start,
            )
        )
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"TRAIN epoch={epoch}, loss={value:.4g}, "
                f"seconds={time.monotonic() - start:.1f}",
                flush=True,
            )
        if time.monotonic() > deadline:
            break

    if not history:
        raise RuntimeError("Time budget exhausted before network training")
    predict = jax.jit(lambda x: network(params, x, window_dev))
    corrected_np = np.concatenate(
        [
            np.asarray(predict(images[i : i + args.batch]))
            for i in range(0, len(images), args.batch)
        ]
    )
    if not np.isfinite(corrected_np).all():
        raise RuntimeError("Network reconstructions are not finite")
    born_np = np.asarray(images)
    (args.out / "training.json").write_text(json.dumps(history, indent=1))
    np.savez_compressed(
        args.out / "network.npz",
        **{
            f"layer_{i}_{name}": np.asarray(value)
            for i, layer in enumerate(params)
            for name, value in zip(("kernel", "bias"), layer)
        },
    )
    np.savez_compressed(
        args.out / "images.npz",
        background=background.astype(np.float32),
        window=window.astype(np.float32),
        truth=delta_truth[test],
        born=born_np[test],
        corrected=corrected_np[test],
        fits=np.zeros((0,) + window.shape),
    )
    print(f"TRAIN complete, epochs={len(history)}", flush=True)

    def measurement_error(delta: jax.Array, index: int) -> float:
        data, _ = forward(background_dev + delta)
        return float(
            jnp.linalg.norm(data - observed[index])
            / jnp.linalg.norm(observed[index])
        )

    fits, fit_records = [], []
    for index in range(args.train, args.train + args.fits):
        fit_start = time.monotonic()
        value_grad = jax.jit(
            jax.value_and_grad(
                lambda theta: fit_objective(
                    theta,
                    forward,
                    observed[index],
                    background_dev,
                    window_dev,
                    args.regularization,
                )
            )
        )

        def evaluate(x: np.ndarray) -> tuple[float, np.ndarray]:
            value, gradient = value_grad(jnp.asarray(x, dtype=jnp.float32))
            return float(value), np.asarray(gradient, dtype=float)

        result = minimize(
            evaluate,
            np.zeros(truth[0].size),
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=args.iterations, ftol=0.0, gtol=0.0),
        )
        delta = 0.1 * window * result.x.reshape(window.shape)
        fits.append(delta.astype(np.float32))
        fit_records.append(
            dict(
                index=index,
                evaluations=int(result.nfev),
                optimizer_message=str(result.message),
                seconds=time.monotonic() - fit_start,
                measurement_error=measurement_error(jnp.asarray(delta), index),
                **sphere_scores(
                    delta_truth[index : index + 1], fits[-1][None], interior
                ),
            )
        )
        print(f"FIT {json.dumps(fit_records[-1])}", flush=True)

    results = dict(
        parameters=vars(args) | dict(out=str(args.out)),
        data_seconds=data_seconds,
        total_seconds=time.monotonic() - start,
        epochs_completed=len(history),
        train=dict(
            born=sphere_scores(delta_truth[train], born_np[train], interior),
            corrected=sphere_scores(
                delta_truth[train], corrected_np[train], interior
            ),
        ),
        test=dict(
            born=sphere_scores(delta_truth[test], born_np[test], interior),
            corrected=sphere_scores(
                delta_truth[test], corrected_np[test], interior
            ),
            born_measurement_error=float(
                np.mean(
                    [
                        measurement_error(images[i], i)
                        for i in range(args.train, args.train + args.test)
                    ]
                )
            ),
            corrected_measurement_error=float(
                np.mean(
                    [
                        measurement_error(jnp.asarray(corrected_np[i]), i)
                        for i in range(args.train, args.train + args.test)
                    ]
                )
            ),
        ),
        fits=fit_records,
        history=history,
        samples=descriptions,
    )
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(
        args.out / "images.npz",
        background=background.astype(np.float32),
        window=window.astype(np.float32),
        truth=delta_truth[test],
        born=born_np[test],
        corrected=corrected_np[test],
        fits=np.stack(fits) if fits else np.zeros((0,) + window.shape),
    )
    print(f"TEST {json.dumps(results['test'])}", flush=True)


def fit_objective(
    theta, forward, observed, background, window, regularization
):
    delta = 0.1 * window * theta.reshape(window.shape)
    predicted, _ = forward(background + delta)
    data_loss = jnp.sum(jnp.abs(predicted - observed) ** 2) / jnp.sum(
        jnp.abs(observed) ** 2
    )
    penalty = sum(
        jnp.mean(jnp.diff(delta / 0.1, axis=i) ** 2) for i in range(3)
    )
    return data_loss + regularization * penalty


def sphere_scores(
    truth: np.ndarray, image: np.ndarray, interior: np.ndarray
) -> dict:
    target = (truth < -THRESHOLD) & interior
    predicted = (image < -THRESHOLD) & interior
    tp = np.sum(target & predicted, axis=(1, 2, 3))
    fp = np.sum(~target & predicted, axis=(1, 2, 3))
    fn = np.sum(target & ~predicted, axis=(1, 2, 3))
    error = np.linalg.norm(
        (image - truth).reshape(len(truth), -1), axis=1
    ) / np.linalg.norm(truth.reshape(len(truth), -1), axis=1)
    return dict(
        relative_image_error=float(np.mean(error)),
        sphere_dice=float(np.mean(2 * tp / np.maximum(1, 2 * tp + fp + fn))),
        sphere_precision=float(np.mean(tp / np.maximum(1, tp + fp))),
        sphere_recall=float(np.mean(tp / np.maximum(1, tp + fn))),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--wavelength-mm", type=float, default=10.0)
    parser.add_argument("--train", type=int, default=160)
    parser.add_argument("--test", type=int, default=40)
    parser.add_argument("--fits", type=int, default=4)
    parser.add_argument("--sources", type=int, default=24)
    parser.add_argument("--receivers", type=int, default=256)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--born-iterations", type=int, default=8)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--seconds", type=float, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
