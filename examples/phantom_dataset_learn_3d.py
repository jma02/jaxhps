"""Learn a Born-approximation correction on a small set of vascular crops.

Random 16 mm crops of the synthetic breast phantom are measured with the
full-wave solver. A linear Born reconstruction is computed from each
measurement set and a small 3D network learns to correct it. Held-out
crops measure recovery that a single-phantom fit cannot.
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lucka_phantom_3d import step_down, vascular_phantom
from phantom_inverse_3d import make_forward, sphere

Params = list[tuple[jax.Array, jax.Array]]


def sample_crops(
    n: int, count: int, seed: int, minimum_core: int, a: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    axis = -a + (np.arange(n) + 0.5) * (2 * a / n)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)
    window = np.prod(step_down(np.abs(points), a, 0.2 * a), axis=-1)
    interior = window > 0.99
    rng = np.random.default_rng(seed)
    images, descriptions = [], []
    while len(images) < count:
        centre = rng.uniform([-0.45, -0.38, -0.62], [0.45, 0.38, -0.2])
        radius = np.linalg.norm(
            (points[interior] + centre) / [0.92, 0.82, 1.0], axis=-1
        )
        if radius.max() > 0.82:
            continue
        phantom_seed = int(rng.integers(1, 10_000))
        shift = centre + [0, 0, 0.6875]
        truth = window * vascular_phantom(
            points + shift, variant="dense", seed=phantom_seed
        )
        core = int(np.sum((truth > 0.04) & interior))
        if core < minimum_core:
            continue
        images.append(truth.astype(np.float32))
        descriptions.append(
            dict(seed=phantom_seed, centre=centre.tolist(), core_voxels=core)
        )
    return points, window, np.stack(images), descriptions


def make_born(
    points: np.ndarray,
    kappa: float,
    directions: np.ndarray,
    receivers: np.ndarray,
    iterations: int,
    damping: float,
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    n = points.shape[0]
    h = float(points[1, 0, 0, 0] - points[0, 0, 0, 0])
    xyz = jnp.asarray(points.reshape(-1, 3))
    incident = jnp.exp(1j * kappa * (jnp.asarray(directions) @ xyz.T))

    def receiver_row(target: jax.Array) -> jax.Array:
        distance = jnp.linalg.norm(target - xyz, axis=-1)
        return (
            -(kappa**2)
            * h**3
            * jnp.exp(1j * kappa * distance)
            / (4 * jnp.pi * distance)
        )

    receiver_map = jax.lax.map(receiver_row, jnp.asarray(receivers))

    def linear(b: jax.Array) -> jax.Array:
        return (incident * b.ravel()) @ receiver_map.T

    def adjoint(data: jax.Array) -> jax.Array:
        return jnp.real(
            jnp.sum(jnp.conj(incident) * (data @ jnp.conj(receiver_map)), 0)
        ).reshape((n, n, n))

    @jax.jit
    def reconstruct(observed: jax.Array) -> jax.Array:
        scale = jnp.sum(jnp.abs(adjoint(linear(jnp.ones((n, n, n))))))
        mu = damping * scale / n**3

        def normal(b: jax.Array) -> jax.Array:
            return adjoint(linear(b)) + mu * b

        rhs = adjoint(observed)
        b = jnp.zeros((n, n, n))
        r = rhs
        p = r
        rr = jnp.sum(r * r)
        for _ in range(iterations):
            q = normal(p)
            alpha = rr / jnp.sum(p * q)
            b = b + alpha * p
            r = r - alpha * q
            rr_new = jnp.sum(r * r)
            p = r + (rr_new / rr) * p
            rr = rr_new
        return b

    return linear, reconstruct


def init_network(key: jax.Array, width: int, depth: int) -> Params:
    shapes = [(2, width)] + [(width, width)] * (depth - 2) + [(width, 1)]
    params = []
    for layer_key, (cin, cout) in zip(jax.random.split(key, depth), shapes):
        scale = jnp.sqrt(2.0 / (27 * cin)) if cout > 1 else 0.0
        params.append(
            (
                scale * jax.random.normal(layer_key, (3, 3, 3, cin, cout)),
                jnp.zeros(cout),
            )
        )
    return params


def network(params: Params, born: jax.Array, window: jax.Array) -> jax.Array:
    x = jnp.stack([born, jnp.broadcast_to(window, born.shape)], -1)
    for i, (kernel, bias) in enumerate(params):
        x = (
            jax.lax.conv_general_dilated(
                x,
                kernel,
                (1, 1, 1),
                "SAME",
                dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
            )
            + bias
        )
        if i < len(params) - 1:
            x = jax.nn.gelu(x)
    return window * (born + x[..., 0])


def core_scores(
    truth: np.ndarray, image: np.ndarray, interior: np.ndarray
) -> dict:
    target = (truth > 0.04) & interior
    predicted = (image > 0.04) & interior
    tp = np.sum(target & predicted, axis=(1, 2, 3))
    fp = np.sum(~target & predicted, axis=(1, 2, 3))
    fn = np.sum(target & ~predicted, axis=(1, 2, 3))
    error = np.linalg.norm(
        (image - truth).reshape(len(truth), -1), axis=1
    ) / np.linalg.norm(truth.reshape(len(truth), -1), axis=1)
    return dict(
        relative_image_error=float(np.mean(error)),
        core_dice=float(np.mean(2 * tp / np.maximum(1, 2 * tp + fp + fn))),
        core_precision=float(np.mean(tp / np.maximum(1, tp + fp))),
        core_recall=float(np.mean(tp / np.maximum(1, tp + fn))),
    )


def run(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    points, window, truth, descriptions = sample_crops(
        args.n, args.train + args.test, args.seed, args.minimum_core
    )
    interior = window > 0.99
    kappa = np.pi * args.waves / 0.2
    directions = sphere(args.sources)
    receivers = 0.36 * sphere(args.receivers, 0.371)
    forward = make_forward(
        points, kappa, directions, receivers, args.tolerance
    )
    _, reconstruct = make_born(
        points,
        kappa,
        directions,
        receivers,
        args.born_iterations,
        args.damping,
    )
    print(
        f"DEVICE {jax.devices()[0]}; n={args.n}, samples={len(truth)}, "
        f"sources={args.sources}, receivers={args.receivers}",
        flush=True,
    )
    observed, born = [], []
    for index, image in enumerate(truth):
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
        born.append(reconstruct(data))
        if index % 20 == 0:
            print(
                f"DATA sample={index}, residual={residual:.2g}, "
                f"seconds={time.monotonic() - start:.1f}",
                flush=True,
            )
    observed = jnp.stack(observed)
    born = jnp.stack(born)
    truth_dev = jnp.asarray(truth)
    window_dev = jnp.asarray(window, dtype=jnp.float32)
    train = slice(0, args.train)
    test = slice(args.train, args.train + args.test)
    data_seconds = time.monotonic() - start

    params = init_network(
        jax.random.PRNGKey(args.seed), args.width, args.depth
    )
    state = [(jnp.zeros_like(k), jnp.zeros_like(b)) for k, b in params]
    moments = (
        state,
        [(jnp.zeros_like(k), jnp.zeros_like(b)) for k, b in params],
    )

    def loss(p: Params, x: jax.Array, y: jax.Array) -> jax.Array:
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

    rng = np.random.default_rng(args.seed)
    m, v = moments
    history = []
    deadline = start + args.seconds
    t = 0
    for epoch in range(args.epochs):
        order = rng.permutation(args.train)
        for batch in range(0, args.train, args.batch):
            if time.monotonic() > deadline:
                break
            idx = jnp.asarray(order[batch : batch + args.batch])
            t += 1
            params, m, v, value = step(
                params, m, v, born[idx], truth_dev[idx], t
            )
        value = float(value)
        if not np.isfinite(value):
            raise RuntimeError("Training loss is not finite")
        history.append(
            dict(epoch=epoch, loss=value, seconds=time.monotonic() - start)
        )
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"TRAIN epoch={epoch}, loss={value:.4g}, "
                f"seconds={time.monotonic() - start:.1f}",
                flush=True,
            )
        if time.monotonic() > deadline:
            break

    predicted = jax.jit(lambda x: network(params, x, window_dev))(born)
    predicted_np = np.asarray(predicted)
    born_np = np.asarray(born)
    results = dict(
        parameters=vars(args) | dict(out=str(args.out)),
        data_seconds=data_seconds,
        total_seconds=time.monotonic() - start,
        epochs_completed=len(history),
        train=dict(
            born=core_scores(truth[train], born_np[train], interior),
            corrected=core_scores(truth[train], predicted_np[train], interior),
        ),
        test=dict(
            born=core_scores(truth[test], born_np[test], interior),
            corrected=core_scores(truth[test], predicted_np[test], interior),
        ),
        history=history,
        samples=descriptions,
    )
    misfits = []
    for index in range(args.train, args.train + args.test):
        data, _ = forward(predicted[index])
        misfits.append(
            float(
                jnp.linalg.norm(data - observed[index])
                / jnp.linalg.norm(observed[index])
            )
        )
    results["test"]["corrected_measurement_error"] = float(np.mean(misfits))
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    np.savez_compressed(
        args.out / "test-images.npz",
        truth=truth[test],
        born=born_np[test],
        corrected=predicted_np[test],
        window=window,
        points=points,
    )
    print(json.dumps({k: results[k] for k in ("train", "test")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--train", type=int, default=160)
    parser.add_argument("--test", type=int, default=40)
    parser.add_argument("--sources", type=int, default=24)
    parser.add_argument("--receivers", type=int, default=256)
    parser.add_argument("--waves", type=float, default=10)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--born-iterations", type=int, default=15)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--minimum-core", type=int, default=150)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--seconds", type=float, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
