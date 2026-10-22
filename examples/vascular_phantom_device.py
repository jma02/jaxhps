"""Device-resident sampling of the unchanged compact-C4 vascular phantom."""

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.special as special
import numpy as np

from lucka_phantom_3d import vascular_ellipsoids

Ellipsoids = tuple[jax.Array, jax.Array, jax.Array]


def _union(points: jax.Array, shapes: Ellipsoids) -> jax.Array:
    centres, axes, rotations = shapes
    local = (
        jnp.einsum(
            "pfi,fij->pfj", points[:, None, :] - centres[None, :, :], rotations
        )
        / axes
    )
    profile = jnp.maximum(1 - jnp.sum(local**2, axis=-1), 0) ** 5
    return 1 - jnp.prod(1 - profile, axis=-1)


@partial(jax.jit, static_argnames=("batch_size",))
def _sample(
    points: jax.Array,
    a: float,
    fibro: Ellipsoids,
    vessels: Ellipsoids,
    batch_size: int,
) -> jax.Array:
    flat = points.reshape(-1, 3)
    padding = (-len(flat)) % batch_size
    tiles = jnp.pad(flat, ((0, padding), (0, 0))).reshape(-1, batch_size, 3)

    def tile(world: jax.Array) -> jax.Array:
        p = (world - jnp.array([0, 0, 0.55 * a])) / (0.8 * a)
        r = jnp.linalg.norm(p / jnp.array([0.92, 0.82, 1.0]), axis=-1)
        envelope = special.betainc(5.0, 5.0, jnp.clip((0.97 - r) / 0.10, 0, 1))
        envelope *= special.betainc(5.0, 5.0, jnp.clip(-p[:, 2] / 0.10, 0, 1))

        def tissue(_: None) -> jax.Array:
            f = _union(p, fibro)
            v = _union(p, vessels)
            fat = -0.041 + 0.006 * jnp.prod(
                jnp.cos(p * jnp.array([11, 13, 9])), axis=-1
            )
            b = (1 - f) * fat + 0.020 * f
            b = (1 - v) * b + 0.103 * v
            skin = jnp.maximum(1 - ((r - 0.9) / 0.04) ** 2, 0) ** 5
            return ((1 - skin) * b + 0.174 * skin) * envelope

        return jax.lax.cond(
            jnp.any(envelope > 0), tissue, lambda _: jnp.zeros_like(r), None
        )

    return (
        jax.lax.map(tile, tiles)
        .ravel()[: len(flat)]
        .reshape(points.shape[:-1])
    )


def vascular_phantom_device(
    points: np.ndarray | jax.Array,
    a: float = 1.25,
    variant: str = "vascular",
    seed: int = 7,
    batch_size: int = 8192,
) -> jax.Array:
    """Sample the CPU phantom's ellipsoids in bounded point tiles on device.

    Enable JAX x64 for equivalence to the double-precision CPU model.
    The result stays on device for use by ``prepare_volume_solver``.
    """
    points = jnp.asarray(points, dtype=jnp.float64)
    if a <= 0 or points.shape[-1:] != (3,) or batch_size < 1:
        raise ValueError(
            "positive half-width/batch size and (..., 3) points required"
        )
    groups = vascular_ellipsoids(variant, seed)
    fibro, vessels = (
        tuple(
            jnp.asarray(np.stack(values))
            for values in zip(*(shape for group in family for shape in group))
        )
        for family in groups
    )
    return _sample(points, a, fibro, vessels, batch_size)
