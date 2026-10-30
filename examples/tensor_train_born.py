"""Three-core TT-SVD of Born fields with slab-wise decompression.

Only a slab of the source/receiver fields is expanded during applications.
Compression of an existing cache still requires that dense cache at setup.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from phantom_inverse_3d import BackgroundBorn


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["first", "middle", "last"],
    meta_fields=[],
)
@dataclass(frozen=True)
class TensorTrainFields:
    first: jax.Array
    middle: jax.Array
    last: jax.Array

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.first.shape[1], self.middle.shape[2], self.last.shape[2]

    @property
    def nbytes(self) -> int:
        return self.first.nbytes + self.middle.nbytes + self.last.nbytes

    def slab(self, index: jax.Array, size: int) -> jax.Array:
        padding = (-self.shape[0]) % size
        first = jnp.pad(self.first, ((0, 0), (0, padding), (0, 0)))
        first = jax.lax.dynamic_slice_in_dim(first, index * size, size, 1)
        middle = jnp.einsum(
            "fxa,fayb->fxyb",
            first,
            self.middle,
            precision=jax.lax.Precision.HIGHEST,
        )
        return jnp.einsum(
            "fxyb,fbz->fxyz",
            middle,
            self.last,
            precision=jax.lax.Precision.HIGHEST,
        ).reshape(self.first.shape[0], -1)


@partial(jax.jit, static_argnames=("shape", "rank"))
def compress_fields(
    fields: jax.Array, shape: tuple[int, int, int], rank: int
) -> TensorTrainFields:
    """Deterministic complex TT-SVD, with each internal rank capped at rank."""
    if rank < 1:
        raise ValueError("rank must be positive")
    nx, ny, nz = shape
    r1 = min(rank, nx, ny * nz)
    r2 = min(rank, r1 * ny, nz)

    def compress(field: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        left, singular, right = jnp.linalg.svd(
            field.reshape(nx, ny * nz), full_matrices=False
        )
        first = left[:, :r1]
        remainder = (singular[:r1, None] * right[:r1]).reshape(r1 * ny, nz)
        left, singular, right = jnp.linalg.svd(remainder, full_matrices=False)
        return (
            first,
            left[:, :r2].reshape(r1, ny, r2),
            singular[:r2, None] * right[:r2],
        )

    return TensorTrainFields(*jax.lax.map(compress, fields))


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["sources", "reciprocal_fields"],
    meta_fields=["shape", "dtype", "slab_size"],
)
@dataclass(frozen=True)
class TensorTrainBorn:
    sources: TensorTrainFields
    reciprocal_fields: TensorTrainFields
    shape: tuple[int, int, int]
    dtype: np.dtype
    slab_size: int = 4

    def __post_init__(self) -> None:
        if self.slab_size < 1:
            raise ValueError("slab_size must be positive")

    @property
    def nbytes(self) -> int:
        return self.sources.nbytes + self.reciprocal_fields.nbytes

    @jax.jit
    def __call__(self, delta: jax.Array) -> jax.Array:
        nx, _, _ = self.shape
        padding = (-nx) % self.slab_size
        padded = jnp.pad(delta, ((0, padding), (0, 0), (0, 0)))

        @jax.checkpoint
        def step(index: jax.Array, data: jax.Array) -> jax.Array:
            sources = self.sources.slab(index, self.slab_size)
            receivers = self.reciprocal_fields.slab(index, self.slab_size)
            values = jax.lax.dynamic_slice_in_dim(
                padded, index * self.slab_size, self.slab_size, 0
            )
            return data + jnp.matmul(
                sources * values.ravel(),
                receivers.T,
                precision=jax.lax.Precision.HIGHEST,
            )

        data = jnp.zeros(
            (
                self.sources.first.shape[0],
                self.reciprocal_fields.first.shape[0],
            ),
            dtype=self.sources.first.dtype,
        )
        return jax.lax.fori_loop(
            0, (nx + padding) // self.slab_size, step, data
        )

    @jax.jit
    def adjoint(self, data: jax.Array) -> jax.Array:
        nx, ny, nz = self.shape
        padding = (-nx) % self.slab_size

        @jax.checkpoint
        def step(index: jax.Array, image: jax.Array) -> jax.Array:
            sources = self.sources.slab(index, self.slab_size)
            receivers = self.reciprocal_fields.slab(index, self.slab_size)
            weighted = jnp.matmul(
                jnp.conj(data),
                receivers,
                precision=jax.lax.Precision.HIGHEST,
            )
            values = jnp.real(jnp.sum(sources * weighted, axis=0))
            return jax.lax.dynamic_update_slice_in_dim(
                image,
                values.reshape(self.slab_size, ny, nz).astype(self.dtype),
                index * self.slab_size,
                0,
            )

        image = jnp.zeros((nx + padding, ny, nz), dtype=self.dtype)
        return jax.lax.fori_loop(
            0, (nx + padding) // self.slab_size, step, image
        )[:nx]


def compress_born(
    operator: BackgroundBorn, rank: int, slab_size: int = 4
) -> TensorTrainBorn:
    return TensorTrainBorn(
        compress_fields(operator.sources, operator.shape, rank),
        compress_fields(operator.reciprocal_fields, operator.shape, rank),
        operator.shape,
        operator.dtype,
        slab_size,
    )
