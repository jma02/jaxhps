"""Spatial Tucker/CP fields and directly contracted block receiver bases.

The block representation compresses the receiver mode locally. It does not
form the Born matrix or reconstruct compressed receiver fields on application.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from phantom_inverse_3d import BackgroundBorn
from tensor_train_born import SlabBorn

PRECISION = jax.lax.Precision.HIGHEST


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["first", "second", "third", "core"],
    meta_fields=[],
)
@dataclass(frozen=True)
class TuckerFields:
    first: jax.Array
    second: jax.Array
    third: jax.Array
    core: jax.Array

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.first.shape[1], self.second.shape[1], self.third.shape[1]

    @property
    def nbytes(self) -> int:
        return sum(x.nbytes for x in jax.tree_util.tree_leaves(self))

    def slab(self, index: jax.Array, size: int) -> jax.Array:
        first = jnp.pad(
            self.first, ((0, 0), (0, (-self.shape[0]) % size), (0, 0))
        )
        first = jax.lax.dynamic_slice_in_dim(first, index * size, size, 1)
        values = jnp.einsum(
            "fxa,fabc->fxbc", first, self.core, precision=PRECISION
        )
        values = jnp.einsum(
            "fxbc,fyb->fxyc", values, self.second, precision=PRECISION
        )
        return jnp.einsum(
            "fxyc,fzc->fxyz", values, self.third, precision=PRECISION
        ).reshape(self.first.shape[0], -1)


@partial(jax.jit, static_argnames=("shape", "rank"))
def tucker_fields(
    fields: jax.Array, shape: tuple[int, int, int], rank: int
) -> TuckerFields:
    """Sequentially truncated HOSVD, with a cap on each spatial mode."""
    if rank < 1:
        raise ValueError("rank must be positive")

    def compress(field: jax.Array) -> tuple[jax.Array, ...]:
        core = field.reshape(shape)
        factors = []
        for axis in range(3):
            unfolded = jnp.moveaxis(core, axis, 0)
            matrix = unfolded.reshape(unfolded.shape[0], -1)
            left, singular, right = jnp.linalg.svd(matrix, full_matrices=False)
            r = min(rank, *matrix.shape)
            factors.append(left[:, :r])
            core = jnp.moveaxis(
                (singular[:r, None] * right[:r]).reshape(
                    (r, *unfolded.shape[1:])
                ),
                0,
                axis,
            )
        return (*factors, core)

    return TuckerFields(*jax.lax.map(compress, fields))


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["first", "second", "third"],
    meta_fields=[],
)
@dataclass(frozen=True)
class CPFields:
    first: jax.Array
    second: jax.Array
    third: jax.Array

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.first.shape[1], self.second.shape[1], self.third.shape[1]

    @property
    def nbytes(self) -> int:
        return sum(x.nbytes for x in jax.tree_util.tree_leaves(self))

    def slab(self, index: jax.Array, size: int) -> jax.Array:
        first = jnp.pad(
            self.first, ((0, 0), (0, (-self.shape[0]) % size), (0, 0))
        )
        first = jax.lax.dynamic_slice_in_dim(first, index * size, size, 1)
        return jnp.einsum(
            "fxa,fya,fza->fxyz",
            first,
            self.second,
            self.third,
            precision=PRECISION,
        ).reshape(self.first.shape[0], -1)


@partial(jax.jit, static_argnames=("rank", "iterations", "restarts"))
def cp_fields(
    tucker: TuckerFields, rank: int, iterations: int = 50, restarts: int = 2
) -> CPFields:
    """Complex CP-ALS on a Tucker core; retain the best deterministic restart."""
    if min(rank, iterations, restarts) < 1:
        raise ValueError("rank, iterations, and restarts must be positive")

    def fit(core: jax.Array) -> tuple[jax.Array, ...]:
        scale = jnp.maximum(jnp.linalg.norm(core), 1e-30)
        target = core / scale

        def restart(key: jax.Array) -> tuple[jax.Array, ...]:
            factors = tuple(
                jax.random.normal(k, (n, rank), dtype=core.dtype) / jnp.sqrt(n)
                for k, n in zip(jax.random.split(key, 3), core.shape)
            )

            def step(_: int, factors: tuple[jax.Array, ...]):
                factors = list(factors)
                for axis in range(3):
                    others = [factors[i] for i in range(3) if i != axis]
                    gram = jnp.prod(
                        jnp.stack([x.conj().T @ x for x in others]), axis=0
                    )
                    rhs = jnp.einsum(
                        "ijk,ja,ka->ia",
                        jnp.moveaxis(target, axis, 0),
                        others[0].conj(),
                        others[1].conj(),
                        precision=PRECISION,
                    )
                    ridge = 1e-6 * jnp.maximum(
                        jnp.trace(gram).real / rank, 1e-12
                    )
                    factors[axis] = jnp.linalg.solve(
                        gram + ridge * jnp.eye(rank, dtype=core.dtype), rhs.T
                    ).T
                norms = jnp.stack(
                    [
                        jnp.maximum(jnp.linalg.norm(x, axis=0), 1e-15)
                        for x in factors
                    ]
                )
                balanced = jnp.exp(jnp.mean(jnp.log(norms), axis=0))
                return tuple(
                    x * (balanced / norms[i])[None, :]
                    for i, x in enumerate(factors)
                )

            return jax.lax.fori_loop(0, iterations, step, factors)

        candidates = jax.lax.map(
            restart, jax.random.split(jax.random.PRNGKey(41), restarts)
        )
        errors = jax.vmap(
            lambda a, b, c: jnp.linalg.norm(
                jnp.einsum("ia,ja,ka->ijk", a, b, c, precision=PRECISION)
                - target
            )
        )(*candidates)
        best = jnp.argmin(errors)
        a, b, c = (x[best] for x in candidates)
        return a * scale, b, c

    factors = jax.lax.map(fit, tucker.core)
    return CPFields(
        *(
            jnp.matmul(basis, factor, precision=PRECISION)
            for basis, factor in zip(
                (tucker.first, tucker.second, tucker.third), factors
            )
        )
    )


def compress_spatial(
    operator: BackgroundBorn,
    method: str,
    rank: int,
    slab_size: int = 32,
    core_rank: int = 64,
    iterations: int = 50,
) -> SlabBorn:
    if method not in ("tucker", "cp"):
        raise ValueError("method must be tucker or cp")
    fields = []
    for dense in (operator.sources, operator.reciprocal_fields):
        compressed = tucker_fields(
            dense, operator.shape, rank if method == "tucker" else core_rank
        )
        if method == "cp":
            compressed = cp_fields(compressed, rank, iterations)
        fields.append(compressed)
    return SlabBorn(*fields, operator.shape, operator.dtype, slab_size)


def to_blocks(values: jax.Array, block_size: int) -> jax.Array:
    """Convert (..., x, y, z) to (blocks, ..., block voxels), zero padding."""
    leading = values.shape[:-3]
    padded = jnp.pad(
        values,
        [(0, 0)] * len(leading)
        + [(0, (-n) % block_size) for n in values.shape[-3:]],
    )
    nx, ny, nz = (n // block_size for n in padded.shape[-3:])
    reshaped = padded.reshape(
        (*leading, nx, block_size, ny, block_size, nz, block_size)
    )
    offset = len(leading)
    return jnp.transpose(
        reshaped,
        (
            offset,
            offset + 2,
            offset + 4,
            *range(offset),
            offset + 1,
            offset + 3,
            offset + 5,
        ),
    ).reshape((nx * ny * nz, *leading, block_size**3))


def from_blocks(
    values: jax.Array, shape: tuple[int, int, int], block_size: int
) -> jax.Array:
    nx, ny, nz = ((n + block_size - 1) // block_size for n in shape)
    image = (
        values.reshape(nx, ny, nz, block_size, block_size, block_size)
        .transpose(0, 3, 1, 4, 2, 5)
        .reshape(nx * block_size, ny * block_size, nz * block_size)
    )
    return image[: shape[0], : shape[1], : shape[2]]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["sources", "receiver_basis", "receiver_core"],
    meta_fields=["shape", "dtype", "block_size"],
)
@dataclass(frozen=True)
class BlockBorn:
    sources: jax.Array
    receiver_basis: jax.Array
    receiver_core: jax.Array
    shape: tuple[int, int, int]
    dtype: np.dtype
    block_size: int

    @property
    def nbytes(self) -> int:
        return sum(x.nbytes for x in jax.tree_util.tree_leaves(self))

    @jax.jit
    def __call__(self, delta: jax.Array) -> jax.Array:
        weighted = self.sources * to_blocks(delta, self.block_size)[:, None, :]
        reduced = jnp.matmul(
            weighted, self.receiver_core.swapaxes(-1, -2), precision=PRECISION
        )
        return jnp.einsum(
            "bsk,rbk->sr", reduced, self.receiver_basis, precision=PRECISION
        )

    @jax.jit
    def adjoint(self, data: jax.Array) -> jax.Array:
        reduced = jnp.einsum(
            "sr,rbk->bsk",
            data.conj(),
            self.receiver_basis,
            precision=PRECISION,
        )
        weighted = jnp.matmul(reduced, self.receiver_core, precision=PRECISION)
        values = jnp.real(jnp.sum(self.sources * weighted, axis=1))
        return from_blocks(values, self.shape, self.block_size).astype(
            self.dtype
        )


@partial(jax.jit, static_argnames=("rank", "block_size", "batch_size"))
def compress_blocks(
    operator: BackgroundBorn, rank: int, block_size: int, batch_size: int = 16
) -> BlockBorn:
    if min(rank, block_size, batch_size) < 1:
        raise ValueError("rank, block_size, and batch_size must be positive")
    count = int(
        np.prod([(n + block_size - 1) // block_size for n in operator.shape])
    )
    basis, core = compress_receiver_blocks(
        operator.reciprocal_fields,
        jnp.arange(count),
        operator.shape,
        rank,
        block_size,
        batch_size,
    )
    return BlockBorn(
        to_blocks(operator.sources.reshape((-1, *operator.shape)), block_size),
        basis.swapaxes(0, 1),
        core,
        operator.shape,
        operator.dtype,
        block_size,
    )


@partial(
    jax.jit, static_argnames=("shape", "rank", "block_size", "batch_size")
)
def compress_receiver_blocks(
    fields: jax.Array,
    indices: jax.Array,
    shape: tuple[int, int, int],
    rank: int,
    block_size: int,
    batch_size: int = 16,
) -> tuple[jax.Array, jax.Array]:
    if min(rank, block_size, batch_size) < 1:
        raise ValueError("rank, block_size, and batch_size must be positive")
    # Padding and extracting small batches avoids a second full receiver cache.
    receivers = jnp.pad(
        fields.reshape((-1, *shape)),
        [(0, 0), *((0, (-n) % block_size) for n in shape)],
    )
    _, ny, nz = (n // block_size for n in receivers.shape[-3:])
    r = min(rank, receivers.shape[0], block_size**3)

    def compress(index: jax.Array) -> tuple[jax.Array, jax.Array]:
        x, yz = index // (ny * nz), index % (ny * nz)
        y, z = yz // nz, yz % nz
        matrix = jax.lax.dynamic_slice(
            receivers,
            (0, x * block_size, y * block_size, z * block_size),
            (receivers.shape[0], block_size, block_size, block_size),
        ).reshape(receivers.shape[0], -1)
        left, singular, right = jnp.linalg.svd(matrix, full_matrices=False)
        return left[:, :r], singular[:r, None] * right[:r]

    return jax.lax.map(compress, indices, batch_size=batch_size)
