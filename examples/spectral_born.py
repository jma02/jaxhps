"""Born application through a shared, truncated orthonormal spatial basis.

Only reciprocal fields are projected. Source-weighted images are transformed
and contracted with retained coefficients, without expanding receiver fields.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.fft import dctn, idctn


def cosine_transform(values: jax.Array, inverse: bool = False) -> jax.Array:
    transform = idctn if inverse else dctn
    return transform(
        values.real, axes=(-3, -2, -1), norm="ortho"
    ) + 1j * transform(values.imag, axes=(-3, -2, -1), norm="ortho")


def spectral_taper(shape: tuple[int, int, int], floor: float) -> jax.Array:
    if not 0 < floor <= 1:
        raise ValueError("taper floor must be in (0, 1]")
    axes = [jnp.sin(jnp.pi * (jnp.arange(n) + 0.5) / n) ** 2 for n in shape]
    return floor + (1 - floor) * (
        axes[0][:, None, None]
        * axes[1][None, :, None]
        * axes[2][None, None, :]
    )


def spectral_indices(energy: np.ndarray, tolerance: float) -> np.ndarray:
    """Smallest shared mode set meeting a receiver Frobenius-tail target."""
    if not 0 <= tolerance < 1:
        raise ValueError("tolerance must be in [0, 1)")
    order = np.argsort(energy.ravel())[::-1]
    tail = np.cumsum(energy.ravel()[order][::-1], dtype=np.float64)
    if tail[-1] == 0:
        return np.array([0], dtype=np.int32)
    count = max(
        1,
        len(order)
        - np.searchsorted(tail, tolerance**2 * tail[-1], side="right"),
    )
    return np.sort(order[:count]).astype(np.int32)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["sources", "coefficients", "indices"],
    meta_fields=["shape", "dtype", "source_batch_size", "basis"],
)
@dataclass(frozen=True)
class SpectralBorn:
    """Store DCT(receiver) or FFT(taper * receiver) coefficients.

    For the Fourier basis, sources must be divided by the same positive taper.
    Truncation errors can be amplified where the taper is small; its spectral
    tail tolerance is not an error bound for the original Born operator.
    """

    sources: jax.Array
    coefficients: jax.Array
    indices: jax.Array
    shape: tuple[int, int, int]
    dtype: np.dtype
    source_batch_size: int = 2
    basis: str = "dct"

    @property
    def nbytes(self) -> int:
        return sum(x.nbytes for x in jax.tree_util.tree_leaves(self))

    @jax.jit
    def __call__(self, delta: jax.Array) -> jax.Array:
        batch = self.source_batch_size
        sources = jnp.pad(
            self.sources, ((0, (-len(self.sources)) % batch), (0, 0))
        ).reshape(-1, batch, *self.shape)

        def apply(source: jax.Array) -> jax.Array:
            values = source * delta
            transformed = (
                cosine_transform(values)
                if self.basis == "dct"
                else jnp.fft.ifftn(values, axes=(-3, -2, -1), norm="ortho")
            ).reshape(batch, -1)
            return jnp.matmul(
                transformed[:, self.indices],
                self.coefficients.T,
                precision=jax.lax.Precision.HIGHEST,
            )

        return jax.lax.map(apply, sources).reshape(
            -1, self.coefficients.shape[0]
        )[: len(self.sources)]

    @jax.jit
    def adjoint(self, data: jax.Array) -> jax.Array:
        batch = self.source_batch_size
        padding = (-len(self.sources)) % batch
        sources = jnp.pad(self.sources, ((0, padding), (0, 0))).reshape(
            -1, batch, *self.shape
        )
        data = jnp.pad(data, ((0, padding), (0, 0))).reshape(
            -1, batch, self.coefficients.shape[0]
        )

        def step(index: int, image: jax.Array) -> jax.Array:
            reduced = jnp.matmul(
                data[index].conj(),
                self.coefficients,
                precision=jax.lax.Precision.HIGHEST,
            )
            spectrum = (
                jnp.zeros((batch, np.prod(self.shape)), dtype=reduced.dtype)
                .at[:, self.indices]
                .set(reduced)
            )
            weighted = (
                cosine_transform(
                    spectrum.reshape(batch, *self.shape), inverse=True
                )
                if self.basis == "dct"
                else jnp.fft.ifftn(
                    spectrum.reshape(batch, *self.shape),
                    axes=(-3, -2, -1),
                    norm="ortho",
                )
            )
            return image + jnp.real(jnp.sum(sources[index] * weighted, axis=0))

        return jax.lax.fori_loop(
            0, len(sources), step, jnp.zeros(self.shape, dtype=self.dtype)
        )
