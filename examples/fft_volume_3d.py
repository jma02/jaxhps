"""GPU Lippmann--Schwinger solver with a truncated free-space Green kernel.

Vico, Greengard & Ferrando, JCP 323 (2016), 191--203,
doi:10.1016/j.jcp.2016.07.028. Kernel construction uses a 4n grid; each
aperiodic convolution uses a 2n grid. G=exp(ikr)/(4*pi*r) satisfies
(Delta+k²)G=-delta, so (I+k² G b)u=u_inc and u_s=-k² G(bu).
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jaxhps._krylov import fgmres


def truncated_green_transform(s, kappa, radius):
    """Analytic Fourier multiplier, including removable s=0 and s=k limits."""
    s = jnp.asarray(s)
    denom = s**2 - kappa**2
    on_shell = jnp.abs(denom) < 1e-12 * kappa**2
    numerator = 1.0 + jnp.exp(1j * kappa * radius) * (
        1j * kappa * radius * jnp.sinc(s * radius / jnp.pi)
        - jnp.cos(s * radius)
    )
    limit = 1j * radius / (2 * kappa) - jnp.expm1(2j * kappa * radius) / (
        4 * kappa**2
    )
    return jnp.where(
        on_shell, limit, numerator / jnp.where(on_shell, 1, denom)
    )


@partial(jax.jit, static_argnames=("n",))
def green_kernel(n, a, kappa):
    """Return Fourier weights for a zero-padded 2n convolution."""
    h = 2 * a / n
    freq = 2 * jnp.pi * jnp.fft.fftfreq(4 * n, d=h)
    s = jnp.sqrt(
        freq[:, None, None] ** 2
        + freq[None, :, None] ** 2
        + freq[None, None, :] ** 2
    )
    weights = jnp.fft.ifftn(truncated_green_transform(s, kappa, 3.6 * a))
    indices = jnp.concatenate([jnp.arange(n), jnp.arange(3 * n, 4 * n)])
    cropped = weights[jnp.ix_(indices, indices, indices)]
    return jnp.fft.fftn(cropped)


def prepare_volume_solver(n, a, kappa, coefficient):
    """Prepare a reusable operator; coefficient accepts (..., 3) points."""
    if n < 4 or a <= 0 or kappa <= 0:
        raise ValueError("n>=4 and positive a,kappa required")
    axis = -a + (np.arange(n) + 0.5) * (2 * a / n)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    b = np.asarray(coefficient(points))
    if b.shape != points.shape[:-1] or not np.all(np.isfinite(b)):
        raise ValueError("coefficient must be finite with shape (n,n,n)")
    b = jnp.asarray(b)
    kernel = green_kernel(n, a, kappa)

    @jax.jit
    def matvec(u):
        charge = b * u.reshape(b.shape)
        padded = jnp.pad(charge, ((0, n),) * 3)
        potential = jnp.fft.ifftn(jnp.fft.fftn(padded) * kernel)[:n, :n, :n]
        return (u.reshape(b.shape) + kappa**2 * potential).ravel()

    jax.block_until_ready(kernel)
    return points, b, kernel, matvec


def solve_volume(matvec, points, kappa, direction, tol=1e-8, restart=30):
    incident = jnp.exp(1j * kappa * jnp.asarray(points @ direction)).ravel()
    stats = {}
    total, info = fgmres(
        matvec, incident, tol=tol, restart=restart, maxiter=600, stats=stats
    )
    total.block_until_ready()
    return total.reshape(points.shape[:-1]), dict(stats, info=int(info))


@partial(jax.jit, static_argnames=("target_batch",))
def evaluate_volume_field(targets, points, density, kappa, h, target_batch=8):
    """Smooth off-support quadrature, with bounded target batching.

    Targets must be outside the volume grid. The uniform-grid weight h³
    applies here; it is already included in the convolution weights above.
    """
    points = points.reshape(-1, 3)
    density = density.ravel()
    count = targets.shape[0]
    padding = (-count) % target_batch
    batches = jnp.pad(targets, ((0, padding), (0, 0)), mode="edge").reshape(
        -1, target_batch, 3
    )

    def evaluate(batch):
        r = jnp.linalg.norm(batch[:, None, :] - points[None, :, :], axis=-1)
        green = jnp.exp(1j * kappa * r) / (4 * jnp.pi * r)
        return -(kappa**2) * h**3 * (green @ density)

    return jax.lax.map(evaluate, batches).ravel()[:count]
