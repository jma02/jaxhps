import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.fft import dctn, idctn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from breast_spheres_learn_3d import reconstruct_born
from phantom_inverse_3d import (
    BackgroundBorn,
    crop,
    make_background_born,
    make_background_field_solver,
    sphere,
)
from spectral_born import SpectralBorn, spectral_indices, spectral_taper


@pytest.mark.parametrize("double", [False, True])
@pytest.mark.parametrize("retained", [17, 60])
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("basis", ["dct", "fft"])
def test_spectral_forward_adjoint_and_gradient(double, retained, batch, basis):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", double)
    try:
        rng = np.random.default_rng(52)
        shape = (5, 4, 3)
        dtype = np.complex128 if double else np.complex64
        sources, receivers = [
            (rng.normal(size=(f, 60)) + 1j * rng.normal(size=(f, 60))).astype(
                dtype
            )
            for f in (3, 7)
        ]
        taper = np.asarray(
            spectral_taper(shape, 0.01 if basis == "fft" else 1.0)
        ).ravel()
        transform = dctn if basis == "dct" else np.fft.fftn
        inverse = idctn if basis == "dct" else np.fft.ifftn
        spectrum = (
            transform(
                (receivers * taper).reshape(7, *shape),
                axes=(-3, -2, -1),
                norm="ortho",
            )
            .reshape(7, -1)
            .astype(dtype)
        )
        indices = np.sort(rng.choice(60, retained, replace=False))
        truncated = np.zeros_like(spectrum)
        truncated[:, indices] = spectrum[:, indices]
        expanded = (
            inverse(
                truncated.reshape(7, *shape), axes=(-3, -2, -1), norm="ortho"
            )
            .reshape(7, -1)
            .astype(dtype)
            / taper
        )
        delta = jnp.asarray(rng.normal(size=shape), dtype=sources.real.dtype)
        data = jnp.asarray(
            rng.normal(size=(3, 7)) + 1j * rng.normal(size=(3, 7)), dtype=dtype
        )
        operator = SpectralBorn(
            jnp.asarray(sources / taper),
            jnp.asarray(spectrum[:, indices]),
            jnp.asarray(indices),
            shape,
            np.dtype(delta.dtype),
            batch,
            basis,
        )
        dense = BackgroundBorn(
            jnp.asarray(sources), jnp.asarray(expanded), shape, delta.dtype
        )
        tol = 1e-10 if double else 4e-5
        np.testing.assert_allclose(
            operator(delta), dense(delta), atol=tol, rtol=tol
        )
        np.testing.assert_allclose(
            operator.adjoint(data), dense.adjoint(data), atol=tol, rtol=tol
        )
        np.testing.assert_allclose(
            jnp.real(jnp.vdot(operator(delta), data)),
            jnp.vdot(delta, operator.adjoint(data)),
            atol=tol,
            rtol=tol,
        )
        np.testing.assert_allclose(
            jax.grad(lambda x: jnp.real(jnp.vdot(operator(x), data)))(delta),
            operator.adjoint(data),
            atol=tol,
            rtol=tol,
        )
        window = jnp.ones(shape, dtype=delta.dtype)
        for candidate in (operator, dense):
            image = reconstruct_born(candidate, data, window, 8, 0.1)
            assert np.isfinite(image).all()
        np.testing.assert_allclose(
            reconstruct_born(operator, data, window, 8, 0.1),
            reconstruct_born(dense, data, window, 8, 0.1),
            atol=tol,
            rtol=tol,
        )
        spectral_gradient = jax.grad(
            lambda y: jnp.sum(
                reconstruct_born(operator, y, window, 8, 0.1) ** 2
            )
        )(data)
        dense_gradient = jax.grad(
            lambda y: jnp.sum(reconstruct_born(dense, y, window, 8, 0.1) ** 2)
        )(data)
        np.testing.assert_allclose(
            spectral_gradient, dense_gradient, atol=tol, rtol=tol
        )
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_mode_selection_tail_and_minimality():
    rng = np.random.default_rng(33)
    energy = rng.exponential(size=120)
    for tolerance in (0, 0.01, 0.1, 0.5, 0.9):
        indices = spectral_indices(energy, tolerance)
        omitted = np.delete(energy, indices).sum()
        assert omitted <= tolerance**2 * energy.sum() + 1e-12
        assert omitted + energy[indices].min() > tolerance**2 * energy.sum()
        assert np.all(np.diff(indices) > 0)
    assert spectral_indices(np.zeros(10), 0.1).size == 1
    assert spectral_indices(np.array([0, 0, 1, 3]), 0).size == 2
    assert spectral_indices(np.array([0, 0, 1, 3]), 0.5).size == 1


def test_streamed_field_solver_matches_full_cache():
    points, _, truth = crop(8)
    background = jnp.asarray(truth)
    directions, receivers = sphere(3), 0.36 * sphere(5)
    dense = make_background_born(
        points, 10.0, directions, receivers, background
    )
    solve = make_background_field_solver(points, 10.0, background)
    for positions, reciprocal, reference in (
        (directions, False, dense.sources),
        (receivers, True, dense.reciprocal_fields),
    ):
        batches = []
        for first in range(0, len(positions), 2):
            fields, residuals = solve(positions[first : first + 2], reciprocal)
            assert float(jnp.max(residuals)) < 5e-6
            batches.append(fields)
        np.testing.assert_allclose(
            jnp.concatenate(batches), reference, rtol=2e-6, atol=2e-7
        )
