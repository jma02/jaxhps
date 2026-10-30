import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from breast_spheres_learn_3d import reconstruct_born
from compressed_born import (
    compress_blocks,
    compress_spatial,
    cp_fields,
    from_blocks,
    to_blocks,
    tucker_fields,
)
from phantom_inverse_3d import BackgroundBorn


@pytest.mark.parametrize("method", ["tucker", "block"])
@pytest.mark.parametrize("double", [False, True])
def test_exact_operator_real_adjoint_and_cg(method, double):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", double)
    try:
        rng = np.random.default_rng(19)
        shape = (5, 4, 3)
        fields = [
            jnp.asarray(
                rng.normal(size=(f, 60)) + 1j * rng.normal(size=(f, 60))
            )
            for f in (3, 7)
        ]
        delta = jnp.asarray(rng.normal(size=shape))
        data = jnp.asarray(
            rng.normal(size=(3, 7)) + 1j * rng.normal(size=(3, 7))
        )
        dense = BackgroundBorn(*fields, shape, delta.dtype)
        compressed = (
            compress_spatial(dense, method, 5, 2)
            if method == "tucker"
            else compress_blocks(dense, 7, 2, 5)
        )
        tol = 1e-10 if double else 4e-5
        np.testing.assert_allclose(
            compressed(delta), dense(delta), atol=tol, rtol=tol
        )
        np.testing.assert_allclose(
            compressed.adjoint(data), dense.adjoint(data), atol=tol, rtol=tol
        )
        window = jnp.ones(shape, dtype=delta.dtype)
        np.testing.assert_allclose(
            reconstruct_born(compressed, data, window, 8, 0.1),
            reconstruct_born(dense, data, window, 8, 0.1),
            atol=tol,
            rtol=tol,
        )
        assert not jax.make_jaxpr(lambda op, x: op(x))(
            compressed, delta
        ).consts
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("method", ["tucker", "cp", "block"])
def test_truncated_adjoint_and_differentiation(method):
    rng = np.random.default_rng(17)
    shape = (5, 4, 3)
    fields = [
        jnp.asarray(
            rng.normal(size=(f, 60)) + 1j * rng.normal(size=(f, 60)),
            dtype=jnp.complex64,
        )
        for f in (3, 7)
    ]
    delta = jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
    data = jnp.asarray(
        rng.normal(size=(3, 7)) + 1j * rng.normal(size=(3, 7)),
        dtype=jnp.complex64,
    )
    dense = BackgroundBorn(*fields, shape, delta.dtype)
    operator = (
        compress_blocks(dense, 2, 2, 5)
        if method == "block"
        else compress_spatial(dense, method, 2, 2, core_rank=3, iterations=10)
    )
    np.testing.assert_allclose(
        jnp.real(jnp.vdot(operator(delta), data)),
        jnp.vdot(delta, operator.adjoint(data)),
        rtol=2e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        jax.grad(lambda x: jnp.real(jnp.vdot(operator(x), data)))(delta),
        operator.adjoint(data),
        rtol=2e-5,
        atol=5e-5,
    )
    gradient = jax.grad(
        lambda y: jnp.sum(
            reconstruct_born(operator, y, jnp.ones(shape), 3, 0.1) ** 2
        )
    )(data)
    assert jnp.all(jnp.isfinite(gradient))
    if method == "block":
        sources = np.stack(
            [
                np.asarray(
                    from_blocks(operator.sources[:, s], shape, 2)
                ).ravel()
                for s in range(3)
            ]
        )
        receivers = (
            operator.receiver_basis.swapaxes(0, 1) @ operator.receiver_core
        )
        receivers = np.stack(
            [
                np.asarray(from_blocks(receivers[:, r], shape, 2)).ravel()
                for r in range(7)
            ]
        )
        decoded = BackgroundBorn(
            jnp.asarray(sources), jnp.asarray(receivers), shape, delta.dtype
        )
        np.testing.assert_allclose(
            operator(delta), decoded(delta), rtol=2e-5, atol=5e-5
        )


@pytest.mark.parametrize("method", ["tucker", "cp"])
def test_complex_separable_and_zero_fields(method):
    rng = np.random.default_rng(7)
    factors = [
        rng.normal(size=(2, n)) + 1j * rng.normal(size=(2, n))
        for n in (6, 5, 4)
    ]
    dense = jnp.asarray(
        np.einsum("fi,fj,fk->fijk", *factors), dtype=jnp.complex64
    )
    dense = dense.at[1].set(0)
    compressed = tucker_fields(dense.reshape(2, -1), (6, 5, 4), 1)
    if method == "cp":
        compressed = cp_fields(compressed, 1, iterations=10)
    decoded = jnp.concatenate(
        [compressed.slab(i, 2) for i in range(3)], axis=1
    )
    np.testing.assert_allclose(
        decoded, dense.reshape(2, -1), atol=4e-5, rtol=4e-5
    )
    assert compressed.nbytes < dense.nbytes


def test_block_padding_roundtrip():
    values = jnp.arange(5 * 4 * 3).reshape(5, 4, 3)
    for size in (2, 4, 8):
        np.testing.assert_array_equal(
            from_blocks(to_blocks(values, size), values.shape, size), values
        )


def test_cp_complex_rank_two_fit():
    rng = np.random.default_rng(42)
    factors = [
        rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
        for n in (6, 5, 4)
    ]
    field = jnp.asarray(
        np.einsum("ia,ja,ka->ijk", *factors), dtype=jnp.complex64
    )
    compressed = cp_fields(
        tucker_fields(field.reshape(1, -1), (6, 5, 4), 2), 2, iterations=100
    )
    decoded = compressed.slab(0, 6).reshape(field.shape)
    assert (
        float(jnp.linalg.norm(decoded - field) / jnp.linalg.norm(field)) < 1e-3
    )
