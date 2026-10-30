import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from breast_spheres_learn_3d import reconstruct_born
from phantom_inverse_3d import BackgroundBorn
from tensor_train_born import compress_born, compress_fields


@pytest.mark.parametrize("shape,slab_size", [((5, 4, 3), 2), ((3, 2, 5), 7)])
@pytest.mark.parametrize("double", [False, True])
def test_full_rank_operator_adjoint_and_cg(shape, slab_size, double):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", double)
    try:
        rng = np.random.default_rng(8)
        n = int(np.prod(shape))
        sources = jnp.asarray(
            rng.normal(size=(3, n)) + 1j * rng.normal(size=(3, n))
        )
        receivers = jnp.asarray(
            rng.normal(size=(7, n)) + 1j * rng.normal(size=(7, n))
        )
        dtype = np.dtype("float64" if double else "float32")
        operator = BackgroundBorn(sources, receivers, shape, dtype)
        compressed = compress_born(operator, max(shape), slab_size)
        image = jnp.asarray(rng.normal(size=shape), dtype=dtype)
        data = jnp.asarray(
            rng.normal(size=(3, 7)) + 1j * rng.normal(size=(3, 7))
        )
        tol = 1e-10 if double else 2e-5
        np.testing.assert_allclose(
            compressed(image), operator(image), rtol=tol, atol=tol
        )
        np.testing.assert_allclose(
            compressed.adjoint(data),
            operator.adjoint(data),
            rtol=tol,
            atol=tol,
        )
        window = jnp.asarray(rng.uniform(0.1, 1, shape), dtype=dtype)
        np.testing.assert_allclose(
            reconstruct_born(compressed, data, window, 8, 0.1),
            reconstruct_born(operator, data, window, 8, 0.1),
            rtol=tol,
            atol=tol,
        )
        assert not jax.make_jaxpr(lambda op, x: op(x))(
            compressed, image
        ).consts
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_rank_one_complex_fields_and_zero_fields():
    rng = np.random.default_rng(2)
    factors = [
        rng.normal(size=(2, n)) + 1j * rng.normal(size=(2, n))
        for n in (6, 5, 4)
    ]
    fields = jnp.asarray(np.einsum("fi,fj,fk->fijk", *factors))
    fields = fields.at[1].set(0)
    compressed = compress_fields(fields.reshape(2, -1), (6, 5, 4), 1)
    decoded = jnp.concatenate(
        [compressed.slab(i, 2) for i in range(3)], axis=1
    )
    np.testing.assert_allclose(
        decoded, fields.reshape(2, -1), rtol=1e-5, atol=1e-5
    )
    assert compressed.nbytes == 2 * (6 + 5 + 4) * fields.dtype.itemsize
    assert compressed.nbytes < fields.nbytes


def test_truncated_operator_is_adjoint_consistent_and_differentiable():
    rng = np.random.default_rng(3)
    shape = (5, 4, 3)
    sources = jnp.asarray(
        rng.normal(size=(3, 60)) + 1j * rng.normal(size=(3, 60))
    )
    receivers = jnp.asarray(
        rng.normal(size=(7, 60)) + 1j * rng.normal(size=(7, 60))
    )
    image = jnp.asarray(rng.normal(size=shape))
    data = jnp.asarray(rng.normal(size=(3, 7)) + 1j * rng.normal(size=(3, 7)))
    operator = compress_born(
        BackgroundBorn(sources, receivers, shape, image.dtype), 2, 2
    )
    decoded = [
        jnp.concatenate(
            [f.slab(i, 2).reshape(-1, 2, 4, 3) for i in range(3)], axis=1
        )[:, :5].reshape(-1, 60)
        for f in (operator.sources, operator.reciprocal_fields)
    ]
    reference = BackgroundBorn(*decoded, shape, image.dtype)
    np.testing.assert_allclose(
        operator(image), reference(image), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        operator.adjoint(data), reference.adjoint(data), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        jax.grad(lambda x: jnp.real(jnp.vdot(data, operator(x))))(image),
        operator.adjoint(data),
        rtol=1e-5,
        atol=1e-5,
    )
    tangent = jax.jvp(operator.adjoint, (data,), (data,))[1]
    np.testing.assert_allclose(
        tangent, operator.adjoint(data), rtol=1e-5, atol=1e-5
    )
    window = jnp.ones(shape, dtype=image.dtype)
    gradient = jax.grad(
        lambda y: jnp.sum(reconstruct_born(operator, y, window, 4, 0.1) ** 2)
    )(data)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    reference_gradient = jax.grad(
        lambda y: jnp.sum(reconstruct_born(reference, y, window, 4, 0.1) ** 2)
    )(data)
    np.testing.assert_allclose(
        gradient, reference_gradient, rtol=1e-5, atol=1e-5
    )


def test_invalid_rank():
    with pytest.raises(ValueError, match="rank must be positive"):
        compress_fields(jnp.ones((1, 8)), (2, 2, 2), 0)
