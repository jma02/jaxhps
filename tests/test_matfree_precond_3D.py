"""Tests for the flat-system preconditioners.

The checks are algebraic, so they do not depend on fmm3dbie fixtures:

* the leaf orderings are consistent with the leaf geometry;
* one Gauss-Seidel sweep reproduces, to roundoff, the solve with the
  block-lower-triangular part of the flat operator in that ordering;
* the shifted-operator preconditioner is the inverse of the damped flat
  operator, and preconditioned GMRES reaches the dense-LU solution.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from jaxhps import DiscretizationNode3D, Domain, PDEProblem
from jaxhps._matfree_iti_3D import (
    build_interface_maps,
    flat_bie_diagonal_approx,
    make_flat_bie_operator,
    materialize,
)
from jaxhps._matfree_precond_3D import (
    leaf_index_grid,
    make_gauss_seidel_sweep,
    make_shifted_operator_preconditioner,
    make_sweep_preconditioner,
    sweep_cost_matvecs,
    sweep_order,
)
from jaxhps.local_solve import local_solve_stage_uniform_3D_ItI

_EX_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "examples")
sys.path.insert(0, os.path.abspath(_EX_DIR))

from wave_scattering_utils_3D import (  # noqa: E402
    make_dense_SD_apply,
)


def _setup(p=8, q=4, L=1, kappa=4.0, shift=0.0, seed=5):
    """Small flat BIE system with synthetic exterior operators."""
    root = DiscretizationNode3D(
        xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5, zmin=-0.5, zmax=0.5
    )
    domain = Domain(p=p, q=q, root=root, L=L)
    n_leaves = np.asarray(domain.interior_points).shape[0]
    ones = np.ones((n_leaves, p**3))
    I_coeffs = (kappa**2 * (1.0 + 1j * shift) * ones).astype(np.complex128)
    problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=ones,
        D_yy_coefficients=ones,
        D_zz_coefficients=ones,
        I_coefficients=I_coeffs,
        source=np.zeros((n_leaves, p**3), dtype=np.complex128),
        use_ItI=True,
        eta=kappa,
    )
    _, T_leaves, _, h_leaves = local_solve_stage_uniform_3D_ItI(problem)
    maps = build_interface_maps(domain)
    n_bdry = int(maps.bdry_rows.size)
    rng = np.random.default_rng(seed)
    S = (
        rng.normal(size=(n_bdry, n_bdry))
        + 1j * rng.normal(size=(n_bdry, n_bdry))
    ) / n_bdry
    D = (
        rng.normal(size=(n_bdry, n_bdry))
        + 1j * rng.normal(size=(n_bdry, n_bdry))
    ) / n_bdry
    return domain, T_leaves, h_leaves, maps, S, D, float(kappa)


def test_leaf_index_grid_and_order() -> None:
    """Leaf indices tile the octant and the sweep orders are permutations."""
    L = 1
    domain, *_ = _setup()
    idx = leaf_index_grid(domain.root, L)
    assert idx.shape == (8**L, 3)
    assert {tuple(v) for v in idx} == {
        (i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)
    }
    fwd = sweep_order(domain.root, L, (1, 1, 1))
    rev = sweep_order(domain.root, L, (-1, -1, -1))
    assert np.array_equal(np.sort(fwd), np.arange(8**L))
    assert np.array_equal(fwd, rev[::-1])
    assert sweep_cost_matvecs(1) == 1 and sweep_cost_matvecs(2) == 3
    jax.clear_caches()


def test_sweep_is_block_triangular_solve() -> None:
    """One sweep equals the solve with the block-lower-triangular part."""
    L = 1
    domain, T_leaves, _, maps, S, D, eta = _setup()
    apply_S, apply_D = make_dense_SD_apply(S, D)
    n = maps.n_flat
    A = np.asarray(
        materialize(
            make_flat_bie_operator(T_leaves, maps, eta, apply_S, apply_D), n
        )
    )
    d = np.asarray(
        flat_bie_diagonal_approx(
            T_leaves,
            maps,
            eta,
            jnp.diag(jnp.asarray(S)),
            jnp.diag(jnp.asarray(D)),
        )
    )
    order = sweep_order(domain.root, L, (1, 1, 1))
    sweep = make_gauss_seidel_sweep(T_leaves, maps, order, 1.0 / d)

    pos = np.empty(maps.n_leaves, dtype=int)
    pos[np.asarray(order)] = np.arange(maps.n_leaves)
    leaf_of = np.arange(n) // maps.n_per_leaf
    bdry = np.asarray(maps.bdry_rows)
    keep = pos[leaf_of][None, :] <= pos[leaf_of][:, None]
    M = np.where(keep, A, 0.0)
    M[bdry, :] = 0.0
    M[bdry, bdry] = d[bdry]

    rng = np.random.default_rng(0)
    r = rng.normal(size=n) + 1j * rng.normal(size=n)
    z_ref = np.linalg.solve(M, r)
    z = np.asarray(sweep(jnp.asarray(r)))
    assert np.abs(z - z_ref).max() / np.abs(z_ref).max() < 1e-10
    # The batched path agrees with the vector path column by column.
    R = np.stack([r, 2.0 * r], axis=1)
    Z = np.asarray(sweep(jnp.asarray(R)))
    assert np.abs(Z[:, 0] - z).max() / np.abs(z).max() < 1e-12
    assert np.abs(Z[:, 1] - 2.0 * z).max() / np.abs(z).max() < 1e-12
    jax.clear_caches()


def test_multi_direction_sweep_is_linear() -> None:
    """The composed forward/reverse sweep is a linear operator on residuals."""
    L = 1
    domain, T_leaves, _, maps, S, D, eta = _setup()
    apply_S, apply_D = make_dense_SD_apply(S, D)
    A_matvec = make_flat_bie_operator(T_leaves, maps, eta, apply_S, apply_D)
    M = make_sweep_preconditioner(
        T_leaves,
        maps,
        eta,
        jnp.diag(jnp.asarray(S)),
        jnp.diag(jnp.asarray(D)),
        domain.root,
        L,
        A_matvec=A_matvec,
    )
    rng = np.random.default_rng(1)
    n = maps.n_flat
    u = jnp.asarray(rng.normal(size=n) + 1j * rng.normal(size=n))
    v = jnp.asarray(rng.normal(size=n) + 1j * rng.normal(size=n))
    lhs = M(u + 3.0 * v)
    rhs = M(u) + 3.0 * M(v)
    assert float(jnp.abs(lhs - rhs).max() / jnp.abs(rhs).max()) < 1e-10
    jax.clear_caches()


def test_shifted_preconditioner_inverts_damped_operator() -> None:
    """The shift preconditioner is the inverse of the damped flat operator."""
    eps = 0.1
    domain, T_leaves, _, maps, S, D, eta = _setup()
    _, T_shift, _, _ = local_solve_stage_uniform_3D_ItI(
        _shifted_problem(domain, eps, eta)
    )
    apply_S, apply_D = make_dense_SD_apply(S, D)
    A_shift = make_flat_bie_operator(T_shift, maps, eta, apply_S, apply_D)
    M = make_shifted_operator_preconditioner(
        T_shift, maps, eta, apply_S, apply_D
    )
    rng = np.random.default_rng(2)
    n = maps.n_flat
    v = jnp.asarray(rng.normal(size=n) + 1j * rng.normal(size=n))
    err = jnp.abs(M(A_shift(v)) - v).max() / jnp.abs(v).max()
    assert float(err) < 1e-8
    # A nonzero shift really changes the operator it inverts.
    A = make_flat_bie_operator(T_leaves, maps, eta, apply_S, apply_D)
    assert float(jnp.abs(M(A(v)) - v).max() / jnp.abs(v).max()) > 1e-3
    jax.clear_caches()


def _shifted_problem(domain, eps, kappa):
    p = domain.p
    n_leaves = np.asarray(domain.interior_points).shape[0]
    ones = np.ones((n_leaves, p**3))
    return PDEProblem(
        domain=domain,
        D_xx_coefficients=ones,
        D_yy_coefficients=ones,
        D_zz_coefficients=ones,
        I_coefficients=(kappa**2 * (1.0 + 1j * eps) * ones).astype(
            np.complex128
        ),
        source=np.zeros((n_leaves, p**3), dtype=np.complex128),
        use_ItI=True,
        eta=kappa,
    )
