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
import pytest

from jaxhps import DiscretizationNode3D, Domain, PDEProblem
from jaxhps._krylov import fgmres
from jaxhps._matfree_iti_3D import (
    build_interface_maps,
    flat_bie_diagonal_approx,
    make_flat_bie_operator,
    materialize,
)
from jaxhps._matfree_precond_3D import (
    leaf_index_grid,
    make_coarse_correction_preconditioner,
    make_face_polynomial_transfer,
    make_face_pair_smoother,
    make_gauss_seidel_sweep,
    make_iterative_shifted_preconditioner,
    make_multilevel_shifted_preconditioner,
    make_parent_face_transfer,
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


@pytest.mark.parametrize("n_rhs", [1, 2])
def test_iterative_shifted_solve_without_dense_setup(monkeypatch, n_rhs):
    _, T_shift, _, maps, S, D, eta = _setup(p=6, q=2, shift=0.1)
    apply_S, apply_D = make_dense_SD_apply(S, D)
    A = make_flat_bie_operator(T_shift, maps, eta, apply_S, apply_D)
    dense = np.asarray(materialize(A, maps.n_flat))
    rng = np.random.default_rng(12)
    b = jnp.asarray(
        rng.normal(size=(maps.n_flat, n_rhs))
        + 1j * rng.normal(size=(maps.n_flat, n_rhs))
    )
    if n_rhs == 1:
        b = b[:, 0]
    reference = np.linalg.solve(dense, b)

    def no_dense(*args, **kwargs):
        raise AssertionError(
            "The iterative preconditioner must stay matrix-free"
        )

    monkeypatch.setattr("jaxhps._matfree_precond_3D.materialize", no_dense)
    monkeypatch.setattr(jax.scipy.linalg, "lu_factor", no_dense)
    stats = {}
    M = make_iterative_shifted_preconditioner(
        T_shift,
        maps,
        eta,
        apply_S,
        apply_D,
        jnp.diag(S),
        jnp.diag(D),
        tol=1e-10,
        restart=b.size,
        maxiter=b.size,
        stats=stats,
    )
    x = M(b)
    np.testing.assert_allclose(x, reference, rtol=1e-8, atol=1e-8)
    assert stats["n_unconverged"] == 0
    assert stats["n_calls"] == 1
    assert stats["n_matvec"] == stats["n_iter"] + 2
    np.testing.assert_array_equal(M(jnp.zeros_like(b)), np.zeros_like(b))
    jax.clear_caches()


def test_inexact_shifted_preconditioner_with_flexible_outer_solver():
    domain, T, _, maps, S, D, eta = _setup(p=6, q=2)
    _, T_shift, _, _ = local_solve_stage_uniform_3D_ItI(
        _shifted_problem(domain, 0.1, eta)
    )
    apply_S, apply_D = make_dense_SD_apply(S, D)
    A = jax.jit(make_flat_bie_operator(T, maps, eta, apply_S, apply_D))
    rng = np.random.default_rng(15)
    exact = jnp.asarray(
        rng.normal(size=maps.n_flat) + 1j * rng.normal(size=maps.n_flat)
    )
    b = A(exact)
    inner_stats = {}
    M = make_iterative_shifted_preconditioner(
        T_shift,
        maps,
        eta,
        apply_S,
        apply_D,
        jnp.diag(S),
        jnp.diag(D),
        tol=0.05,
        restart=20,
        maxiter=20,
        stats=inner_stats,
    )
    stats = {}
    x, info = fgmres(
        A,
        b,
        precond=M,
        tol=1e-9,
        restart=maps.n_flat,
        maxiter=maps.n_flat,
        stats=stats,
    )
    assert info == 0
    assert float(jnp.linalg.norm(b - A(x)) / jnp.linalg.norm(b)) <= 1e-9
    np.testing.assert_allclose(x, exact, rtol=1e-6, atol=1e-6)
    assert inner_stats["n_calls"] == stats["n_iter"]
    assert inner_stats["n_unconverged"] > 0
    assert inner_stats["n_iter"] <= 20 * inner_stats["n_calls"]
    jax.clear_caches()


@pytest.mark.parametrize("coarse_q", [1, 2])
def test_face_polynomial_transfer_is_an_isometry(coarse_q):
    transfer = make_face_polynomial_transfer(8, 4, coarse_q)
    rng = np.random.default_rng(18)
    x = jnp.asarray(rng.normal(size=transfer.n_fine))
    y = jnp.asarray(rng.normal(size=transfer.n_coarse))
    np.testing.assert_allclose(
        jnp.vdot(x, transfer.prolong(y)),
        jnp.vdot(transfer.restrict(x), y),
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        transfer.restrict(transfer.prolong(y)), y, rtol=1e-13, atol=1e-13
    )


def test_parent_face_transfer_is_an_isometry():
    domain, *_ = _setup(L=2)
    transfer = make_parent_face_transfer(domain.root, 2)
    rng = np.random.default_rng(19)
    y = jnp.asarray(rng.normal(size=transfer.n_coarse))
    np.testing.assert_allclose(
        transfer.restrict(transfer.prolong(y)), y, rtol=1e-13, atol=1e-13
    )


@pytest.mark.parametrize("kind", ["coarse", "multilevel"])
def test_shifted_hierarchy_preconditions_without_flat_materialization(
    monkeypatch, kind
):
    domain, T, _, maps, S, D, eta = _setup(p=6, q=2)
    _, T_shift, _, _ = local_solve_stage_uniform_3D_ItI(
        _shifted_problem(domain, 0.1, eta)
    )
    apply_S, apply_D = make_dense_SD_apply(S, D)
    A = jax.jit(make_flat_bie_operator(T, maps, eta, apply_S, apply_D))
    A_shift = make_flat_bie_operator(T_shift, maps, eta, apply_S, apply_D)
    diagonal = flat_bie_diagonal_approx(
        T_shift, maps, eta, jnp.diag(S), jnp.diag(D)
    )
    dense_sizes = []
    original_materialize = materialize

    def coarse_only(operator, n):
        dense_sizes.append(n)
        assert n < maps.n_flat
        return original_materialize(operator, n)

    monkeypatch.setattr("jaxhps._matfree_precond_3D.materialize", coarse_only)
    stats = {}
    if kind == "coarse":
        M = make_coarse_correction_preconditioner(
            A_shift,
            diagonal,
            maps,
            2,
            coarse_q=1,
            T_leaves=T_shift,
            stats=stats,
        )
    else:
        M = make_multilevel_shifted_preconditioner(
            A_shift,
            diagonal,
            maps,
            2,
            domain.root,
            1,
            coarse_q=1,
            T_leaves=T_shift,
            stats=stats,
        )

    assert dense_sizes == [stats["coarse_dim"]]
    rng = np.random.default_rng(20)
    exact = jnp.asarray(
        rng.normal(size=maps.n_flat) + 1j * rng.normal(size=maps.n_flat)
    )
    b = A(exact)
    x, info = fgmres(
        A, b, precond=M, tol=1e-8, restart=maps.n_flat, maxiter=maps.n_flat
    )
    assert info == 0
    assert float(jnp.linalg.norm(b - A(x)) / jnp.linalg.norm(b)) <= 1e-8
    np.testing.assert_allclose(x, exact, rtol=2e-5, atol=2e-5)
    assert stats["n_calls"] > 0
    assert stats["coarse_dim"] < maps.n_flat
    jax.clear_caches()


def test_polynomial_levels_preserve_the_nested_coarse_space():
    first = make_face_polynomial_transfer(8, 8, 4)
    second = make_face_polynomial_transfer(8, 4, 2, coefficient_space=True)
    direct = make_face_polynomial_transfer(8, 8, 2)
    rng = np.random.default_rng(24)
    y = jnp.asarray(rng.normal(size=(direct.n_coarse, 2)))
    np.testing.assert_allclose(
        first.prolong(second.prolong(y)), direct.prolong(y), atol=1e-13
    )


def test_face_pair_smoother_matches_the_block_diagonal_solve():
    _, T, _, maps, S, D, eta = _setup(p=6, q=2, shift=0.1)
    apply_S, apply_D = make_dense_SD_apply(S, D)
    A = np.asarray(
        materialize(
            make_flat_bie_operator(T, maps, eta, apply_S, apply_D), maps.n_flat
        )
    )
    diagonal = flat_bie_diagonal_approx(T, maps, eta, jnp.diag(S), jnp.diag(D))
    blocks = np.diag(np.asarray(diagonal)).copy()
    partner = np.asarray(maps.partner)
    for face in range(maps.n_flat // 4):
        first = np.arange(4 * face, 4 * face + 4)
        if partner[first[0]] < 0 or first[0] > partner[first[0]]:
            continue
        rows = np.concatenate([first, partner[first]])
        blocks[np.ix_(rows, rows)] = A[np.ix_(rows, rows)]
    rng = np.random.default_rng(25)
    r = jnp.asarray(
        rng.normal(size=(maps.n_flat, 2))
        + 1j * rng.normal(size=(maps.n_flat, 2))
    )
    smoother = make_face_pair_smoother(diagonal, T, maps.partner, omega=1)
    np.testing.assert_allclose(
        smoother(r), np.linalg.solve(blocks, r), rtol=1e-12, atol=1e-12
    )
    jax.clear_caches()


def test_multilevel_octree_reduction_respects_the_dense_size_limit(
    monkeypatch,
):
    root = DiscretizationNode3D(
        xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1
    )
    domain = Domain(p=4, q=2, root=root, L=2)
    maps = build_interface_maps(domain)
    d = jnp.linspace(2.0, 3.0, maps.n_flat).astype(jnp.complex128)

    def operator(x):
        diag_x = d * x if x.ndim == 1 else d[:, None] * x
        return diag_x + 0.1 * (
            jnp.roll(x, 1, axis=0) + jnp.roll(x, -1, axis=0)
        )

    dense_sizes = []

    def coarse_only(op, n):
        dense_sizes.append(n)
        assert n <= 6
        return materialize(op, n)

    monkeypatch.setattr("jaxhps._matfree_precond_3D.materialize", coarse_only)
    stats = {}
    M = make_multilevel_shifted_preconditioner(
        operator, d, maps, 2, root, 2, coarse_q=1, direct_limit=6, stats=stats
    )
    rng = np.random.default_rng(26)
    exact = jnp.asarray(rng.normal(size=(maps.n_flat, 2)))
    b = operator(exact)
    solution, info = fgmres(
        operator, b, precond=M, tol=1e-10, restart=30, maxiter=30
    )
    assert info == 0
    np.testing.assert_allclose(solution, exact, rtol=1e-8, atol=1e-8)
    assert dense_sizes == [6]
    assert stats["n_levels"] == 4
    jax.clear_caches()


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
