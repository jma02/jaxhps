r"""Preconditioners for the flat leaf-interface system, and what they do.

The flat operator of :mod:`jaxhps._matfree_iti_3D` has a very particular
structure: on an interior row the only diagonal entry is the node's own trace,
and *all* coupling leaves the leaf,

.. math::
   (A z)_i = z_i + \big(T^{(\ell')} z^{(\ell')}\big)_{{\rm partner}(i)},
   \qquad \ell' \neq \ell(i),

so, grouping the unknowns by leaf, :math:`A = I + N` with :math:`N` purely
off-block-diagonal on the interior rows: the block diagonal is the identity and
diagonal scaling can only ever act through the boundary rows. Since the leaf
ItI maps are close to unitary for real :math:`\kappa` and real coefficients,
the eigenvalues of :math:`I + N` sit near the circle :math:`|\lambda - 1| = 1`,
which passes through the origin. That is the observed obstruction: the
Jacobi-preconditioned operator is *well conditioned* yet slow, because its
spectrum fills an annulus around the origin, and refinement adds interface
modes that fill it more densely -- iteration counts grow with :math:`q` at
fixed :math:`\kappa`.

Two families are provided, with very different behaviour:

* :func:`make_sweep_preconditioner` -- Gauss-Seidel sweeps over the leaves in a
  diagonal ordering, applying :math:`(I + N_<)^{-1}` by forward substitution at
  the cost of one matvec per sweep and no factorization (the impedance
  double-sweep idea, free here because the HPS interface unknowns already *are*
  impedance traces). Measured on the real coupled system it does **not** help:
  the associated stationary iteration has spectral radius above one, and the
  spectrum is not clustered by any triangular part.
* :func:`make_shifted_operator_preconditioner` -- invert a *damped* copy of the
  same flat operator, built from leaf ItI maps of the complex-shifted problem
  :math:`\kappa^2 \to \kappa^2 (1 + i\varepsilon)`. Damping pulls the interface
  spectrum off the origin and clusters the preconditioned spectrum, giving
  iteration counts an order of magnitude smaller and, in the measured range,
  independent of :math:`q`. Its cost is a solve with the damped operator, which
  here is a dense factorization and therefore a diagnostic rather than a
  scalable solver.

Boundary rows are closed by the exterior integral equation, whose row is dense
over the whole domain boundary; the sweeps treat them by their approximate
diagonal (:func:`jaxhps._matfree_iti_3D.flat_bie_diagonal_approx`).
"""

from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ._matfree_iti_3D import (
    InterfaceMaps,
    flat_bie_diagonal_approx,
    leaf_bounds_uniform_3D,
    make_flat_bie_operator,
    materialize,
)
from ._discretization_tree import DiscretizationNode3D

# Sweep directions: the 8 diagonal orderings of a uniform octree, labelled by
# the signs applied to the (x, y, z) leaf indices before sorting.
DIAGONAL_SIGNS = [
    (sx, sy, sz) for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)
]


def leaf_index_grid(root: DiscretizationNode3D, L: int) -> np.ndarray:
    """Integer ``(i, j, k)`` cell index of every leaf, in local-solve order.

    Returns an array of shape ``(8**L, 3)``; ``i`` increases with ``x``.
    """
    bounds = leaf_bounds_uniform_3D(root, L)
    n = 2**L
    hx = (root.xmax - root.xmin) / n
    hy = (root.ymax - root.ymin) / n
    hz = (root.zmax - root.zmin) / n
    i = np.rint((bounds[:, 0] - root.xmin) / hx).astype(np.int64)
    j = np.rint((bounds[:, 2] - root.ymin) / hy).astype(np.int64)
    k = np.rint((bounds[:, 4] - root.zmin) / hz).astype(np.int64)
    return np.stack([i, j, k], axis=-1)


def sweep_order(
    root: DiscretizationNode3D, L: int, signs: Tuple[int, int, int]
) -> np.ndarray:
    """Leaf visiting order for one diagonal sweep direction.

    ``signs`` flips the sort key per axis, so ``(1, 1, 1)`` sweeps from the
    ``(xmin, ymin, zmin)`` corner and ``(-1, -1, -1)`` is its reverse.
    """
    idx = leaf_index_grid(root, L)
    n = 2**L
    key = np.zeros(idx.shape[0], dtype=np.int64)
    for axis, s in enumerate(signs):
        a = idx[:, axis] if s > 0 else (n - 1 - idx[:, axis])
        key = key * n + a
    return np.argsort(key, kind="stable").astype(np.int32)


def sweep_orders(
    root: DiscretizationNode3D,
    L: int,
    directions: Sequence[Tuple[int, int, int]],
) -> np.ndarray:
    """Stack of leaf orderings, shape ``(len(directions), 8**L)``."""
    return np.stack([sweep_order(root, L, s) for s in directions], axis=0)


def _leafwise_maps(maps: InterfaceMaps):
    partner = np.asarray(maps.partner).reshape(maps.n_leaves, maps.n_per_leaf)
    interior = partner >= 0
    p_safe = np.where(interior, partner, 0)
    return (
        jnp.asarray(p_safe // maps.n_per_leaf),
        jnp.asarray(p_safe % maps.n_per_leaf),
        jnp.asarray(interior),
    )


def make_gauss_seidel_sweep(
    T_leaves: jax.Array,
    maps: InterfaceMaps,
    order: np.ndarray,
    diag_inv: jax.Array,
) -> Callable[[jax.Array], jax.Array]:
    r"""One Gauss-Seidel sweep over the leaves, i.e. apply :math:`(I+N_<)^{-1}`.

    Parameters
    ----------
    order : np.ndarray
        Leaf visiting order, e.g. from :func:`sweep_order`.
    diag_inv : jax.Array
        Reciprocal diagonal used on the boundary rows, shape ``(n_flat,)``;
        interior entries are ignored (their diagonal is exactly 1).

    Returns a callable acting on flat vectors of shape ``(n_flat,)``.
    """
    n_leaves, n_per_leaf = maps.n_leaves, maps.n_per_leaf
    p_leaf, p_node, interior = _leafwise_maps(maps)
    order_j = jnp.asarray(order)
    dinv = jnp.asarray(diag_inv).reshape(n_leaves, n_per_leaf)

    def sweep(r: jax.Array) -> jax.Array:
        batched = r.ndim == 2
        R = (
            r.reshape(n_leaves, n_per_leaf, -1)
            if batched
            else r.reshape(n_leaves, n_per_leaf)
        )
        z0 = jnp.zeros_like(R)
        g0 = jnp.zeros_like(R)
        mask = interior[..., None] if batched else interior
        dscale = dinv[..., None] if batched else dinv

        def body(step, carry):
            z, g_out = carry
            ell = order_j[step]
            gathered = g_out[p_leaf[ell], p_node[ell]]
            z_l = jnp.where(mask[ell], R[ell] - gathered, R[ell] * dscale[ell])
            g_l = T_leaves[ell] @ z_l
            return z.at[ell].set(z_l), g_out.at[ell].set(g_l)

        z, _ = jax.lax.fori_loop(0, n_leaves, body, (z0, g0))
        return z.reshape(r.shape)

    return sweep


def make_sweep_preconditioner(
    T_leaves: jax.Array,
    maps: InterfaceMaps,
    eta: float,
    S_diag: jax.Array,
    D_diag: jax.Array,
    root: DiscretizationNode3D,
    L: int,
    directions: Optional[Sequence[Tuple[int, int, int]]] = None,
    A_matvec: Optional[Callable[[jax.Array], jax.Array]] = None,
) -> Callable[[jax.Array], jax.Array]:
    r"""Multi-directional Gauss-Seidel sweep preconditioner.

    With one direction this is a plain forward substitution. With several, the
    sweeps are composed multiplicatively on the running residual,

    .. math::
       z \mathrel{+}= (I+N_<^{(d)})^{-1}\,(r - A z),
       \qquad d = 1, \dots, n_{\rm dir},

    which needs ``A_matvec`` for the residual updates: the total cost is
    ``2 n_dir - 1`` matvec-equivalents per application (``n_dir`` sweeps plus
    ``n_dir - 1`` residual matvecs).

    ``directions`` defaults to the forward and reverse diagonal sweeps
    ``(1,1,1)`` and ``(-1,-1,-1)``; pass :data:`DIAGONAL_SIGNS` for all eight.
    """
    if directions is None:
        directions = [(1, 1, 1), (-1, -1, -1)]
    if len(directions) > 1 and A_matvec is None:
        raise ValueError(
            "multi-directional sweeps need A_matvec for the residual updates"
        )
    d = flat_bie_diagonal_approx(T_leaves, maps, eta, S_diag, D_diag)
    diag_inv = 1.0 / d
    sweeps = [
        make_gauss_seidel_sweep(
            T_leaves, maps, sweep_order(root, L, s), diag_inv
        )
        for s in directions
    ]

    def apply(r: jax.Array) -> jax.Array:
        z = sweeps[0](r)
        for s in sweeps[1:]:
            z = z + s(r - A_matvec(z))
        return z

    return apply


def sweep_cost_matvecs(n_directions: int) -> int:
    """Matvec-equivalent cost of one preconditioner application."""
    return max(1, 2 * n_directions - 1)


def make_shifted_operator_preconditioner(
    T_shift: jax.Array,
    maps: InterfaceMaps,
    eta: float,
    apply_S: Callable[[jax.Array], jax.Array],
    apply_D: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array], jax.Array]:
    r"""Invert the flat operator of a complex-shifted (damped) problem.

    ``T_shift`` are the leaf ItI maps of the same domain and discretization but
    with :math:`\kappa^2` replaced by :math:`\kappa^2 (1 + i\varepsilon)`; the
    exterior operators are reused unchanged, since the shift is only there to
    damp the interior interface coupling.

    .. warning::
       The damped operator is materialized and factored densely, which costs
       :math:`O(n_{\rm flat}^2)` memory: this is a spectral diagnostic and a
       reference for how much a damped-operator preconditioner can buy, not a
       scalable preconditioner. Making it scalable means replacing the dense
       factorization by an approximate damped solve -- the damped interface
       coupling decays away from the diagonal, which is what a localized or
       hierarchically compressed solve would exploit.
    """
    A_shift = materialize(
        make_flat_bie_operator(T_shift, maps, eta, apply_S, apply_D),
        maps.n_flat,
    )
    lu, piv = jax.scipy.linalg.lu_factor(A_shift)

    def apply(r: jax.Array) -> jax.Array:
        if r.ndim == 1:
            return jax.scipy.linalg.lu_solve((lu, piv), r)
        return jnp.stack(
            [
                jax.scipy.linalg.lu_solve((lu, piv), r[:, j])
                for j in range(r.shape[1])
            ],
            axis=1,
        )

    return apply
