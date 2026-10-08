r"""Preconditioners for the flat leaf-interface system.

On interior rows, :math:`(Az)_i = z_i + (T^{(\ell')}z^{(\ell')})_{partner(i)}`
with :math:`\ell' \ne \ell(i)`. Thus the leaf-block diagonal is the identity;
Jacobi scaling only affects boundary rows. Nearly unitary lossless leaf maps
can produce an interface spectrum near :math:`|\lambda-1|=1`, reaching the
origin. In the measured cases, refinement fills this annulus more densely
and increases GMRES iterations despite modest condition numbers.

* :func:`make_sweep_preconditioner` applies triangular Gauss-Seidel solves
  in diagonal leaf orderings without factorization. In the measured coupled
  systems, sweeps did not cluster the spectrum or improve convergence; the
  associated stationary iteration had spectral radius above one.
* :func:`make_shifted_operator_preconditioner` inverts the flat operator for
  :math:`\kappa^2 \to \kappa^2(1+i\varepsilon)`. The measured damped solves
  reduced iterations by an order of magnitude, independently of ``q`` in
  that range. This implementation uses dense LU: it is a diagnostic, not a
  scalable solver.

The exterior BIE couples all boundary rows. Sweeps approximate these rows
using :func:`jaxhps._matfree_iti_3D.flat_bie_diagonal_approx`.
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
    lower = np.array([root.xmin, root.ymin, root.zmin])
    upper = np.array([root.xmax, root.ymax, root.zmax])
    spacing = (upper - lower) / 2**L
    return np.rint((bounds[:, ::2] - lower) / spacing).astype(np.int64)


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
    partner = maps.partner.reshape(n_leaves, n_per_leaf)
    interior = partner >= 0
    p_safe = jnp.where(interior, partner, 0)
    p_leaf, p_node = p_safe // n_per_leaf, p_safe % n_per_leaf
    order_j = jnp.asarray(order)
    dinv = jnp.asarray(diag_inv).reshape(n_leaves, n_per_leaf)

    def sweep(r: jax.Array) -> jax.Array:
        batched = r.ndim == 2
        R = r.reshape(n_leaves, n_per_leaf, *r.shape[1:])
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
        return jax.scipy.linalg.lu_solve((lu, piv), r)

    return apply
