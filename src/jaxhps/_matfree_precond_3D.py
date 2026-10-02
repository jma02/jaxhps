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

from dataclasses import dataclass
import time
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
from ._krylov import fgmres

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


def make_iterative_shifted_preconditioner(
    T_shift: jax.Array,
    maps: InterfaceMaps,
    eta: float,
    apply_S: Callable[[jax.Array], jax.Array],
    apply_D: Callable[[jax.Array], jax.Array],
    S_diag: jax.Array,
    D_diag: jax.Array,
    *,
    tol: float = 0.1,
    restart: int = 30,
    maxiter: int = 60,
    stats: Optional[dict] = None,
) -> Callable[[jax.Array], jax.Array]:
    r"""Approximately solve the shifted flat system without materializing it.

    Each application starts from zero and runs Jacobi-preconditioned GMRES,
    stopping on the true shifted residual or the iteration budget. This is
    a nonlinear map even with a fixed iteration budget: the outer solver
    must be flexible, e.g. :func:`jaxhps._krylov.fgmres`.

    Additional persistent storage is one shifted set of leaf ItI maps and
    a diagonal. Inner Krylov storage is ``O(n_flat * restart)`` per RHS;
    exterior storage depends on the supplied ``apply_S`` and ``apply_D``.
    ``stats`` accumulates inner work and unmet tolerances across calls.
    No frequency-independent convergence or cost reduction is assumed.
    """
    if not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("tol must be finite and between zero and one")
    if restart < 1 or maxiter < 1:
        raise ValueError("restart and maxiter must be positive")
    A_shift = jax.jit(
        make_flat_bie_operator(T_shift, maps, eta, apply_S, apply_D)
    )
    d_inv = 1.0 / flat_bie_diagonal_approx(T_shift, maps, eta, S_diag, D_diag)
    if stats is None:
        stats = {}
    stats.update(
        n_calls=0, n_matvec=0, n_iter=0, n_unconverged=0, rel_res_history=[]
    )

    def jacobi(v):
        return d_inv * v if v.ndim == 1 else d_inv[:, None] * v

    def apply(r: jax.Array) -> jax.Array:
        inner_stats = {}
        z, info = fgmres(
            A_shift,
            r,
            precond=jacobi,
            tol=tol,
            restart=restart,
            maxiter=maxiter,
            stats=inner_stats,
        )
        stats["n_calls"] += 1
        stats["n_matvec"] += inner_stats["n_matvec"]
        stats["n_iter"] += inner_stats["n_iter"]
        stats["n_unconverged"] += int(info != 0)
        stats["rel_res_history"].append(inner_stats["final_rel_res"])
        return z

    return apply


@dataclass(frozen=True)
class GalerkinTransfer:
    """Matrix-free prolongation/restriction between two trace spaces."""

    n_fine: int
    n_coarse: int
    prolong: Callable[[jax.Array], jax.Array]
    restrict: Callable[[jax.Array], jax.Array]
    restrict_diagonal: Callable[[jax.Array], jax.Array]


def make_face_polynomial_transfer(
    n_leaves: int, q: int, coarse_q: int, *, coefficient_space: bool = False
) -> GalerkinTransfer:
    """Tensor-product polynomial transfer on every leaf face."""
    if not 1 <= coarse_q < q:
        raise ValueError("coarse_q must be positive and smaller than q")
    if coefficient_space:
        basis = jnp.eye(q)[:, :coarse_q]
    else:
        nodes = np.polynomial.legendre.leggauss(q)[0]
        basis = np.polynomial.legendre.legvander(nodes, coarse_q - 1)
        basis = jnp.asarray(np.linalg.qr(basis)[0])
    n_fine = n_leaves * 6 * q**2
    n_coarse = n_leaves * 6 * coarse_q**2

    def prolong(y: jax.Array) -> jax.Array:
        batched = y.ndim == 2
        Y = y.reshape(n_leaves, 6, coarse_q, coarse_q, -1)
        X = jnp.einsum("ia,lfabm,jb->lfijm", basis, Y, basis)
        out = X.reshape(n_fine, -1)
        return out if batched else out[:, 0]

    def restrict(x: jax.Array) -> jax.Array:
        batched = x.ndim == 2
        X = x.reshape(n_leaves, 6, q, q, -1)
        Y = jnp.einsum("ia,lfijm,jb->lfabm", basis.conj(), X, basis.conj())
        out = Y.reshape(n_coarse, -1)
        return out if batched else out[:, 0]

    basis2 = jnp.abs(basis) ** 2

    def restrict_diagonal(d: jax.Array) -> jax.Array:
        D = d.reshape(n_leaves, 6, q, q)
        return jnp.einsum("ia,lfij,jb->lfab", basis2, D, basis2).ravel()

    return GalerkinTransfer(
        n_fine, n_coarse, prolong, restrict, restrict_diagonal
    )


def make_parent_face_transfer(
    root: DiscretizationNode3D, L: int
) -> GalerkinTransfer:
    """Aggregate constant exterior child faces onto their parent faces."""
    if L < 1:
        raise ValueError("L must be positive")
    idx = leaf_index_grid(root, L)
    parent_grid = idx // 2
    parent_lookup = {
        tuple(cell): leaf
        for leaf, cell in enumerate(leaf_index_grid(root, L - 1))
    }
    parent_leaf = np.array(
        [parent_lookup[tuple(cell)] for cell in parent_grid]
    )
    local = idx % 2
    n_leaves = idx.shape[0]
    fine_to_coarse = np.full((n_leaves, 6), -1, dtype=np.int32)
    for face, (axis, side) in enumerate(
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))
    ):
        use = local[:, axis] == side
        fine_to_coarse[use, face] = parent_leaf[use] * 6 + face
    fine_to_coarse = fine_to_coarse.ravel()
    valid = fine_to_coarse >= 0
    valid_rows = jnp.asarray(np.flatnonzero(valid), dtype=jnp.int32)
    coarse_rows = jnp.asarray(fine_to_coarse[valid], dtype=jnp.int32)
    gather = jnp.asarray(np.where(valid, fine_to_coarse, 0), dtype=jnp.int32)
    mask = jnp.asarray(valid)
    n_fine = n_leaves * 6
    n_coarse = (n_leaves // 8) * 6

    def prolong(y: jax.Array) -> jax.Array:
        values = y[gather] * 0.5
        return (
            jnp.where(mask[:, None], values, 0)
            if y.ndim == 2
            else jnp.where(mask, values, 0)
        )

    def restrict(x: jax.Array) -> jax.Array:
        if x.ndim == 1:
            return (
                jnp.zeros(n_coarse, dtype=x.dtype)
                .at[coarse_rows]
                .add(0.5 * x[valid_rows])
            )
        return (
            jnp.zeros((n_coarse, x.shape[1]), dtype=x.dtype)
            .at[coarse_rows]
            .add(0.5 * x[valid_rows])
        )

    def restrict_diagonal(d: jax.Array) -> jax.Array:
        return (
            jnp.zeros(n_coarse, dtype=d.dtype)
            .at[coarse_rows]
            .add(0.25 * d[valid_rows])
        )

    return GalerkinTransfer(
        n_fine, n_coarse, prolong, restrict, restrict_diagonal
    )


def _galerkin_operator(
    operator: Callable[[jax.Array], jax.Array], transfer: GalerkinTransfer
) -> Callable[[jax.Array], jax.Array]:
    return lambda x: transfer.restrict(operator(transfer.prolong(x)))


def _factorized_solver(operator: Callable, n: int) -> Callable:
    A = materialize(operator, n)
    lu, piv = jax.scipy.linalg.lu_factor(A)

    @jax.jit
    def solve(rhs):
        return jax.scipy.linalg.lu_solve((lu, piv), rhs)

    return solve


def make_interface_pair_smoother(
    diagonal: jax.Array,
    outgoing_diagonal: jax.Array,
    partner: jax.Array,
    *,
    omega: float = 0.6,
) -> Callable[[jax.Array], jax.Array]:
    """Invert the 2-by-2 coupling of each paired interior trace node."""
    interior = partner >= 0
    safe = jnp.where(interior, partner, 0)
    tp = outgoing_diagonal[safe]
    denominator = jnp.where(interior, 1.0 - tp * outgoing_diagonal, diagonal)
    if bool(jnp.any(jnp.abs(denominator) < 1e-14)):
        raise ValueError("singular interface-pair smoother")

    @jax.jit
    def apply(r: jax.Array) -> jax.Array:
        if r.ndim == 1:
            coupled = jnp.where(interior, tp * r[safe], 0)
            return omega * (r - coupled) / denominator
        coupled = jnp.where(interior[:, None], tp[:, None] * r[safe], 0)
        return omega * (r - coupled) / denominator[:, None]

    return apply


def _coarsen_leaf_maps(
    T: jax.Array, q: int, coarse_q: int, *, coefficient_space: bool
) -> jax.Array:
    transfer = make_face_polynomial_transfer(
        1, q, coarse_q, coefficient_space=coefficient_space
    )
    P = transfer.prolong(jnp.eye(transfer.n_coarse, dtype=T.dtype))
    return jnp.einsum("ia,lij,jb->lab", P.conj(), T, P)


def make_face_pair_smoother(
    diagonal: jax.Array,
    T: jax.Array,
    partner: jax.Array,
    *,
    omega: float = 0.6,
    stats: Optional[dict] = None,
) -> Callable[[jax.Array], jax.Array]:
    """Block Jacobi on complete paired faces; diagonal on exterior rows."""
    n_face = T.shape[1] // 6
    host_partner = np.asarray(partner).reshape(-1, n_face)
    faces = np.arange(host_partner.shape[0])
    paired_faces = faces[
        (host_partner[:, 0] >= 0) & (faces < host_partner[:, 0] // n_face)
    ]
    first = paired_faces[:, None] * n_face + np.arange(n_face)[None, :]
    second = np.asarray(partner)[first]
    rows = jnp.asarray(np.concatenate([first, second], axis=1))
    local_first, local_second = first % T.shape[1], second % T.shape[1]
    Tf = T[
        jnp.asarray(first[:, 0] // T.shape[1])[:, None, None],
        jnp.asarray(local_first)[:, :, None],
        jnp.asarray(local_first)[:, None, :],
    ]
    Tg = T[
        jnp.asarray(second[:, 0] // T.shape[1])[:, None, None],
        jnp.asarray(local_second)[:, :, None],
        jnp.asarray(local_second)[:, None, :],
    ]
    identity = jnp.broadcast_to(jnp.eye(n_face, dtype=T.dtype), Tf.shape)
    blocks = jnp.concatenate(
        [
            jnp.concatenate([identity, Tg], axis=2),
            jnp.concatenate([Tf, identity], axis=2),
        ],
        axis=1,
    )
    lu, piv = jax.vmap(jax.scipy.linalg.lu_factor)(blocks)
    if stats is not None:
        stats["factor_bytes"] = (
            stats.get("factor_bytes", 0) + lu.nbytes + piv.nbytes
        )

    @jax.jit
    def apply(r: jax.Array) -> jax.Array:
        out = r / diagonal if r.ndim == 1 else r / diagonal[:, None]
        pair_values = jax.vmap(jax.scipy.linalg.lu_solve)((lu, piv), r[rows])
        return omega * out.at[rows].set(pair_values)

    return apply


def _partners_at_order(partner: jax.Array, q: int, coarse_q: int) -> jax.Array:
    host = np.asarray(partner).reshape(-1, q**2)
    interior = host[:, 0] >= 0
    if np.any(host[interior] % q**2 != np.arange(q**2)):
        raise ValueError(
            "polynomial transfer requires aligned face-node ordering"
        )
    faces = partner.reshape(-1, q**2)[:, 0] // q**2
    nodes = faces[:, None] * coarse_q**2 + jnp.arange(coarse_q**2)[None, :]
    return jnp.where(faces[:, None] >= 0, nodes, -1).ravel()


def _parent_partners(root: DiscretizationNode3D, L: int) -> jax.Array:
    grid = leaf_index_grid(root, L)
    lookup = {tuple(cell): leaf for leaf, cell in enumerate(grid)}
    partners = np.full((len(grid), 6), -1, dtype=np.int32)
    for leaf, cell in enumerate(grid):
        for face in range(6):
            neighbor = cell.copy()
            neighbor[face // 2] += 1 if face % 2 else -1
            neighbor_leaf = lookup.get(tuple(neighbor))
            if neighbor_leaf is not None:
                partners[leaf, face] = 6 * neighbor_leaf + (face ^ 1)
    return jnp.asarray(partners.ravel())


def make_coarse_correction_preconditioner(
    operator: Callable[[jax.Array], jax.Array],
    diagonal: jax.Array,
    maps: InterfaceMaps,
    q: int,
    *,
    coarse_q: int = 2,
    omega: float = 0.6,
    T_leaves: Optional[jax.Array] = None,
    direct_limit: int = 1024,
    stats: Optional[dict] = None,
) -> Callable[[jax.Array], jax.Array]:
    r"""Local smoothing followed by an exact face-polynomial correction.

    ``T_leaves`` enables paired-face block smoothing; otherwise use Jacobi.
    The coarse inverse is formed only below ``direct_limit`` unknowns.
    """
    transfer = make_face_polynomial_transfer(maps.n_leaves, q, coarse_q)
    if transfer.n_coarse > direct_limit:
        raise ValueError(
            "coarse system exceeds direct_limit; reduce coarse_q or use multilevel"
        )
    coarse_operator = _galerkin_operator(operator, transfer)
    coarse_solve = _factorized_solver(coarse_operator, transfer.n_coarse)
    d_inv = omega / diagonal
    smoother_stats = {}
    smoother = (
        None
        if T_leaves is None
        else make_face_pair_smoother(
            diagonal, T_leaves, maps.partner, omega=omega, stats=smoother_stats
        )
    )
    if stats is None:
        stats = {}
    stats.update(
        n_calls=0,
        n_matvec=0,
        n_levels=2,
        coarse_dim=transfer.n_coarse,
        setup_columns=transfer.n_coarse,
        coarse_factor_bytes=16 * transfer.n_coarse**2 + 4 * transfer.n_coarse,
        smoother_factor_bytes=smoother_stats.get("factor_bytes", 0),
        apply_seconds=0.0,
        coarse_solves=0,
    )

    def correction(r: jax.Array) -> jax.Array:
        smooth = (
            smoother(r)
            if smoother is not None
            else d_inv * r
            if r.ndim == 1
            else d_inv[:, None] * r
        )
        residual = r - operator(smooth)
        return smooth + transfer.prolong(
            coarse_solve(transfer.restrict(residual))
        )

    def apply(r: jax.Array) -> jax.Array:
        start = time.perf_counter()
        out = correction(r)
        jax.block_until_ready(out)
        stats["n_calls"] += 1
        stats["n_matvec"] += 1
        stats["apply_seconds"] += time.perf_counter() - start
        stats["coarse_solves"] += 1
        return out

    return apply


def make_multilevel_shifted_preconditioner(
    operator: Callable[[jax.Array], jax.Array],
    diagonal: jax.Array,
    maps: InterfaceMaps,
    q: int,
    root: DiscretizationNode3D,
    L: int,
    *,
    coarse_q: int = 2,
    omega: float = 0.6,
    direct_limit: int = 1024,
    T_leaves: Optional[jax.Array] = None,
    stats: Optional[dict] = None,
) -> Callable[[jax.Array], jax.Array]:
    r"""V-cycle with pre/post smoothing and bounded coarse LU.

    First reduce face polynomial order by factors of two. If necessary,
    truncate to constants and aggregate exterior child faces onto parents.
    Coarse operator actions use ``P^H A P`` through the fine callback.
    """
    if not 1 <= coarse_q < q:
        raise ValueError("coarse_q must be positive and smaller than q")
    if direct_limit < 6:
        raise ValueError("direct_limit must be at least 6")
    if maps.n_leaves != 8**L:
        raise ValueError("maps and octree depth are inconsistent")
    operators = [operator]
    diagonals = [jnp.asarray(diagonal)]
    leaf_maps = T_leaves
    level_leaf_maps = [leaf_maps]
    outgoing_diagonals = (
        []
        if leaf_maps is None
        else [jnp.diagonal(leaf_maps, axis1=1, axis2=2).ravel()]
    )
    partners = [maps.partner]
    transfers = []
    level_q = q
    while level_q > coarse_q:
        next_q = max(coarse_q, level_q // 2)
        coefficient_space = bool(transfers)
        transfer = make_face_polynomial_transfer(
            maps.n_leaves, level_q, next_q, coefficient_space=coefficient_space
        )
        transfers.append(transfer)
        operators.append(_galerkin_operator(operators[-1], transfer))
        diagonals.append(transfer.restrict_diagonal(diagonals[-1]))
        partners.append(_partners_at_order(partners[-1], level_q, next_q))
        if leaf_maps is not None:
            leaf_maps = _coarsen_leaf_maps(
                leaf_maps, level_q, next_q, coefficient_space=coefficient_space
            )
            outgoing_diagonals.append(
                jnp.diagonal(leaf_maps, axis1=1, axis2=2).ravel()
            )
        level_leaf_maps.append(leaf_maps)
        level_q = next_q
    if diagonals[-1].size > direct_limit and level_q > 1:
        coefficient_space = bool(transfers)
        transfer = make_face_polynomial_transfer(
            maps.n_leaves, level_q, 1, coefficient_space=coefficient_space
        )
        transfers.append(transfer)
        operators.append(_galerkin_operator(operators[-1], transfer))
        diagonals.append(transfer.restrict_diagonal(diagonals[-1]))
        partners.append(_partners_at_order(partners[-1], level_q, 1))
        if leaf_maps is not None:
            leaf_maps = _coarsen_leaf_maps(
                leaf_maps, level_q, 1, coefficient_space=coefficient_space
            )
            outgoing_diagonals.append(
                jnp.diagonal(leaf_maps, axis1=1, axis2=2).ravel()
            )
        level_leaf_maps.append(leaf_maps)
    for level in range(L, 0, -1):
        if diagonals[-1].size <= direct_limit:
            break
        transfer = make_parent_face_transfer(root, level)
        transfers.append(transfer)
        operators.append(_galerkin_operator(operators[-1], transfer))
        diagonals.append(transfer.restrict_diagonal(diagonals[-1]))
        partners.append(_parent_partners(root, level - 1))
        level_leaf_maps.append(None)
        if leaf_maps is not None:
            outgoing_diagonals.append(
                transfer.restrict_diagonal(outgoing_diagonals[-1])
            )
    coarse_solve = _factorized_solver(operators[-1], diagonals[-1].size)
    inverses = [omega / d for d in diagonals[:-1]]
    smoother_stats = {}
    smoothers = (
        []
        if T_leaves is None
        else [
            make_interface_pair_smoother(d, t, p, omega=omega)
            if T is None
            else make_face_pair_smoother(
                d, T, p, omega=omega, stats=smoother_stats
            )
            for d, t, p, T in zip(
                diagonals[:-1],
                outgoing_diagonals[:-1],
                partners[:-1],
                level_leaf_maps[:-1],
            )
        ]
    )
    if stats is None:
        stats = {}
    n_matvec_per_call = 2 * len(transfers)
    stats.update(
        n_calls=0,
        n_matvec=0,
        n_levels=len(operators),
        coarse_dim=int(diagonals[-1].size),
        setup_columns=int(diagonals[-1].size),
        coarse_factor_bytes=16 * int(diagonals[-1].size) ** 2
        + 4 * int(diagonals[-1].size),
        smoother_factor_bytes=smoother_stats.get("factor_bytes", 0),
        apply_seconds=0.0,
        coarse_solves=0,
    )

    def cycle(level: int, r: jax.Array) -> jax.Array:
        if level == len(transfers):
            return coarse_solve(r)
        d_inv = inverses[level]
        x = (
            smoothers[level](r)
            if smoothers
            else d_inv * r
            if r.ndim == 1
            else d_inv[:, None] * r
        )
        residual = r - operators[level](x)
        x = x + transfers[level].prolong(
            cycle(level + 1, transfers[level].restrict(residual))
        )
        residual = r - operators[level](x)
        return x + (
            smoothers[level](residual)
            if smoothers
            else d_inv * residual
            if r.ndim == 1
            else d_inv[:, None] * residual
        )

    def apply(r: jax.Array) -> jax.Array:
        start = time.perf_counter()
        out = cycle(0, r)
        jax.block_until_ready(out)
        stats["n_calls"] += 1
        stats["n_matvec"] += n_matvec_per_call
        stats["apply_seconds"] += time.perf_counter() - start
        stats["coarse_solves"] += 1
        return out

    return apply
