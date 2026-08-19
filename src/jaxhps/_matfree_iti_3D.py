r"""Matrix-free interior HPS response for 3D uniform ItI discretizations.

The standard pipeline (:func:`jaxhps.build_solver` with ``return_top_T=True``)
merges the per-leaf impedance-to-impedance (ItI) maps up to the root and
returns a dense operator of size :math:`6 (2^L q)^2`, which the exterior
boundary-integral coupling then turns into a Dirichlet-to-Neumann map. That
root operator is the memory bottleneck of the coupled solver.

This module keeps the hierarchy unmerged. The leaf ItI maps
:math:`T^{(\ell)}` are used directly in one flat linear system for the
per-leaf incoming impedance traces

.. math::
   z^{(\ell)} = u_n^{(\ell)} + i \eta u \quad \text{on } \partial \tau_\ell,

where :math:`u_n^{(\ell)}` uses the outward normal of leaf
:math:`\tau_\ell`. Two nodes of neighboring leaves at the same physical point
carry opposite outward normals, so continuity of :math:`u` and of the flux
gives the gluing condition

.. math::
   z^{(\ell)}(x) + g_{\rm out}^{(\ell')}(x) = 0,
   \qquad
   g_{\rm out}^{(\ell')} = T^{(\ell')} z^{(\ell')} + h^{(\ell')},

one equation per interior interface node per side. Nodes on
:math:`\partial\Omega` are closed either by prescribing the incoming
impedance (giving the action of the root ItI map without forming it, see
:func:`root_ItI_matvec`) or by the exterior boundary-integral equation (see
:func:`make_flat_bie_operator`), which is what the coupled scattering solver
needs.

Memory: the flat system stores the leaf ItI blocks,
:math:`8^L (6q^2)^2` entries, versus :math:`(6 \cdot 4^L q^2)^2` for the root
operator -- a factor :math:`4.5 \cdot 2^L` fewer entries -- and neither the
intermediate merge operators nor the dense root DtN map is ever built.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from ._discretization_tree import DiscretizationNode3D
from ._domain import Domain
from ._grid_creation_3D import (
    bounds_for_oct_subdivision,
    compute_boundary_Gauss_points_uniform_3D,
)

# Outward normal of each of the 6 leaf faces, in the face order used by the
# local solve stage (x=xmin, x=xmax, y=ymin, y=ymax, z=zmin, z=zmax).
FACE_NORMALS = np.array(
    [
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
    ]
)


def leaf_bounds_uniform_3D(root: DiscretizationNode3D, L: int) -> np.ndarray:
    """Bounding boxes of the uniform-octree leaves, in local-solve order.

    Returns an array of shape ``(8**L, 6)`` with rows
    ``(xmin, xmax, ymin, ymax, zmin, zmax)``, ordered like the leaf axis of
    the arrays returned by the local solve stage.
    """
    bounds = jnp.array(
        [[root.xmin, root.xmax, root.ymin, root.ymax, root.zmin, root.zmax]]
    )
    for _ in range(L):
        bounds = jax.vmap(bounds_for_oct_subdivision)(bounds).reshape(-1, 6)
    return np.asarray(bounds)


def leaf_boundary_gauss_points(
    root: DiscretizationNode3D, L: int, q: int
) -> np.ndarray:
    """Gauss boundary nodes of every leaf, matching the leaf ItI ordering.

    Returns an array of shape ``(8**L, 6 q**2, 3)``. Within a leaf the nodes
    are ordered face by face exactly like the rows and columns of the leaf
    ItI matrix from
    :func:`jaxhps.local_solve.local_solve_stage_uniform_3D_ItI`.
    """
    bounds = leaf_bounds_uniform_3D(root, L)
    out = np.empty((bounds.shape[0], 6 * q**2, 3))
    for i, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(bounds):
        leaf = DiscretizationNode3D(
            xmin=float(xmin),
            xmax=float(xmax),
            ymin=float(ymin),
            ymax=float(ymax),
            zmin=float(zmin),
            zmax=float(zmax),
        )
        out[i] = np.asarray(
            compute_boundary_Gauss_points_uniform_3D(leaf, 0, q)
        )
    return out


def leaf_boundary_normals(q: int) -> np.ndarray:
    """Outward normals at the ``6 q**2`` boundary nodes of one leaf."""
    return np.repeat(FACE_NORMALS, q**2, axis=0)


def leaf_boundary_quad_weights(
    root: DiscretizationNode3D, L: int, q: int
) -> np.ndarray:
    """Surface quadrature weights at the ``6 q**2`` nodes of one leaf.

    Tensor-product Gauss-Legendre weights on each face, in the node ordering
    of :func:`leaf_boundary_gauss_points`. All leaves of a uniform octree are
    congruent, so one set of weights applies to every leaf.
    """
    gw = np.polynomial.legendre.leggauss(q)[1]
    hx = (root.xmax - root.xmin) / 2**L
    hy = (root.ymax - root.ymin) / 2**L
    hz = (root.zmax - root.zmin) / 2**L
    face_scales = [
        (hy / 2) * (hz / 2),
        (hy / 2) * (hz / 2),
        (hx / 2) * (hz / 2),
        (hx / 2) * (hz / 2),
        (hx / 2) * (hy / 2),
        (hx / 2) * (hy / 2),
    ]
    w_face = np.outer(gw, gw).ravel()
    return np.concatenate([s * w_face for s in face_scales])


@dataclass
class InterfaceMaps:
    """Index bookkeeping for the flat (unmerged) leaf interface system.

    Flat indexing is ``i = leaf * n_per_leaf + node`` with
    ``n_per_leaf = 6 q**2``.

    Attributes
    ----------
    partner : jax.Array
        Shape ``(N,)``. ``partner[i]`` is the flat index of the node of the
        neighboring leaf at the same physical point, or ``-1`` if node ``i``
        lies on the domain boundary.
    is_interior : jax.Array
        Shape ``(N,)`` bool, ``True`` where ``partner >= 0``.
    bdry_rows : jax.Array
        Shape ``(n_bdry,)``. ``bdry_rows[m]`` is the flat index of the leaf
        node coinciding with the ``m``-th node of ``Domain.boundary_points``
        (flattened), so boundary data in domain ordering maps to flat rows
        without further permutation.
    n_leaves : int
    n_per_leaf : int
    """

    partner: jax.Array
    is_interior: jax.Array
    bdry_rows: jax.Array
    n_leaves: int
    n_per_leaf: int

    @property
    def n_flat(self) -> int:
        return self.n_leaves * self.n_per_leaf


def build_interface_maps(domain: Domain, atol: float = 1e-10) -> InterfaceMaps:
    """Pair up coincident leaf boundary nodes of a uniform 3D octree.

    Every Gauss node of a leaf face either coincides with exactly one node of
    a neighboring leaf (interior interface) or lies on the domain boundary.
    Gauss nodes are strictly interior to their face, so the pairing is
    one-to-one and no edge or corner bookkeeping is needed.
    """
    q = domain.q
    L = domain.L
    pts = leaf_boundary_gauss_points(domain.root, L, q)
    n_leaves, n_per_leaf = pts.shape[0], pts.shape[1]
    flat = pts.reshape(-1, 3)

    tree = cKDTree(flat)
    pairs = tree.query_pairs(r=atol, output_type="ndarray")
    partner = np.full(flat.shape[0], -1, dtype=np.int32)
    if pairs.size:
        same_leaf = (pairs[:, 0] // n_per_leaf) == (pairs[:, 1] // n_per_leaf)
        if same_leaf.any():
            raise ValueError(
                "found coincident boundary nodes within a single leaf; "
                "check the Gauss node layout"
            )
        counts = np.bincount(pairs.ravel(), minlength=flat.shape[0])
        if counts.max() > 1:
            raise ValueError(
                "a leaf boundary node matched more than one partner; "
                f"tolerance atol={atol:g} may be too large"
            )
        partner[pairs[:, 0]] = pairs[:, 1]
        partner[pairs[:, 1]] = pairs[:, 0]

    bdry_flat = np.flatnonzero(partner < 0)
    domain_bdry = np.asarray(domain.boundary_points).reshape(-1, 3)
    if bdry_flat.shape[0] != domain_bdry.shape[0]:
        raise ValueError(
            f"expected {domain_bdry.shape[0]} unmatched (boundary) leaf "
            f"nodes, found {bdry_flat.shape[0]}"
        )
    dist, idx = cKDTree(flat[bdry_flat]).query(domain_bdry)
    if dist.max() > atol or np.unique(idx).size != idx.size:
        raise ValueError(
            "could not match Domain.boundary_points to leaf boundary nodes "
            f"(max distance {dist.max():.3e})"
        )
    bdry_rows = bdry_flat[idx].astype(np.int32)

    return InterfaceMaps(
        partner=jnp.asarray(partner),
        is_interior=jnp.asarray(partner >= 0),
        bdry_rows=jnp.asarray(bdry_rows),
        n_leaves=int(n_leaves),
        n_per_leaf=int(n_per_leaf),
    )


def leaf_outgoing(T_leaves: jax.Array, z: jax.Array) -> jax.Array:
    """Apply the block-diagonal leaf ItI operator to flat incoming traces.

    Parameters
    ----------
    T_leaves : jax.Array
        Shape ``(n_leaves, n_per_leaf, n_per_leaf)``.
    z : jax.Array
        Shape ``(N,)`` or ``(N, n_src)``, ``N = n_leaves * n_per_leaf``.

    Returns
    -------
    jax.Array
        Same shape as ``z``.
    """
    n_leaves, n_per_leaf = T_leaves.shape[0], T_leaves.shape[1]
    squeeze = z.ndim == 1
    Z = z.reshape(n_leaves, n_per_leaf, -1)
    W = jnp.einsum("lij,ljk->lik", T_leaves, Z)
    W = W.reshape(n_leaves * n_per_leaf, -1)
    return W[:, 0] if squeeze else W


def _flatten_leafwise(a: jax.Array) -> jax.Array:
    """Flatten a ``(n_leaves, n_per_leaf, ...)`` array over its first 2 axes."""
    return a.reshape(a.shape[0] * a.shape[1], *a.shape[2:])


def _gluing_rows(
    z: jax.Array, g_out: jax.Array, maps: InterfaceMaps
) -> jax.Array:
    """``z_i + g_out_{partner(i)}`` on interior rows, ``0`` on boundary rows."""
    p_safe = jnp.where(maps.is_interior, maps.partner, 0)
    gathered = g_out[p_safe]
    mask = maps.is_interior
    if z.ndim == 2:
        mask = mask[:, None]
    return jnp.where(mask, z + gathered, jnp.zeros_like(z))


def interface_residual(
    T_leaves: jax.Array,
    h_leaves: Optional[jax.Array],
    maps: InterfaceMaps,
    z: jax.Array,
) -> jax.Array:
    r"""Residual of the interior gluing conditions.

    Computes :math:`z + \mathcal{P}(T z + h)`, where :math:`\mathcal{P}`
    gathers a node's partner across the interface, and is zero on boundary
    rows. It vanishes, up to discretization error, for the traces of a global
    solution of the PDE.
    """
    g_out = leaf_outgoing(T_leaves, z)
    if h_leaves is not None:
        g_out = g_out + _flatten_leafwise(jnp.asarray(h_leaves))
    return _gluing_rows(z, g_out, maps)


def make_root_ItI_operator(
    T_leaves: jax.Array, maps: InterfaceMaps
) -> Callable[[jax.Array], jax.Array]:
    """Flat operator whose boundary rows prescribe the incoming impedance.

    The returned ``matvec`` acts on flat leaf traces ``z`` with

    * interior rows ``z_i + (T z)_partner(i)``,
    * boundary rows ``z_i``,

    so solving ``matvec(z) = rhs`` with ``rhs`` from :func:`root_ItI_rhs`
    reproduces the action of the root-level ItI map without merging.
    """

    def matvec(z: jax.Array) -> jax.Array:
        g_out = leaf_outgoing(T_leaves, z)
        rows = _gluing_rows(z, g_out, maps)
        mask = maps.is_interior
        if z.ndim == 2:
            mask = mask[:, None]
        return jnp.where(mask, rows, z)

    return matvec


def root_ItI_rhs(
    h_leaves: Optional[jax.Array],
    maps: InterfaceMaps,
    g_in_bdry: jax.Array,
) -> jax.Array:
    """Right-hand side matching :func:`make_root_ItI_operator`.

    Interior rows get ``-h_partner(i)``; boundary rows get the prescribed
    incoming impedance ``g_in_bdry`` in ``Domain.boundary_points`` ordering.
    """
    n_flat = maps.n_flat
    shape = (n_flat,) if g_in_bdry.ndim == 1 else (n_flat, g_in_bdry.shape[-1])
    rhs = jnp.zeros(shape, dtype=jnp.complex128)
    if h_leaves is not None:
        h_flat = _flatten_leafwise(jnp.asarray(h_leaves))
        p_safe = jnp.where(maps.is_interior, maps.partner, 0)
        gathered = -h_flat[p_safe]
        mask = maps.is_interior
        if rhs.ndim == 2:
            mask = mask[:, None]
        rhs = jnp.where(mask, gathered, rhs)
    return rhs.at[maps.bdry_rows].set(g_in_bdry)


def boundary_traces_from_z(
    T_leaves: jax.Array,
    h_leaves: Optional[jax.Array],
    maps: InterfaceMaps,
    z: jax.Array,
    eta: float,
) -> Tuple[jax.Array, jax.Array]:
    r"""Dirichlet and Neumann traces on the domain boundary, given ``z``.

    With :math:`g_{\rm in} = u_n + i\eta u` and
    :math:`g_{\rm out} = u_n - i\eta u`,

    .. math::
       u = \frac{g_{\rm in} - g_{\rm out}}{2 i \eta},
       \qquad
       u_n = \frac{g_{\rm in} + g_{\rm out}}{2}.

    Returns ``(u, u_n)`` in ``Domain.boundary_points`` ordering.
    """
    g_out = leaf_outgoing(T_leaves, z)
    if h_leaves is not None:
        g_out = g_out + _flatten_leafwise(jnp.asarray(h_leaves))
    g_in_b = z[maps.bdry_rows]
    g_out_b = g_out[maps.bdry_rows]
    u = (g_in_b - g_out_b) / (2j * eta)
    u_n = (g_in_b + g_out_b) / 2.0
    return u, u_n


def materialize(matvec: Callable[[jax.Array], jax.Array], n: int) -> jax.Array:
    """Dense matrix of a flat operator, by applying it to the identity.

    Only meant for small verification problems and spectral diagnostics.
    """
    return matvec(jnp.eye(n, dtype=jnp.complex128))


def root_ItI_matvec(
    T_leaves: jax.Array,
    maps: InterfaceMaps,
    g_in_bdry: jax.Array,
    solve_flat: Callable[[Callable, jax.Array], jax.Array],
    h_leaves: Optional[jax.Array] = None,
    eta: Optional[float] = None,
) -> jax.Array:
    r"""Action of the root ItI map, computed without merging.

    Solves the flat interface system with the boundary incoming impedance set
    to ``g_in_bdry`` and returns the outgoing impedance on the domain
    boundary, i.e. the same quantity as ``T_root @ g_in_bdry`` plus the
    particular contribution when ``h_leaves`` is given.

    Parameters
    ----------
    solve_flat : Callable
        ``solve_flat(matvec, rhs) -> z``: any solver for the flat system,
        e.g. dense LU on the materialized operator for small problems, or a
        Krylov method.
    eta : float, optional
        Unused; accepted so callers can pass the impedance parameter
        uniformly.
    """
    matvec = make_root_ItI_operator(T_leaves, maps)
    rhs = root_ItI_rhs(h_leaves, maps, jnp.asarray(g_in_bdry))
    z = solve_flat(matvec, rhs)
    g_out = leaf_outgoing(T_leaves, z)
    if h_leaves is not None:
        g_out = g_out + _flatten_leafwise(jnp.asarray(h_leaves))
    return g_out[maps.bdry_rows]


def dense_solve_flat(
    matvec: Callable[[jax.Array], jax.Array], rhs: jax.Array
) -> jax.Array:
    """Solve the flat system by materializing it and calling a dense LU."""
    A = materialize(matvec, rhs.shape[0])
    return jnp.linalg.solve(A, rhs)


def make_flat_bie_operator(
    T_leaves: jax.Array,
    maps: InterfaceMaps,
    eta: float,
    apply_S: Callable[[jax.Array], jax.Array],
    apply_D: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array], jax.Array]:
    r"""Coupled interior/exterior operator on the flat leaf traces.

    Interior rows are the gluing conditions; boundary rows are the exterior
    integral equation of Gillman-Barnett-Martinsson,

    .. math::
       \tfrac12 u^s - D u^s + S u^s_n = 0,

    with the traces of the homogeneous part of the interior solution obtained
    from ``z`` by :func:`boundary_traces_from_z`. Together with
    :func:`flat_bie_rhs` this is the dense-DtN-free version of
    :math:`(\tfrac12 I - D + S T_{\rm int}) u^s = S(u^{\rm inc}_n -
    T_{\rm int} u^{\rm inc})`: the Dirichlet-to-Neumann map is never formed,
    and the impedance traces of the interior solution are unknowns.
    """

    def matvec(z: jax.Array) -> jax.Array:
        g_out = leaf_outgoing(T_leaves, z)
        rows = _gluing_rows(z, g_out, maps)
        g_in_b = z[maps.bdry_rows]
        g_out_b = g_out[maps.bdry_rows]
        u = (g_in_b - g_out_b) / (2j * eta)
        u_n = (g_in_b + g_out_b) / 2.0
        bdry = 0.5 * u - apply_D(u) + apply_S(u_n)
        return rows.at[maps.bdry_rows].set(bdry)

    return matvec


def flat_bie_diagonal_approx(
    T_leaves: jax.Array,
    maps: InterfaceMaps,
    eta: float,
    S_diag: jax.Array,
    D_diag: jax.Array,
) -> jax.Array:
    r"""Approximate diagonal of :func:`make_flat_bie_operator`.

    Interior (gluing) rows are exact: their diagonal is 1, since the partner
    term always belongs to a different leaf. On a boundary row, the terms kept
    are those in which the node's own trace reaches its own row directly,

    .. math::
       \frac{\partial u}{\partial z_i} = \frac{1 - T_{ii}}{2 i \eta},
       \qquad
       \frac{\partial u_n}{\partial z_i} = \frac{1 + T_{ii}}{2},

    giving :math:`(\tfrac12 - D_{kk})\partial_z u + S_{kk}\partial_z u_n`.
    Dropped are the paths through the off-diagonal of the leaf ItI map, in
    which :math:`z_i` moves the traces at the *other* boundary nodes of the
    same leaf and those reach row :math:`k` through :math:`D_{kj}, S_{kj}`;
    at ``p=6, q=4, kappa=4`` these amount to about 10% of the true diagonal.
    The result is therefore a preconditioner, not the diagonal: it becomes
    exact only when the leaf ItI blocks are diagonal.

    ``S_diag`` and ``D_diag`` are the diagonals of the boundary operators in
    ``Domain.boundary_points`` ordering, so this needs only the self-interaction
    entries of the quadrature -- no dense operator.
    """
    T_diag_leafwise = jnp.diagonal(T_leaves, axis1=1, axis2=2)
    T_diag = T_diag_leafwise.reshape(-1)
    diag = jnp.ones(maps.n_flat, dtype=jnp.complex128)
    t_b = T_diag[maps.bdry_rows]
    du = (1.0 - t_b) / (2j * eta)
    du_n = (1.0 + t_b) / 2.0
    bdry = (0.5 - jnp.asarray(D_diag)) * du + jnp.asarray(S_diag) * du_n
    return diag.at[maps.bdry_rows].set(bdry)


def flat_bie_rhs(
    h_leaves: Optional[jax.Array],
    maps: InterfaceMaps,
    eta: float,
    apply_S: Callable[[jax.Array], jax.Array],
    apply_D: Callable[[jax.Array], jax.Array],
    uin: Optional[jax.Array] = None,
    uin_dn: Optional[jax.Array] = None,
    n_src: Optional[int] = None,
) -> jax.Array:
    r"""Right-hand side matching :func:`make_flat_bie_operator`.

    Two equivalent ways to bring in the incident field, selected by which
    data are supplied:

    * *scattered* (``h_leaves`` given, ``uin`` omitted): the interior unknown
      is the scattered field, which satisfies
      :math:`\Delta u^s + \kappa^2 (1-b) u^s = \kappa^2 b u^{\rm inc}`, so
      the incident field enters only through the volume source and its
      outgoing impedance data ``h_leaves``.
    * *total* (``uin``/``uin_dn`` given, ``h_leaves`` omitted): the interior
      unknown is the total field, which is source-free, and the incident
      traces are subtracted on the boundary rows.  This is the formulation of
      the dense-DtN drivers.

    ``uin`` and ``uin_dn`` are in ``Domain.boundary_points`` ordering.
    ``n_src`` sets the number of right-hand sides when neither ``uin`` nor a
    multi-source ``h_leaves`` fixes it.
    """
    n_flat = maps.n_flat
    if uin is not None:
        uin = jnp.asarray(uin)
        n_src = None if uin.ndim == 1 else uin.shape[-1]
    elif h_leaves is not None and jnp.asarray(h_leaves).ndim == 3:
        n_src = jnp.asarray(h_leaves).shape[-1]
    shape = (n_flat,) if n_src is None else (n_flat, n_src)
    rhs = jnp.zeros(shape, dtype=jnp.complex128)

    if h_leaves is not None:
        h_flat = _flatten_leafwise(jnp.asarray(h_leaves))
        p_safe = jnp.where(maps.is_interior, maps.partner, 0)
        gathered = -h_flat[p_safe]
        mask = maps.is_interior
        if rhs.ndim == 2:
            mask = mask[:, None]
        rhs = jnp.where(mask, gathered, rhs)
        h_b = h_flat[maps.bdry_rows]
    else:
        h_b = jnp.zeros(rhs[maps.bdry_rows].shape, dtype=jnp.complex128)

    # Known part of the scattered traces: the particular solution's
    # contribution to (u, u_n), minus the incident field when the interior
    # unknown is the total field.
    us_known = -h_b / (2j * eta)
    us_n_known = h_b / 2.0
    if uin is not None:
        us_known = us_known - uin
        us_n_known = us_n_known - jnp.asarray(uin_dn)
    bdry = -(0.5 * us_known - apply_D(us_known) + apply_S(us_n_known))
    return rhs.at[maps.bdry_rows].set(bdry)
