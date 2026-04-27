from typing import Tuple
from functools import partial
import jax.numpy as jnp
import jax

from ._grid_creation_3D import rearrange_indices_ext_int
from .quadrature import (
    chebyshev_points,
    gauss_points,
    differentiation_matrix_1D,
    barycentric_lagrange_interpolation_matrix_2D,
    barycentric_lagrange_interpolation_matrix_3D,
    affine_transform,
)


@partial(jax.jit, static_argnums=(0,))
def precompute_diff_operators_3D(
    p: int, half_side_len: float
) -> Tuple[jnp.ndarray]:
    """
    Returns D_x, D_y, D_z, D_xx, D_yy, D_zz, D_xy, D_xz, D_yz
    """

    r = rearrange_indices_ext_int(p)

    pts = chebyshev_points(p)
    cheby_diff_matrix = differentiation_matrix_1D(pts) / half_side_len

    # Precompute du/dx, du/dy, du/dz
    du_dx = jnp.kron(
        cheby_diff_matrix,
        jnp.kron(jnp.eye(p), jnp.eye(p)),
    )
    du_dy = jnp.kron(
        jnp.eye(p),
        jnp.kron(cheby_diff_matrix, jnp.eye(p)),
    )
    du_dz = jnp.kron(
        jnp.eye(p),
        jnp.kron(jnp.eye(p), cheby_diff_matrix),
    )

    du_dx = du_dx[r, :]
    du_dx = du_dx[:, r]

    du_dy = du_dy[r, :]
    du_dy = du_dy[:, r]

    du_dz = du_dz[r, :]
    du_dz = du_dz[:, r]

    # Precompute D_xx, D_yy, D_zz, D_xy, D_xz, D_yz
    d_xx = du_dx @ du_dx
    d_yy = du_dy @ du_dy
    d_zz = du_dz @ du_dz
    d_xy = du_dx @ du_dy
    d_xz = du_dx @ du_dz
    d_yz = du_dy @ du_dz

    return (du_dx, du_dy, du_dz, d_xx, d_yy, d_zz, d_xy, d_xz, d_yz)


@partial(jax.jit, static_argnums=(0, 1))
def precompute_P_3D_DtN(p: int, q: int) -> jnp.ndarray:
    """Puts together an operator interpolating from the Gauss grid on the boundary to the Cheby points on the boundary of the 3D cube.

    Args:
        p (int): Number of Chebyshev points in 1D.
        q (int): Number of Gauss-Legendre points in 1D.

    Returns:
        jnp.ndarray: Has shape (p**3 - (p-2)**3, 6 * q**2)
    """
    gauss_pts = gauss_points(q)
    cheby_pts = chebyshev_points(p)

    n_cheby_bdry_pts = p**3 - (p - 2) ** 3

    P = barycentric_lagrange_interpolation_matrix_2D(
        gauss_pts, gauss_pts, cheby_pts, cheby_pts
    )

    I_P = jnp.zeros((n_cheby_bdry_pts, 6 * q**2))

    # First face
    idxes = get_face_1_idxes(p)
    I_P = I_P.at[idxes, 0 : q**2].set(P)

    # Second face
    idxes = get_face_2_idxes(p)
    I_P = I_P.at[idxes, q**2 : 2 * q**2].set(P)

    # Third face
    idxes = get_face_3_idxes(p)
    I_P = I_P.at[idxes, 2 * q**2 : 3 * q**2].set(P)

    # Fourth face
    idxes = get_face_4_idxes(p)
    I_P = I_P.at[idxes, 3 * q**2 : 4 * q**2].set(P)

    # Fifth face
    idxes = get_face_5_idxes(p)
    I_P = I_P.at[idxes, 4 * q**2 : 5 * q**2].set(P)

    # Sixth face
    idxes = get_face_6_idxes(p)
    I_P = I_P.at[idxes, 5 * q**2 :].set(P)

    # Take averages of the corners of the cube. They occur at
    # indices 0, p-1, p**2 - p, p**2 - 1, p**2, p**2 + p - 1, 2 * p**2 - p, 2 * p**2 - 1.
    corner_idxes = jnp.array(
        [
            0,
            p - 1,
            p**2 - p,
            p**2 - 1,
            p**2,
            p**2 + p - 1,
            2 * p**2 - p,
            2 * p**2 - 1,
        ]
    )
    I_P = I_P.at[corner_idxes, :].set(I_P[corner_idxes, :] / 3)

    # Take averages of the edges of the cube. There are 12 edges. They are all of length p-2 because we have
    # already averaged the corners.
    edge_idxes = jnp.concatenate(
        [
            jnp.arange(1, p - 1),
            jnp.arange(p**2 + 1, p**2 + p - 1),
            jnp.arange(p**2 - p + 1, p**2 - 1),
            jnp.arange(2 * p**2 - p + 1, 2 * p**2 - 1),
            jnp.arange(1, p - 1) * p,
            jnp.arange(p, p**2 - p, p) + p**2,
            jnp.arange(2 * p**2, 3 * p**2 - 2 * p, p),
            jnp.arange(3 * p**2 - 2 * p, 4 * p**2 - 4 * p, p),
            jnp.arange(2 * p - 1, p**2 - p, p),
            jnp.arange(p**2 + 2 * p - 1, 2 * p**2 - p, p),
            jnp.arange(2 * p**2 + p - 1, 3 * p**2 - 2 * p, p),
            jnp.arange(3 * p**2 - 2 * p + p - 1, 4 * p**2 - 4 * p, p),
        ]
    )

    I_P = I_P.at[edge_idxes, :].set(I_P[edge_idxes, :] / 2)

    return I_P


@partial(jax.jit, static_argnums=(0, 1))
def precompute_Q_3D_DtN(
    p: int, q: int, du_dx: jnp.ndarray, du_dy: jnp.ndarray, du_dz: jnp.ndarray
) -> jnp.ndarray:
    """Precomputes an operator which maps a solution on the
    3D Chebyshev points to the outward normal derivative of the solution on the boundary Gauss grid.

    Args:
        p (int): Number of Chebyshev points in 1D.
        q (int): Number of Gauss-Legendre points in 1D.
        du_dx (jnp.ndarray): Precomputed derivative operator in the x-direction. Has shape (p**3, p**3).
        du_dy (jnp.ndarray): Precomputed derivative operator in the y-direction. Has shape (p**3, p**3).
        du_dz (jnp.ndarray): Precomputed derivative operator in the z-direction. Has shape (p**3, p**3).

    Returns:
        jnp.ndarray: Has shape (6 * q**2, p**3)
    """

    # First assemble N, a matrix of shape (6 p**2, p**3) which maps from a solution on all of the
    # Chebyshev points to the outward normal derivative of the solution on the boundary Chebyshev points.
    # Note this operator is double-counting
    # grid points on the edges.
    N = jnp.full((6 * p**2, p**3), jnp.nan)
    # Grab the rows of the appropriate derivative operator and put them into N.
    face_1_idxes = get_face_1_idxes(p)
    N = N.at[: p**2].set(-1 * du_dx[face_1_idxes])
    face_2_idxes = get_face_2_idxes(p)
    N = N.at[p**2 : 2 * p**2].set(du_dx[face_2_idxes])
    face_3_idxes = get_face_3_idxes(p)
    N = N.at[2 * p**2 : 3 * p**2].set(-1 * du_dy[face_3_idxes])
    face_4_idxes = get_face_4_idxes(p)
    N = N.at[3 * p**2 : 4 * p**2].set(du_dy[face_4_idxes])
    face_5_idxes = get_face_5_idxes(p)
    N = N.at[4 * p**2 : 5 * p**2].set(-1 * du_dz[face_5_idxes])
    face_6_idxes = get_face_6_idxes(p)
    N = N.at[5 * p**2 :].set(du_dz[face_6_idxes])

    # Now assemble Q_D, a matrix of shape (6 * q**2, p**3) which maps from the Chebyshev points
    # on the boundary to the Gauss points on the boundary.

    gauss_pts = gauss_points(q)
    cheby_pts = chebyshev_points(p)
    Q = barycentric_lagrange_interpolation_matrix_2D(
        cheby_pts, cheby_pts, gauss_pts, gauss_pts
    )
    Q_I = jnp.kron(jnp.eye(6), Q)
    # Q_I = jnp.zeros((6 * q**2, 6 * p**2), dtype=jnp.float64)
    # # Face 1
    # Q_I = Q_I.at[0 : q**2, 0 : p**2].set(jnp.flip(Q, axis=(0, 1)))
    # # Face 2
    # # Q_I = Q_I.at[q**2 : 2 * q**2].set(Q)
    # # # Face 3
    # # Q_I = Q_I.at[2 * q**2 : 3 * q**2].set(Q)
    # # # Face 4
    # # Q_I = Q_I.at[3 * q**2 : 4 * q**2].set(Q)
    # # # Face 5
    # # Q_I = Q_I.at[4 * q**2 : 5 * q**2].set(Q)
    # # # Face 6
    # # Q_I = Q_I.at[5 * q**2 :].set(Q)
    Q_D = Q_I @ N
    return Q_D


############################################
# These next functions are designed to get the
# indices of certain faces of the p^3 Chebyshev grid.
# I am using numbers [1,...,6] for the faces; look at my notes for
# the definition of these numbers.


@partial(jax.jit, static_argnums=(0,))
def get_face_1_idxes(p: int) -> jax.Array:
    """Face lying paralel to the (y,z) plane farthest in the -x direction."""
    return jnp.arange(p**2)


@partial(jax.jit, static_argnums=(0,))
def get_face_2_idxes(p: int) -> jax.Array:
    """Face lying paralel to the (y,z) plane farthest in the +x direction."""
    return jnp.arange(p**2) + p**2


@partial(jax.jit, static_argnums=(0,))
def get_face_3_idxes(p: int) -> jax.Array:
    """Face lying paralel to the (x,z) plane farthest in the -y direction."""
    o = jnp.concatenate(
        [
            jnp.arange(p),
            jnp.arange(2 * p**2, 3 * p**2 - 2 * p),
            jnp.arange(p**2, p**2 + p),
        ]
    )
    return o


@partial(jax.jit, static_argnums=(0,))
def get_face_4_idxes(p: int) -> jax.Array:
    """Face lying paralel to the (x,z) plane farthest in the +y direction."""
    o = jnp.concatenate(
        [
            jnp.arange(p * (p - 1), p**2),
            jnp.arange(3 * p**2 - 2 * p, 4 * p**2 - 4 * p),
            jnp.arange(2 * p**2 - p, 2 * p**2),
        ]
    )
    return o


@partial(jax.jit, static_argnums=(0,))
def get_face_5_idxes(p: int) -> jax.Array:
    """Face lying paralel to the (x,y) plane farthest in the -z direction."""
    first_col = jnp.arange(0, p**2, p)
    last_col = jnp.arange(p**2, 2 * p**2, p)

    first_row = jnp.arange(2 * p**2, 3 * p**2 - 2 * p, p)
    last_row = jnp.arange(3 * p**2 - 2 * p, 4 * p**2 - 4 * p, p)

    int_idxes = jnp.arange(4 * p**2 - 4 * p, 4 * p**2 - 4 - p + (p - 2) ** 2)

    # The order we put these togethr in is first_col, then the interior is one from first_row,
    # then p-2 from int_idxes, then one from last_row. Repeat until out of elements in last_row.
    # Finally, add the last_col.
    out = jnp.zeros(p**2, dtype=int)
    out = out.at[:p].set(first_col)
    out = out.at[-p:].set(last_col)

    counter = p
    for i in range(p - 2):
        out = out.at[counter].set(first_row[i])
        counter += 1
        out = out.at[counter : counter + p - 2].set(
            int_idxes[i * (p - 2) : (i + 1) * (p - 2)]
        )
        counter += p - 2
        out = out.at[counter].set(last_row[i])
        counter += 1

    return out


@partial(jax.jit, static_argnums=(0,))
def get_face_6_idxes(p: int) -> jax.Array:
    """Face lying paralel to the (x,y) plane farthest in the +z direction."""
    first_col = jnp.arange(p - 1, p**2, p)
    last_col = jnp.arange(p**2 + p - 1, 2 * p**2, p)

    first_row = jnp.arange(2 * p**2 + p - 1, 3 * p**2 - 2 * p, p)
    last_row = jnp.arange(3 * p**2 - 2 * p + p - 1, 4 * p**2 - 4 * p, p)

    u = p**3 - (p - 2) ** 3
    l = u - (p - 2) ** 2
    int_idxes = jnp.arange(l, u)
    # The order we put these togethr in is first_col, then the interior is one from first_row,
    # then p-2 from int_idxes, then one from last_row. Repeat until out of elements in last_row.
    # Finally, add the last_col.
    out = jnp.zeros(p**2, dtype=int)
    out = out.at[:p].set(first_col)
    out = out.at[-p:].set(last_col)

    counter = p
    for i in range(p - 2):
        out = out.at[counter].set(first_row[i])
        counter += 1
        out = out.at[counter : counter + p - 2].set(
            int_idxes[i * (p - 2) : (i + 1) * (p - 2)]
        )
        counter += p - 2
        out = out.at[counter].set(last_row[i])
        counter += 1

    return out


def precompute_projection_ops_3D(
    q: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    The refining operator maps from a Gauss-Legendre grid to four Gauss-Legendre grids on
    the same domain.

    For reference in this section, imagine the four panesl are laid out like this:

    --------
    | D | C |
    --------
    | A | B |
    --------

    The coarsening operator maps from four Gauss-Legendre grids to a single Gauss-Legendre grid.
    """

    if q % 2:
        raise ValueError("q must be even.")

    gauss_pts = gauss_points(q)

    idxes = jnp.arange(q**2)

    first_half = affine_transform(gauss_pts, jnp.array([-1, 0]))
    second_half = affine_transform(gauss_pts, jnp.array([0, 1]))

    # Refining operator

    to_A = barycentric_lagrange_interpolation_matrix_2D(
        gauss_pts, gauss_pts, first_half, first_half
    )
    to_B = barycentric_lagrange_interpolation_matrix_2D(
        gauss_pts, gauss_pts, second_half, first_half
    )
    to_C = barycentric_lagrange_interpolation_matrix_2D(
        gauss_pts, gauss_pts, second_half, second_half
    )
    to_D = barycentric_lagrange_interpolation_matrix_2D(
        gauss_pts, gauss_pts, first_half, second_half
    )
    refining_operator = jnp.concatenate([to_A, to_B, to_C, to_D], axis=0)

    L_4f1 = refining_operator

    # Coarsening operator
    coarsening_op_out = jnp.zeros((q**2, 4 * q**2), dtype=jnp.float64)

    gauss_pts_first_half = gauss_pts[: q // 2]
    gauss_pts_second_half = gauss_pts[q // 2 :]

    from_A = barycentric_lagrange_interpolation_matrix_2D(
        first_half, first_half, gauss_pts_first_half, gauss_pts_first_half
    )
    A_bools = idxes % q < q / 2
    A_bools = jnp.logical_and(A_bools, idxes < q**2 / 2)
    coarsening_op_out = coarsening_op_out.at[A_bools, 0 : q**2].set(from_A)

    from_B = barycentric_lagrange_interpolation_matrix_2D(
        second_half, first_half, gauss_pts_second_half, gauss_pts_first_half
    )
    B_bools = idxes % q < q / 2
    B_bools = jnp.logical_and(B_bools, idxes >= q**2 / 2)
    coarsening_op_out = coarsening_op_out.at[B_bools, q**2 : 2 * (q**2)].set(
        from_B
    )

    from_C = barycentric_lagrange_interpolation_matrix_2D(
        second_half, second_half, gauss_pts_second_half, gauss_pts_second_half
    )
    C_bools = idxes % q >= q / 2
    C_bools = jnp.logical_and(C_bools, idxes >= q**2 / 2)
    coarsening_op_out = coarsening_op_out.at[
        C_bools, 2 * q**2 : 3 * (q**2)
    ].set(from_C)

    from_D = barycentric_lagrange_interpolation_matrix_2D(
        first_half, second_half, gauss_pts_first_half, gauss_pts_second_half
    )
    D_bools = idxes % q >= q / 2
    D_bools = jnp.logical_and(D_bools, idxes < q**2 / 2)
    coarsening_op_out = coarsening_op_out.at[D_bools, 3 * q**2 :].set(from_D)

    L_1f4 = coarsening_op_out

    return L_4f1, L_1f4


def precompute_L_8f1(p: int) -> jnp.array:
    cheby_pts_1d = chebyshev_points(p)
    cheby_pts_refined = jnp.concatenate(
        [
            affine_transform(cheby_pts_1d, jnp.array([-1, 0])),
            affine_transform(cheby_pts_1d, jnp.array([0, 1])),
        ]
    )

    # This thing is scale and translation invariant so no need
    # to worry about recomputing it for different sizes
    I_refined = barycentric_lagrange_interpolation_matrix_3D(
        cheby_pts_1d,
        cheby_pts_1d,
        cheby_pts_1d,
        cheby_pts_refined,
        cheby_pts_refined,
        cheby_pts_refined,
    )

    r, c = indexing_for_refinement_operator(p)

    # I_refined has cols that are a standard meshgrid of (cheby_pts_1d, cheby_pts_1d, cheby_pts_1d).
    # We want to rearrange the columns so that the first p**3 - (p-2)**3 columns are the exterior points
    # and the rest are the interior points.
    I_refined = I_refined[:, c]

    # I_refined has rows that are a standard meshgrid of (cheby_pts_refined, cheby_pts_refined, cheby_pts_refined).
    # We first want to rearrange them so that each of the 8 blocks are together. Then we want to rearrange each
    # block so that the first p**3 - (p-2)**3 rows are the exterior points and the rest are the interior points.
    I_refined = I_refined[r, :]
    return I_refined


def indexing_for_refinement_operator(p: int) -> jnp.array:
    col_idxes = rearrange_indices_ext_int(p)

    ii = jnp.arange(8 * p**3)

    # part a is where x is in the first half, y is in the first half, and z is in the second half.
    a_bools = jnp.logical_and(
        ii % (2 * p) >= p,
        jnp.logical_and(
            (ii // (2 * p)) % (2 * p) < p, (ii // ((2 * p) ** 2)) % (2 * p) < p
        ),
    )
    a_idxes = ii[a_bools]
    a_idxes = a_idxes[col_idxes]

    # part b is where x is in the second half, y is in the first half, and z is in the second half.
    b_bools = jnp.logical_and(
        ii % (2 * p) >= p,
        jnp.logical_and(
            (ii // (2 * p)) % (2 * p) < p,
            (ii // ((2 * p) ** 2)) % (2 * p) >= p,
        ),
    )
    b_idxes = ii[b_bools]
    b_idxes = b_idxes[col_idxes]

    # part c is where x is in the second half, y is in the second half, and z is in the second half.
    c_bools = jnp.logical_and(
        ii % (2 * p) >= p,
        jnp.logical_and(
            (ii // (2 * p)) % (2 * p) >= p,
            (ii // ((2 * p) ** 2)) % (2 * p) >= p,
        ),
    )
    c_idxes = ii[c_bools]
    c_idxes = c_idxes[col_idxes]

    # part d is where x is in the first half, y is in the second half, and z is in the second half.
    d_bools = jnp.logical_and(
        ii % (2 * p) >= p,
        jnp.logical_and(
            (ii // (2 * p)) % (2 * p) >= p,
            (ii // ((2 * p) ** 2)) % (2 * p) < p,
        ),
    )
    d_idxes = ii[d_bools]
    d_idxes = d_idxes[col_idxes]

    # part e is where x is in the first half, y is in the first half, and z is in the first half.
    e_bools = jnp.logical_and(
        ii % (2 * p) < p,
        jnp.logical_and(
            (ii // (2 * p)) % (2 * p) < p, (ii // ((2 * p) ** 2)) % (2 * p) < p
        ),
    )
    e_idxes = ii[e_bools]
    e_idxes = e_idxes[col_idxes]

    # part f is where x is in the second half, y is in the first half, and z is in the first half.
    f_bools = jnp.logical_and(
        ii % (2 * p) < p,
        jnp.logical_and(
            (ii // (2 * p)) % (2 * p) < p,
            (ii // ((2 * p) ** 2)) % (2 * p) >= p,
        ),
    )
    f_idxes = ii[f_bools]
    f_idxes = f_idxes[col_idxes]

    # part g is where x is in the second half, y is in the second half, and z is in the first half.
    g_bools = jnp.logical_and(
        ii % (2 * p) < p,
        jnp.logical_and(
            (ii // (2 * p)) % (2 * p) >= p,
            (ii // ((2 * p) ** 2)) % (2 * p) >= p,
        ),
    )
    g_idxes = ii[g_bools]
    g_idxes = g_idxes[col_idxes]

    # part h is where x is in the first half, y is in the second half, and z is in the first half.
    h_bools = jnp.logical_and(
        ii % (2 * p) < p,
        jnp.logical_and(
            (ii // (2 * p)) % (2 * p) >= p,
            (ii // ((2 * p) ** 2)) % (2 * p) < p,
        ),
    )
    h_idxes = ii[h_bools]
    h_idxes = h_idxes[col_idxes]

    row_idxes = jnp.concatenate(
        [
            a_idxes,
            b_idxes,
            c_idxes,
            d_idxes,
            e_idxes,
            f_idxes,
            g_idxes,
            h_idxes,
        ]
    )

    return row_idxes, col_idxes


############################################
# 3D ItI precomputed operators.
#
# Conventions (mirroring the 2D ItI implementation in _precompute_operators_2D.py):
#  - The Cheby boundary nodes are single-counted: shape (p**3 - (p-2)**3,).
#  - At edges shared by 2 faces and corners shared by 3 faces, all face
#    contributions are AVERAGED (similar to the DtN P_3D averaging scheme).
#  - Outgoing impedance traces live on the 6q**2 Gauss boundary nodes; the QH
#    operator first builds a double-counted (6 p**2 row) impedance map on the
#    Cheby boundary and then interpolates Cheby -> Gauss face-wise via
#    Q_I = kron(I_6, Q_2D).
############################################


@partial(jax.jit, static_argnums=(0, 1))
def precompute_P_3D_ItI(p: int, q: int) -> jnp.ndarray:
    """Maps boundary impedance data on the 6 q**2 Gauss boundary nodes to the
    p**3 - (p-2)**3 Cheby boundary nodes (single-counted).

    Implementation: identical block-and-average construction as
    :func:`precompute_P_3D_DtN`. The averaging at shared edges/corners is
    consistent with the averaging convention used by the 3D ItI G and N_tilde
    operators below, so the local solve's incoming-impedance residual is
    well-defined at shared boundary points when the merged Gauss inputs are
    consistent across adjacent faces.
    """
    return precompute_P_3D_DtN(p, q)


@partial(jax.jit, static_argnums=(3,))
def precompute_N_matrix_3D(
    du_dx: jnp.ndarray, du_dy: jnp.ndarray, du_dz: jnp.ndarray, p: int
) -> jnp.ndarray:
    """Outward-normal-derivative operator on the **double-counted** 6 p**2 Cheby
    boundary nodes. Returns a matrix of shape (6 p**2, p**3) which maps a
    function on all p**3 Chebyshev points to its outward normal derivative on
    each of the 6 faces, taking the values at edges/corners independently per
    face (so each shared point appears once per face that contains it).

    This matches the inner construction of :func:`precompute_Q_3D_DtN`.
    """
    N = jnp.zeros((6 * p**2, p**3), dtype=du_dx.dtype)
    f1 = get_face_1_idxes(p)
    f2 = get_face_2_idxes(p)
    f3 = get_face_3_idxes(p)
    f4 = get_face_4_idxes(p)
    f5 = get_face_5_idxes(p)
    f6 = get_face_6_idxes(p)
    N = N.at[: p**2].set(-1 * du_dx[f1])
    N = N.at[p**2 : 2 * p**2].set(du_dx[f2])
    N = N.at[2 * p**2 : 3 * p**2].set(-1 * du_dy[f3])
    N = N.at[3 * p**2 : 4 * p**2].set(du_dy[f4])
    N = N.at[4 * p**2 : 5 * p**2].set(-1 * du_dz[f5])
    N = N.at[5 * p**2 :].set(du_dz[f6])
    return N


@partial(jax.jit, static_argnums=(3,))
def precompute_N_tilde_matrix_3D(
    du_dx: jnp.ndarray, du_dy: jnp.ndarray, du_dz: jnp.ndarray, p: int
) -> jnp.ndarray:
    """Outward-normal-derivative operator on the **single-counted** boundary
    Cheby nodes, with averaging at shared edges and corners.

    Returns a matrix of shape (p**3 - (p-2)**3, p**3).

    For each face f, the rows of N_tilde indexed by ``get_face_X_idxes(p)`` are
    incremented by the outward-normal contribution from that face. After all
    six faces have contributed, the rows corresponding to the 8 cube corners
    are divided by 3 (each corner is touched by 3 faces) and the rows
    corresponding to the 12 cube edges (each of length p-2) are divided by 2.
    Interior face rows already have a single contribution.
    """
    n_bdry = p**3 - (p - 2) ** 3
    N_tilde = jnp.zeros((n_bdry, p**3), dtype=du_dx.dtype)

    f1 = get_face_1_idxes(p)
    f2 = get_face_2_idxes(p)
    f3 = get_face_3_idxes(p)
    f4 = get_face_4_idxes(p)
    f5 = get_face_5_idxes(p)
    f6 = get_face_6_idxes(p)

    # The columns of du_d* are the rearranged-Cheby-ordering Cheby points;
    # rows of du_d* are the same ordering. ``get_face_*_idxes`` produces
    # boundary-Cheby indices in this rearranged ordering.
    N_tilde = N_tilde.at[f1].add(-1 * du_dx[f1])
    N_tilde = N_tilde.at[f2].add(du_dx[f2])
    N_tilde = N_tilde.at[f3].add(-1 * du_dy[f3])
    N_tilde = N_tilde.at[f4].add(du_dy[f4])
    N_tilde = N_tilde.at[f5].add(-1 * du_dz[f5])
    N_tilde = N_tilde.at[f6].add(du_dz[f6])

    # 8 corners (each shared by 3 faces) -> divide by 3.
    corner_idxes = jnp.array(
        [
            0,
            p - 1,
            p**2 - p,
            p**2 - 1,
            p**2,
            p**2 + p - 1,
            2 * p**2 - p,
            2 * p**2 - 1,
        ]
    )
    N_tilde = N_tilde.at[corner_idxes, :].set(N_tilde[corner_idxes, :] / 3)

    # 12 edges (each shared by 2 faces, length p-2 after corners removed).
    edge_idxes = jnp.concatenate(
        [
            jnp.arange(1, p - 1),
            jnp.arange(p**2 + 1, p**2 + p - 1),
            jnp.arange(p**2 - p + 1, p**2 - 1),
            jnp.arange(2 * p**2 - p + 1, 2 * p**2 - 1),
            jnp.arange(1, p - 1) * p,
            jnp.arange(p, p**2 - p, p) + p**2,
            jnp.arange(2 * p**2, 3 * p**2 - 2 * p, p),
            jnp.arange(3 * p**2 - 2 * p, 4 * p**2 - 4 * p, p),
            jnp.arange(2 * p - 1, p**2 - p, p),
            jnp.arange(p**2 + 2 * p - 1, 2 * p**2 - p, p),
            jnp.arange(2 * p**2 + p - 1, 3 * p**2 - 2 * p, p),
            jnp.arange(3 * p**2 - 2 * p + p - 1, 4 * p**2 - 4 * p, p),
        ]
    )
    N_tilde = N_tilde.at[edge_idxes, :].set(N_tilde[edge_idxes, :] / 2)

    return N_tilde


@jax.jit
def precompute_G_3D_ItI(N_tilde: jnp.ndarray, eta: float) -> jnp.ndarray:
    """Incoming-impedance operator on the single-counted Cheby boundary.

    G = N_tilde + i*eta * I[:n_bdry]   (added to the first n_bdry columns).

    Returns shape (n_bdry, p**3) where n_bdry = p**3 - (p-2)**3. Acts on a
    solution u on the rearranged Chebyshev grid (boundary first, then interior)
    and returns the incoming impedance trace ``u_n + i*eta*u`` at each
    single-counted boundary Cheby node.
    """
    shape_0 = N_tilde.shape[0]
    F = N_tilde.astype(jnp.complex128)
    F = F.at[:shape_0, :shape_0].set(
        F[:shape_0, :shape_0] + 1j * eta * jnp.eye(shape_0)
    )
    return F


@partial(jax.jit, static_argnums=(1, 2))
def precompute_QH_3D_ItI(
    N: jnp.ndarray, p: int, q: int, eta: float
) -> jnp.ndarray:
    """Outgoing-impedance operator at the 6 q**2 Gauss boundary nodes.

    Builds the double-counted impedance map H = N - i*eta * I_face on the 6 p**2
    Cheby boundary (i.e. for each face block, subtract i*eta along the diagonal
    consisting of the columns indexed by that face's Cheby boundary indices).
    Then interpolates Cheby -> Gauss face-wise using Q_I = kron(I_6, Q_2D).

    Args:
        N: Output of :func:`precompute_N_matrix_3D`. Shape (6 p**2, p**3).
        p, q: Shape parameters.
        eta: Real impedance coefficient.

    Returns:
        QH: shape (6 q**2, p**3).
    """
    H = N.astype(jnp.complex128)

    f1 = get_face_1_idxes(p)
    f2 = get_face_2_idxes(p)
    f3 = get_face_3_idxes(p)
    f4 = get_face_4_idxes(p)
    f5 = get_face_5_idxes(p)
    f6 = get_face_6_idxes(p)

    rows = jnp.arange(p**2)
    H = H.at[rows, f1].add(-1j * eta)
    H = H.at[rows + p**2, f2].add(-1j * eta)
    H = H.at[rows + 2 * p**2, f3].add(-1j * eta)
    H = H.at[rows + 3 * p**2, f4].add(-1j * eta)
    H = H.at[rows + 4 * p**2, f5].add(-1j * eta)
    H = H.at[rows + 5 * p**2, f6].add(-1j * eta)

    cheby_pts = chebyshev_points(p)
    gauss_pts = gauss_points(q)
    Q = barycentric_lagrange_interpolation_matrix_2D(
        cheby_pts, cheby_pts, gauss_pts, gauss_pts
    )
    Q_I = jnp.kron(jnp.eye(6), Q)
    return Q_I @ H
