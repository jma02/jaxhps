import jax
import jax.numpy as jnp
from functools import partial
from ._discretization_tree import (
    DiscretizationNode2D,
    DiscretizationNode3D,
)
from .quadrature import (
    chebyshev_points,
    barycentric_lagrange_interpolation_matrix_2D,
    barycentric_lagrange_interpolation_matrix_3D,
    affine_transform,
)
from ._grid_creation_2D import (
    rearrange_indices_ext_int as rearrange_indices_ext_int_2D,
)
from ._grid_creation_3D import (
    rearrange_indices_ext_int as rearrange_indices_ext_int_3D,
)
from typing import Tuple


@partial(jax.jit, static_argnums=(1,))
def interp_from_hps_2D(
    leaves: Tuple[DiscretizationNode2D],
    p: int,
    f_evals: jax.Array,
    x_vals: jax.Array,
    y_vals: jax.Array,
) -> Tuple[jax.Array]:
    # xmin = root.xmin
    # xmax = root.xmax
    # ymin = root.ymin
    # ymax = root.ymax

    # Create the regular grid
    # x = jnp.linspace(xmin, xmax, n_pts, endpoint=False, dtype=jnp.float64)
    # y = jnp.linspace(ymin, ymax, n_pts, endpoint=False, dtype=jnp.float64)
    # y = jnp.flip(y)

    bool_multi_source = f_evals.ndim == 3
    n_x = x_vals.shape[0]
    n_y = y_vals.shape[0]
    X, Y = jnp.meshgrid(x_vals, y_vals)
    target_pts = jnp.concatenate(
        (jnp.expand_dims(X, 2), jnp.expand_dims(Y, 2)), axis=2
    )
    pts_lst = target_pts.reshape(-1, 2)

    corners_lst = [
        jnp.array(
            [
                [node.xmin, node.ymin],
                [node.xmax, node.ymin],
                [node.xmax, node.ymax],
                [node.xmin, node.ymax],
            ]
        )
        for node in leaves
    ]
    corners_iter = jnp.stack(corners_lst)

    # Find which patch the point is in
    # These should have shape (n_pts^2, 4^l)
    satisfies_xmin = pts_lst[:, 0, None] >= corners_iter[None, :, 0, 0]
    satisfies_xmax = pts_lst[:, 0, None] <= corners_iter[None, :, 1, 0]
    satisfies_ymin = pts_lst[:, 1, None] >= corners_iter[None, :, 0, 1]
    satisfies_ymax = pts_lst[:, 1, None] <= corners_iter[None, :, 2, 1]

    x_bools = jnp.logical_and(satisfies_xmin, satisfies_xmax)
    y_bools = jnp.logical_and(satisfies_ymin, satisfies_ymax)

    # Has shape (n_pts^2, 4^l)
    x_and_y = jnp.logical_and(x_bools, y_bools)

    # Find the indexes of the patches that contain each point
    patch_idx = jnp.argmax(x_and_y, axis=1)

    corners_for_vmap = corners_iter[patch_idx]
    f_for_vmap = f_evals[patch_idx]

    xvals_for_vmap = pts_lst[:, 0]
    yvals_for_vmap = pts_lst[:, 1]

    # Interpolate to the target points
    vals = vmapped_interp_to_point_2D(
        xvals_for_vmap, yvals_for_vmap, corners_for_vmap, f_for_vmap, p
    )
    if bool_multi_source:
        vals = vals.reshape(n_x, n_y, f_evals.shape[-1])
    else:
        vals = vals.reshape(n_x, n_y)
    return vals, target_pts


@partial(jax.jit, static_argnums=(4,))
def _interp_to_point_2D(
    xval: jax.Array,
    yval: jax.Array,
    corners: jax.Array,
    f: jax.Array,
    p: int,
) -> jax.Array:
    """
    For a particular point (xval, yval) this function finds the patch
    that contains the point, constructs a polynomial interpolation matrix
    to that point, and evaluates the interpolant at the point.

    Args:
        xval (jax.Array): x coordinate of the target point. Has shape () or (n,)
        yval (jax.Array): y coordinate of the target point. Has shape () or (n,)
        p (int): Polynomial order of interpolation
        cheby_grid (jax.Array): HPS discretization. Has size (n_leaves, p^2, 2).
        all_corners (jax.Array): Corners of the patches. Has size (n_leaves, 4, 2).
        f (jax.Array): Function evals on the Cheby grid. Has size (n_leaves, p^2)

    Returns:
        jax.Array: Has shape (1,) or (n^2,) depending on the size of input xval and yval. The output
        is the interpolated value at the target point(s).
    """
    # If xval is a scalar, need to reshape it to be a 1D array for the barycentric interp function.
    # If xval is a 1D vector, this doesn't change anything.
    xval = xval.reshape(-1)
    yval = yval.reshape(-1)
    # print("_interp_to_point: xval: ", xval, " xval shape: ", xval.shape)
    # print("_interp_to_point: yval: ", yval, " yval shape: ", yval.shape)

    cheby_pts = chebyshev_points(p)

    # out = jnp.zeros_like(xval)

    xmin_i, ymin_i = corners[0]
    xmax_i, ymax_i = corners[2]

    from_x = affine_transform(cheby_pts, jnp.array([xmin_i, xmax_i]))
    from_y = affine_transform(cheby_pts, jnp.array([ymin_i, ymax_i]))
    # Annoyingly this is how the y vals are ordered
    from_y = jnp.flip(from_y)

    I = barycentric_lagrange_interpolation_matrix_2D(
        from_x, from_y, xval, yval
    )

    rearrange_idxes = rearrange_indices_ext_int_2D(p)
    I = I[:, rearrange_idxes]

    return I @ f


vmapped_interp_to_point_2D = jax.vmap(
    _interp_to_point_2D, in_axes=(0, 0, 0, 0, None)
)


@partial(jax.jit, static_argnums=(1,))
def interp_from_hps_3D(
    leaves: Tuple[DiscretizationNode3D],
    p: int,
    f_evals: jax.Array,
    x_vals: jax.Array,
    y_vals: jax.Array,
    z_vals: jax.Array,
) -> Tuple[jax.Array]:
    n_x = x_vals.shape[0]
    n_y = y_vals.shape[0]
    n_z = z_vals.shape[0]

    X, Y, Z = jnp.meshgrid(x_vals, y_vals, z_vals)
    target_pts = jnp.concatenate(
        (jnp.expand_dims(X, 3), jnp.expand_dims(Y, 3), jnp.expand_dims(Z, 3)),
        axis=3,
    )

    pts = jnp.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)
    corners_lst = [
        jnp.array(
            [
                [node.xmin, node.ymin, node.zmin],
                [node.xmax, node.ymax, node.zmax],
            ]
        )
        for node in leaves
    ]
    corners_iter = jnp.stack(corners_lst)

    satisfies_xmin = pts[:, 0, None] >= corners_iter[None, :, 0, 0]
    satisfies_xmax = pts[:, 0, None] <= corners_iter[None, :, 1, 0]
    satisfies_ymin = pts[:, 1, None] >= corners_iter[None, :, 0, 1]
    satisfies_ymax = pts[:, 1, None] <= corners_iter[None, :, 1, 1]
    satisfies_zmin = pts[:, 2, None] >= corners_iter[None, :, 0, 2]
    satisfies_zmax = pts[:, 2, None] <= corners_iter[None, :, 1, 2]

    x_bools = jnp.logical_and(satisfies_xmin, satisfies_xmax)
    y_bools = jnp.logical_and(satisfies_ymin, satisfies_ymax)
    z_bools = jnp.logical_and(satisfies_zmin, satisfies_zmax)

    all_bools = jnp.logical_and(x_bools, jnp.logical_and(y_bools, z_bools))

    # Find the indexes of the patches that contain each point
    patch_idx = jnp.argmax(all_bools, axis=1)

    corners_for_vmap = corners_iter[patch_idx]
    f_for_vmap = f_evals[patch_idx]

    xvals_for_vmap = pts[:, 0]
    yvals_for_vmap = pts[:, 1]
    zvals_for_vmap = pts[:, 2]

    # Interpolate to the target points
    vals = vmapped_interp_to_point_3D(
        xvals_for_vmap,
        yvals_for_vmap,
        zvals_for_vmap,
        corners_for_vmap,
        f_for_vmap,
        p,
    )
    
    # Correctly handle multi-source data during reshape
    if f_evals.ndim == 3:
        n_sources = f_evals.shape[-1]
        vals = vals.reshape(n_x, n_y, n_z, n_sources)
    else:
        vals = vals.reshape(n_x, n_y, n_z)
    return vals, target_pts


@partial(jax.jit, static_argnums=(5,))
def _interp_to_point_3D(
    xval: jnp.array,
    yval: jnp.array,
    zval: jnp.array,
    corners: jnp.array,
    f: jnp.array,
    p: int,
) -> jnp.array:
    """
    For a particular point (xval, yval) this function finds the patch
    that contains the point, constructs a polynomial interpolation matrix
    to that point, and evaluates the interpolant at the point.

    Args:
        xval (jnp.array): x coordinate of the target point. Has shape () or (n,)
        yval (jnp.array): y coordinate of the target point. Has shape () or (n,)
        p (int): Polynomial order of interpolation
        cheby_grid (jnp.array): HPS discretization. Has size (n_leaves, p^2, 2).
        all_corners (jnp.array): Corners of the patches. Has size (n_leaves, 4, 2).
        f (jnp.array): Function evals on the Cheby grid. Has size (n_leaves, p^2)

    Returns:
        jnp.array: Has shape (1,) or (n^2,) depending on the size of input xval and yval. The output
        is the interpolated value at the target point(s).
    """
    # If xval is a scalar, need to reshape it to be a 1D array for the barycentric interp function.
    # If xval is a 1D vector, this doesn't change anything.
    xval = xval.reshape(-1)
    yval = yval.reshape(-1)
    zval = zval.reshape(-1)
    # print("_interp_to_point: xval: ", xval, " xval shape: ", xval.shape)
    # print("_interp_to_point: yval: ", yval, " yval shape: ", yval.shape)

    cheby_pts = chebyshev_points(p)

    from_x = affine_transform(cheby_pts, corners[:, 0])
    from_y = affine_transform(cheby_pts, corners[:, 1])
    from_z = affine_transform(cheby_pts, corners[:, 2])

    I = barycentric_lagrange_interpolation_matrix_3D(
        from_x, from_y, from_z, xval, yval, zval
    )

    rearrange_idxes = rearrange_indices_ext_int_3D(p)
    I = I[:, rearrange_idxes]

    return I @ f


vmapped_interp_to_point_3D = jax.vmap(
    _interp_to_point_3D, in_axes=(0, 0, 0, 0, 0, None)
)


@partial(jax.jit, static_argnums=(2,))
def interp_to_single_Chebyshev_panel_2D(
    node_bounds: jax.Array,
    samples: jax.Array,
    p: int,
    from_x: jax.Array,
    from_y: jax.Array,
) -> jax.Array:
    # Get the points for this node
    c = chebyshev_points(p)
    to_x = affine_transform(c, node_bounds[:2])
    to_y = affine_transform(c, node_bounds[2:])
    to_y = jnp.flip(to_y)  # Annoyingly this is how the y vals are ordered

    # Create the interpolation matrix
    I = barycentric_lagrange_interpolation_matrix_2D(
        from_x,
        from_y,
        to_x,
        to_y,
    )

    rearrange_idxes = rearrange_indices_ext_int_2D(p)
    I = I[rearrange_idxes, :]

    # Now we can evaluate the function at the sample points
    return I @ samples.flatten()


interp_to_hps_2D = jax.vmap(
    interp_to_single_Chebyshev_panel_2D, in_axes=(0, None, None, None, None)
)


@partial(jax.jit, static_argnums=(2,))
def interp_to_single_Chebyshev_panel_3D(
    node_bounds: jax.Array,
    samples: jax.Array,
    p: int,
    from_x: jax.Array,
    from_y: jax.Array,
    from_z: jax.Array,
) -> jax.Array:
    # Get the points for this node
    c = chebyshev_points(p)
    to_x = affine_transform(c, node_bounds[:2])
    to_y = affine_transform(c, node_bounds[2:4])
    to_z = affine_transform(c, node_bounds[4:])

    # Create the interpolation matrix
    I = barycentric_lagrange_interpolation_matrix_3D(
        from_x, from_y, from_z, to_x, to_y, to_z
    )
    rearrange_idxes = rearrange_indices_ext_int_3D(p)
    I = I[rearrange_idxes, :]

    # Now we can evaluate the function at the sample points
    return I @ samples.flatten()


interp_to_hps_3D = jax.vmap(
    interp_to_single_Chebyshev_panel_3D,
    in_axes=(0, None, None, None, None, None),
)
