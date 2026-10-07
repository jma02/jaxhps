from ._discretization_tree import (
    DiscretizationNode2D,
    DiscretizationNode3D,
    get_all_leaves,
)
from ._grid_creation_2D import (
    compute_interior_Chebyshev_points_uniform_2D,
    compute_interior_Chebyshev_points_adaptive_2D,
    compute_boundary_Gauss_points_uniform_2D,
    compute_boundary_Gauss_points_adaptive_2D,
    get_all_uniform_leaves_2D,
)
from ._grid_creation_3D import (
    compute_interior_Chebyshev_points_uniform_3D,
    compute_interior_Chebyshev_points_adaptive_3D,
    compute_boundary_Gauss_points_uniform_3D,
    compute_boundary_Gauss_points_adaptive_3D,
    get_all_uniform_leaves_3D,
)
from ._interpolation_methods import (
    interp_from_hps_2D,
    interp_from_hps_3D,
    interp_to_hps_2D,
    interp_to_hps_3D,
)
from ._adaptive_discretization_3D import (
    generate_adaptive_mesh_level_restriction as generate_adaptive_mesh_level_restriction_3D,
)
from ._adaptive_discretization_2D import (
    generate_adaptive_mesh_level_restriction_2D,
)
import jax
import jax.numpy as jnp
import logging
from typing import Callable, List, Tuple


class Domain:
    def __init__(
        self,
        p: int,
        q: int,
        root: DiscretizationNode2D | DiscretizationNode3D,
        L: int | None = None,
    ):
        self.p: int = p  #: Polynomial order for Chebyshev points.
        self.q: int = q  #:  Polynomial order for Gauss-Legendre points.
        #: Root node of the discretization tree
        self.root: DiscretizationNode2D | DiscretizationNode3D = root
        #: Number of levels of uniform refinement, or None for adaptive refinement.
        self.L: int | None = L

        self.bool_2D = isinstance(root, DiscretizationNode2D)

        if self.L is not None:
            self.bool_uniform = True

            # Depending on whether root is a DiscretizationNode2D or
            # DiscretizationNode3D, we compute the grid points differently
            if self.bool_2D:
                #: Interior Chebyshev points with shape (n_leaves, p^d, d)
                self.interior_points: jax.Array = (
                    compute_interior_Chebyshev_points_uniform_2D(root, L, p)
                )
                #: Boundary Gauss points with shape (x q^{d-1}, d), where x is the number of leaves touching the boundary.
                self.boundary_points: jax.Array = (
                    compute_boundary_Gauss_points_uniform_2D(root, L, q)
                )
                #: The number of leaves in the discretization tree.
                self.n_leaves: int = 4**L
            else:
                self.interior_points = (
                    compute_interior_Chebyshev_points_uniform_3D(root, L, p)
                )
                self.boundary_points = (
                    compute_boundary_Gauss_points_uniform_3D(root, L, q)
                )
                self.n_leaves = 8**L

        else:
            # If L is None, we're using an adaptive discretization
            self.bool_uniform = False
            self.n_leaves = len(get_all_leaves(root))
            if self.bool_2D:
                self.interior_points = (
                    compute_interior_Chebyshev_points_adaptive_2D(root, p)
                )
                self.boundary_points = (
                    compute_boundary_Gauss_points_adaptive_2D(root, q)
                )
            else:
                self.interior_points = (
                    compute_interior_Chebyshev_points_adaptive_3D(root, p)
                )
                self.boundary_points = (
                    compute_boundary_Gauss_points_adaptive_3D(root, q)
                )

    def interp_to_interior_points(
        self,
        values: jax.Array,
        sample_points_x: jax.Array,
        sample_points_y: jax.Array,
        sample_points_z: jax.Array = None,
    ) -> jax.Array:
        """

        This is a utility for interpolating from values on a rectangular grid to values on
        the HPS grid. The interpolation method builds a separate barycentric Lagrange interpolation
        matrix for each leaf on the HPS grid and maps the values to the leaf discretization points.

        The values are assumed to be defined on ``sample_points``:

        .. code:: python

           X, Y, Z = jnp.meshgrid(sample_points_x,
                                sample_points_y,
                                sample_points_z,
                                indexing="ij")
           sample_points = jnp.concatenate((jnp.expand_dims(X, 3),
                                          jnp.expand_dims(Y, 3),
                                          jnp.expand_dims(Z, 3)),
                                         axis=3
                                        )


        Parameters
        ----------
        values : (jax.Array)
            Has shape (n_x, n_y) or (n_x, n_y, n_z), and specifies the values of the function on the rectangular grid.

        sample_points_x : (jax.Array)
            Has shape (n_x,). Specifies the x-coordinates of the rectangular grid.

        sample_points_y : (jax.Array)
            Has shape (n_y,). Specifies the y-coordinates of the rectangular grid.

        sample_points_z : (optional, jax.Array)
            Has shape (n_z,). Specifies the z-coordinates of the rectangular grid. This parameter is optional and should be provided only for 3D cases. Defaults to None.

        Returns
        -------
        jax.Array
            Samples on the HPS grid. Has shape (n_leaves, p^d), where d is the dimension of the problem.

        """
        n_x = sample_points_x.shape[0]
        n_y = sample_points_y.shape[0]
        # 2D vs 3D checking
        if isinstance(self.root, DiscretizationNode2D):
            bool_2D = True
            assert sample_points_z is None
            # Check shape of values
            assert values.shape == (n_x, n_y)
        else:
            bool_2D = False
            assert sample_points_z is not None
            n_z = sample_points_z.shape[0]
            # Check shape of values
            assert values.shape == (n_x, n_y, n_z)

        if bool_2D:
            if self.bool_uniform:
                leaves = get_all_uniform_leaves_2D(self.root, self.L)
            else:
                leaves = get_all_leaves(self.root)
            logging.debug("interp_to_interior_points: leaves: %s", len(leaves))
            leaf_bounds = jnp.array(
                [
                    [leaf.xmin, leaf.xmax, leaf.ymin, leaf.ymax]
                    for leaf in leaves
                ]
            )
            logging.debug(
                "interp_to_interior_points: leaf_bounds: %s", leaf_bounds.shape
            )

            return interp_to_hps_2D(
                leaf_bounds, values, self.p, sample_points_x, sample_points_y
            )
        else:
            # 3D case
            if self.bool_uniform:
                leaves = get_all_uniform_leaves_3D(self.root, self.L)
            else:
                leaves = get_all_leaves(self.root)

            leaf_bounds = jnp.array(
                [
                    [
                        leaf.xmin,
                        leaf.xmax,
                        leaf.ymin,
                        leaf.ymax,
                        leaf.zmin,
                        leaf.zmax,
                    ]
                    for leaf in leaves
                ]
            )
            return interp_to_hps_3D(
                leaf_bounds,
                values,
                self.p,
                sample_points_x,
                sample_points_y,
                sample_points_z,
            )

    def interp_to_boundary_points(
        self,
        sample_points: jax.Array,
        sample_values: jax.Array,
        sample_f: Callable,
    ) -> jax.Array:
        raise NotImplementedError(
            "interp_to_boundary_points is not implemented yet."
        )

    def interp_from_interior_points(
        self,
        samples: jax.Array,
        eval_points_x: jax.Array,
        eval_points_y: jax.Array,
        eval_points_z: jax.Array = None,
    ) -> Tuple[jax.Array]:
        """
        This is a method for interpolating from the HPS grid to a rectangular grid. For each point in the rectangular grid,
        a barycentric Lagrange interpolation matrix is built from the containing leaf to the point, and the function is interpolated
        from the leaf to that point.


        Parameters
        ----------
            samples : jax.Array
                Function sampled on the HPS grid. Has shape (n_leaves, p^d).

            eval_points_x : jax.Array
                Evaluation points in the x dimension. Has shape (n_x,).

            eval_points_y : jax.Array
                Evaluation points in the y dimension. Has shape (n_y,).

            eval_points_z : optional, Jax.Array
                Evaluation points in the z dimension. Has shape (n_z,). Defaults to None.

        Returns
        -------
        vals : jax.Array
            Function sampled on a rectangular grid of shape (n_x, n_y) or (n_x, n_y, n_z)

        target_pts : jax.Array
            The target points that the vals are sampled on. Has shape (n_x, n_y, 2) or (n_x, n_y, n_z, 3).

        """
        # 2D vs 3D checking
        if isinstance(self.root, DiscretizationNode2D):
            bool_2D = True
            assert eval_points_z is None
        else:
            bool_2D = False
            assert eval_points_z is not None

        n_leaves, n_per_leaf, _ = self.interior_points.shape

        # It's possible that we have samples shaped like (n_leaves, n_per_leaf, n_sources)
        assert samples.shape[:2] == (n_leaves, n_per_leaf)

        if bool_2D:
            if self.bool_uniform:
                leaves = get_all_uniform_leaves_2D(self.root, self.L)
            else:
                leaves = get_all_leaves(self.root)
            return interp_from_hps_2D(
                leaves=leaves,
                p=self.p,
                f_evals=samples,
                x_vals=eval_points_x,
                y_vals=eval_points_y,
            )

        else:
            if self.bool_uniform:
                leaves = get_all_uniform_leaves_3D(self.root, self.L)
            else:
                leaves = get_all_leaves(self.root)
            return interp_from_hps_3D(
                leaves=leaves,
                p=self.p,
                f_evals=samples,
                x_vals=eval_points_x,
                y_vals=eval_points_y,
                z_vals=eval_points_z,
            )

    def get_adaptive_boundary_data_lst(
        self, f: Callable[[jax.Array], jax.Array]
    ) -> List[jax.Array]:
        """
        Given a callable object ``f``, this function evaluates the function at the
        boundary discretization points, and organizes the results into a list. This
        is helpful for specifying boundary conditions in the adaptive case,
        where it is not clear a priori how many points will be on each part of the boundary.

        Parameters
        ----------
            f : Callable[[jax.Array], jax.Array]
                Must have signature [..., d] -> [...].

        Returns
        -------
        List[jax.Array]
            Each element of the list corresponds to a side (2D) or face (3D) of the boundary.
        """
        if self.bool_2D:
            side_0_pts = self.boundary_points[
                self.boundary_points[:, 1] == self.root.ymin
            ]

            side_1_pts = self.boundary_points[
                self.boundary_points[:, 0] == self.root.xmax
            ]
            side_2_pts = self.boundary_points[
                self.boundary_points[:, 1] == self.root.ymax
            ]
            side_3_pts = self.boundary_points[
                self.boundary_points[:, 0] == self.root.xmin
            ]

            bdry_data_lst = [
                f(side_0_pts),
                f(side_1_pts),
                f(side_2_pts),
                f(side_3_pts),
            ]
        else:
            face_1_pts = self.boundary_points[
                self.boundary_points[:, 0] == self.root.xmin
            ]
            face_2_pts = self.boundary_points[
                self.boundary_points[:, 0] == self.root.xmax
            ]
            face_3_pts = self.boundary_points[
                self.boundary_points[:, 1] == self.root.ymin
            ]
            face_4_pts = self.boundary_points[
                self.boundary_points[:, 1] == self.root.ymax
            ]
            face_5_pts = self.boundary_points[
                self.boundary_points[:, 2] == self.root.zmin
            ]
            face_6_pts = self.boundary_points[
                self.boundary_points[:, 2] == self.root.zmax
            ]

            bdry_data_lst = [
                f(face_1_pts),
                f(face_2_pts),
                f(face_3_pts),
                f(face_4_pts),
                f(face_5_pts),
                f(face_6_pts),
            ]

        return bdry_data_lst

    @classmethod
    def from_adaptive_discretization(
        cls,
        p: int,
        q: int,
        root: DiscretizationNode2D | DiscretizationNode3D,
        f: Callable[[jax.Array], jax.Array]
        | List[Callable[[jax.Array], jax.Array]],
        tol: float,
        use_level_restriction: bool = True,
        use_l_2_norm: bool = False,
    ) -> "Domain":
        """
        This is a constructor for creating a ``Domain`` when using an adaptive discretization.
        Given the root of the domain and a function ``f`` to be evaluated on the domain, this method
        will adaptively refine the HPS grid until reaching a specified tolerance ``tol``. Multiple
        functions for adaptive refinement can be specified in a list.

        The tolerance is enforced in the :math:`\\ell_\\infty` norm by default, but can also be enforced in :math:`\\ell_2`.

        Args:
            p (int): Polynomial order for Chebyshev points.

            q (int): Polynomial order for Gauss-Legendre points.

            root (DiscretizationNode2D | DiscretizationNode3D): Specifies the root of the discretization tree.

            f (Callable[[jax.Array], jax.Array] | List[Callable[[jax.Array], jax.Array]]): Function(s) to be used in the adaptive refinement method.

            tol (float): Tolerance parameter.

            use_level_restriction (bool, optional): Whether to enforce the level restriction criterion. Defaults to True. If set to False, could cause errors in the merge stage.

            use_l_2_norm (bool, optional): If set to True, the refinement uses a relative L_2 criterion instead of L_infinity.

        Returns:
            Domain: The domain object with an adaptively-refined discretization tree.
        """
        # Check if f is a list of functions
        if not isinstance(f, list):
            f = [f]

        bool_2D = isinstance(root, DiscretizationNode2D)
        if bool_2D:
            for i, func in enumerate(f):
                generate_adaptive_mesh_level_restriction_2D(
                    root=root,
                    f_fn=func,
                    tol=tol,
                    p=p,
                    q=q,
                    restrict_bool=use_level_restriction,
                    l2_norm=use_l_2_norm,
                )

        else:
            for i, func in enumerate(f):
                generate_adaptive_mesh_level_restriction_3D(
                    root=root,
                    f_fn=func,
                    tol=tol,
                    p=p,
                    q=q,
                    restrict_bool=use_level_restriction,
                    l2_norm=use_l_2_norm,
                )

        return cls(p=p, q=q, root=root)
