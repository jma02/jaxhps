import logging
import jax.numpy as jnp
import numpy as np
from jaxhps._build_solver import build_solver
from jaxhps._domain import Domain
from jaxhps._discretization_tree import (
    DiscretizationNode2D,
    DiscretizationNode3D,
)
from jaxhps._discretization_tree_operations_2D import add_four_children
from jaxhps._discretization_tree_operations_3D import add_eight_children

from jaxhps._pdeproblem import PDEProblem


class Test_build_solver:
    def test_0(self, caplog) -> None:
        """Uniform 2D DtN"""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        L = 2

        # Create a uniform 2D domain
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        domain = Domain(p=p, q=q, root=root, L=L)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )
        source = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
            source=source,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

        assert len(pde_problem.S_lst) == L
        assert len(pde_problem.g_tilde_lst) == L

        assert pde_problem.S_lst[-1].shape == (1, n_bdry // 2, n_bdry)
        assert pde_problem.g_tilde_lst[-1].shape == (1, n_bdry // 2)

    def test_1(self, caplog) -> None:
        """Uniform 2D ItI"""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        L = 2

        # Create a uniform 2D domain
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        domain = Domain(p=p, q=q, root=root, L=L)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )
        source = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
            source=source,
            use_ItI=True,
            eta=1.0,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

        assert len(pde_problem.S_lst) == L
        assert len(pde_problem.g_tilde_lst) == L

        # g_tilde has shape n_bdry in the ItI case.
        assert pde_problem.S_lst[-1].shape == (1, n_bdry, n_bdry)
        assert pde_problem.g_tilde_lst[-1].shape == (1, n_bdry)

    def test_2(self, caplog) -> None:
        """Uniform 3D DtN"""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        L = 2

        # Create a uniform 2D domain
        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0
        )

        domain = Domain(p=p, q=q, root=root, L=L)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )
        source = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
            source=source,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

        assert len(pde_problem.S_lst) == L
        assert len(pde_problem.g_tilde_lst) == L

        logging.debug(
            "Here are S_lst shapes: %s", [s.shape for s in pde_problem.S_lst]
        )

        assert pde_problem.S_lst[-1].shape == (n_bdry // 2, n_bdry)
        assert pde_problem.g_tilde_lst[-1].shape == (n_bdry // 2,)

    def test_3(self, caplog) -> None:
        """Adaptive 2D DtN"""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4

        # Create a uniform 2D domain
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        add_four_children(root, root=root, q=q)
        add_four_children(root.children[0], root=root, q=q)
        add_four_children(root.children[1], root=root, q=q)
        add_four_children(root.children[0].children[1], root=root, q=q)

        domain = Domain(p=p, q=q, root=root)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )
        source = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
            source=source,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

    def test_4(self, caplog) -> None:
        """Adaptive 3D DtN"""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4

        # Create a uniform 2D domain
        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0
        )

        add_eight_children(root, root=root, q=q)
        add_eight_children(root.children[0], root=root, q=q)
        add_eight_children(root.children[1], root=root, q=q)

        domain = Domain(p=p, q=q, root=root)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )
        source = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
            source=source,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

    def test_5(self, caplog) -> None:
        """Uniform 2D DtN with Batching"""
        caplog.set_level(logging.DEBUG)
        p = 7
        q = 4
        L = 3

        # Create a uniform 2D domain
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        domain = Domain(p=p, q=q, root=root, L=L)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )
        source = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
            source=source,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

        assert len(pde_problem.S_lst) == L
        assert len(pde_problem.g_tilde_lst) == L

        assert pde_problem.S_lst[-1].shape == (1, n_bdry // 2, n_bdry)
        assert pde_problem.g_tilde_lst[-1].shape == (1, n_bdry // 2)

    def test_6(self, caplog) -> None:
        """Uniform 2D ItI"""
        caplog.set_level(logging.DEBUG)
        p = 7
        q = 4
        L = 3

        # Create a uniform 2D domain
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        domain = Domain(p=p, q=q, root=root, L=L)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )
        source = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
            source=source,
            use_ItI=True,
            eta=1.0,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

        assert len(pde_problem.S_lst) == L
        assert len(pde_problem.g_tilde_lst) == L

        # g_tilde has shape n_bdry in the ItI case.
        assert pde_problem.S_lst[-1].shape == (1, n_bdry, n_bdry)
        assert pde_problem.g_tilde_lst[-1].shape == (1, n_bdry)

    def test_7(self, caplog) -> None:
        """Uniform 2D ItI with no source"""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        L = 2

        # Create a uniform 2D domain
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        domain = Domain(p=p, q=q, root=root, L=L)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
            use_ItI=True,
            eta=1.0,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

        assert len(pde_problem.S_lst) == L

        # g_tilde has shape n_bdry in the ItI case.
        assert pde_problem.S_lst[-1].shape == (1, n_bdry, n_bdry)

    def test_8(self, caplog) -> None:
        """Uniform 2D DtN with no source"""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        L = 2

        # Create a uniform 2D domain
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        domain = Domain(p=p, q=q, root=root, L=L)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

        assert len(pde_problem.S_lst) == L

        # g_tilde has shape n_bdry in the ItI case.
        assert pde_problem.S_lst[-1].shape == (1, n_bdry // 2, n_bdry)
