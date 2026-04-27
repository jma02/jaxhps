"""End-to-end pipeline test for the uniform 3D ItI solver path.

Builds the full solver (local solve + 3D oct merge) for the homogeneous
constant-coefficient Helmholtz equation

    Δu + κ² u = 0

on the cube [-0.5, 0.5]^3 (one oct merge, L=1, 8 leaves), drives it with
the impedance trace of the analytical plane-wave solution
``u(x) = exp(i k . x)`` with ``|k|=κ``, runs the down-pass, and checks
that the recovered interior solution matches the analytical one to
spectral precision.
"""
import logging

import numpy as np
import jax
import jax.numpy as jnp

from jaxhps._discretization_tree import DiscretizationNode3D
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem
from jaxhps._build_solver import build_solver
from jaxhps._solve import solve


def _outward_normals_for_boundary(boundary_points: np.ndarray, root) -> np.ndarray:
    n = np.zeros_like(boundary_points)
    eps = 1e-9
    n[np.abs(boundary_points[:, 0] - root.xmin) < eps] = [-1, 0, 0]
    n[np.abs(boundary_points[:, 0] - root.xmax) < eps] = [1, 0, 0]
    n[np.abs(boundary_points[:, 1] - root.ymin) < eps] = [0, -1, 0]
    n[np.abs(boundary_points[:, 1] - root.ymax) < eps] = [0, 1, 0]
    n[np.abs(boundary_points[:, 2] - root.zmin) < eps] = [0, 0, -1]
    n[np.abs(boundary_points[:, 2] - root.zmax) < eps] = [0, 0, 1]
    return n


class TestPipelineUniform3DItIPlanewave:
    def test_planewave_pipeline(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        kappa = 4.0
        eta = kappa
        # k_vec direction (1,2,2)/3 -> |k|=κ
        k_vec = kappa * np.array([1.0, 2.0, 2.0]) / 3.0
        L = 1
        results = []
        for p, q in [(8, 6), (12, 8), (16, 10)]:
            root = DiscretizationNode3D(
                xmin=-0.5, xmax=0.5,
                ymin=-0.5, ymax=0.5,
                zmin=-0.5, zmax=0.5,
            )
            n_leaves = 8 ** L
            domain = Domain(p=p, q=q, root=root, L=L)

            ones = np.ones((n_leaves, p**3))
            I_coeffs = (kappa ** 2) * ones
            src = np.zeros((n_leaves, p**3))

            pde_problem = PDEProblem(
                domain=domain,
                D_xx_coefficients=ones,
                D_yy_coefficients=ones,
                D_zz_coefficients=ones,
                I_coefficients=I_coeffs,
                source=src,
                use_ItI=True,
                eta=eta,
            )

            T_top = build_solver(pde_problem, return_top_T=True)

            bp = np.asarray(domain.boundary_points).reshape(-1, 3)
            n_dirs = _outward_normals_for_boundary(bp, root)
            assert np.all(np.linalg.norm(n_dirs, axis=1) > 0.5)

            u_b = np.exp(1j * (bp @ k_vec))
            kn = n_dirs @ k_vec
            g_in = 1j * (kn + eta) * u_b
            g_out = 1j * (kn - eta) * u_b

            T_top_np = np.asarray(T_top)
            pred_out = T_top_np @ g_in
            err_top = np.max(np.abs(pred_out - g_out)) / np.max(np.abs(g_out))

            # Run the down-pass
            solns = solve(pde_problem, jnp.asarray(g_in))
            assert solns.shape == (n_leaves, p**3)

            # Compare to analytical plane-wave at every Cheby interior point
            interior_pts = np.asarray(domain.interior_points).reshape(
                n_leaves, p**3, 3
            )
            u_exact = np.exp(1j * (interior_pts @ k_vec))
            u_solver = np.asarray(solns)

            err_int = np.max(np.abs(u_solver - u_exact)) / np.max(np.abs(u_exact))
            results.append((p, q, float(err_top), float(err_int)))
            logging.info(
                "p=%d q=%d  T-err=%.3e  interior-err=%.3e",
                p, q, err_top, err_int,
            )
            jax.clear_caches()

        # The finest grid should be near spectral precision and convergent.
        assert results[-1][3] < 1e-3, f"Pipeline planewave test failed: {results}"
        assert results[-1][3] < results[0][3], (
            f"Convergence not observed: {results}"
        )

    def test_planewave_pipeline_L2(self, caplog) -> None:
        """Same plane-wave test but on a 64-leaf (L=2) octree.  This
        exercises the multi-level merge + multi-level down-pass."""
        caplog.set_level(logging.DEBUG)
        kappa = 4.0
        eta = kappa
        k_vec = kappa * np.array([1.0, 2.0, 2.0]) / 3.0
        L = 2
        p, q = 8, 6

        root = DiscretizationNode3D(
            xmin=-0.5, xmax=0.5,
            ymin=-0.5, ymax=0.5,
            zmin=-0.5, zmax=0.5,
        )
        n_leaves = 8 ** L
        domain = Domain(p=p, q=q, root=root, L=L)

        ones = np.ones((n_leaves, p**3))
        I_coeffs = (kappa ** 2) * ones
        src = np.zeros((n_leaves, p**3))

        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=ones,
            D_yy_coefficients=ones,
            D_zz_coefficients=ones,
            I_coefficients=I_coeffs,
            source=src,
            use_ItI=True,
            eta=eta,
        )

        T_top = build_solver(pde_problem, return_top_T=True)

        bp = np.asarray(domain.boundary_points).reshape(-1, 3)
        n_dirs = _outward_normals_for_boundary(bp, root)
        u_b = np.exp(1j * (bp @ k_vec))
        kn = n_dirs @ k_vec
        g_in = 1j * (kn + eta) * u_b
        g_out = 1j * (kn - eta) * u_b

        pred_out = np.asarray(T_top) @ g_in
        err_top = np.max(np.abs(pred_out - g_out)) / np.max(np.abs(g_out))

        solns = solve(pde_problem, jnp.asarray(g_in))
        interior_pts = np.asarray(domain.interior_points).reshape(
            n_leaves, p**3, 3
        )
        u_exact = np.exp(1j * (interior_pts @ k_vec))
        u_solver = np.asarray(solns)
        err_int = np.max(np.abs(u_solver - u_exact)) / np.max(np.abs(u_exact))
        logging.info(
            "L=2 p=%d q=%d  T-err=%.3e  interior-err=%.3e",
            p, q, err_top, err_int,
        )
        assert err_int < 1e-3, f"L=2 pipeline test failed: err={err_int}"

    def test_manufactured_source_variable_coefficients(self, caplog) -> None:
        """Manufactured solution with variable I coefficient and non-zero
        source.  Verifies the source path of the pipeline (not exercised
        by the homogeneous plane-wave test).

        u(x,y,z) = sin(x) cos(y) sin(z)  ->  Lap u = -3 u
        I(x,y,z) = kappa^2 + 2 + sin(x) cos(y) cos(z) x  (variable)
        Equation: Lap u + I u = source  =>  source = (I - 3) u
        """
        caplog.set_level(logging.DEBUG)
        kappa = 4.0
        eta = kappa  # any non-zero eta works for this test (it is not an absorbing BC test)
        p, q, L = 12, 10, 1
        root = DiscretizationNode3D(
            xmin=-0.4, xmax=0.4, ymin=-0.4, ymax=0.4, zmin=-0.4, zmax=0.4,
        )
        domain = Domain(p=p, q=q, root=root, L=L)
        pts = np.asarray(domain.interior_points)
        xg, yg, zg = pts[..., 0], pts[..., 1], pts[..., 2]
        u_int = np.sin(xg) * np.cos(yg) * np.sin(zg)
        I_var = (kappa ** 2 + 2.0 + np.sin(xg) * np.cos(yg) * np.cos(zg) * xg)
        src = (-3.0 * u_int + I_var * u_int).astype(np.complex128)

        n_leaves = pts.shape[0]
        ones = np.ones((n_leaves, p**3))
        problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=ones, D_yy_coefficients=ones, D_zz_coefficients=ones,
            I_coefficients=I_var.astype(np.complex128), source=src,
            use_ItI=True, eta=eta,
        )
        build_solver(problem)

        bp = np.asarray(domain.boundary_points).reshape(-1, 3)
        nrm = _outward_normals_for_boundary(bp, root)
        u_b = np.sin(bp[:, 0]) * np.cos(bp[:, 1]) * np.sin(bp[:, 2])
        gn = (np.cos(bp[:, 0]) * np.cos(bp[:, 1]) * np.sin(bp[:, 2]) * nrm[:, 0]
              - np.sin(bp[:, 0]) * np.sin(bp[:, 1]) * np.sin(bp[:, 2]) * nrm[:, 1]
              + np.sin(bp[:, 0]) * np.cos(bp[:, 1]) * np.cos(bp[:, 2]) * nrm[:, 2])
        g_in = (gn + 1j * eta * u_b).astype(np.complex128)
        solns = np.asarray(solve(problem, jnp.asarray(g_in)))
        err = np.max(np.abs(solns - u_int)) / np.max(np.abs(u_int))
        logging.info("manufactured-source variable-coeff err: %.3e", err)
        assert err < 1e-9, f"manufactured-source test failed: err={err}"
        jax.clear_caches()
