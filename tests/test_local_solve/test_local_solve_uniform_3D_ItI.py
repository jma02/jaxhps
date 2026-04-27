"""Tests for the 3D uniform ItI local solve stage.

Verifies:
* Output shape contracts (single-source and multi-source).
* Spectral convergence of the impedance-to-impedance (T) and
  impedance-to-interior-solution (Y) maps against a manufactured plane-wave
  Helmholtz solution.
"""
import numpy as np
import jax.numpy as jnp
import jax

from jaxhps.local_solve._uniform_3D_ItI import (
    local_solve_stage_uniform_3D_ItI,
    get_ItI_3D,
)
from jaxhps._discretization_tree import DiscretizationNode3D
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem


def _outward_normals(q: int) -> np.ndarray:
    """Outward unit normal at each Gauss boundary node, ordered by face."""
    n_dirs = np.zeros((6 * q * q, 3))
    for f, nd in enumerate(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    ):
        n_dirs[f * q * q : (f + 1) * q * q] = nd
    return n_dirs


class Test_local_solve_stage_uniform_3D_ItI:
    def test_shapes_single_leaf(self) -> None:
        """Single-leaf shapes are correct for a single source term."""
        p, q = 8, 6
        eta = 4.0
        root = DiscretizationNode3D(
            xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5, zmin=-0.5, zmax=0.5
        )
        domain = Domain(p=p, q=q, root=root, L=0)
        ones = np.ones((1, p**3))
        src = np.zeros((1, p**3))
        problem = PDEProblem(
            domain=domain,
            source=src,
            D_xx_coefficients=ones,
            D_yy_coefficients=ones,
            D_zz_coefficients=ones,
            I_coefficients=16.0 * ones,
            use_ItI=True,
            eta=eta,
        )
        Y, T, v, h = local_solve_stage_uniform_3D_ItI(problem)
        assert Y.shape == (1, p**3, 6 * q**2)
        assert T.shape == (1, 6 * q**2, 6 * q**2)
        assert v.shape == (1, p**3)
        assert h.shape == (1, 6 * q**2)
        jax.clear_caches()

    def test_shapes_multisource(self) -> None:
        """Shapes propagate the source axis when given a multi-source RHS."""
        p, q = 8, 6
        eta = 4.0
        n_src = 4
        root = DiscretizationNode3D(
            xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5, zmin=-0.5, zmax=0.5
        )
        domain = Domain(p=p, q=q, root=root, L=0)
        ones = np.ones((1, p**3))
        src = np.zeros((1, p**3, n_src))
        problem = PDEProblem(
            domain=domain,
            source=src,
            D_xx_coefficients=ones,
            D_yy_coefficients=ones,
            D_zz_coefficients=ones,
            I_coefficients=16.0 * ones,
            use_ItI=True,
            eta=eta,
        )
        Y, T, v, h = local_solve_stage_uniform_3D_ItI(problem)
        assert v.shape == (1, p**3, n_src)
        assert h.shape == (1, 6 * q**2, n_src)
        jax.clear_caches()

    def test_planewave_spectral_convergence(self) -> None:
        """Verify spectral convergence on the homogeneous Helmholtz equation
        (Δ + κ²)u = 0 with manufactured plane-wave solution u = exp(i k·x),
        |k| = κ. Asserts that the impedance-to-impedance and
        impedance-to-interior maps converge spectrally as p increases.
        """
        kappa = 4.0
        eta = kappa
        k_vec = kappa * np.array([1.0, 2.0, 2.0]) / 3.0
        half = 0.5
        errors_T = []
        errors_Y = []
        for p, q in [(8, 6), (12, 10), (16, 12)]:
            root = DiscretizationNode3D(
                xmin=-half,
                xmax=half,
                ymin=-half,
                ymax=half,
                zmin=-half,
                zmax=half,
            )
            domain = Domain(p=p, q=q, root=root, L=0)
            ones = np.ones((1, p**3))
            src = np.zeros((1, p**3))
            problem = PDEProblem(
                domain=domain,
                source=src,
                D_xx_coefficients=ones,
                D_yy_coefficients=ones,
                D_zz_coefficients=ones,
                I_coefficients=kappa**2 * ones,
                use_ItI=True,
                eta=eta,
            )
            Y, T, _, _ = local_solve_stage_uniform_3D_ItI(problem)

            cheby = np.asarray(domain.interior_points).reshape(-1, 3)
            bp = np.asarray(domain.boundary_points).reshape(-1, 3)
            u_cheby = np.exp(1j * (cheby @ k_vec))
            u_g = np.exp(1j * (bp @ k_vec))
            n_dirs = _outward_normals(q)
            g_in = (1j * (n_dirs @ k_vec) + 1j * eta) * u_g
            g_out = (1j * (n_dirs @ k_vec) - 1j * eta) * u_g

            T0 = np.asarray(T[0])
            Y0 = np.asarray(Y[0])
            err_T = np.max(np.abs(T0 @ g_in - g_out)) / np.max(np.abs(g_out))
            err_Y = np.max(np.abs(Y0 @ g_in - u_cheby)) / np.max(
                np.abs(u_cheby)
            )
            errors_T.append(float(err_T))
            errors_Y.append(float(err_Y))
            jax.clear_caches()

        # Loose absolute tolerances: the tightest (p=16) point should be near
        # double precision, the coarsest (p=8) within 1e-3.
        assert errors_T[0] < 1e-3
        assert errors_T[2] < 1e-9
        assert errors_Y[0] < 1e-3
        assert errors_Y[2] < 1e-9
        # Spectral convergence: errors should decrease monotonically.
        assert errors_T[1] < errors_T[0]
        assert errors_T[2] < errors_T[1]
        assert errors_Y[1] < errors_Y[0]
        assert errors_Y[2] < errors_Y[1]


class Test_get_ItI_3D:
    def test_shapes(self) -> None:
        """Verify shapes from a single get_ItI_3D call with random operators."""
        p, q = 6, 4
        n_cheby = p**3
        n_bdry = p**3 - (p - 2) ** 3
        n_gauss = 6 * q**2

        rng = np.random.default_rng(0)
        diff_op = jnp.array(rng.normal(size=(n_cheby, n_cheby)))
        source = jnp.array(rng.normal(size=(n_cheby, 1)))
        P = jnp.array(rng.normal(size=(n_bdry, n_gauss)))
        QH = jnp.array(
            rng.normal(size=(n_gauss, n_cheby))
            + 1j * rng.normal(size=(n_gauss, n_cheby))
        )
        G = jnp.array(
            rng.normal(size=(n_bdry, n_cheby))
            + 1j * rng.normal(size=(n_bdry, n_cheby))
        )
        T, Y, h, v = get_ItI_3D(
            diff_operator=diff_op,
            source_term=source,
            P=P,
            QH=QH,
            G=G,
        )
        assert T.shape == (n_gauss, n_gauss)
        assert Y.shape == (n_cheby, n_gauss)
        assert h.shape == (n_gauss, 1)
        assert v.shape == (n_cheby, 1)
        jax.clear_caches()
