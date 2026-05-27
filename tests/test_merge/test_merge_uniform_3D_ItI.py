import numpy as np
import logging
from jaxhps.merge._uniform_3D_ItI import (
    merge_stage_uniform_3D_ItI,
    _uniform_oct_merge_ItI,
)
import jax
import jax.numpy as jnp

from jaxhps.local_solve._uniform_3D_ItI import (
    local_solve_stage_uniform_3D_ItI,
)
from jaxhps._discretization_tree import DiscretizationNode3D
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem


class Test_merge_stage_uniform_3D_ItI:
    def test_0(self, caplog) -> None:
        # Smoke: shape contract for a single oct merge with constant
        # Helmholtz coefficient (so the local solve has a well-defined
        # ItI map).
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        l = 1
        eta = 4.0
        root = DiscretizationNode3D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        n_leaves = 8**l
        domain = Domain(p=p, q=q, root=root, L=l)

        d_xx = np.ones((n_leaves, p**3))
        d_yy = np.ones((n_leaves, p**3))
        d_zz = np.ones((n_leaves, p**3))
        I_coeffs = np.full((n_leaves, p**3), eta**2)
        source_term = np.zeros((n_leaves, p**3))

        t = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx,
            D_yy_coefficients=d_yy,
            D_zz_coefficients=d_zz,
            I_coefficients=I_coeffs,
            source=source_term,
            use_ItI=True,
            eta=eta,
        )

        Y_arr, T_arr, v, h = local_solve_stage_uniform_3D_ItI(t)
        assert Y_arr.shape == (n_leaves, p**3, 6 * q**2)
        assert T_arr.shape == (n_leaves, 6 * q**2, 6 * q**2)

        S_arr_lst, g_tilde_lst = merge_stage_uniform_3D_ItI(
            T_arr=T_arr, h_arr=h, l=l
        )
        assert len(S_arr_lst) == l
        assert len(g_tilde_lst) == l
        for i in range(l):
            logging.debug(
                "S[%d].shape = %s, g_tilde[%d].shape = %s",
                i,
                S_arr_lst[i].shape,
                i,
                g_tilde_lst[i].shape,
            )
            assert S_arr_lst[i].shape[-2] == g_tilde_lst[i].shape[-1]
        jax.clear_caches()

    def test_1(self, caplog) -> None:
        # Multi-source smoke: same as test_0 with nsrc > 1.
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        l = 1
        nsrc = 2
        eta = 4.0
        root = DiscretizationNode3D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        n_leaves = 8**l
        domain = Domain(p=p, q=q, root=root, L=l)

        d_xx = np.ones((n_leaves, p**3))
        d_yy = np.ones((n_leaves, p**3))
        d_zz = np.ones((n_leaves, p**3))
        I_coeffs = np.full((n_leaves, p**3), eta**2)
        source_term = np.zeros((n_leaves, p**3, nsrc))

        t = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx,
            D_yy_coefficients=d_yy,
            D_zz_coefficients=d_zz,
            I_coefficients=I_coeffs,
            source=source_term,
            use_ItI=True,
            eta=eta,
        )

        Y_arr, T_arr, v, h = local_solve_stage_uniform_3D_ItI(t)
        assert Y_arr.shape == (n_leaves, p**3, 6 * q**2)
        assert T_arr.shape == (n_leaves, 6 * q**2, 6 * q**2)

        S_arr_lst, g_tilde_lst = merge_stage_uniform_3D_ItI(
            T_arr=T_arr, h_arr=h, l=l
        )
        assert len(S_arr_lst) == l
        assert len(g_tilde_lst) == l
        jax.clear_caches()


class Test_merge_planewave_correctness:
    r"""Verify the merged T maps the global impedance trace correctly for a
    manufactured plane-wave solution u = \exp(i k \cdot x) on the 8-leaf cube.

    With (\Delta + \kappa^2) u = 0 satisfied identically,
    \partial u / \partial n = i (k \cdot n) u.  The ItI traces are
        g_in  = (\partial u / \partial n) + i \eta u = i (k \cdot n + \eta) u
        g_out = (\partial u / \partial n) - i \eta u = i (k \cdot n - \eta) u
    so we expect T_merged @ g_in \approx g_out at spectral precision.
    """

    @staticmethod
    def _outward_normals_for_boundary(
        boundary_points: np.ndarray, root
    ) -> np.ndarray:
        """Return unit outward normal at each boundary Gauss point."""
        n = np.zeros_like(boundary_points)
        eps = 1e-9
        # Faces: x=xmin, x=xmax, y=ymin, y=ymax, z=zmin, z=zmax
        n[np.abs(boundary_points[:, 0] - root.xmin) < eps] = [-1, 0, 0]
        n[np.abs(boundary_points[:, 0] - root.xmax) < eps] = [1, 0, 0]
        n[np.abs(boundary_points[:, 1] - root.ymin) < eps] = [0, -1, 0]
        n[np.abs(boundary_points[:, 1] - root.ymax) < eps] = [0, 1, 0]
        n[np.abs(boundary_points[:, 2] - root.zmin) < eps] = [0, 0, -1]
        n[np.abs(boundary_points[:, 2] - root.zmax) < eps] = [0, 0, 1]
        return n

    def test_planewave(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        kappa = 4.0
        eta = kappa
        k_vec = kappa * np.array([1.0, 2.0, 2.0]) / 3.0
        results = []
        for p, q in [(8, 6), (12, 8), (16, 10)]:
            l = 1
            root = DiscretizationNode3D(
                xmin=-0.5,
                xmax=0.5,
                ymin=-0.5,
                ymax=0.5,
                zmin=-0.5,
                zmax=0.5,
                depth=0,
            )
            n_leaves = 8**l
            domain = Domain(p=p, q=q, root=root, L=l)

            ones = np.ones((n_leaves, p**3))
            I_coeffs = kappa**2 * ones
            src = np.zeros((n_leaves, p**3))

            problem = PDEProblem(
                domain=domain,
                D_xx_coefficients=ones,
                D_yy_coefficients=ones,
                D_zz_coefficients=ones,
                I_coefficients=I_coeffs,
                source=src,
                use_ItI=True,
                eta=eta,
            )

            _, T_arr, _, h_arr = local_solve_stage_uniform_3D_ItI(problem)
            S_lst, g_tilde_lst, T_top = merge_stage_uniform_3D_ItI(
                T_arr=T_arr,
                h_arr=h_arr,
                l=l,
                return_T=True,
            )

            bp = np.asarray(domain.boundary_points).reshape(-1, 3)
            n_dirs = self._outward_normals_for_boundary(bp, root)
            assert np.all(np.linalg.norm(n_dirs, axis=1) > 0.5), (
                "Some boundary points were not assigned a normal"
            )

            u_b = np.exp(1j * (bp @ k_vec))
            kn = n_dirs @ k_vec
            g_in = 1j * (kn + eta) * u_b
            g_out = 1j * (kn - eta) * u_b

            T_top_np = np.asarray(T_top)
            pred = T_top_np @ g_in
            err = np.max(np.abs(pred - g_out)) / np.max(np.abs(g_out))
            results.append((p, q, float(err)))
            logging.debug("p=%d q=%d -> rel err = %.3e", p, q, err)
            jax.clear_caches()

        # Sanity: error should drop with p, and the tightest grid should
        # reach near-spectral precision (well below 1e-3).
        assert results[-1][2] < 1e-3, f"Spectral test failed: {results}"
        assert results[-1][2] < results[0][2], (
            f"Convergence not observed: {results}"
        )


class Test__uniform_oct_merge_ItI:
    def test_0(self):
        # Random complex inputs of the right shape - check shape contract.
        q = 3
        n_gauss_bdry = 6 * q**2
        rng = np.random.default_rng(0)

        def rand_T():
            return rng.normal(
                size=(n_gauss_bdry, n_gauss_bdry)
            ) + 1j * rng.normal(size=(n_gauss_bdry, n_gauss_bdry))

        def rand_h():
            return rng.normal(size=(n_gauss_bdry,)) + 1j * rng.normal(
                size=(n_gauss_bdry,)
            )

        T_a, T_b, T_c, T_d = rand_T(), rand_T(), rand_T(), rand_T()
        T_e, T_f, T_g, T_h = rand_T(), rand_T(), rand_T(), rand_T()
        h_a, h_b, h_c, h_d = rand_h(), rand_h(), rand_h(), rand_h()
        h_e, h_f, h_g, h_h = rand_h(), rand_h(), rand_h(), rand_h()
        q_idxes = jnp.arange(q)

        S, T, h_ext, g_int = _uniform_oct_merge_ItI(
            q_idxes,
            jnp.asarray(T_a, dtype=jnp.complex128),
            jnp.asarray(T_b, dtype=jnp.complex128),
            jnp.asarray(T_c, dtype=jnp.complex128),
            jnp.asarray(T_d, dtype=jnp.complex128),
            jnp.asarray(T_e, dtype=jnp.complex128),
            jnp.asarray(T_f, dtype=jnp.complex128),
            jnp.asarray(T_g, dtype=jnp.complex128),
            jnp.asarray(T_h, dtype=jnp.complex128),
            jnp.asarray(h_a, dtype=jnp.complex128),
            jnp.asarray(h_b, dtype=jnp.complex128),
            jnp.asarray(h_c, dtype=jnp.complex128),
            jnp.asarray(h_d, dtype=jnp.complex128),
            jnp.asarray(h_e, dtype=jnp.complex128),
            jnp.asarray(h_f, dtype=jnp.complex128),
            jnp.asarray(h_g, dtype=jnp.complex128),
            jnp.asarray(h_h, dtype=jnp.complex128),
        )

        assert T.shape == (24 * q**2, 24 * q**2)
        assert S.shape == (24 * q**2, 24 * q**2)
        assert h_ext.shape == (24 * q**2,)
        assert g_int.shape == (24 * q**2,)
        jax.clear_caches()
