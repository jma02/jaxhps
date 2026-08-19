"""Tests for the matrix-free (unmerged) 3D ItI interface system.

Three checks, in increasing strength:

* the interface gluing residual vanishes for the traces of an exact
  plane-wave solution (validates the partner map and the normal-flip sign
  convention);
* the flat system reproduces the action of the dense root ItI map returned by
  ``build_solver(..., return_top_T=True)``, both for the homogeneous problem
  and with a source term;
* the impedance-to-Dirichlet/Neumann conversion on the domain boundary
  matches the Cayley-transformed dense DtN map.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from jaxhps import DiscretizationNode3D, Domain, PDEProblem, build_solver
from jaxhps._matfree_iti_3D import (
    boundary_traces_from_z,
    build_interface_maps,
    dense_solve_flat,
    interface_residual,
    leaf_boundary_gauss_points,
    leaf_boundary_normals,
    make_root_ItI_operator,
    root_ItI_matvec,
    root_ItI_rhs,
)
from jaxhps.local_solve import local_solve_stage_uniform_3D_ItI

_EX_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "examples")
sys.path.insert(0, os.path.abspath(_EX_DIR))

from wave_scattering_utils_3D import (  # noqa: E402
    make_dense_SD_apply,
    solve_bie_flat_matfree,
)


def _problem(p, q, L, kappa, eta, half=0.5, n_src=None, bump=False):
    root = DiscretizationNode3D(
        xmin=-half,
        xmax=half,
        ymin=-half,
        ymax=half,
        zmin=-half,
        zmax=half,
    )
    domain = Domain(p=p, q=q, root=root, L=L)
    int_pts = np.asarray(domain.interior_points)
    n_leaves = int_pts.shape[0]
    ones = np.ones((n_leaves, p**3))
    if bump:
        r = np.linalg.norm(int_pts, axis=-1)
        R = 0.3
        b = np.where(r < R, -0.4 * (1.0 - (r / R) ** 2) ** 4, 0.0)
    else:
        b = np.zeros((n_leaves, p**3))
    I_coeffs = kappa**2 * (1.0 - b)
    if n_src is None:
        src = np.zeros((n_leaves, p**3), dtype=np.complex128)
    else:
        rng = np.random.default_rng(3)
        src = rng.normal(size=(n_leaves, p**3, n_src)) + 1j * rng.normal(
            size=(n_leaves, p**3, n_src)
        )
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
    return domain, problem


def test_interface_maps_pairing() -> None:
    """Each leaf boundary node has either one partner or lies on the boundary."""
    q, L = 4, 1
    domain, _ = _problem(p=8, q=q, L=L, kappa=4.0, eta=4.0)
    maps = build_interface_maps(domain)
    partner = np.asarray(maps.partner)
    assert maps.n_leaves == 8**L
    assert maps.n_per_leaf == 6 * q**2
    # 12 interior face-quads of the octant, each with q**2 nodes on 2 sides.
    assert int((partner >= 0).sum()) == 2 * 12 * q**2
    # The pairing is an involution on the interior nodes.
    interior = np.flatnonzero(partner >= 0)
    assert np.array_equal(partner[partner[interior]], interior)
    # Boundary rows cover every node of Domain.boundary_points exactly once.
    bdry_rows = np.asarray(maps.bdry_rows)
    assert (
        bdry_rows.size
        == np.asarray(domain.boundary_points).reshape(-1, 3).shape[0]
    )
    assert np.unique(bdry_rows).size == bdry_rows.size
    jax.clear_caches()


def test_planewave_interface_residual() -> None:
    """Exact plane-wave traces satisfy the gluing conditions."""
    p, q, L = 12, 8, 1
    kappa, eta = 4.0, 4.0
    domain, problem = _problem(p=p, q=q, L=L, kappa=kappa, eta=eta)
    _, T_leaves, _, _ = local_solve_stage_uniform_3D_ItI(problem)
    maps = build_interface_maps(domain)

    k_vec = kappa * np.array([1.0, 2.0, 2.0]) / 3.0
    pts = leaf_boundary_gauss_points(domain.root, L, q).reshape(-1, 3)
    nrm = np.tile(leaf_boundary_normals(q), (maps.n_leaves, 1))
    u = np.exp(1j * (pts @ k_vec))
    g_in = (1j * (nrm @ k_vec) + 1j * eta) * u

    res = np.asarray(
        interface_residual(T_leaves, None, maps, jnp.asarray(g_in))
    )
    rel = np.max(np.abs(res)) / np.max(np.abs(g_in))
    # At p = 12 the leaf ItI maps themselves are accurate to ~1e-9, so this
    # is at the level of the local solve's discretization error.
    assert rel < 1e-8, f"interface residual too large: {rel:.3e}"
    jax.clear_caches()


def test_root_ItI_action_matches_dense() -> None:
    """Matrix-free root ItI action equals the merged dense top-level map."""
    p, q, L = 8, 4, 1
    kappa, eta = 4.0, 4.0
    domain, problem = _problem(p=p, q=q, L=L, kappa=kappa, eta=eta, bump=True)
    _, T_leaves, _, _ = local_solve_stage_uniform_3D_ItI(problem)
    maps = build_interface_maps(domain)
    T_root = np.asarray(build_solver(problem, return_top_T=True))

    n_bdry = T_root.shape[0]
    rng = np.random.default_rng(0)
    g_in = rng.normal(size=n_bdry) + 1j * rng.normal(size=n_bdry)

    g_out_mf = np.asarray(
        root_ItI_matvec(T_leaves, maps, jnp.asarray(g_in), dense_solve_flat)
    )
    g_out_dense = T_root @ g_in
    rel = np.linalg.norm(g_out_mf - g_out_dense) / np.linalg.norm(g_out_dense)
    assert rel < 1e-10, f"root ItI action mismatch: {rel:.3e}"
    jax.clear_caches()


def test_root_ItI_with_source_matches_dense() -> None:
    """With a source term, the flat solve matches the merged solver's traces.

    ``build_solver`` returns the top-level ItI map and stores the merge
    hierarchy, but not the top-level particular impedance data, so this test
    compares the total outgoing impedance obtained from the flat system
    against the merged ItI map applied to the same incoming data plus the
    particular part recovered from a second flat solve with zero boundary
    data (an internal consistency check of linearity plus the dense map).
    """
    p, q, L = 8, 4, 1
    kappa, eta = 4.0, 4.0
    domain, problem = _problem(
        p=p, q=q, L=L, kappa=kappa, eta=eta, n_src=2, bump=True
    )
    _, T_leaves, _, h_leaves = local_solve_stage_uniform_3D_ItI(problem)
    maps = build_interface_maps(domain)
    T_root = np.asarray(build_solver(problem, return_top_T=True))

    n_bdry = T_root.shape[0]
    rng = np.random.default_rng(1)
    g_in = rng.normal(size=(n_bdry, 2)) + 1j * rng.normal(size=(n_bdry, 2))

    g_tot = np.asarray(
        root_ItI_matvec(
            T_leaves,
            maps,
            jnp.asarray(g_in),
            dense_solve_flat,
            h_leaves=h_leaves,
        )
    )
    g_part = np.asarray(
        root_ItI_matvec(
            T_leaves,
            maps,
            jnp.zeros_like(jnp.asarray(g_in)),
            dense_solve_flat,
            h_leaves=h_leaves,
        )
    )
    g_hom = g_tot - g_part
    rel = np.linalg.norm(g_hom - T_root @ g_in) / np.linalg.norm(T_root @ g_in)
    assert rel < 1e-10, f"homogeneous part mismatch: {rel:.3e}"
    jax.clear_caches()


def test_boundary_traces_match_dense_DtN() -> None:
    """Traces from the flat solve satisfy the dense Cayley DtN relation."""
    p, q, L = 8, 4, 1
    kappa, eta = 4.0, 4.0
    domain, problem = _problem(p=p, q=q, L=L, kappa=kappa, eta=eta, bump=True)
    _, T_leaves, _, _ = local_solve_stage_uniform_3D_ItI(problem)
    maps = build_interface_maps(domain)
    T_root = np.asarray(build_solver(problem, return_top_T=True))
    n_bdry = T_root.shape[0]
    T_DtN = (
        -1j
        * eta
        * np.linalg.solve(T_root - np.eye(n_bdry), T_root + np.eye(n_bdry))
    )

    rng = np.random.default_rng(2)
    g_in = rng.normal(size=n_bdry) + 1j * rng.normal(size=n_bdry)
    matvec = make_root_ItI_operator(T_leaves, maps)
    z = dense_solve_flat(matvec, root_ItI_rhs(None, maps, jnp.asarray(g_in)))
    u, u_n = boundary_traces_from_z(T_leaves, None, maps, z, eta)
    u = np.asarray(u)
    u_n = np.asarray(u_n)
    rel = np.linalg.norm(T_DtN @ u - u_n) / np.linalg.norm(u_n)
    assert rel < 1e-9, f"DtN relation violated: {rel:.3e}"
    jax.clear_caches()


def _synthetic_SD(n, seed=7):
    """Well-conditioned stand-ins for the single/double-layer matrices.

    The flat-vs-dense comparison is an algebraic identity for *any* pair of
    boundary operators, so the tests below use random matrices rather than
    fmm3dbie fixtures; this keeps the equivalence check independent of the
    quadrature and available without the Fortran dependency.
    """
    rng = np.random.default_rng(seed)
    S = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / (
        10.0 * np.sqrt(n)
    )
    D = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / (
        10.0 * np.sqrt(n)
    )
    return S, D


def _dense_bie_reference(T_root, S, D, uin, uin_dn, eta):
    n = uin.shape[0]
    T_DtN = -1j * eta * np.linalg.solve(T_root - np.eye(n), T_root + np.eye(n))
    A = 0.5 * np.eye(n) - D + S @ T_DtN
    rhs = S @ (uin_dn - T_DtN @ uin)
    uscat_b = np.linalg.solve(A, rhs)
    uscat_dn_b = T_DtN @ (uscat_b + uin) - uin_dn
    return uscat_b, uscat_dn_b


def _plane_wave_bdry_traces(domain, kappa, dirs):
    bp = np.asarray(domain.boundary_points).reshape(-1, 3)
    a = domain.root.xmax
    nrm = np.zeros_like(bp)
    for axis in range(3):
        nrm[np.isclose(bp[:, axis], -a), axis] = -1.0
        nrm[np.isclose(bp[:, axis], a), axis] = 1.0
    k_vecs = kappa * np.asarray(dirs)
    phases = bp @ k_vecs.T
    uin = np.exp(1j * phases)
    uin_dn = 1j * (nrm @ k_vecs.T) * uin
    return uin, uin_dn


def test_flat_bie_matches_dense_DtN_bie() -> None:
    """The flat coupled system reproduces the dense-DtN BIE solution.

    Same boundary operators and same leaf data on both sides, so the two
    solves are algebraically identical systems and must agree to round-off:
    this is the gate on the coupling, independent of any quadrature.
    """
    p, q, L = 8, 4, 1
    kappa, eta = 4.0, 4.0
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])
    domain, problem = _problem(p=p, q=q, L=L, kappa=kappa, eta=eta, bump=True)
    _, T_leaves, _, h_leaves = local_solve_stage_uniform_3D_ItI(problem)
    maps = build_interface_maps(domain)
    T_root = np.asarray(build_solver(problem, return_top_T=True))
    n_bdry = T_root.shape[0]

    S, D = _synthetic_SD(n_bdry)
    uin, uin_dn = _plane_wave_bdry_traces(domain, kappa, dirs)

    apply_S, apply_D = make_dense_SD_apply(S, D)
    us_flat, us_dn_flat, info = solve_bie_flat_matfree(
        T_leaves,
        h_leaves,
        maps,
        eta,
        apply_S,
        apply_D,
        uin,
        uin_dn,
        formulation="total",
        method="dense",
    )
    assert info["converged"]

    us_ref = np.zeros_like(us_flat)
    us_dn_ref = np.zeros_like(us_dn_flat)
    for s in range(dirs.shape[0]):
        us_ref[:, s], us_dn_ref[:, s] = _dense_bie_reference(
            T_root, S, D, uin[:, s], uin_dn[:, s], eta
        )

    rel = np.linalg.norm(us_flat - us_ref) / np.linalg.norm(us_ref)
    rel_dn = np.linalg.norm(us_dn_flat - us_dn_ref) / np.linalg.norm(us_dn_ref)
    assert rel < 1e-9, f"flat vs dense Dirichlet trace: {rel:.3e}"
    assert rel_dn < 1e-9, f"flat vs dense Neumann trace: {rel_dn:.3e}"
    jax.clear_caches()


def test_flat_bie_gmres_matches_dense_lu() -> None:
    """GMRES on the flat operator reaches the dense-LU solution.

    The preconditioner here is an LU factorization of the materialized flat
    operator, so this checks the iterative path (matvec, right-hand side and
    trace recovery) against the direct one rather than the conditioning of
    the flat system: with the random ``_synthetic_SD`` operators used here,
    unpreconditioned restarted GMRES stagnates (with the real fmm3dbie S, D
    of the experiments it does not), which is what the spectral experiments
    investigate.
    """
    from scipy.linalg import lu_factor, lu_solve

    from jaxhps._matfree_iti_3D import (
        make_flat_bie_operator,
        materialize,
    )

    p, q, L = 8, 4, 1
    kappa, eta = 4.0, 4.0
    dirs = np.array([[1.0, 0.0, 0.0]])
    domain, problem = _problem(p=p, q=q, L=L, kappa=kappa, eta=eta, bump=True)
    _, T_leaves, _, h_leaves = local_solve_stage_uniform_3D_ItI(problem)
    maps = build_interface_maps(domain)
    n_bdry = np.asarray(domain.boundary_points).reshape(-1, 3).shape[0]

    S, D = _synthetic_SD(n_bdry)
    uin, uin_dn = _plane_wave_bdry_traces(domain, kappa, dirs)
    apply_S, apply_D = make_dense_SD_apply(S, D)

    args = (T_leaves, h_leaves, maps, eta, apply_S, apply_D, uin, uin_dn)
    us_lu, us_dn_lu, _ = solve_bie_flat_matfree(
        *args, formulation="total", method="dense"
    )
    A_flat = np.asarray(
        materialize(
            make_flat_bie_operator(T_leaves, maps, eta, apply_S, apply_D),
            maps.n_flat,
        )
    )
    lu = lu_factor(A_flat)

    def precond(x):
        return jnp.asarray(lu_solve(lu, np.asarray(x)))

    stats = {}
    us_it, us_dn_it, info = solve_bie_flat_matfree(
        *args,
        formulation="total",
        method="gmres",
        tol=1e-10,
        restart=50,
        maxiter=100,
        precond=precond,
        stats=stats,
    )
    assert info["converged"], f"GMRES did not converge: {info}"
    assert stats["n_matvec"] > 0
    rel = np.linalg.norm(us_it - us_lu) / np.linalg.norm(us_lu)
    rel_dn = np.linalg.norm(us_dn_it - us_dn_lu) / np.linalg.norm(us_dn_lu)
    assert rel < 1e-7, f"GMRES vs LU Dirichlet trace: {rel:.3e}"
    assert rel_dn < 1e-7, f"GMRES vs LU Neumann trace: {rel_dn:.3e}"
    jax.clear_caches()


def _diagonal_part(T_leaves):
    """Leafwise ItI blocks with their off-diagonal entries removed."""
    d = jnp.diagonal(T_leaves, axis1=1, axis2=2)
    return jnp.einsum("li,ij->lij", d, jnp.eye(T_leaves.shape[1]))


def test_flat_bie_diagonal_approx_vs_materialized() -> None:
    """The Jacobi preconditioner against the true flat-operator diagonal.

    Interior rows and the diagonal-leaf-ItI limit are reproduced exactly; with
    the full leaf ItI blocks the boundary rows are only approximated, since the
    paths in which a node's trace moves the other boundary nodes of its own
    leaf through the ItI off-diagonal are dropped.
    """
    from jaxhps._matfree_iti_3D import (
        flat_bie_diagonal_approx,
        make_flat_bie_operator,
        materialize,
    )

    p, q, L = 6, 4, 1
    kappa, eta = 4.0, 4.0
    domain, problem = _problem(p=p, q=q, L=L, kappa=kappa, eta=eta, bump=True)
    _, T_leaves, _, _ = local_solve_stage_uniform_3D_ItI(problem)
    maps = build_interface_maps(domain)
    n_bdry = np.asarray(domain.boundary_points).reshape(-1, 3).shape[0]

    S, D = _synthetic_SD(n_bdry)
    apply_S, apply_D = make_dense_SD_apply(S, D)
    S_diag, D_diag = jnp.diag(jnp.asarray(S)), jnp.diag(jnp.asarray(D))
    br = np.asarray(maps.bdry_rows)
    interior = np.setdiff1d(np.arange(maps.n_flat), br)

    for name, T in (("full", T_leaves), ("diag", _diagonal_part(T_leaves))):
        A_flat = np.asarray(
            materialize(
                make_flat_bie_operator(T, maps, eta, apply_S, apply_D),
                maps.n_flat,
            )
        )
        true_diag = np.diagonal(A_flat)
        d = np.asarray(flat_bie_diagonal_approx(T, maps, eta, S_diag, D_diag))
        assert np.abs(d[interior] - true_diag[interior]).max() == 0.0
        rel = np.abs(d[br] - true_diag[br]).max() / np.abs(true_diag[br]).max()
        if name == "full":
            assert rel < 0.2, f"approximation unexpectedly poor: {rel:.3e}"
        else:
            assert rel < 1e-12, f"diagonal-ItI case not exact: {rel:.3e}"
    jax.clear_caches()


def _scattered_vs_total_formulation_err(p, q, L):
    kappa, eta = 4.0, 4.0
    a = 0.5
    d = np.array([[1.0, 0.0, 0.0]])
    root = DiscretizationNode3D(
        xmin=-a, xmax=a, ymin=-a, ymax=a, zmin=-a, zmax=a
    )
    domain = Domain(p=p, q=q, root=root, L=L)
    int_pts = np.asarray(domain.interior_points)
    r = np.linalg.norm(int_pts, axis=-1)
    R = 0.3
    b = np.where(r < R, -0.4 * (1.0 - (r / R) ** 2) ** 4, 0.0)
    ones = np.ones_like(b)
    uin_int = np.exp(1j * kappa * np.einsum("lpd,sd->lps", int_pts, d))
    src = (kappa**2 * b[..., None] * uin_int).astype(np.complex128)
    problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=ones,
        D_yy_coefficients=ones,
        D_zz_coefficients=ones,
        I_coefficients=kappa**2 * (1.0 - b),
        source=src,
        use_ItI=True,
        eta=eta,
    )
    _, T_leaves, _, h_leaves = local_solve_stage_uniform_3D_ItI(problem)
    maps = build_interface_maps(domain)
    n_bdry = np.asarray(domain.boundary_points).reshape(-1, 3).shape[0]
    S, D = _synthetic_SD(n_bdry)
    uin, uin_dn = _plane_wave_bdry_traces(domain, kappa, d)
    apply_S, apply_D = make_dense_SD_apply(S, D)

    args = (T_leaves, h_leaves, maps, eta, apply_S, apply_D, uin, uin_dn)
    us_tot, us_dn_tot, _ = solve_bie_flat_matfree(
        *args, formulation="total", method="dense"
    )
    us_sc, us_dn_sc, _ = solve_bie_flat_matfree(
        *args, formulation="scattered", method="dense"
    )
    rel = np.linalg.norm(us_sc - us_tot) / np.linalg.norm(us_tot)
    rel_dn = np.linalg.norm(us_dn_sc - us_dn_tot) / np.linalg.norm(us_dn_tot)
    jax.clear_caches()
    return rel, rel_dn


def test_flat_bie_scattered_matches_total_formulation() -> None:
    """The two ways of injecting the incident field agree as p grows.

    ``formulation="scattered"`` puts the incident field into the interior
    volume source only; ``formulation="total"`` puts it into the boundary
    rows.  The two are equivalent for the exact solution but differ by the
    interior discretization error of the source term, so the check is that
    the gap shrinks under refinement rather than that it is at round-off.
    Measured at (q=4, L=1): 3.5e-2 at p=8, 1.1e-2 at p=12, and 1.1e-4 at
    (p=16, q=6, L=1).
    """
    rel_coarse, rel_dn_coarse = _scattered_vs_total_formulation_err(8, 4, 1)
    rel_fine, rel_dn_fine = _scattered_vs_total_formulation_err(12, 4, 1)
    assert rel_fine < 0.5 * rel_coarse, (
        f"Dirichlet gap did not shrink: {rel_coarse:.3e} -> {rel_fine:.3e}"
    )
    assert rel_dn_fine < 0.5 * rel_dn_coarse, (
        f"Neumann gap did not shrink: {rel_dn_coarse:.3e} -> {rel_dn_fine:.3e}"
    )
    assert rel_fine < 2e-2, f"Dirichlet gap too large at p=12: {rel_fine:.3e}"
