"""3D uniform local solve stage for ItI merges.

Mirrors :mod:`jaxhps.local_solve._uniform_2D_ItI` but for 3D leaves. The
boundary impedance traces live on a single-counted Cheby boundary
(``n_bdry = p**3 - (p-2)**3``) for the incoming side, and on the
``6 q**2`` Gauss boundary for the outgoing side. The local "Robin" problem

.. math::
   B u = \\begin{pmatrix} G \\\\ A_\\text{int} \\end{pmatrix} u
       = \\begin{pmatrix} P\\, g_\\text{in} \\\\ f_\\text{int} \\end{pmatrix}

is square (``p**3`` x ``p**3``) and invertible because the ItI map is coercive
for any kappa, unlike the DtN local Dirichlet solve which can have interior
resonances.

Operators (precomputed in :mod:`jaxhps._precompute_operators_3D`):

* ``P`` : ``(n_bdry, 6 q**2)`` -- Gauss -> single-counted Cheby boundary.
* ``QH``: ``(6 q**2, p**3)``    -- Cheby -> outgoing impedance on Gauss boundary.
* ``G`` : ``(n_bdry, p**3)``    -- incoming impedance ``u_n + i*eta*u`` on Cheby boundary.
"""

import jax.numpy as jnp
import jax

from .._pdeproblem import PDEProblem
from ._uniform_3D_DtN import _gather_coeffs_3D
from ._uniform_2D_DtN import vmapped_assemble_diff_operator
from typing import Tuple
import logging


def local_solve_stage_uniform_3D_ItI(
    pde_problem: PDEProblem,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Local solve stage for 3D problems with a uniform octree, producing ItI matrices.

    Parameters
    ----------
    pde_problem : PDEProblem
        Specifies the discretization, differential operator, source function,
        and keeps track of the pre-computed differentiation and interpolation
        matrices. Must have ``use_ItI=True`` and a 3D uniform domain.
    device : jax.Device
        Where to perform the computation.
    host_device : jax.Device
        Where to place the output.

    Returns
    -------
    Y : jax.Array
        Solution operators mapping incoming impedance boundary data to
        homogeneous solutions on the leaf interiors. Shape
        ``(n_leaves, p**3, 6 q**2)``.
    T : jax.Array
        Impedance-to-Impedance matrices for each leaf. Shape
        ``(n_leaves, 6 q**2, 6 q**2)``.
    v : jax.Array
        Particular solutions on the Cheby grid. Shape ``(n_leaves, p**3)`` or
        ``(n_leaves, p**3, n_src)`` for multi-source.
    h : jax.Array
        Outgoing impedance trace ``v_n - i*eta*v`` of the particular solutions.
        Shape ``(n_leaves, 6 q**2)`` or ``(n_leaves, 6 q**2, n_src)``.
    """
    logging.debug(
        "local_solve_stage_uniform_3D_ItI: started. device=%s", device
    )

    coeffs_gathered, which_coeffs = _gather_coeffs_3D(
        D_xx_coeffs=pde_problem.D_xx_coefficients,
        D_xy_coeffs=pde_problem.D_xy_coefficients,
        D_yy_coeffs=pde_problem.D_yy_coefficients,
        D_xz_coeffs=pde_problem.D_xz_coefficients,
        D_yz_coeffs=pde_problem.D_yz_coefficients,
        D_zz_coeffs=pde_problem.D_zz_coefficients,
        D_x_coeffs=pde_problem.D_x_coefficients,
        D_y_coeffs=pde_problem.D_y_coefficients,
        D_z_coeffs=pde_problem.D_z_coefficients,
        I_coeffs=pde_problem.I_coefficients,
    )
    source_term = pde_problem.source
    bool_multi_source = source_term.ndim == 3
    source_term = jax.device_put(source_term, device)

    # Stack the precomputed differential operators into a single array,
    # matching the order expected by ``vmapped_assemble_diff_operator``.
    diff_ops = jnp.stack(
        [
            pde_problem.D_xx,
            pde_problem.D_xy,
            pde_problem.D_yy,
            pde_problem.D_xz,
            pde_problem.D_yz,
            pde_problem.D_zz,
            pde_problem.D_x,
            pde_problem.D_y,
            pde_problem.D_z,
            jnp.eye(pde_problem.domain.p**3),
        ]
    )
    diff_ops = jax.device_put(diff_ops, device)
    coeffs_gathered = jax.device_put(coeffs_gathered, device)
    diff_operators = vmapped_assemble_diff_operator(
        coeffs_gathered, which_coeffs, diff_ops
    )
    if not bool_multi_source:
        source_term = jnp.expand_dims(source_term, axis=-1)

    T_arr, Y_arr, h, v = vmapped_get_ItI_3D(
        diff_operators,
        source_term,
        pde_problem.P,
        pde_problem.QH,
        pde_problem.G,
    )

    if not bool_multi_source:
        h = h[..., 0]
        v = v[..., 0]

    T_arr_host = jax.device_put(T_arr, host_device)
    del T_arr
    v_host = jax.device_put(v, host_device)
    del v
    h_host = jax.device_put(h, host_device)
    del h
    Y_arr_host = jax.device_put(Y_arr, host_device)
    del Y_arr

    return Y_arr_host, T_arr_host, v_host, h_host


@jax.jit
def get_ItI_3D(
    diff_operator: jax.Array,
    source_term: jax.Array,
    P: jax.Array,
    QH: jax.Array,
    G: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Solve a single 3D leaf's local impedance problem.

    The leaf's PDE is :math:`A u = f` with `A` the assembled differential operator
    on the ``p**3`` Chebyshev nodes (rearranged so boundary nodes come first).
    We replace the first ``n_bdry = p**3 - (p-2)**3`` rows of `A` with the
    incoming-impedance operator `G`, giving a square system

        ``B u = [P @ g_in ; f_int]``

    that is invertible without interior-resonance pathology.

    Args:
        diff_operator: shape ``(p**3, p**3)``.
        source_term:   shape ``(p**3, n_src)``.
        P:             shape ``(n_bdry, 6 q**2)``. Gauss -> Cheby boundary interp.
        QH:            shape ``(6 q**2, p**3)``. Cheby -> outgoing impedance on Gauss.
        G:             shape ``(n_bdry, p**3)``. Incoming impedance on Cheby boundary.

    Returns:
        T:    ``(6 q**2, 6 q**2)``    -- impedance-to-impedance map.
        Y:    ``(p**3, 6 q**2)``      -- impedance-to-interior solution map.
        h:    ``(6 q**2, n_src)``     -- outgoing impedance of particular solutions.
        v:    ``(p**3, n_src)``       -- particular solutions on Cheby grid.
    """
    n_cheby_pts = diff_operator.shape[-1]
    n_bdry = P.shape[0]
    A = diff_operator

    # Build B by replacing the top n_bdry rows of A with G.
    B = jnp.zeros((n_cheby_pts, n_cheby_pts), dtype=jnp.complex128)
    B = B.at[:n_bdry].set(G)
    B = B.at[n_bdry:].set(A[n_bdry:].astype(jnp.complex128))
    B_inv = jnp.linalg.inv(B)

    # Phi maps from interior source values to particular solutions on all Cheby nodes.
    Phi = B_inv[:, n_bdry:]

    # Y maps incoming impedance on Gauss -> homogeneous solution on Cheby.
    Y = B_inv[:, :n_bdry] @ P

    source_int = source_term[n_bdry:]
    v = Phi @ source_int
    h = QH @ v
    T = QH @ Y
    return (T, Y, h, v)


vmapped_get_ItI_3D = jax.vmap(
    get_ItI_3D,
    in_axes=(0, 0, None, None, None),
    out_axes=(0, 0, 0, 0),
)
