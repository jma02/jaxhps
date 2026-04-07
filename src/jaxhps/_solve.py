import jax
import jax.numpy as jnp
from typing import List
from ._pdeproblem import PDEProblem


from .down_pass import (
    down_pass_uniform_2D_ItI,
    down_pass_uniform_2D_DtN,
    down_pass_uniform_3D_DtN,
    down_pass_adaptive_2D_DtN,
    down_pass_adaptive_3D_DtN,
)

from .up_pass import up_pass_uniform_2D_ItI, up_pass_uniform_2D_DtN


def solve(
    pde_problem: PDEProblem,
    boundary_data: jax.Array | List[jax.Array],
    source: jax.Array = None,
    compute_device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
) -> jax.Array:
    """
    This function performs the downward pass of the HPS algorithm, after the solution operators have
    been formed by a call to :func:`jaxhps.build_solver`.

    If the problem is a 2D uniform problem, the source term can be specified here. For other problems, the source term must be specified at the time the solver is built.


    Args:
        pde_problem (PDEProblem): Specifies the differential operator, source, domain, and precomputed interpolation and differentiation matrices. Also contains all of the solution operators computed by :func:`jaxhps.build_solver`.


        boundary_data (jax.Array | List[jax.Array]): This specifies the data on the boundary of the domain that will be propagated down to the interior of the leaves. If using an adaptive discretization, this must be specified as a list of arrays, one for each side or face of the root boundary. This list can be specified using the :func:`Domain.get_adaptive_boundary_data_lst` utility. For uniform discretizations, this argument can be a jax.Array of shape (n_bdry,) or a list.

        source (jax.Array): The source term for the PDE. Currently, this can only be specified for 2D uniform ItI problems. For other versions, the source must be specified at the time the solver is built.

        compute_device (jax.Device, optional): Where the computation should happen. Defaults to jax.devices()[0].

        host_device (jax.Device, optional): Where the solution operators should be stored. Defaults to jax.devices("cpu")[0].

    Returns:
        jax.Array: The solution on the HPS grid. This has shape (n_leaves, p^d).
    """
    if source is not None:
        # Check that we're dealing with a 2D uniform ItI problem
        if (
            not pde_problem.domain.bool_2D
            or not pde_problem.domain.bool_uniform
        ):
            raise ValueError(
                "Source can only be specified for 2D uniform ItI problems. For other problems, the source must be specified at the time the solver is built."
            )
        return _up_then_down_pass(
            pde_problem=pde_problem,
            boundary_data=boundary_data,
            source=source,
            compute_device=compute_device,
            host_device=host_device,
        )

    if not pde_problem.domain.bool_uniform:
        # interface is different for the adaptive down pass so we'll
        # pass to a different function.
        return _adaptive_solve(
            pde_problem=pde_problem,
            boundary_data_lst=boundary_data,
        )

    if isinstance(boundary_data, list):
        # If the boundary data is a list, we need to concatenate it
        # into a single array. Concatenate along the first axis to 
        # correctly handle multi-source problems (which have sources on the last axis).
        boundary_data = jnp.concatenate(boundary_data, axis=0)

    if pde_problem.use_ItI:
        down_pass_fn = down_pass_uniform_2D_ItI

    elif pde_problem.domain.bool_2D:
        down_pass_fn = down_pass_uniform_2D_DtN
    else:
        down_pass_fn = down_pass_uniform_3D_DtN

    return down_pass_fn(
        boundary_data,
        pde_problem.S_lst,
        pde_problem.g_tilde_lst,
        pde_problem.Y,
        pde_problem.v,
        device=compute_device,
        host_device=host_device,
    )


def _adaptive_solve(
    pde_problem: PDEProblem,
    boundary_data_lst: List[jax.Array],
) -> jax.Array:
    if not isinstance(boundary_data_lst, list):
        raise ValueError(
            "For adaptive solves, boundary data needs to be a list. Try using the Domain.get_adaptive_boundary_data_lst() utility."
        )
    # Figure out which function to use
    if pde_problem.domain.bool_2D:
        down_pass_fn = down_pass_adaptive_2D_DtN
    else:
        down_pass_fn = down_pass_adaptive_3D_DtN

    return down_pass_fn(
        pde_problem=pde_problem, boundary_data=boundary_data_lst
    )


def _up_then_down_pass(
    pde_problem: PDEProblem,
    boundary_data: jax.Array,
    source: jax.Array,
    compute_device: jax.Device,
    host_device: jax.Device,
) -> jax.Array:
    # Figure out if ItI or DtN is being used
    if pde_problem.use_ItI:
        # If using ItI, we need to compute the g_tilde_lst
        # and v_arr in the upward pass.
        up_pass_fn = up_pass_uniform_2D_ItI
        down_pass_fn = down_pass_uniform_2D_ItI
    else:
        # If using DtN, we can just use the precomputed g_tilde_lst and v_arr.
        up_pass_fn = up_pass_uniform_2D_DtN
        down_pass_fn = down_pass_uniform_2D_DtN

    # First, do the upward pass
    v, g_tilde_lst = up_pass_fn(
        pde_problem=pde_problem,
        source=source,
        device=compute_device,
        host_device=host_device,
    )

    # Now, do the downward pass
    solns = down_pass_fn(
        boundary_data=boundary_data,
        S_lst=pde_problem.S_lst,
        g_tilde_lst=g_tilde_lst,
        Y_arr=pde_problem.Y,
        v_arr=v,
        device=compute_device,
        host_device=host_device,
    )
    return solns
