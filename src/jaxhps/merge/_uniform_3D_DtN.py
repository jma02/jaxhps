from typing import Tuple, List

import jax
import jax.numpy as jnp
import logging

from ._schur_complement import (
    _oct_merge_from_submatrices,
)


def merge_stage_uniform_3D_DtN(
    T_arr: jax.Array,
    h_arr: jax.Array,
    l: int,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
    return_T: bool = False,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Implements uniform 3D merges of DtN matrices. Merges the nodes in the quadtree eight at a time.
    This function returns lists containing :math:`S` and :math:`\\tilde{g}`, giving enough information
    to propagate boundary data back down the tree in a later part of the algorithm.

    If this function is called with the argument ``return_T=True``, the top-level DtN matrix is also returned.

    Parameters
    ----------
    pde_problem : PDEProblem
        Specifies the discretization, differential operator, source function, and keeps track of the pre-computed differentiation and interpolation matrices.

    T_arr : jax.Array
        Array of DtN matrices from the local solve stage. Has shape (n_leaves, 6q^2, 6q^2)

    h_arr : jax.Array
        Array of outgoing boundary data from the local solve stage. Has shape (n_leaves, 6q^2)

    l : int
        The number of levels in the quadtree.

    device : jax.Device
        Where to perform the computation. Defaults to jax.devices()[0].

    host_device : jax.Device
        Where to place the output. Defaults to jax.devices("cpu")[0].

    return_T : bool
        If True, the top-level DtN matrix is returned. Defaults to False.

    Returns
    -------
    S_lst : List[jax.Array]
        A list of propagation operators. The first element of the list are the propagation operators for the nodes just above the leaves, and the last element of the list is the propagation operator for the root of the quadtree.

    g_tilde_lst: List[jax.Array]
        A list of incoming particular solution data along the merge interfaces. The first element of the list corresponds to the nodes just above the leaves, and the last element of the list corresponds to the root of the quadtree.

    T_last : jax.Array
        The top-level DtN matrix, which is only returned if ``return_T=True``. Has shape (4q, 4q).

    """
    if l == 0:
        # JAX clamps the out-of-bounds indexing in the oct-merge below, so a
        # single-leaf input would silently produce a wrong-shape top-level T
        # instead of crashing. Reject it here.
        raise ValueError(
            "merge_stage_uniform_3D_DtN requires a tree with at least one"
            " level of refinement (l >= 1); got l=0, which is a single leaf"
            " with nothing to merge."
        )

    # Move the data to the compute device if necessary
    T_arr = jax.device_put(T_arr, device)
    h_arr = jax.device_put(h_arr, device)

    S_lst = []
    g_tilde_lst = []

    q = int(jnp.sqrt(T_arr.shape[-1] // 6))

    for i in range(l - 1, 0, -1):
        logging.debug("merge_stage_uniform_3D_DtN: i = %d", i)
        logging.debug(
            "merge_stage_uniform_3D_DtN: T_arr shape: %s", T_arr.shape
        )
        logging.debug(
            "merge_stage_uniform_3D_DtN: h_arr shape: %s", h_arr.shape
        )
        S_arr, T_arr, h_arr, g_tilde_arr = vmapped_uniform_oct_merge_DtN(
            jnp.arange(q), T_arr, h_arr
        )
        S_host = jax.device_put(S_arr, host_device)
        g_tilde_host = jax.device_put(g_tilde_arr, host_device)

        S_lst.append(S_host)
        g_tilde_lst.append(g_tilde_host)

        # Do the deletion if there is a GPU available. Otherwise, we need to keep
        # this data
        if host_device != device:
            S_arr.delete()
            g_tilde_arr.delete()

    # Do the last oct-merge without the reshaping operation.
    S_last, T_last, h_last, g_tilde_last = _uniform_oct_merge_DtN(
        jnp.arange(q),
        T_arr[0],
        T_arr[1],
        T_arr[2],
        T_arr[3],
        T_arr[4],
        T_arr[5],
        T_arr[6],
        T_arr[7],
        h_arr[0],
        h_arr[1],
        h_arr[2],
        h_arr[3],
        h_arr[4],
        h_arr[5],
        h_arr[6],
        h_arr[7],
    )
    # May want to report D shape at a future point.
    D_shape = S_last.shape[0]  # noqa: F841
    S_lst.append(jax.device_put(S_last, host_device))
    g_tilde_lst.append(jax.device_put(g_tilde_last, host_device))

    # logging.debug("_build_stage: done with merging.")
    if return_T:
        T_last_out = jax.device_put(T_last, host_device)
        return S_lst, g_tilde_lst, T_last_out
    else:
        return S_lst, g_tilde_lst


@jax.jit
def _uniform_oct_merge_DtN(
    q_idxes: jnp.array,
    T_a: jnp.ndarray,
    T_b: jnp.ndarray,
    T_c: jnp.ndarray,
    T_d: jnp.ndarray,
    T_e: jnp.ndarray,
    T_f: jnp.ndarray,
    T_g: jnp.ndarray,
    T_h: jnp.ndarray,
    v_prime_a: jnp.ndarray,
    v_prime_b: jnp.ndarray,
    v_prime_c: jnp.ndarray,
    v_prime_d: jnp.ndarray,
    v_prime_e: jnp.ndarray,
    v_prime_f: jnp.ndarray,
    v_prime_g: jnp.ndarray,
    v_prime_h: jnp.ndarray,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    a_submatrices_subvecs = get_a_submatrices(T_a, v_prime_a)
    del T_a, v_prime_a

    b_submatrices_subvecs = get_b_submatrices(T_b, v_prime_b)
    del T_b, v_prime_b

    c_submatrices_subvecs = get_c_submatrices(T_c, v_prime_c)
    del T_c, v_prime_c

    d_submatrices_subvecs = get_d_submatrices(T_d, v_prime_d)
    del T_d, v_prime_d

    e_submatrices_subvecs = get_e_submatrices(T_e, v_prime_e)
    del T_e, v_prime_e

    f_submatrices_subvecs = get_f_submatrices(T_f, v_prime_f)
    del T_f, v_prime_f

    g_submatrices_subvecs = get_g_submatrices(T_g, v_prime_g)
    del T_g, v_prime_g

    h_submatrices_subvecs = get_h_submatrices(T_h, v_prime_h)
    del T_h, v_prime_h

    T, S, v_prime_ext_out, v_int = _oct_merge_from_submatrices(
        a_submatrices_subvecs=a_submatrices_subvecs,
        b_submatrices_subvecs=b_submatrices_subvecs,
        c_submatrices_subvecs=c_submatrices_subvecs,
        d_submatrices_subvecs=d_submatrices_subvecs,
        e_submatrices_subvecs=e_submatrices_subvecs,
        f_submatrices_subvecs=f_submatrices_subvecs,
        g_submatrices_subvecs=g_submatrices_subvecs,
        h_submatrices_subvecs=h_submatrices_subvecs,
    )

    r = get_rearrange_indices(jnp.arange(T.shape[0]), q_idxes)
    v_prime_ext_out = v_prime_ext_out[r]
    T = T[r][:, r]
    S = S[:, r]

    return S, T, v_prime_ext_out, v_int


_vmapped_uniform_oct_merge_DtN = jax.vmap(
    _uniform_oct_merge_DtN,
    in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    out_axes=(0, 0, 0, 0),
)


@jax.jit
def vmapped_uniform_oct_merge_DtN(
    q_idxes: jnp.array,
    T_in: jnp.ndarray,
    v_prime_in: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # print("vmapped_uniform_oct_merge: T_in shape: ", T_in.shape)
    # print("vmapped_uniform_oct_merge: v_prime_in shape: ", v_prime_in.shape)
    n_leaves, a, b = T_in.shape
    T_in = T_in.reshape((-1, 8, a, b))
    if v_prime_in.ndim == 2:
        v_prime_in = v_prime_in.reshape((-1, 8, a))
    else:
        nsrc = v_prime_in.shape[-1]
        v_prime_in = v_prime_in.reshape((-1, 8, a, nsrc))
    # print("vmapped_uniform_oct_merge: T_in shape: ", T_in.shape)
    # print("vmapped_uniform_oct_merge: v_prime_in shape: ", v_prime_in.shape)
    S, T_out, v_prime_ext_out, v_int = _vmapped_uniform_oct_merge_DtN(
        q_idxes,
        T_in[:, 0],
        T_in[:, 1],
        T_in[:, 2],
        T_in[:, 3],
        T_in[:, 4],
        T_in[:, 5],
        T_in[:, 6],
        T_in[:, 7],
        v_prime_in[:, 0],
        v_prime_in[:, 1],
        v_prime_in[:, 2],
        v_prime_in[:, 3],
        v_prime_in[:, 4],
        v_prime_in[:, 5],
        v_prime_in[:, 6],
        v_prime_in[:, 7],
    )

    return S, T_out, v_prime_ext_out, v_int


@jax.jit
def get_a_submatrices(T: jnp.array, v: jnp.array) -> Tuple[jnp.array]:
    n_per_face = T.shape[0] // 6
    idxes = jnp.arange(T.shape[0])
    idxes_1 = jnp.concatenate(
        [
            idxes[:n_per_face],
            idxes[2 * n_per_face : 3 * n_per_face],
            idxes[5 * n_per_face :],
        ]
    )
    idxes_9 = idxes[n_per_face : 2 * n_per_face]
    idxes_12 = idxes[3 * n_per_face : 4 * n_per_face]
    idxes_17 = idxes[4 * n_per_face : 5 * n_per_face]
    return _return_submatrices_subvecs(
        T, v, idxes_1, idxes_9, idxes_12, idxes_17
    )


@jax.jit
def get_b_submatrices(T: jnp.array, v: jnp.array) -> Tuple[jnp.array]:
    n_per_face = T.shape[0] // 6
    idxes = jnp.arange(T.shape[0])
    idxes_2 = jnp.concatenate(
        [
            idxes[n_per_face : 3 * n_per_face],
            idxes[5 * n_per_face :],
        ]
    )
    idxes_9 = idxes[:n_per_face]
    idxes_10 = idxes[3 * n_per_face : 4 * n_per_face]
    idxes_18 = idxes[4 * n_per_face : 5 * n_per_face]
    return _return_submatrices_subvecs(
        T, v, idxes_2, idxes_9, idxes_10, idxes_18
    )


@jax.jit
def get_c_submatrices(T: jnp.array, v: jnp.array) -> Tuple[jnp.array]:
    n_per_face = T.shape[0] // 6
    idxes = jnp.arange(T.shape[0])
    idxes_3 = jnp.concatenate(
        [
            idxes[n_per_face : 2 * n_per_face],
            idxes[3 * n_per_face : 4 * n_per_face],
            idxes[5 * n_per_face :],
        ]
    )
    idxes_10 = idxes[2 * n_per_face : 3 * n_per_face]
    idxes_11 = idxes[:n_per_face]
    idxes_19 = idxes[4 * n_per_face : 5 * n_per_face]
    return _return_submatrices_subvecs(
        T, v, idxes_3, idxes_10, idxes_11, idxes_19
    )


@jax.jit
def get_d_submatrices(T: jnp.array, v: jnp.array) -> Tuple[jnp.array]:
    n_per_face = T.shape[0] // 6
    idxes = jnp.arange(T.shape[0])
    idxes_4 = jnp.concatenate(
        [
            idxes[:n_per_face],
            idxes[3 * n_per_face : 4 * n_per_face],
            idxes[5 * n_per_face :],
        ]
    )
    idxes_11 = idxes[n_per_face : 2 * n_per_face]
    idxes_12 = idxes[2 * n_per_face : 3 * n_per_face]
    idxes_20 = idxes[4 * n_per_face : 5 * n_per_face]
    return _return_submatrices_subvecs(
        T, v, idxes_4, idxes_11, idxes_12, idxes_20
    )


@jax.jit
def get_e_submatrices(T: jnp.array, v: jnp.array) -> Tuple[jnp.array]:
    n_per_face = T.shape[0] // 6
    idxes = jnp.arange(T.shape[0])
    idxes_5 = jnp.concatenate(
        [
            idxes[:n_per_face],
            idxes[2 * n_per_face : 3 * n_per_face],
            idxes[4 * n_per_face : 5 * n_per_face],
        ]
    )
    idxes_13 = idxes[n_per_face : 2 * n_per_face]
    idxes_16 = idxes[3 * n_per_face : 4 * n_per_face]
    idxes_17 = idxes[5 * n_per_face :]
    return _return_submatrices_subvecs(
        T, v, idxes_5, idxes_13, idxes_16, idxes_17
    )


@jax.jit
def get_f_submatrices(T: jnp.array, v: jnp.array) -> Tuple[jnp.array]:
    n_per_face = T.shape[0] // 6
    idxes = jnp.arange(T.shape[0])
    idxes_6 = jnp.concatenate(
        [
            idxes[n_per_face : 3 * n_per_face],
            idxes[4 * n_per_face : 5 * n_per_face],
        ]
    )
    idxes_13 = idxes[:n_per_face]
    idxes_14 = idxes[3 * n_per_face : 4 * n_per_face]
    idxes_18 = idxes[5 * n_per_face :]
    return _return_submatrices_subvecs(
        T, v, idxes_6, idxes_13, idxes_14, idxes_18
    )


@jax.jit
def get_g_submatrices(T: jnp.array, v: jnp.array) -> Tuple[jnp.array]:
    n_per_face = T.shape[0] // 6
    idxes = jnp.arange(T.shape[0])
    idxes_7 = jnp.concatenate(
        [
            idxes[n_per_face : 2 * n_per_face],
            idxes[3 * n_per_face : 5 * n_per_face],
        ]
    )
    idxes_14 = idxes[2 * n_per_face : 3 * n_per_face]
    idxes_15 = idxes[:n_per_face]
    idxes_19 = idxes[5 * n_per_face :]
    return _return_submatrices_subvecs(
        T, v, idxes_7, idxes_14, idxes_15, idxes_19
    )


@jax.jit
def get_h_submatrices(T: jnp.array, v: jnp.array) -> Tuple[jnp.array]:
    n_per_face = T.shape[0] // 6
    idxes = jnp.arange(T.shape[0])
    idxes_8 = jnp.concatenate(
        [idxes[:n_per_face], idxes[3 * n_per_face : 5 * n_per_face]]
    )
    idxes_15 = idxes[n_per_face : 2 * n_per_face]
    idxes_16 = idxes[2 * n_per_face : 3 * n_per_face]
    idxes_20 = idxes[5 * n_per_face :]
    return _return_submatrices_subvecs(
        T, v, idxes_8, idxes_15, idxes_16, idxes_20
    )


@jax.jit
def _return_submatrices_subvecs(
    T: jnp.array,
    v: jnp.array,
    idxes_0: jnp.array,
    idxes_1: jnp.array,
    idxes_2: jnp.array,
    idxes_3: jnp.array,
) -> Tuple[jnp.array]:
    T_0_0 = T[idxes_0][:, idxes_0]
    T_0_1 = T[idxes_0][:, idxes_1]
    T_0_2 = T[idxes_0][:, idxes_2]
    T_0_3 = T[idxes_0][:, idxes_3]
    T_1_0 = T[idxes_1][:, idxes_0]
    T_1_1 = T[idxes_1][:, idxes_1]
    T_1_2 = T[idxes_1][:, idxes_2]
    T_1_3 = T[idxes_1][:, idxes_3]
    T_2_0 = T[idxes_2][:, idxes_0]
    T_2_1 = T[idxes_2][:, idxes_1]
    T_2_2 = T[idxes_2][:, idxes_2]
    T_2_3 = T[idxes_2][:, idxes_3]
    T_3_0 = T[idxes_3][:, idxes_0]
    T_3_1 = T[idxes_3][:, idxes_1]
    T_3_2 = T[idxes_3][:, idxes_2]
    T_3_3 = T[idxes_3][:, idxes_3]
    v_0 = v[idxes_0]
    v_1 = v[idxes_1]
    v_2 = v[idxes_2]
    v_3 = v[idxes_3]

    return (
        T_0_0,
        T_0_1,
        T_0_2,
        T_0_3,
        T_1_0,
        T_1_1,
        T_1_2,
        T_1_3,
        T_2_0,
        T_2_1,
        T_2_2,
        T_2_3,
        T_3_0,
        T_3_1,
        T_3_2,
        T_3_3,
        v_0,
        v_1,
        v_2,
        v_3,
    )


##############################################
# These functions are the last part of the _oct_merge
# function. It is designed to give the indices for re-arranging
# the outputs to comply with the expected ordering.


@jax.jit
def get_rearrange_indices(idxes: jnp.array, q_idxes: jnp.array) -> jnp.array:
    """
    After the _oct_merge computation, the outputs are ordered
    [region_1, ..., region_8]. This function returns the indices
    which will re-arrange the outputs to be in the order
    [face 1, ..., face 6], where the quad points in each face
    are ordered in column-first order

    Args:
        idxes (jnp.array): Has shape (24 * q**2,)

    Returns:
        jnp.array: Has shape (24 * q**2,)
    """
    n_per_region = idxes.shape[0] // 8
    n_per_face = n_per_region // 3

    n_a = n_per_region
    region_a_idxes = idxes[:n_a]
    region_a_yz = region_a_idxes[:n_per_face]
    region_a_xz = region_a_idxes[n_per_face : 2 * n_per_face]
    region_a_xy = region_a_idxes[2 * n_per_face :]

    n_b = 2 * n_per_region
    region_b_idxes = idxes[n_a:n_b]
    region_b_yz = region_b_idxes[:n_per_face]
    region_b_xz = region_b_idxes[n_per_face : 2 * n_per_face]
    region_b_xy = region_b_idxes[2 * n_per_face :]

    n_c = 3 * n_per_region
    region_c_idxes = idxes[n_b:n_c]
    region_c_yz = region_c_idxes[:n_per_face]
    region_c_xz = region_c_idxes[n_per_face : 2 * n_per_face]
    region_c_xy = region_c_idxes[2 * n_per_face :]

    n_d = 4 * n_per_region
    region_d_idxes = idxes[n_c:n_d]
    region_d_yz = region_d_idxes[:n_per_face]
    region_d_xz = region_d_idxes[n_per_face : 2 * n_per_face]
    region_d_xy = region_d_idxes[2 * n_per_face :]

    n_e = 5 * n_per_region
    region_e_idxes = idxes[n_d:n_e]
    region_e_yz = region_e_idxes[:n_per_face]
    region_e_xz = region_e_idxes[n_per_face : 2 * n_per_face]
    region_e_xy = region_e_idxes[2 * n_per_face :]

    n_f = 6 * n_per_region
    region_f_idxes = idxes[n_e:n_f]
    region_f_yz = region_f_idxes[:n_per_face]
    region_f_xz = region_f_idxes[n_per_face : 2 * n_per_face]
    region_f_xy = region_f_idxes[2 * n_per_face :]

    n_g = 7 * n_per_region
    region_g_idxes = idxes[n_f:n_g]
    region_g_yz = region_g_idxes[:n_per_face]
    region_g_xz = region_g_idxes[n_per_face : 2 * n_per_face]
    region_g_xy = region_g_idxes[2 * n_per_face :]

    region_h_idxes = idxes[n_g:]
    region_h_yz = region_h_idxes[:n_per_face]
    region_h_xz = region_h_idxes[n_per_face : 2 * n_per_face]
    region_h_xy = region_h_idxes[2 * n_per_face :]

    out = jnp.concatenate(
        [
            # Face 0 [e, h, d, a]
            region_e_yz,
            region_h_yz,
            region_d_yz,
            region_a_yz,
            # Face 1 [f, g, c, b]
            region_f_yz,
            region_g_yz,
            region_c_yz,
            region_b_yz,
            # Face 2 [e, f, b, a]
            region_e_xz,
            region_f_xz,
            region_b_xz,
            region_a_xz,
            # Face 3 [h, g, c, d]
            region_h_xz,
            region_g_xz,
            region_c_xz,
            region_d_xz,
            # Face 4
            region_e_xy,
            region_f_xy,
            region_g_xy,
            region_h_xy,
            # Face 5
            region_a_xy,
            region_b_xy,
            region_c_xy,
            region_d_xy,
        ]
    )
    return out
