"""3D uniform oct merge stage for ItI matrices.

Mirrors :mod:`jaxhps.merge._uniform_2D_ItI` but for 3D leaves arranged in an
octree.  Each oct-merge takes 8 sibling leaves (a..h) with their local ItI
matrices and outgoing-impedance particular sources, and returns the merged
ItI map plus the propagation operator ``S`` and incoming particular data
``g_tilde_int`` on the 12 interior face-quadrants needed by the down-pass.

The leaf labelling convention follows :mod:`jaxhps.merge._uniform_3D_DtN`:

*  Bottom layer (z low):  a (0,0,0), b (1,0,0), c (1,1,0), d (0,1,0)
*  Top layer    (z high): e (0,0,1), f (1,0,1), g (1,1,1), h (0,1,1)

Interior interface labels (the 12 face-quads internal to the octant):

==  ===================  ====================
nb  interface            shared between
==  ===================  ====================
 9  bottom xz-mid (front)  a <-> b
10  bottom yz-mid (right)  b <-> c
11  bottom xz-mid (back)   c <-> d
12  bottom yz-mid (left)   d <-> a
13  top xz-mid (front)     e <-> f
14  top yz-mid (right)     f <-> g
15  top xz-mid (back)      g <-> h
16  top yz-mid (left)      h <-> e
17  vertical front-left    a <-> e
18  vertical front-right   b <-> f
19  vertical back-right    c <-> g
20  vertical back-left     d <-> h
==  ===================  ====================

For ItI merges we follow the 2D pattern: split the 24 interior unknowns
(each interface contributes one trace per side) into two groups by
"checkerboard" parity of the owning leaf.  Each interface joins one
even-parity leaf to one odd-parity leaf, so each interface contributes
exactly one entry to each group.

*  Group 1 ("odd" owners):   b, d, e, g  -> 12 unknowns
*  Group 2 ("even" owners):  a, c, f, h  -> 12 unknowns

The interior block ``D`` of the merge linear system has the structure
``D = I + [[0, D_12], [D_21, 0]]`` with ``D_12: group 2 -> group 1`` and
``D_21: group 1 -> group 2``.  This is the structure expected by
:func:`jaxhps.merge._schur_complement.assemble_merge_outputs_ItI`.
"""

from typing import Tuple, List

import jax
import jax.numpy as jnp
import logging

from ._schur_complement import assemble_merge_outputs_ItI
from ._uniform_3D_DtN import (
    get_a_submatrices,
    get_b_submatrices,
    get_c_submatrices,
    get_d_submatrices,
    get_e_submatrices,
    get_f_submatrices,
    get_g_submatrices,
    get_h_submatrices,
    get_rearrange_indices,
)


def merge_stage_uniform_3D_ItI(
    T_arr: jax.Array,
    h_arr: jax.Array,
    l: int,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
    return_T: bool = False,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
    """Implements uniform 3D merges of ItI matrices, eight leaves at a time.

    Mirrors :func:`merge_stage_uniform_3D_DtN` but with the ItI Schur
    complement structure.

    Parameters
    ----------
    T_arr : jax.Array
        ItI matrices from the local solve stage. Has shape
        ``(n_leaves, 6 q**2, 6 q**2)`` and dtype ``complex128``.

    h_arr : jax.Array
        Outgoing-impedance boundary data from the local solve stage. Has
        shape ``(n_leaves, 6 q**2)`` or ``(n_leaves, 6 q**2, n_src)``.

    l : int
        Number of levels in the octree.

    Returns
    -------
    S_lst : List[jax.Array]
        Propagation operators per merge level.

    g_tilde_lst : List[jax.Array]
        Incoming particular data on merge interfaces per level.

    T_last : jax.Array
        Top-level ItI matrix (only returned if ``return_T=True``).
    """
    logging.debug("merge_stage_uniform_3D_ItI: started. device=%s", device)

    T_arr = jax.device_put(T_arr, device)
    h_arr = jax.device_put(h_arr, device)

    bool_multi_source = h_arr.ndim == 3

    S_lst: List[jnp.ndarray] = []
    g_tilde_lst: List[jnp.ndarray] = []

    q = int(jnp.sqrt(T_arr.shape[-1] // 6))

    # Add a trailing source axis to h_arr if it isn't there so the vmap
    # machinery has a uniform shape; we'll squeeze it back at the end.
    if not bool_multi_source:
        h_arr = h_arr[..., None]

    for i in range(l - 1, 0, -1):
        logging.debug("merge_stage_uniform_3D_ItI: i = %d", i)
        S_arr, T_arr, h_arr, g_tilde_arr = vmapped_uniform_oct_merge_ItI(
            jnp.arange(q), T_arr, h_arr
        )

        if not bool_multi_source:
            g_tilde_arr_out = jnp.squeeze(g_tilde_arr, axis=-1)
        else:
            g_tilde_arr_out = g_tilde_arr

        S_lst.append(jax.device_put(S_arr, host_device))
        g_tilde_lst.append(jax.device_put(g_tilde_arr_out, host_device))

        if host_device != device:
            S_arr.delete()
            g_tilde_arr.delete()

    # Final oct-merge without reshaping the outputs.
    S_last, T_last, h_last, g_tilde_last = _uniform_oct_merge_ItI(
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

    if not bool_multi_source:
        g_tilde_last = jnp.squeeze(g_tilde_last, axis=-1)
        h_last = jnp.squeeze(h_last, axis=-1)

    S_lst.append(jax.device_put(jnp.expand_dims(S_last, axis=0), host_device))
    g_tilde_lst.append(
        jax.device_put(jnp.expand_dims(g_tilde_last, axis=0), host_device)
    )

    if return_T:
        T_last_out = jax.device_put(T_last, host_device)
        return S_lst, g_tilde_lst, T_last_out
    return S_lst, g_tilde_lst


@jax.jit
def _uniform_oct_merge_ItI(
    q_idxes: jnp.ndarray,
    T_a: jnp.ndarray,
    T_b: jnp.ndarray,
    T_c: jnp.ndarray,
    T_d: jnp.ndarray,
    T_e: jnp.ndarray,
    T_f: jnp.ndarray,
    T_g: jnp.ndarray,
    T_h: jnp.ndarray,
    h_a: jnp.ndarray,
    h_b: jnp.ndarray,
    h_c: jnp.ndarray,
    h_d: jnp.ndarray,
    h_e: jnp.ndarray,
    h_f: jnp.ndarray,
    h_g: jnp.ndarray,
    h_h: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ItI 8-way oct-merge of sibling leaves.

    Parameters
    ----------
    q_idxes : jax.Array
        Index array used by ``get_rearrange_indices`` for the final
        leaf-block -> face-quad rearrangement.
    T_a..T_h : jax.Array (complex128)
        Local ItI matrices for the 8 children. Shape ``(6 q**2, 6 q**2)``.
    h_a..h_h : jax.Array (complex128)
        Local outgoing-impedance particular data. Shape ``(6 q**2,)`` or
        ``(6 q**2, n_src)``.

    Returns
    -------
    S : jax.Array
        Propagation operator. Shape ``(24 q**2_int, 24 q**2_ext)``.
    T : jax.Array
        Merged ItI matrix. Shape ``(24 q**2_ext, 24 q**2_ext)``.
    h_ext_out : jax.Array
        Outgoing-impedance particular data on the merged exterior. Shape
        ``(24 q**2_ext,)`` or ``(24 q**2_ext, n_src)``.
    g_tilde_int : jax.Array
        Incoming-impedance particular data on the 12 interior interfaces.
        Shape ``(24 q**2_int,)`` or ``(24 q**2_int, n_src)``.
    """
    # Extract per-leaf submatrices and subvectors. The DtN helpers do
    # generic indexing and work on complex matrices unchanged.
    a_sub = get_a_submatrices(T_a, h_a)
    b_sub = get_b_submatrices(T_b, h_b)
    c_sub = get_c_submatrices(T_c, h_c)
    d_sub = get_d_submatrices(T_d, h_d)
    e_sub = get_e_submatrices(T_e, h_e)
    f_sub = get_f_submatrices(T_f, h_f)
    g_sub = get_g_submatrices(T_g, h_g)
    h_sub = get_h_submatrices(T_h, h_h)

    T, S, h_ext_out, g_tilde_int = _oct_merge_from_submatrices_ItI(
        a_sub, b_sub, c_sub, d_sub, e_sub, f_sub, g_sub, h_sub
    )

    r = get_rearrange_indices(jnp.arange(T.shape[0]), q_idxes)
    h_ext_out = h_ext_out[r]
    T = T[r][:, r]
    S = S[:, r]

    return S, T, h_ext_out, g_tilde_int


@jax.jit
def _oct_merge_from_submatrices_ItI(
    a_sub: Tuple[jax.Array],
    b_sub: Tuple[jax.Array],
    c_sub: Tuple[jax.Array],
    d_sub: Tuple[jax.Array],
    e_sub: Tuple[jax.Array],
    f_sub: Tuple[jax.Array],
    g_sub: Tuple[jax.Array],
    h_sub: Tuple[jax.Array],
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build the ItI Schur-complement system and call ``assemble_merge_outputs_ItI``.

    Each ``*_sub`` tuple has the structure produced by
    :func:`jaxhps.merge._uniform_3D_DtN._return_submatrices_subvecs`:

    For leaf ``a`` with index sets ``(idx_1, idx_9, idx_12, idx_17)``::

        T_a_1_1, T_a_1_9, T_a_1_12, T_a_1_17,
        T_a_9_1, T_a_9_9, T_a_9_12, T_a_9_17,
        T_a_12_1, T_a_12_9, T_a_12_12, T_a_12_17,
        T_a_17_1, T_a_17_9, T_a_17_12, T_a_17_17,
        h_a_1, h_a_9, h_a_12, h_a_17

    Other leaves follow the same structure, with their own face indices.
    """
    (
        T_a_1_1,
        T_a_1_9,
        T_a_1_12,
        T_a_1_17,
        T_a_9_1,
        T_a_9_9,
        T_a_9_12,
        T_a_9_17,
        T_a_12_1,
        T_a_12_9,
        T_a_12_12,
        T_a_12_17,
        T_a_17_1,
        T_a_17_9,
        T_a_17_12,
        T_a_17_17,
        h_a_1,
        h_a_9,
        h_a_12,
        h_a_17,
    ) = a_sub

    (
        T_b_2_2,
        T_b_2_9,
        T_b_2_10,
        T_b_2_18,
        T_b_9_2,
        T_b_9_9,
        T_b_9_10,
        T_b_9_18,
        T_b_10_2,
        T_b_10_9,
        T_b_10_10,
        T_b_10_18,
        T_b_18_2,
        T_b_18_9,
        T_b_18_10,
        T_b_18_18,
        h_b_2,
        h_b_9,
        h_b_10,
        h_b_18,
    ) = b_sub

    (
        T_c_3_3,
        T_c_3_10,
        T_c_3_11,
        T_c_3_19,
        T_c_10_3,
        T_c_10_10,
        T_c_10_11,
        T_c_10_19,
        T_c_11_3,
        T_c_11_10,
        T_c_11_11,
        T_c_11_19,
        T_c_19_3,
        T_c_19_10,
        T_c_19_11,
        T_c_19_19,
        h_c_3,
        h_c_10,
        h_c_11,
        h_c_19,
    ) = c_sub

    (
        T_d_4_4,
        T_d_4_11,
        T_d_4_12,
        T_d_4_20,
        T_d_11_4,
        T_d_11_11,
        T_d_11_12,
        T_d_11_20,
        T_d_12_4,
        T_d_12_11,
        T_d_12_12,
        T_d_12_20,
        T_d_20_4,
        T_d_20_11,
        T_d_20_12,
        T_d_20_20,
        h_d_4,
        h_d_11,
        h_d_12,
        h_d_20,
    ) = d_sub

    (
        T_e_5_5,
        T_e_5_13,
        T_e_5_16,
        T_e_5_17,
        T_e_13_5,
        T_e_13_13,
        T_e_13_16,
        T_e_13_17,
        T_e_16_5,
        T_e_16_13,
        T_e_16_16,
        T_e_16_17,
        T_e_17_5,
        T_e_17_13,
        T_e_17_16,
        T_e_17_17,
        h_e_5,
        h_e_13,
        h_e_16,
        h_e_17,
    ) = e_sub

    (
        T_f_6_6,
        T_f_6_13,
        T_f_6_14,
        T_f_6_18,
        T_f_13_6,
        T_f_13_13,
        T_f_13_14,
        T_f_13_18,
        T_f_14_6,
        T_f_14_13,
        T_f_14_14,
        T_f_14_18,
        T_f_18_6,
        T_f_18_13,
        T_f_18_14,
        T_f_18_18,
        h_f_6,
        h_f_13,
        h_f_14,
        h_f_18,
    ) = f_sub

    (
        T_g_7_7,
        T_g_7_14,
        T_g_7_15,
        T_g_7_19,
        T_g_14_7,
        T_g_14_14,
        T_g_14_15,
        T_g_14_19,
        T_g_15_7,
        T_g_15_14,
        T_g_15_15,
        T_g_15_19,
        T_g_19_7,
        T_g_19_14,
        T_g_19_15,
        T_g_19_19,
        h_g_7,
        h_g_14,
        h_g_15,
        h_g_19,
    ) = g_sub

    (
        T_h_8_8,
        T_h_8_15,
        T_h_8_16,
        T_h_8_20,
        T_h_15_8,
        T_h_15_15,
        T_h_15_16,
        T_h_15_20,
        T_h_16_8,
        T_h_16_15,
        T_h_16_16,
        T_h_16_20,
        T_h_20_8,
        T_h_20_15,
        T_h_20_16,
        T_h_20_20,
        h_h_8,
        h_h_15,
        h_h_16,
        h_h_20,
    ) = h_sub

    # Per-face block sizes. All face-quads have the same n_per_face = q**2
    # in the uniform case, but compute them generically.
    n_1 = T_a_1_1.shape[0]
    n_2 = T_b_2_2.shape[0]
    n_3 = T_c_3_3.shape[0]
    n_4 = T_d_4_4.shape[0]
    n_5 = T_e_5_5.shape[0]
    n_6 = T_f_6_6.shape[0]
    n_7 = T_g_7_7.shape[0]
    n_8 = T_h_8_8.shape[0]
    n_9 = T_a_9_9.shape[0]
    n_10 = T_b_10_10.shape[0]
    n_11 = T_c_11_11.shape[0]
    n_12 = T_a_12_12.shape[0]
    n_13 = T_e_13_13.shape[0]
    n_14 = T_f_14_14.shape[0]
    n_15 = T_g_15_15.shape[0]
    n_16 = T_e_16_16.shape[0]
    n_17 = T_a_17_17.shape[0]
    n_18 = T_b_18_18.shape[0]
    n_19 = T_c_19_19.shape[0]
    n_20 = T_d_20_20.shape[0]

    n_ext_pts = n_1 + n_2 + n_3 + n_4 + n_5 + n_6 + n_7 + n_8

    # Group 1 (odd-leaf owners): b_9, b_10, b_18, d_11, d_12, d_20,
    #                            e_13, e_16, e_17, g_14, g_15, g_19
    g1_sizes = [
        n_9,
        n_10,
        n_18,  # b
        n_11,
        n_12,
        n_20,  # d
        n_13,
        n_16,
        n_17,  # e
        n_14,
        n_15,
        n_19,  # g
    ]
    # Use Python int arithmetic so offsets are static (shapes are known
    # at trace time inside @jax.jit).
    g1_offsets = [0]
    for sz in g1_sizes:
        g1_offsets.append(g1_offsets[-1] + sz)
    n_g1 = g1_offsets[-1]

    # Group 2 (even-leaf owners): a_9, a_12, a_17, c_10, c_11, c_19,
    #                             f_13, f_14, f_18, h_15, h_16, h_20
    g2_sizes = [
        n_9,
        n_12,
        n_17,  # a
        n_10,
        n_11,
        n_19,  # c
        n_13,
        n_14,
        n_18,  # f
        n_15,
        n_16,
        n_20,  # h
    ]
    g2_offsets = [0]
    for sz in g2_sizes:
        g2_offsets.append(g2_offsets[-1] + sz)
    n_g2 = g2_offsets[-1]

    def g1(i: int) -> Tuple[int, int]:
        return g1_offsets[i], g1_offsets[i + 1]

    def g2(i: int) -> Tuple[int, int]:
        return g2_offsets[i], g2_offsets[i + 1]

    # Exterior block offsets.
    ext_starts = [0]
    for sz in [n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8]:
        ext_starts.append(ext_starts[-1] + sz)

    def ext(i: int) -> Tuple[int, int]:
        return ext_starts[i], ext_starts[i + 1]

    dtype = T_a_1_1.dtype

    # ------------------------------------------------------------------
    # Build B (n_ext_pts x (n_g1 + n_g2)).  For each leaf alpha, alpha's
    # exterior block has nonzero entries at the columns corresponding to
    # alpha's three interior face variables.  Even-leaf interior vars live
    # in Group 2; odd-leaf interior vars in Group 1.
    # ------------------------------------------------------------------
    B = jnp.zeros((n_ext_pts, n_g1 + n_g2), dtype=dtype)

    # B[alpha_outer, beta_neighbor_face] = T_alpha[outer, interior_face]
    # because g_in_alpha(g) = t_int(beta_at_g, g), where beta is in the
    # OPPOSITE parity group from alpha.

    # Leaf a (even) -> neighbors b, d, e (Group 1).
    # a's interior faces: 9 (b), 12 (d), 17 (e).
    s, e = ext(0)
    s1, e1 = g1(0)  # b_9
    B = B.at[s:e, s1:e1].set(T_a_1_9)
    s1, e1 = g1(4)  # d_12
    B = B.at[s:e, s1:e1].set(T_a_1_12)
    s1, e1 = g1(8)  # e_17
    B = B.at[s:e, s1:e1].set(T_a_1_17)

    # Leaf b (odd) -> neighbors a, c, f (Group 2).
    # b's interior faces: 9 (a), 10 (c), 18 (f).
    s, e = ext(1)
    s2, e2 = g2(0)  # a_9
    B = B.at[s:e, n_g1 + s2 : n_g1 + e2].set(T_b_2_9)
    s2, e2 = g2(3)  # c_10
    B = B.at[s:e, n_g1 + s2 : n_g1 + e2].set(T_b_2_10)
    s2, e2 = g2(8)  # f_18
    B = B.at[s:e, n_g1 + s2 : n_g1 + e2].set(T_b_2_18)

    # Leaf c (even) -> neighbors b, d, g (Group 1).
    # c's interior faces: 10 (b), 11 (d), 19 (g).
    s, e = ext(2)
    s1, e1 = g1(1)  # b_10
    B = B.at[s:e, s1:e1].set(T_c_3_10)
    s1, e1 = g1(3)  # d_11
    B = B.at[s:e, s1:e1].set(T_c_3_11)
    s1, e1 = g1(11)  # g_19
    B = B.at[s:e, s1:e1].set(T_c_3_19)

    # Leaf d (odd) -> neighbors c, a, h (Group 2).
    # d's interior faces: 11 (c), 12 (a), 20 (h).
    s, e = ext(3)
    s2, e2 = g2(4)  # c_11
    B = B.at[s:e, n_g1 + s2 : n_g1 + e2].set(T_d_4_11)
    s2, e2 = g2(1)  # a_12
    B = B.at[s:e, n_g1 + s2 : n_g1 + e2].set(T_d_4_12)
    s2, e2 = g2(11)  # h_20
    B = B.at[s:e, n_g1 + s2 : n_g1 + e2].set(T_d_4_20)

    # Leaf e (odd) -> neighbors f, h, a (Group 2).
    # e's interior faces: 13 (f), 16 (h), 17 (a).
    s, e = ext(4)
    s2, e2 = g2(6)  # f_13
    B = B.at[s:e, n_g1 + s2 : n_g1 + e2].set(T_e_5_13)
    s2, e2 = g2(10)  # h_16
    B = B.at[s:e, n_g1 + s2 : n_g1 + e2].set(T_e_5_16)
    s2, e2 = g2(2)  # a_17
    B = B.at[s:e, n_g1 + s2 : n_g1 + e2].set(T_e_5_17)

    # Leaf f (even) -> neighbors e, g, b (Group 1).
    # f's interior faces: 13 (e), 14 (g), 18 (b).
    s, e = ext(5)
    s1, e1 = g1(6)  # e_13
    B = B.at[s:e, s1:e1].set(T_f_6_13)
    s1, e1 = g1(9)  # g_14
    B = B.at[s:e, s1:e1].set(T_f_6_14)
    s1, e1 = g1(2)  # b_18
    B = B.at[s:e, s1:e1].set(T_f_6_18)

    # Leaf g (odd) -> neighbors f, h, c (Group 2).
    # g's interior faces: 14 (f), 15 (h), 19 (c).
    s, e = ext(6)
    s2, e2 = g2(7)  # f_14
    B = B.at[s:e, n_g1 + s2 : n_g1 + e2].set(T_g_7_14)
    s2, e2 = g2(9)  # h_15
    B = B.at[s:e, n_g1 + s2 : n_g1 + e2].set(T_g_7_15)
    s2, e2 = g2(5)  # c_19
    B = B.at[s:e, n_g1 + s2 : n_g1 + e2].set(T_g_7_19)

    # Leaf h (even) -> neighbors g, e, d (Group 1).
    # h's interior faces: 15 (g), 16 (e), 20 (d).
    s, e = ext(7)
    s1, e1 = g1(10)  # g_15
    B = B.at[s:e, s1:e1].set(T_h_8_15)
    s1, e1 = g1(7)  # e_16
    B = B.at[s:e, s1:e1].set(T_h_8_16)
    s1, e1 = g1(5)  # d_20
    B = B.at[s:e, s1:e1].set(T_h_8_20)

    # ------------------------------------------------------------------
    # Build C ((n_g1 + n_g2) x n_ext_pts).  For each interior unknown
    # "leaf alpha at interface k", C entry at exterior column ext_alpha
    # is R_alpha[int_k, ext_alpha].
    # ------------------------------------------------------------------
    C = jnp.zeros((n_g1 + n_g2, n_ext_pts), dtype=dtype)

    # Group 1 rows.
    s1, e1 = g1(0)
    s, e = ext(1)
    C = C.at[s1:e1, s:e].set(T_b_9_2)
    s1, e1 = g1(1)
    s, e = ext(1)
    C = C.at[s1:e1, s:e].set(T_b_10_2)
    s1, e1 = g1(2)
    s, e = ext(1)
    C = C.at[s1:e1, s:e].set(T_b_18_2)
    s1, e1 = g1(3)
    s, e = ext(3)
    C = C.at[s1:e1, s:e].set(T_d_11_4)
    s1, e1 = g1(4)
    s, e = ext(3)
    C = C.at[s1:e1, s:e].set(T_d_12_4)
    s1, e1 = g1(5)
    s, e = ext(3)
    C = C.at[s1:e1, s:e].set(T_d_20_4)
    s1, e1 = g1(6)
    s, e = ext(4)
    C = C.at[s1:e1, s:e].set(T_e_13_5)
    s1, e1 = g1(7)
    s, e = ext(4)
    C = C.at[s1:e1, s:e].set(T_e_16_5)
    s1, e1 = g1(8)
    s, e = ext(4)
    C = C.at[s1:e1, s:e].set(T_e_17_5)
    s1, e1 = g1(9)
    s, e = ext(6)
    C = C.at[s1:e1, s:e].set(T_g_14_7)
    s1, e1 = g1(10)
    s, e = ext(6)
    C = C.at[s1:e1, s:e].set(T_g_15_7)
    s1, e1 = g1(11)
    s, e = ext(6)
    C = C.at[s1:e1, s:e].set(T_g_19_7)

    # Group 2 rows.
    s2, e2 = g2(0)
    s, e = ext(0)
    C = C.at[n_g1 + s2 : n_g1 + e2, s:e].set(T_a_9_1)
    s2, e2 = g2(1)
    s, e = ext(0)
    C = C.at[n_g1 + s2 : n_g1 + e2, s:e].set(T_a_12_1)
    s2, e2 = g2(2)
    s, e = ext(0)
    C = C.at[n_g1 + s2 : n_g1 + e2, s:e].set(T_a_17_1)
    s2, e2 = g2(3)
    s, e = ext(2)
    C = C.at[n_g1 + s2 : n_g1 + e2, s:e].set(T_c_10_3)
    s2, e2 = g2(4)
    s, e = ext(2)
    C = C.at[n_g1 + s2 : n_g1 + e2, s:e].set(T_c_11_3)
    s2, e2 = g2(5)
    s, e = ext(2)
    C = C.at[n_g1 + s2 : n_g1 + e2, s:e].set(T_c_19_3)
    s2, e2 = g2(6)
    s, e = ext(5)
    C = C.at[n_g1 + s2 : n_g1 + e2, s:e].set(T_f_13_6)
    s2, e2 = g2(7)
    s, e = ext(5)
    C = C.at[n_g1 + s2 : n_g1 + e2, s:e].set(T_f_14_6)
    s2, e2 = g2(8)
    s, e = ext(5)
    C = C.at[n_g1 + s2 : n_g1 + e2, s:e].set(T_f_18_6)
    s2, e2 = g2(9)
    s, e = ext(7)
    C = C.at[n_g1 + s2 : n_g1 + e2, s:e].set(T_h_15_8)
    s2, e2 = g2(10)
    s, e = ext(7)
    C = C.at[n_g1 + s2 : n_g1 + e2, s:e].set(T_h_16_8)
    s2, e2 = g2(11)
    s, e = ext(7)
    C = C.at[n_g1 + s2 : n_g1 + e2, s:e].set(T_h_20_8)

    # ------------------------------------------------------------------
    # Build D_12 (n_g1 x n_g2).  For each row "owner beta at interface k"
    # in Group 1, the nonzero columns are the Group 2 entries
    # corresponding to the OTHER side of each of beta's three interior
    # faces, with values R_beta[face_k, face_j] for j in beta's interior
    # face set.
    # ------------------------------------------------------------------
    D_12 = jnp.zeros((n_g1, n_g2), dtype=dtype)

    # Row b_9 (g1 idx 0): b's interior faces (9, 10, 18) -> opposite owners
    # (a, c, f).  Group 2 col positions: a_9 = g2(0), c_10 = g2(3), f_18 = g2(8).
    s1, e1 = g1(0)
    s2, e2 = g2(0)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_b_9_9)
    s2, e2 = g2(3)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_b_9_10)
    s2, e2 = g2(8)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_b_9_18)

    # Row b_10 (g1 idx 1)
    s1, e1 = g1(1)
    s2, e2 = g2(0)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_b_10_9)
    s2, e2 = g2(3)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_b_10_10)
    s2, e2 = g2(8)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_b_10_18)

    # Row b_18 (g1 idx 2)
    s1, e1 = g1(2)
    s2, e2 = g2(0)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_b_18_9)
    s2, e2 = g2(3)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_b_18_10)
    s2, e2 = g2(8)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_b_18_18)

    # Row d_11 (g1 idx 3): d's faces (11, 12, 20) -> owners (c, a, h).
    # Group 2 cols: c_11 = g2(4), a_12 = g2(1), h_20 = g2(11).
    s1, e1 = g1(3)
    s2, e2 = g2(4)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_d_11_11)
    s2, e2 = g2(1)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_d_11_12)
    s2, e2 = g2(11)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_d_11_20)

    # Row d_12 (g1 idx 4)
    s1, e1 = g1(4)
    s2, e2 = g2(4)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_d_12_11)
    s2, e2 = g2(1)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_d_12_12)
    s2, e2 = g2(11)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_d_12_20)

    # Row d_20 (g1 idx 5)
    s1, e1 = g1(5)
    s2, e2 = g2(4)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_d_20_11)
    s2, e2 = g2(1)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_d_20_12)
    s2, e2 = g2(11)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_d_20_20)

    # Row e_13 (g1 idx 6): e's faces (13, 16, 17) -> owners (f, h, a).
    # Group 2 cols: f_13 = g2(6), h_16 = g2(10), a_17 = g2(2).
    s1, e1 = g1(6)
    s2, e2 = g2(6)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_e_13_13)
    s2, e2 = g2(10)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_e_13_16)
    s2, e2 = g2(2)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_e_13_17)

    # Row e_16 (g1 idx 7)
    s1, e1 = g1(7)
    s2, e2 = g2(6)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_e_16_13)
    s2, e2 = g2(10)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_e_16_16)
    s2, e2 = g2(2)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_e_16_17)

    # Row e_17 (g1 idx 8)
    s1, e1 = g1(8)
    s2, e2 = g2(6)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_e_17_13)
    s2, e2 = g2(10)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_e_17_16)
    s2, e2 = g2(2)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_e_17_17)

    # Row g_14 (g1 idx 9): g's faces (14, 15, 19) -> owners (f, h, c).
    # Group 2 cols: f_14 = g2(7), h_15 = g2(9), c_19 = g2(5).
    s1, e1 = g1(9)
    s2, e2 = g2(7)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_g_14_14)
    s2, e2 = g2(9)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_g_14_15)
    s2, e2 = g2(5)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_g_14_19)

    # Row g_15 (g1 idx 10)
    s1, e1 = g1(10)
    s2, e2 = g2(7)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_g_15_14)
    s2, e2 = g2(9)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_g_15_15)
    s2, e2 = g2(5)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_g_15_19)

    # Row g_19 (g1 idx 11)
    s1, e1 = g1(11)
    s2, e2 = g2(7)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_g_19_14)
    s2, e2 = g2(9)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_g_19_15)
    s2, e2 = g2(5)
    D_12 = D_12.at[s1:e1, s2:e2].set(T_g_19_19)

    # ------------------------------------------------------------------
    # Build D_21 (n_g2 x n_g1).  Symmetric structure to D_12.
    # ------------------------------------------------------------------
    D_21 = jnp.zeros((n_g2, n_g1), dtype=dtype)

    # Row a_9 (g2 idx 0): a's faces (9, 12, 17) -> owners (b, d, e).
    # Group 1 cols: b_9 = g1(0), d_12 = g1(4), e_17 = g1(8).
    s2, e2 = g2(0)
    s1, e1 = g1(0)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_a_9_9)
    s1, e1 = g1(4)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_a_9_12)
    s1, e1 = g1(8)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_a_9_17)

    # Row a_12 (g2 idx 1)
    s2, e2 = g2(1)
    s1, e1 = g1(0)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_a_12_9)
    s1, e1 = g1(4)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_a_12_12)
    s1, e1 = g1(8)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_a_12_17)

    # Row a_17 (g2 idx 2)
    s2, e2 = g2(2)
    s1, e1 = g1(0)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_a_17_9)
    s1, e1 = g1(4)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_a_17_12)
    s1, e1 = g1(8)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_a_17_17)

    # Row c_10 (g2 idx 3): c's faces (10, 11, 19) -> owners (b, d, g).
    # Group 1 cols: b_10 = g1(1), d_11 = g1(3), g_19 = g1(11).
    s2, e2 = g2(3)
    s1, e1 = g1(1)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_c_10_10)
    s1, e1 = g1(3)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_c_10_11)
    s1, e1 = g1(11)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_c_10_19)

    # Row c_11 (g2 idx 4)
    s2, e2 = g2(4)
    s1, e1 = g1(1)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_c_11_10)
    s1, e1 = g1(3)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_c_11_11)
    s1, e1 = g1(11)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_c_11_19)

    # Row c_19 (g2 idx 5)
    s2, e2 = g2(5)
    s1, e1 = g1(1)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_c_19_10)
    s1, e1 = g1(3)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_c_19_11)
    s1, e1 = g1(11)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_c_19_19)

    # Row f_13 (g2 idx 6): f's faces (13, 14, 18) -> owners (e, g, b).
    # Group 1 cols: e_13 = g1(6), g_14 = g1(9), b_18 = g1(2).
    s2, e2 = g2(6)
    s1, e1 = g1(6)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_f_13_13)
    s1, e1 = g1(9)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_f_13_14)
    s1, e1 = g1(2)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_f_13_18)

    # Row f_14 (g2 idx 7)
    s2, e2 = g2(7)
    s1, e1 = g1(6)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_f_14_13)
    s1, e1 = g1(9)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_f_14_14)
    s1, e1 = g1(2)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_f_14_18)

    # Row f_18 (g2 idx 8)
    s2, e2 = g2(8)
    s1, e1 = g1(6)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_f_18_13)
    s1, e1 = g1(9)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_f_18_14)
    s1, e1 = g1(2)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_f_18_18)

    # Row h_15 (g2 idx 9): h's faces (15, 16, 20) -> owners (g, e, d).
    # Group 1 cols: g_15 = g1(10), e_16 = g1(7), d_20 = g1(5).
    s2, e2 = g2(9)
    s1, e1 = g1(10)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_h_15_15)
    s1, e1 = g1(7)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_h_15_16)
    s1, e1 = g1(5)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_h_15_20)

    # Row h_16 (g2 idx 10)
    s2, e2 = g2(10)
    s1, e1 = g1(10)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_h_16_15)
    s1, e1 = g1(7)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_h_16_16)
    s1, e1 = g1(5)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_h_16_20)

    # Row h_20 (g2 idx 11)
    s2, e2 = g2(11)
    s1, e1 = g1(10)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_h_20_15)
    s1, e1 = g1(7)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_h_20_16)
    s1, e1 = g1(5)
    D_21 = D_21.at[s2:e2, s1:e1].set(T_h_20_20)

    # ------------------------------------------------------------------
    # Assemble h_int and h_ext.  h_int is concat of (Group 1 owners' int
    # h-pieces) followed by (Group 2 owners' int h-pieces).
    # ------------------------------------------------------------------
    h_int = jnp.concatenate(
        [
            # Group 1
            h_b_9,
            h_b_10,
            h_b_18,
            h_d_11,
            h_d_12,
            h_d_20,
            h_e_13,
            h_e_16,
            h_e_17,
            h_g_14,
            h_g_15,
            h_g_19,
            # Group 2
            h_a_9,
            h_a_12,
            h_a_17,
            h_c_10,
            h_c_11,
            h_c_19,
            h_f_13,
            h_f_14,
            h_f_18,
            h_h_15,
            h_h_16,
            h_h_20,
        ],
        axis=0,
    )

    h_ext = jnp.concatenate(
        [h_a_1, h_b_2, h_c_3, h_d_4, h_e_5, h_f_6, h_g_7, h_h_8],
        axis=0,
    )

    A_lst = [
        T_a_1_1,
        T_b_2_2,
        T_c_3_3,
        T_d_4_4,
        T_e_5_5,
        T_f_6_6,
        T_g_7_7,
        T_h_8_8,
    ]

    T, S, h_ext_out, g_tilde_int = assemble_merge_outputs_ItI(
        A_lst, B, C, D_12, D_21, h_ext, h_int
    )
    return T, S, h_ext_out, g_tilde_int


_vmapped_uniform_oct_merge_ItI = jax.vmap(
    _uniform_oct_merge_ItI,
    in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    out_axes=(0, 0, 0, 0),
)


@jax.jit
def vmapped_uniform_oct_merge_ItI(
    q_idxes: jnp.ndarray,
    T_in: jnp.ndarray,
    h_in: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """vmap'd ItI oct-merge over groups of 8 sibling leaves."""
    n_leaves, a, b = T_in.shape
    T_in = T_in.reshape((-1, 8, a, b))
    if h_in.ndim == 2:
        h_in = h_in.reshape((-1, 8, a))
    else:
        nsrc = h_in.shape[-1]
        h_in = h_in.reshape((-1, 8, a, nsrc))

    S, T_out, h_ext_out, g_tilde_int = _vmapped_uniform_oct_merge_ItI(
        q_idxes,
        T_in[:, 0],
        T_in[:, 1],
        T_in[:, 2],
        T_in[:, 3],
        T_in[:, 4],
        T_in[:, 5],
        T_in[:, 6],
        T_in[:, 7],
        h_in[:, 0],
        h_in[:, 1],
        h_in[:, 2],
        h_in[:, 3],
        h_in[:, 4],
        h_in[:, 5],
        h_in[:, 6],
        h_in[:, 7],
    )

    return S, T_out, h_ext_out, g_tilde_int
