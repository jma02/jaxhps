"""3D ItI down-pass for uniform octrees.

This file mirrors the structure of ``_uniform_2D_ItI.py`` but for the 3D
oct-merge case.  The merge stage in ``merge/_uniform_3D_ItI.py`` produces
two outputs per level:

* ``S``, the propagation operator that maps the merged-node's incoming
  impedance trace ``g_in_ext`` to the interior-face traces ``t_int``
  (via ``t_int_homog = S @ g_in_ext``).
* ``g_tilde_int``, the source-only contribution to ``t_int``
  (``t_int = t_int_homog + g_tilde_int``).

The down-pass walks the octree from the root to the leaves.  At each
level it:

1. Computes ``t_int = S @ bdry_data + g_tilde_int``.
2. Splits ``bdry_data`` (24 face-quads) and ``t_int``
   (24 interior-face-quads) into the boundary-data inputs for the eight
   children.

The convention used in the 3D ItI merge is that each interior unknown
``"X_f"`` in ``t_int`` stores ``-g_out_X(f)``, which by impedance
continuity equals ``g_in to X's neighbor across face f``.  In other
words, ``t_int["b_9"] = g_in_a(9)`` (the data leaf ``a`` needs as its
incoming impedance on face 9, the a-b interface).

Both sides of an interior face use the same intra-face 2D ordering
(unlike 2D ItI, where the natural CCW orientation of an edge flips
between the two leaves), so no per-face flip is required.
"""

from typing import List

import jax
import jax.numpy as jnp
import logging


def down_pass_uniform_3D_ItI(
    boundary_data: jax.Array,
    S_lst: List[jax.Array],
    g_tilde_lst: List[jax.Array],
    Y_arr: jax.Array,
    v_arr: jax.Array,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
) -> jax.Array:
    """Downward pass for 3D ItI on a uniform octree.

    Parameters
    ----------
    boundary_data : jax.Array
        Incoming impedance data on the merged-domain exterior, in
        face-quad order (24 face-quads of ``n_per_face`` entries each at
        the root level).  Shape ``(n_bdry,)`` or ``(n_bdry, n_src)``.
    S_lst : List[jax.Array]
        Per-level propagation operators ``S``.  ``S_lst[0]`` is the
        per-leaf-parent merge (level just above the leaves) and
        ``S_lst[-1]`` is the root.
    g_tilde_lst : List[jax.Array]
        Per-level source-only interior data ``g_tilde``.
    Y_arr : jax.Array
        Per-leaf interior-from-boundary maps, shape
        ``(n_leaf, p**3, 6 * q**2)``.  If ``None``, the function returns
        the per-leaf incoming impedance traces instead.
    v_arr : jax.Array
        Per-leaf particular solution at the interior nodes,
        ``(n_leaf, p**3)`` or ``(n_leaf, p**3, n_src)``.

    Returns
    -------
    jax.Array
        Per-leaf interior solutions, shape ``(n_leaf, p**3)`` or
        ``(n_leaf, p**3, n_src)``.
    """
    logging.debug(
        "down_pass_uniform_3D_ItI: started. boundary_imp_data shape: %s, "
        "len(g_tilde_lst): %s, len(S_lst): %s",
        boundary_data.shape,
        len(g_tilde_lst),
        len(S_lst),
    )

    bdry_data = jax.device_put(boundary_data, device)
    if Y_arr is not None:
        Y_arr = jax.device_put(Y_arr, device)
    if v_arr is not None:
        v_arr = jax.device_put(v_arr, device)
    S_lst = [jax.device_put(S, device) for S in S_lst]
    g_tilde_lst = [jax.device_put(g_tilde, device) for g_tilde in g_tilde_lst]

    n_levels = len(S_lst)

    bool_multi_source = len(g_tilde_lst) and g_tilde_lst[0].ndim == 3
    if bool_multi_source and bdry_data.ndim == 1:
        raise ValueError(
            "For multi-source downward pass, need to specify boundary"
            " data for each source."
        )

    # Reshape to (1, n_bdry) or (1, n_bdry, n_src)
    if (bool_multi_source and bdry_data.ndim == 2) or (
        not bool_multi_source and bdry_data.ndim == 1
    ):
        bdry_data = jnp.expand_dims(bdry_data, axis=0)

    # Walk the octree from the root down to the leaves.  At each level
    # we propagate (S, bdry_data, g_tilde) -> 8 children's bdry_data.
    for level in range(n_levels - 1, -1, -1):
        S_arr = S_lst[level]
        g_tilde = g_tilde_lst[level]

        bdry_data = vmapped_propogate_down_3D_ItI(S_arr, bdry_data, g_tilde)

        # Output shape from vmap: (n_parent_at_level, 8, 6 * n_per_face[, n_src])
        # Flatten the (n_parent, 8) axes for the next level.
        if bool_multi_source:
            n_bdry_child = bdry_data.shape[2]
            n_src = bdry_data.shape[-1]
            bdry_data = bdry_data.reshape((-1, n_bdry_child, n_src))
        else:
            n_bdry_child = bdry_data.shape[2]
            bdry_data = bdry_data.reshape((-1, n_bdry_child))

    leaf_incoming_imp = bdry_data

    if Y_arr is None:
        return leaf_incoming_imp

    if bool_multi_source:
        leaf_homog = jnp.einsum("ijk,ikl->ijl", Y_arr, leaf_incoming_imp)
    else:
        leaf_homog = jnp.einsum("ijk,ik->ij", Y_arr, leaf_incoming_imp)

    leaf_solns = leaf_homog + v_arr
    leaf_solns = jax.device_put(leaf_solns, host_device)
    return leaf_solns


@jax.jit
def _propogate_down_3D_ItI(
    S_arr: jax.Array,
    bdry_data: jax.Array,
    f_data: jax.Array,
) -> jax.Array:
    """Propagate impedance data one level down for a single oct-merge.

    Args
    ----
    S_arr : (24 * n_per_face, 24 * n_per_face)
    bdry_data : (24 * n_per_face,) or (24 * n_per_face, n_src)
    f_data : (24 * n_per_face,) or (24 * n_per_face, n_src)

    Returns
    -------
    jax.Array
        Shape (8, 6 * n_per_face[, n_src]).  Each child's outer-face
        boundary data, in leaf-local face order
        ``[face_0, face_1, face_2, face_3, face_4, face_5]``.
    """
    n_per_face = bdry_data.shape[0] // 24
    n = n_per_face

    # t_int has 24 interior face-quad blocks, organised as
    #   [G1 (12 blocks), G2 (12 blocks)]
    # G1 = odd-leaf-owners' interior faces:
    #   b_9, b_10, b_18, d_11, d_12, d_20, e_13, e_16, e_17, g_14, g_15, g_19
    # G2 = even-leaf-owners' interior faces:
    #   a_9, a_12, a_17, c_10, c_11, c_19, f_13, f_14, f_18, h_15, h_16, h_20
    t_int_homog = S_arr @ bdry_data
    t_int = t_int_homog + f_data

    n_g1 = 12 * n  # offset where G2 begins

    def pf(f: int, q: int) -> jax.Array:
        """Parent face f, quad q -> a single n_per_face block."""
        s = (4 * f + q) * n
        return bdry_data[s : s + n]

    def t_g1(i: int) -> jax.Array:
        return t_int[i * n : (i + 1) * n]

    def t_g2(i: int) -> jax.Array:
        return t_int[n_g1 + i * n : n_g1 + (i + 1) * n]

    # ------------------------------------------------------------------
    # Parent face-quad ordering (from get_rearrange_indices in 3D DtN):
    #     face 0 (xmin): [e, h, d, a]
    #     face 1 (xmax): [f, g, c, b]
    #     face 2 (ymin): [e, f, b, a]
    #     face 3 (ymax): [h, g, c, d]
    #     face 4 (zmax): [e, f, g, h]
    #     face 5 (zmin): [a, b, c, d]
    #
    # Each leaf alpha owns 3 outer faces (in leaf-local
    # [0=xmin, 1=xmax, 2=ymin, 3=ymax, 4=zmax, 5=zmin] order) and 3
    # interior faces.  For the interior faces, the value we feed leaf
    # alpha at face f is g_in_alpha(f), which by compatibility equals
    # ``"beta_f"`` in t_int, where beta is alpha's neighbor across f.
    # ------------------------------------------------------------------

    # Leaf a (0,0,0): outer={0,2,5}, interior={1=9 b, 3=12 d, 4=17 e}
    g_a = jnp.concatenate(
        [
            pf(0, 3),  # face 0 xmin
            t_g1(0),  # face 1 xmax  = "b_9"
            pf(2, 3),  # face 2 ymin
            t_g1(4),  # face 3 ymax  = "d_12"
            t_g1(8),  # face 4 zmax  = "e_17"
            pf(5, 0),  # face 5 zmin
        ],
        axis=0,
    )

    # Leaf b (1,0,0): outer={1,2,5}, interior={0=9 a, 3=10 c, 4=18 f}
    g_b = jnp.concatenate(
        [
            t_g2(0),  # face 0 xmin = "a_9"
            pf(1, 3),  # face 1 xmax
            pf(2, 2),  # face 2 ymin
            t_g2(3),  # face 3 ymax = "c_10"
            t_g2(8),  # face 4 zmax = "f_18"
            pf(5, 1),  # face 5 zmin
        ],
        axis=0,
    )

    # Leaf c (1,1,0): outer={1,3,5}, interior={0=11 d, 2=10 b, 4=19 g}
    g_c = jnp.concatenate(
        [
            t_g1(3),  # face 0 xmin = "d_11"
            pf(1, 2),  # face 1 xmax
            t_g1(1),  # face 2 ymin = "b_10"
            pf(3, 2),  # face 3 ymax
            t_g1(11),  # face 4 zmax = "g_19"
            pf(5, 2),  # face 5 zmin
        ],
        axis=0,
    )

    # Leaf d (0,1,0): outer={0,3,5}, interior={1=11 c, 2=12 a, 4=20 h}
    g_d = jnp.concatenate(
        [
            pf(0, 2),  # face 0 xmin
            t_g2(4),  # face 1 xmax = "c_11"
            t_g2(1),  # face 2 ymin = "a_12"
            pf(3, 3),  # face 3 ymax
            t_g2(11),  # face 4 zmax = "h_20"
            pf(5, 3),  # face 5 zmin
        ],
        axis=0,
    )

    # Leaf e (0,0,1): outer={0,2,4}, interior={1=13 f, 3=16 h, 5=17 a}
    g_e = jnp.concatenate(
        [
            pf(0, 0),  # face 0 xmin
            t_g2(6),  # face 1 xmax = "f_13"
            pf(2, 0),  # face 2 ymin
            t_g2(10),  # face 3 ymax = "h_16"
            pf(4, 0),  # face 4 zmax
            t_g2(2),  # face 5 zmin = "a_17"
        ],
        axis=0,
    )

    # Leaf f (1,0,1): outer={1,2,4}, interior={0=13 e, 3=14 g, 5=18 b}
    g_f = jnp.concatenate(
        [
            t_g1(6),  # face 0 xmin = "e_13"
            pf(1, 0),  # face 1 xmax
            pf(2, 1),  # face 2 ymin
            t_g1(9),  # face 3 ymax = "g_14"
            pf(4, 1),  # face 4 zmax
            t_g1(2),  # face 5 zmin = "b_18"
        ],
        axis=0,
    )

    # Leaf g (1,1,1): outer={1,3,4}, interior={0=15 h, 2=14 f, 5=19 c}
    g_g = jnp.concatenate(
        [
            t_g2(9),  # face 0 xmin = "h_15"
            pf(1, 1),  # face 1 xmax
            t_g2(7),  # face 2 ymin = "f_14"
            pf(3, 1),  # face 3 ymax
            pf(4, 2),  # face 4 zmax
            t_g2(5),  # face 5 zmin = "c_19"
        ],
        axis=0,
    )

    # Leaf h (0,1,1): outer={0,3,4}, interior={1=15 g, 2=16 e, 5=20 d}
    g_h = jnp.concatenate(
        [
            pf(0, 1),  # face 0 xmin
            t_g1(10),  # face 1 xmax = "g_15"
            t_g1(7),  # face 2 ymin = "e_16"
            pf(3, 0),  # face 3 ymax
            pf(4, 3),  # face 4 zmax
            t_g1(5),  # face 5 zmin = "d_20"
        ],
        axis=0,
    )

    return jnp.stack([g_a, g_b, g_c, g_d, g_e, g_f, g_g, g_h], axis=0)


vmapped_propogate_down_3D_ItI = jax.vmap(
    _propogate_down_3D_ItI, in_axes=(0, 0, 0), out_axes=0
)
