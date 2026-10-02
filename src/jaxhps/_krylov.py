"""Restarted flexible GMRES for matrix-free JAX operators."""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def _orthogonalize(V: jax.Array, w: jax.Array):
    h = V.conj() @ w
    w = w - V.T @ h
    correction = V.conj() @ w
    return h + correction, w - V.T @ correction


def fgmres(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    *,
    precond: Optional[Callable[[jax.Array], jax.Array]] = None,
    x0: Optional[jax.Array] = None,
    tol: float = 1e-8,
    restart: int = 50,
    maxiter: int = 200,
    stats: Optional[dict] = None,
) -> tuple[jax.Array, int]:
    r"""Right-preconditioned FGMRES, accepting variable or nonlinear solves.

    Store ``z_j = precond(v_j)`` separately from the Arnoldi basis and solve
    the small least-squares problem for ``A Z = V H``. Success is certified
    only by :math:`\|b-Ax\|_2 / \|b\|_2 \leq \mathrm{tol}`. ``maxiter`` bounds
    Arnoldi steps, including a partial final restart; residual checks count
    separately in ``stats['n_matvec']``. Arrays of right-hand sides use one
    shared scalar Krylov space with the Frobenius norm (global FGMRES).

    Retains at most ``2 * restart + 1`` vectors plus a small Hessenberg
    matrix. Python controls iterations; vector operations stay on device.
    ``stats`` is overwritten for each solve. ``true_res_history`` records
    recomputed residuals; ``res_history`` records least-squares estimates.
    """
    if not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("tol must be finite and between zero and one")
    if restart < 1 or maxiter < 1:
        raise ValueError("restart and maxiter must be positive")
    b = jnp.asarray(b, dtype=jnp.result_type(b, jnp.complex128))
    shape = b.shape
    b = b.ravel()
    x = (
        jnp.zeros_like(b)
        if x0 is None
        else jnp.asarray(x0, dtype=b.dtype).ravel()
    )
    if x.shape != b.shape:
        raise ValueError("x0 and b must have the same size")
    if stats is None:
        stats = {}
    stats.update(
        n_matvec=0,
        n_precond=0,
        n_iter=0,
        n_cycles=0,
        res_history=[],
        true_res_history=[],
    )

    def apply(v):
        stats["n_matvec"] += 1
        return matvec(v.reshape(shape)).ravel()

    b_norm = float(jnp.linalg.norm(b))
    if b_norm == 0:
        stats["final_rel_res"] = 0.0
        stats["true_res_history"].append(0.0)
        return jnp.zeros_like(b).reshape(shape), 0
    r = b - apply(x)
    beta = float(jnp.linalg.norm(r))
    stats["true_res_history"].append(beta / b_norm)
    atol = tol * b_norm
    while beta > atol and stats["n_iter"] < maxiter:
        stats["n_cycles"] += 1
        m = min(restart, maxiter - stats["n_iter"])
        V = jnp.zeros((m + 1, b.size), dtype=b.dtype).at[0].set(r / beta)
        Z = jnp.zeros((m, b.size), dtype=b.dtype)
        H = np.zeros((m + 1, m), dtype=np.complex128)
        e1 = np.zeros(m + 1, dtype=np.complex128)
        e1[0] = beta
        for j in range(m):
            z = V[j]
            if precond is not None:
                z = precond(z.reshape(shape)).ravel()
                stats["n_precond"] += 1
            Z = Z.at[j].set(z)
            w = apply(z)
            w_norm = float(jnp.linalg.norm(w))
            if not np.isfinite(w_norm):
                raise ValueError(
                    "operator or preconditioner returned nonfinite values"
                )
            h, w = _orthogonalize(V, w)
            H[: j + 1, j] = np.asarray(h)[: j + 1]
            h_norm = float(jnp.linalg.norm(w))
            H[j + 1, j] = h_norm
            k = j + 1
            stats["n_iter"] += 1
            y, *_ = np.linalg.lstsq(H[: k + 1, :k], e1[: k + 1], rcond=None)
            estimate = np.linalg.norm(H[: k + 1, :k] @ y - e1[: k + 1])
            stats["res_history"].append(float(estimate / b_norm))
            breakdown = h_norm <= np.finfo(np.float64).eps * w_norm
            if estimate <= atol or breakdown or k == m:
                x = x + Z[:k].T @ jnp.asarray(y, dtype=x.dtype)
                r = b - apply(x)
                beta = float(jnp.linalg.norm(r))
                stats["true_res_history"].append(beta / b_norm)
                break
            V = V.at[k].set(w / h_norm)
        if breakdown and beta > atol:
            break
    stats["final_rel_res"] = beta / b_norm
    info = 0 if beta <= atol else 1
    return x.reshape(shape), info
