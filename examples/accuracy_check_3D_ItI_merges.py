"""Accuracy check for the 3D ItI (Impedance-to-Impedance) merges.

Sweeps over refinement levels ``L`` and polynomial orders ``p`` for two
manufactured problems with impedance (Robin-type) boundary data and reports
max-norm relative errors against the analytic solution.

Reproduced from the upstream-author test gist:
    https://gist.github.com/meliao/9b42e6effe81f6dd1c6decefe7a4db05

Usage
-----
    python examples/accuracy_check_3D_ItI_merges.py \
        --problem_1 --problem_2 --p_vals 4 6 8 --l_vals 1 2

See also
--------
``examples/hp_convergence_2D_problems.py`` — the 2D counterpart.
"""

import argparse
import logging
import os
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np

from jaxhps import (
    DiscretizationNode3D,
    Domain,
    PDEProblem,
    build_solver,
    solve,
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)

jax.config.update("jax_default_device", jax.devices("cpu")[0])

XMIN, XMAX = -0.5, 0.5
YMIN, YMAX = -0.5, 0.5
ZMIN, ZMAX = -0.5, 0.5


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="data/examples/accuracy_check_3D_ItI",
        help="Directory for plot/.npz outputs (created if needed).",
    )
    parser.add_argument("--problem_1", action="store_true")
    parser.add_argument("--problem_2", action="store_true")
    parser.add_argument(
        "--p_vals", type=int, nargs="+", default=[4, 6, 8],
        help="Chebyshev polynomial orders to sweep.",
    )
    parser.add_argument(
        "--l_vals", type=int, nargs="+", default=[1, 2],
        help="Octree refinement levels to sweep.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


class Problem3DItI(ABC):
    """Abstract base for 3D ItI accuracy-check problems.

    Subclasses supply the analytic solution, PDE coefficients, and source
    term; ``run`` handles the (L, p) sweep and error computation.
    """

    # Impedance parameter — override in subclasses as needed.
    eta: float = 1.0

    @abstractmethod
    def soln(self, pts: jax.Array) -> jax.Array:
        """Analytic solution at points ``pts`` of shape (..., 3)."""

    @abstractmethod
    def source(self, pts: jax.Array) -> jax.Array:
        """Source term ``f`` at interior Chebyshev points."""

    @abstractmethod
    def _dx(self, pts: jax.Array) -> jax.Array:
        """``du/dx`` at points ``pts``."""

    @abstractmethod
    def _dy(self, pts: jax.Array) -> jax.Array:
        """``du/dy`` at points ``pts``."""

    @abstractmethod
    def _dz(self, pts: jax.Array) -> jax.Array:
        """``du/dz`` at points ``pts``."""

    def D_xx_coefficients(self, pts: jax.Array) -> jax.Array:
        return jnp.ones(pts.shape[:-1])

    def D_yy_coefficients(self, pts: jax.Array) -> jax.Array:
        return jnp.ones(pts.shape[:-1])

    def D_zz_coefficients(self, pts: jax.Array) -> jax.Array:
        return jnp.ones(pts.shape[:-1])

    def I_coefficients(self, pts: jax.Array) -> jax.Array | None:
        return None

    def D_x_coefficients(self, pts: jax.Array) -> jax.Array | None:
        return None

    def D_y_coefficients(self, pts: jax.Array) -> jax.Array | None:
        return None

    def D_z_coefficients(self, pts: jax.Array) -> jax.Array | None:
        return None

    def boundary_data(
        self, boundary_points: jax.Array, root: DiscretizationNode3D
    ) -> list[jax.Array]:
        """Impedance data ``du/dn + i*eta*u`` on each of the 6 cube faces.

        Face ordering: x=xmin, x=xmax, y=ymin, y=ymax, z=zmin, z=zmax.
        """
        n_per_face = boundary_points.shape[0] // 6
        f1 = boundary_points[:n_per_face]
        f2 = boundary_points[n_per_face : 2 * n_per_face]
        f3 = boundary_points[2 * n_per_face : 3 * n_per_face]
        f4 = boundary_points[3 * n_per_face : 4 * n_per_face]
        f5 = boundary_points[4 * n_per_face : 5 * n_per_face]
        f6 = boundary_points[5 * n_per_face :]
        return [
            -self._dx(f1) + 1j * self.eta * self.soln(f1),
            self._dx(f2) + 1j * self.eta * self.soln(f2),
            -self._dy(f3) + 1j * self.eta * self.soln(f3),
            self._dy(f4) + 1j * self.eta * self.soln(f4),
            -self._dz(f5) + 1j * self.eta * self.soln(f5),
            self._dz(f6) + 1j * self.eta * self.soln(f6),
        ]

    def run(self, l_vals: list[int], p_vals: list[int]) -> np.ndarray:
        """Sweep (L, p) and return a 2-D array of max-norm relative errors."""
        errors = np.zeros((len(l_vals), len(p_vals)))
        for i, l in enumerate(l_vals):
            l = int(l)
            for j, p in enumerate(p_vals):
                p = int(p)
                logging.info(
                    "Running %s with l=%i, p=%i", type(self).__name__, l, p
                )
                root = DiscretizationNode3D(
                    xmin=XMIN, xmax=XMAX,
                    ymin=YMIN, ymax=YMAX,
                    zmin=ZMIN, zmax=ZMAX,
                )
                domain = Domain(p=p, q=p - 2, root=root, L=l)
                pts = domain.interior_points

                pde_problem = PDEProblem(
                    domain=domain,
                    D_xx_coefficients=self.D_xx_coefficients(pts),
                    D_yy_coefficients=self.D_yy_coefficients(pts),
                    D_zz_coefficients=self.D_zz_coefficients(pts),
                    D_x_coefficients=self.D_x_coefficients(pts),
                    D_y_coefficients=self.D_y_coefficients(pts),
                    D_z_coefficients=self.D_z_coefficients(pts),
                    I_coefficients=self.I_coefficients(pts),
                    source=self.source(pts),
                    use_ItI=True,
                    eta=self.eta,
                )

                build_solver(pde_problem)
                g = self.boundary_data(domain.boundary_points, root)
                computed_soln = solve(pde_problem, g)
                expected_soln = self.soln(pts)

                err = float(
                    np.max(np.abs(np.asarray(computed_soln) - np.asarray(expected_soln)))
                )
                nrm = float(np.max(np.abs(np.asarray(expected_soln))))
                errors[i, j] = err / nrm
                logging.info(
                    "%s: l=%i, p=%i, err=%.3e",
                    type(self).__name__, l, p, errors[i, j],
                )
                jax.clear_caches()
        return errors


# ---------------------------------------------------------------------------
# Problem 1 — polynomial coefficients, polynomial solution.
# ---------------------------------------------------------------------------

class Problem1(Problem3DItI):
    """Operator: ``Delta u + x * u_x + 3 z^2 * u_z = f``.

    Solution ``u = 4 x^3 y^4 z - z^2 + 4 y`` is a polynomial of degree 8
    that the spectral collocation should resolve to machine precision once
    ``p`` is large enough.
    """

    eta = 1.0

    def soln(self, pts):
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        return 4 * x**3 * y**4 * z - z**2 + 4 * y

    def _dx(self, pts):
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        return 12 * x**2 * y**4 * z

    def _dy(self, pts):
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        return 16 * x**3 * y**3 * z + 4

    def _dz(self, pts):
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        return 4 * x**3 * y**4 - 2 * z

    def source(self, pts):
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        # Laplacian: u_xx + u_yy + u_zz
        term1 = 24 * x * y**4 * z         # u_xx of 4 x^3 y^4 z
        term2 = 48 * x**3 * y**2 * z      # u_yy of 4 x^3 y^4 z
        term3 = -2                         # u_zz of -z^2
        # x * u_x
        term4 = 12 * x**3 * y**4 * z
        # 3 z^2 * u_z
        term5 = 12 * x**3 * y**4 * z**2
        term6 = -6 * z**3
        return term1 + term2 + term3 + term4 + term5 + term6

    def D_x_coefficients(self, pts):
        return pts[..., 0]

    def D_z_coefficients(self, pts):
        return 3 * pts[..., 2] ** 2


# ---------------------------------------------------------------------------
# Problem 2 — Gravity Helmholtz; solution is a plane wave.
# ---------------------------------------------------------------------------

class Problem2(Problem3DItI):
    """``Delta u + (1 - z) * eta^2 * u = -z * eta^2 * u``, plane-wave solution.

    Solution ``u = exp(i * eta * d . x)`` with direction
    ``d = (1, 1, 1) / sqrt(3)``. With ``eta = 16`` the plane wave oscillates
    roughly five times across the unit cube, which probes the solver's
    high-wavenumber accuracy at modest ``p`` / ``L``.
    """

    eta = 16.0
    source_dir = jnp.array([1.0, 1.0, 1.0]) / jnp.sqrt(3)

    def soln(self, pts):
        return jnp.exp(1j * self.eta * pts @ self.source_dir)

    def source(self, pts):
        z = pts[..., 2]
        return -z * self.eta**2 * self.soln(pts)

    def I_coefficients(self, pts):
        z = pts[..., 2]
        return (1 - z) * self.eta**2

    def _dx(self, pts):
        return 1j * self.eta * self.source_dir[0] * self.soln(pts)

    def _dy(self, pts):
        return 1j * self.eta * self.source_dir[1] * self.soln(pts)

    def _dz(self, pts):
        return 1j * self.eta * self.source_dir[2] * self.soln(pts)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(
    errors: np.ndarray,
    l_vals: list[int],
    p_vals: list[int],
    name: str,
    plots_dir: str,
) -> None:
    """Save the error grid to ``<plots_dir>/<name>.npz`` (creates dir)."""
    os.makedirs(plots_dir, exist_ok=True)
    out = os.path.join(plots_dir, f"{name}.npz")
    np.savez(out, errors=errors, l_vals=np.array(l_vals), p_vals=np.array(p_vals))
    logging.info("Wrote %s", out)


def maybe_plot(
    errors: np.ndarray,
    l_vals: list[int],
    p_vals: list[int],
    title: str,
    save_path: str | None,
) -> None:
    """Plot max-norm relative error vs ``p`` (one curve per ``L``) if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.info("matplotlib not available; skipping plot")
        return

    fig, ax = plt.subplots()
    for i, l in enumerate(l_vals):
        ax.semilogy(p_vals, errors[i], marker="o", label=f"L={l}")
    ax.set_xticks(p_vals)
    ax.set_xlabel("p")
    ax.set_ylabel(r"Relative $\ell_{\infty}$ error")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        logging.info("Wrote %s", save_path)
    plt.close(fig)


def main() -> None:
    args = setup_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not (args.problem_1 or args.problem_2):
        # Default: run both.
        args.problem_1 = args.problem_2 = True

    if args.problem_1:
        e = Problem1().run(args.l_vals, args.p_vals)
        print("\nProblem1 errors (rows=L, cols=p):")
        print(np.array2string(e, formatter={"float_kind": lambda x: f"{x:.3e}"}))
        save_results(e, args.l_vals, args.p_vals, "problem_1", args.plots_dir)
        maybe_plot(
            e, args.l_vals, args.p_vals,
            "Problem 1: variable-coefficient polynomial",
            os.path.join(args.plots_dir, "problem_1.png"),
        )

    if args.problem_2:
        e = Problem2().run(args.l_vals, args.p_vals)
        print("\nProblem2 errors (rows=L, cols=p):")
        print(np.array2string(e, formatter={"float_kind": lambda x: f"{x:.3e}"}))
        save_results(e, args.l_vals, args.p_vals, "problem_2", args.plots_dir)
        maybe_plot(
            e, args.l_vals, args.p_vals,
            "Problem 2: gravity Helmholtz plane wave",
            os.path.join(args.plots_dir, "problem_2.png"),
        )


if __name__ == "__main__":
    main()
