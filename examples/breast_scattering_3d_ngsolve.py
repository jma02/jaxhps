"""Smoothed free-space ngsolve version of py-helm's 3D breast problem.

Reference solver for ``breast_scattering_3d.py``.  This is a modified version
of py-helm's ``helmholtz_penetrable_3D.py`` (https://github.com/nibj/py-helm)
with two changes that make it directly comparable to the HPS+BIE solver:

* the squared refractive index ``n(x)`` is the same smoothed coefficient as
  ``breast_phantom_3d.breast_n_of_x`` (tanh-mollified interfaces, steepness
  ``kval``), expressed as a global ngsolve ``CoefficientFunction`` instead of
  piecewise-constant materials on a fitted mesh;
* the problem is posed in free space -- the chest-wall impedance plane at
  ``y = 0`` is removed, and the radiation condition is approximated by a
  radial PML on a full ball (py-helm uses a half-ball + PML).

Everything else mirrors py-helm: point sources at the sensor locations,
scattered-field formulation ``Lap u^s + kappa^2 n u^s = kappa^2 (1 - n) u^inc``
solved per transmitter with a factored FEM matrix, and the scattered field
recorded at all sensors, giving ``umeas`` of shape ``(n_rx, n_tx)``.

The sensors are the same Fibonacci cap points as the HPS driver, so the two
``umeas`` matrices are entry-wise comparable (see ``breast_compare_3d.py``).

Usage:
    python examples/breast_scattering_3d_ngsolve.py --n_sensors 64 \
        --out breast_umeas_ngsolve.npz
"""

import argparse
import os
import sys
import time

import numpy as np
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    GridFunction,
    H1,
    LinearForm,
    Mesh,
    SetNumThreads,
    SymbolicBFI,
    SymbolicLFI,
    TaskManager,
    exp,
    grad,
    ngsglobals,
    pml,
    sqrt,
    x,
    y,
    z,
)
from netgen.csg import CSGeometry, Pnt, Sphere

sys.path.insert(0, os.path.dirname(__file__))
from breast_phantom_3d import (  # noqa: E402
    B_RADIUS,
    DEFAULT_CENTERS,
    DEFAULT_MVALS,
    DEFAULT_RADII,
    DELTA_SKIN,
    KAPPA,
    KVAL,
    SKINVAL,
    TISSUEVAL,
    fibonacci_cap_points,
)

ngsglobals.msg_level = 0
SetNumThreads(16)


def cf_tanh(s):
    """Overflow-safe tanh of an ngsolve CoefficientFunction."""
    return 1.0 - 2.0 / (exp(2.0 * s) + 1.0)


def cf_chi(d, kval=KVAL):
    """Smoothed indicator chi(d) = (1 + tanh(kval*d)) / 2."""
    return 0.5 * (1.0 + cf_tanh(kval * d))


def smoothed_ncoef():
    """ngsolve CF mirroring ``breast_phantom_3d.breast_n_of_x``."""
    r = sqrt(x * x + y * y + z * z)
    chi_hemi = cf_chi(y)
    chi_outer = cf_chi(B_RADIUS - r)
    chi_inner = cf_chi(B_RADIUS - DELTA_SKIN - r)
    bump = (SKINVAL - 1.0) * chi_outer + (TISSUEVAL - SKINVAL) * chi_inner
    for c, rad, m in zip(DEFAULT_CENTERS, DEFAULT_RADII, DEFAULT_MVALS):
        d = rad - sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2)
        bump = bump + (m - TISSUEVAL) * cf_chi(d)
    return 1.0 + chi_hemi * bump


def make_mesh(hmax_air, hmax_breast, pmlmin, delta_pml):
    """Full ball: breast-refinement sphere + water + PML shell."""
    geo = CSGeometry()
    breast_zone = Sphere(Pnt(0, 0, 0), 1.2 * B_RADIUS).maxh(hmax_breast)
    water = Sphere(Pnt(0, 0, 0), pmlmin)
    outer = Sphere(Pnt(0, 0, 0), pmlmin + delta_pml).bc("outer")
    geo.Add(breast_zone.mat("Breast"))
    geo.Add((water - breast_zone).mat("Water"))
    geo.Add((outer - water).mat("PML"))
    mesh = Mesh(geo.GenerateMesh(maxh=hmax_air))
    mesh.Curve(2)
    return mesh


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_sensors", type=int, default=64)
    ap.add_argument("--porder", type=int, default=3)
    ap.add_argument(
        "--ppw", type=float, default=8.0, help="points per wavelength in air"
    )
    ap.add_argument(
        "--hmax_breast",
        type=float,
        default=None,
        help="mesh size in the breast region; default resolves the smoothed"
        " interfaces (min(hmax_air/sqrt(max n), 2/kval))",
    )
    ap.add_argument(
        "--water_layers",
        type=float,
        default=2.0,
        help="water thickness between the breast and the PML, in wavelengths"
        " (py-helm uses 2; must keep the sensors inside the water region)",
    )
    ap.add_argument("--out", default="breast_umeas_ngsolve.npz")
    args = ap.parse_args()

    kappa = KAPPA
    lam_water = 2.0 * np.pi / kappa
    pmlmin = B_RADIUS + args.water_layers * lam_water
    delta_pml = lam_water / 2.0
    hmax_air = lam_water / args.ppw
    # The smoothed interfaces have width ~1/KVAL; resolve them.
    hmax_breast = args.hmax_breast
    if hmax_breast is None:
        hmax_breast = min(hmax_air / np.sqrt(max(DEFAULT_MVALS)), 2.0 / KVAL)

    sensors = fibonacci_cap_points(args.n_sensors)
    sensor_rad = float(np.linalg.norm(sensors[0]))
    if sensor_rad >= pmlmin:
        raise ValueError(
            f"sensors (radius {sensor_rad:.3f}) must lie inside the water"
            f" region (PML starts at {pmlmin:.3f}); increase --water_layers."
        )
    print(
        f"kappa={kappa}, hmax_air={hmax_air:.4f}, hmax_breast={hmax_breast:.4f},"
        f" n_sensors={sensors.shape[0]}"
    )

    mesh = make_mesh(hmax_air, hmax_breast, pmlmin, delta_pml)
    mesh.SetPML(pml.Radial((0, 0, 0), rad=pmlmin, alpha=1j), "PML")
    ncoef = smoothed_ncoef()

    fes = H1(mesh, order=args.porder, complex=True)
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += SymbolicBFI(grad(u) * grad(v) - kappa**2 * ncoef * u * v)
    a += SymbolicBFI(-1j * kappa * u * v, definedon=mesh.Boundaries("outer"))
    print("Number of DoFs:", fes.ndof)
    t0 = time.perf_counter()
    with TaskManager():
        a.Assemble()
        Ainv = a.mat.Inverse()
    print(f"assemble+factor: {time.perf_counter() - t0:.1f}s")

    n_s = sensors.shape[0]
    umeas = np.zeros((n_s, n_s), dtype=complex)
    gfu = GridFunction(fes)
    for itx, (xtx, ytx, ztx) in enumerate(sensors):
        dist = sqrt((x - xtx) ** 2 + (y - ytx) ** 2 + (z - ztx) ** 2)
        ui = CoefficientFunction(exp(1j * kappa * dist) / (4.0 * np.pi * dist))
        b = LinearForm(fes)
        b += SymbolicLFI(kappa * kappa * (ncoef - 1.0) * ui * v)
        with TaskManager():
            b.Assemble()
            gfu.vec.data = Ainv * b.vec
            umeas[:, itx] = [gfu(mesh(*p)) for p in sensors]
        if itx % 10 == 0:
            print(f"tx {itx}/{n_s} done")

    np.savez(
        args.out,
        umeas=umeas,
        sensors=sensors,
        kappa=kappa,
        cen=DEFAULT_CENTERS.T,
        rad=DEFAULT_RADII,
        mval=DEFAULT_MVALS,
    )
    print(f"wrote {args.out}  umeas shape {umeas.shape}")


if __name__ == "__main__":
    main()
