"""3D breast-phantom scattering with HPS+BIE, replicating py-helm's test.

Replicates the forward problem of py-helm's ``helmholtz_penetrable_3D.py``
(see https://github.com/nibj/py-helm) in free space: a smoothed hemispherical
breast phantom (skin shell + tissue + spherical tumors, see
``breast_phantom_3d.py``) is illuminated by point sources placed on a
spherical sensor cap, and the scattered field is recorded at the same sensor
locations, producing the ``(n_rx, n_tx)`` measurement matrix ``umeas``.

Differences from py-helm:

* free space instead of a chest-wall impedance plane at ``y = 0`` (the
  Sommerfeld condition is imposed exactly via the exterior BIE rather than
  with a PML),
* smoothed material coefficients instead of piecewise-constant materials on a
  fitted mesh,
* deterministic Fibonacci-lattice sensors instead of surface-mesh vertices.

Requires precomputed single/double-layer matrices on a cube that strictly
contains the breast (``a > b_radius = 1``), e.g.:

    python examples/gen_SD_3D.py --q 8 --L 2 --kappa 4.0 --a 1.25 \
        --out data/examples/SD_3D/SD_k4_q8_L2_a1.25.npz

Usage:
    python examples/breast_scattering_3d.py --npz data/examples/SD_3D/SD_k4_q8_L2_a1.25.npz \
        --n_sensors 64 --out breast_umeas_hps.npz
"""

import argparse
import os
import sys
import time

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
from breast_phantom_3d import (  # noqa: E402
    B_RADIUS,
    DEFAULT_CENTERS,
    DEFAULT_MVALS,
    DEFAULT_RADII,
    breast_b_of_x,
    fibonacci_cap_points,
)
from wave_scattering_utils_3D import (  # noqa: E402
    eval_uscat_offsurface_3D,
    load_SD_matrices_3D,
    solve_scattering_bie_3D_pointsource,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--npz", required=True, help="SD matrices from gen_SD_3D.py"
    )
    ap.add_argument(
        "--n_sensors",
        type=int,
        default=64,
        help="number of collocated tx/rx points (py-helm targets 500)",
    )
    ap.add_argument("--p", type=int, default=None, help="Chebyshev order")
    ap.add_argument("--out", default="breast_umeas_hps.npz")
    args = ap.parse_args()

    sd = load_SD_matrices_3D(args.npz)
    a, kappa = sd["a"], sd["kappa"]
    if a <= B_RADIUS:
        raise ValueError(
            f"cube half-side a={a} must exceed the breast radius {B_RADIUS}; "
            f"regenerate the SD matrices with a larger --a."
        )

    sensors = fibonacci_cap_points(args.n_sensors)
    print(f"kappa={kappa}, a={a}, n_sensors={sensors.shape[0]}")

    t0 = time.perf_counter()
    out = solve_scattering_bie_3D_pointsource(
        sd, breast_b_of_x, sensors, p=args.p
    )
    t_solve = time.perf_counter() - t0
    print(f"HPS build + BIE solve: {t_solve:.1f}s")

    # Sensors are strictly outside the cube -> exterior representation.
    t0 = time.perf_counter()
    umeas = np.asarray(
        eval_uscat_offsurface_3D(
            target_pts=jnp.asarray(sensors),
            src_pts=jnp.asarray(out["boundary_points"]),
            src_normals=jnp.asarray(out["normals"]),
            src_weights=jnp.asarray(out["sdp"]["wts"]),
            uscat_b=jnp.asarray(out["uscat_b"]),
            uscat_dn_b=jnp.asarray(out["uscat_dn_b"]),
            k=float(kappa),
        )
    )  # (n_rx, n_tx)
    print(f"off-surface eval: {time.perf_counter() - t0:.1f}s")

    np.savez(
        args.out,
        umeas=umeas,
        sensors=sensors,
        kappa=kappa,
        a=a,
        cen=DEFAULT_CENTERS.T,
        rad=DEFAULT_RADII,
        mval=DEFAULT_MVALS,
    )
    print(f"wrote {args.out}  umeas shape {umeas.shape}")


if __name__ == "__main__":
    main()
