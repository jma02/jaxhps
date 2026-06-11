"""Smoothed 3D breast phantom and tx/rx sensor geometry.

Replicates the penetrable-breast forward-scattering test from py-helm
(``helmholtz_penetrable_3D.py`` + ``test_params.py``), posed in free space so
it can be driven by the HPS+BIE Sommerfeld coupling in
``wave_scattering_utils_3D.py``.

Geometry (lengths in units where the breast radius is ``b_radius``):

* a hemispherical breast of radius ``b_radius`` occupying ``y > 0``,
* a skin shell of thickness ``delta_skin`` (refractive index ``skinval``),
* interior tissue (``tissueval``) containing up to three spherical tumors
  with indices ``mval[j]``,
* water (index 1) everywhere else.

py-helm builds the squared refractive index ``n(x)`` as a piecewise-constant
material function on a fitted FE mesh.  HPS uses a fixed tensor-product
Chebyshev grid, so the coefficient must be smooth; here every material
interface is mollified with a tanh transition of steepness ``kval``
(py-helm's ``test_params.kval = 40``):

    chi(d) = (1 + tanh(kval * d)) / 2        (smoothed indicator of d > 0)

    n(x) = 1 + chi(y) * [ (skinval - 1) * chi(b_radius - r)
                          + (tissueval - skinval) * chi(b_radius - delta_skin - r)
                          + sum_j (mval_j - tissueval) * chi(rad_j - |x - cen_j|) ]

The scattering potential consumed by the HPS solver is ``b(x) = 1 - n(x)``
(so that ``Lap u + kappa^2 (1 - b) u = 0``).

Sensors: py-helm meshes a spherical cap of radius ``sensor_radius`` cut by the
plane ``y = sensor_offset`` and uses the mesh vertices as collocated
transmitter/receiver locations.  Here we generate a deterministic
quasi-uniform point set of the same cardinality on the same cap with a
Fibonacci lattice.

This module is intentionally NumPy-only so the ngsolve comparison script can
import it without pulling in JAX.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Default parameters -- mirrors py-helm's test_params.py
# ---------------------------------------------------------------------------

KAPPA = 4.0  # wavenumber (test value; the paper uses 16)
KVAL = 40.0  # steepness of the smoothed material transitions
B_RADIUS = 1.0  # breast radius (corresponds to 6 cm)
DELTA_SKIN = B_RADIUS / 30.0  # 0.2 cm skin on a 6 cm breast
SKINVAL = (1524.0 / 1610.0) ** 2  # squared index of skin
TISSUEVAL = (1524.0 / 1485.0) ** 2  # squared index of tissue
LAM_WATER = 2.0 * np.pi / KAPPA
SENSOR_RADIUS = B_RADIUS + LAM_WATER
SENSOR_OFFSET = 0.1 * B_RADIUS
ITARGET = 500  # number of tx/rx points py-helm asks for

# Fixed three-tumor configuration from test_params.py.
DEFAULT_CENTERS = np.array(
    [
        [0.0, B_RADIUS / 2.0, -B_RADIUS / 4.0],
        [0.0, B_RADIUS / 2.0, 0.0],
        [0.0, B_RADIUS / 2.0, B_RADIUS / 4.0],
    ]
)
DEFAULT_RADII = np.array([0.1 * B_RADIUS, 0.1 * B_RADIUS, 0.05 * B_RADIUS])
# test_params.py draws mval ~ (1524/1500)^2 * U(1.01, 1.8); fix representative
# deterministic values so runs are reproducible.
DEFAULT_MVALS = (1524.0 / 1500.0) ** 2 * np.array([1.2, 1.5, 1.1])


def smoothed_indicator(d: np.ndarray, kval: float = KVAL) -> np.ndarray:
    """``chi(d) ~ 1`` for ``d >> 1/kval``, ``~ 0`` for ``d << -1/kval``."""
    return 0.5 * (1.0 + np.tanh(kval * np.asarray(d)))


def breast_n_of_x(
    pts: np.ndarray,
    centers: np.ndarray = DEFAULT_CENTERS,
    radii: np.ndarray = DEFAULT_RADII,
    mvals: np.ndarray = DEFAULT_MVALS,
    b_radius: float = B_RADIUS,
    delta_skin: float = DELTA_SKIN,
    skinval: float = SKINVAL,
    tissueval: float = TISSUEVAL,
    kval: float = KVAL,
) -> np.ndarray:
    """Smoothed squared refractive index ``n(x)`` at ``pts`` (shape (..., 3))."""
    pts = np.asarray(pts)
    r = np.linalg.norm(pts, axis=-1)
    y = pts[..., 1]

    chi_hemi = smoothed_indicator(y, kval)
    chi_outer = smoothed_indicator(b_radius - r, kval)
    chi_inner = smoothed_indicator(b_radius - delta_skin - r, kval)

    bump = (skinval - 1.0) * chi_outer + (tissueval - skinval) * chi_inner
    for c, rad, m in zip(centers, radii, mvals):
        d = rad - np.linalg.norm(pts - np.asarray(c), axis=-1)
        bump = bump + (m - tissueval) * smoothed_indicator(d, kval)
    return 1.0 + chi_hemi * bump


def breast_b_of_x(pts: np.ndarray, **kwargs) -> np.ndarray:
    """Scattering potential ``b(x) = 1 - n(x)`` for the HPS convention."""
    return 1.0 - breast_n_of_x(pts, **kwargs)


def fibonacci_cap_points(
    n_pts: int = ITARGET,
    sensor_radius: float = SENSOR_RADIUS,
    sensor_offset: float = SENSOR_OFFSET,
) -> np.ndarray:
    """Quasi-uniform points on the spherical cap ``|x| = R, y >= offset``.

    Deterministic Fibonacci lattice on the cap, mirroring py-helm's use of
    surface-mesh vertices of the cap as collocated tx/rx locations.
    Returns ``(n_pts, 3)``.
    """
    # Cap in "polar axis = +y" coordinates: cos(theta) in [offset/R, 1].
    c_min = sensor_offset / sensor_radius
    i = np.arange(n_pts)
    cos_t = c_min + (1.0 - c_min) * (i + 0.5) / n_pts
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t**2))
    golden = np.pi * (3.0 - np.sqrt(5.0))
    phi = golden * i
    x = sensor_radius * sin_t * np.cos(phi)
    y = sensor_radius * cos_t
    z = sensor_radius * sin_t * np.sin(phi)
    return np.stack([x, y, z], axis=-1)
