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


def build_lucka_phantom(int_pts, a):
    """Spherical legacy tissue potential, preserving thresholded layers.
    Polynomial ramps vanish to third order at their support edge, but
    threshold-based tissue overrides can introduce internal jumps.
    This geometry alone is not a field-accuracy validation.
    """
    b_fat = -0.041
    b_fibro = 0.020
    b_vessel = 0.103
    b_skin = 0.174

    shape = int_pts.shape
    pts = int_pts.reshape(-1, 3)
    r = np.linalg.norm(pts, axis=-1)

    t = np.abs(r - 0.9 * a) / (0.08 * a)
    skin_mask = np.where(t < 1.0, (1.0 - t**2) ** 4, 0.0)

    fat_mask = np.where(r < 0.75 * a, 1.0, 0.0)
    trans = (r - 0.75 * a) / (0.10 * a)
    trans = np.clip(trans, 0, 1)
    fat_mask = np.where(
        (r >= 0.75 * a) & (r < 0.85 * a), (1.0 - trans**2) ** 4, fat_mask
    )

    fibro_mask = np.where(r < 0.30 * a, 1.0, 0.0)
    trans_f = (r - 0.30 * a) / (0.10 * a)
    trans_f = np.clip(trans_f, 0, 1)
    fibro_mask = np.where(
        (r >= 0.30 * a) & (r < 0.40 * a),
        (1.0 - trans_f**2) ** 4,
        fibro_mask,
    )

    vessel_radius = 0.05 * a
    distances = (
        np.sqrt((pts[:, 0] - 0.2 * a) ** 2 + (pts[:, 1] - 0.15 * a) ** 2),
        np.sqrt((pts[:, 1] + 0.1 * a) ** 2 + (pts[:, 2] - 0.2 * a) ** 2),
        np.sqrt((pts[:, 0] + 0.15 * a) ** 2 + (pts[:, 2] - (-0.1 * a)) ** 2),
    )
    vessel_mask = np.zeros(len(pts))
    for distance in distances:
        mask = np.where(
            distance < vessel_radius,
            (1.0 - (distance / vessel_radius) ** 2) ** 4,
            0.0,
        )
        vessel_mask = np.maximum(vessel_mask, mask)
    vessel_mask *= r < 0.8 * a

    b = fat_mask * b_fat
    b = np.where(fibro_mask > 0.5, fibro_mask * b_fibro, b)
    b = np.where(vessel_mask > 0.5, vessel_mask * b_vessel, b)
    b = b * (1.0 - skin_mask) + skin_mask * b_skin

    b *= np.where(r < 0.98 * a, 1.0, 0.0)

    return b.reshape(shape[:-1])


def build_lucka_phantom_hemisphere(int_pts, a):
    """Pendant hemispherical legacy tissue potential, preserving thresholded layers.
    Polynomial ramps vanish to third order at their support edge, but
    threshold-based tissue overrides can introduce internal jumps.
    This geometry alone is not a field-accuracy validation.
    """
    b_fat = -0.041
    b_fibro = 0.020
    b_vessel = 0.103
    b_skin = 0.174

    shape = int_pts.shape
    pts = int_pts.reshape(-1, 3)

    R = 0.80 * a
    z0 = 0.55 * a
    c = np.array([0.0, 0.0, z0])
    d = pts - c[None, :]
    r = np.linalg.norm(d, axis=-1)
    z = pts[:, 2]

    t = np.clip((z - (z0 - 0.10 * R)) / (0.10 * R), 0.0, 1.0)
    zcut = (1.0 - t**2) ** 4
    t = np.abs(r - 0.9 * R) / (0.08 * R)
    skin_mask = np.where(t < 1.0, (1.0 - t**2) ** 4, 0.0) * zcut

    fat_mask = np.where(r < 0.75 * R, 1.0, 0.0)
    trans = np.clip((r - 0.75 * R) / (0.10 * R), 0, 1)
    fat_mask = np.where(
        (r >= 0.75 * R) & (r < 0.85 * R), (1.0 - trans**2) ** 4, fat_mask
    )
    fat_mask = fat_mask * zcut

    c_fib = c - np.array([0.0, 0.0, 0.45 * R])
    r_fib = np.linalg.norm(pts - c_fib[None, :], axis=-1)
    R_fib = 0.35 * R
    fibro_mask = np.where(r_fib < 0.75 * R_fib, 1.0, 0.0)
    trans_f = np.clip((r_fib - 0.75 * R_fib) / (0.25 * R_fib), 0, 1)
    fibro_mask = np.where(
        (r_fib >= 0.75 * R_fib) & (r_fib < R_fib),
        (1.0 - trans_f**2) ** 4,
        fibro_mask,
    )

    vessel_radius = 0.05 * R
    distances = (
        np.sqrt((pts[:, 0] - 0.2 * R) ** 2 + (pts[:, 1] - 0.15 * R) ** 2),
        np.sqrt(
            (pts[:, 1] + 0.1 * R) ** 2 + (pts[:, 2] - (z0 - 0.5 * R)) ** 2
        ),
        np.sqrt(
            (pts[:, 0] + 0.15 * R) ** 2 + (pts[:, 2] - (z0 - 0.6 * R)) ** 2
        ),
    )
    vessel_mask = np.zeros(len(pts))
    for distance in distances:
        mask = np.where(
            distance < vessel_radius,
            (1.0 - (distance / vessel_radius) ** 2) ** 4,
            0.0,
        )
        vessel_mask = np.maximum(vessel_mask, mask)
    vessel_mask *= (r < 0.8 * R) * zcut

    b = fat_mask * b_fat
    b = np.where(fibro_mask > 0.5, fibro_mask * b_fibro, b)
    b = np.where(vessel_mask > 0.5, vessel_mask * b_vessel, b)
    b = b * (1.0 - skin_mask) + skin_mask * b_skin

    rad_all = np.linalg.norm(pts, axis=-1)
    b *= np.where(rad_all < 0.98 * np.sqrt(3) * a, 1.0, 0.0)

    return b.reshape(shape[:-1])
