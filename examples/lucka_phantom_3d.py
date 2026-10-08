"""Synthetic tissue contrasts with compact C4 transitions.

Tissue values follow the historical Lucka-inspired example, not a segmented
clinical phantom. All joins have four continuous derivatives. Feature widths
are geometric parameters and must still be resolved by the discretization.
"""

import numpy as np
from scipy.special import betainc


def step_down(x, end, width):
    """One before ``end-width``, zero after ``end``; C4 at both joins."""
    if width <= 0:
        raise ValueError("width must be positive")
    t = np.clip((end - np.asarray(x)) / width, 0.0, 1.0)
    return betainc(5, 5, t)


def bump(t):
    """Compact C4 profile, polynomial in squared distance near its centre."""
    return np.maximum(1.0 - np.asarray(t) ** 2, 0.0) ** 5


def tissue_phantom(points, a=1.25, geometry="hemisphere"):
    """Evaluate b=1-n² on (..., 3) points; exactly zero near the box faces."""
    points = np.asarray(points, dtype=float)
    if a <= 0 or points.shape[-1:] != (3,):
        raise ValueError("positive half-width and (..., 3) points required")
    if geometry not in ("hemisphere", "sphere"):
        raise ValueError("geometry must be hemisphere or sphere")
    pendant = geometry == "hemisphere"
    radius = (0.8 if pendant else 1.0) * a
    z0 = 0.55 * a if pendant else 0.0
    centre = np.array([0.0, 0.0, z0])
    r = np.linalg.norm(points - centre, axis=-1)
    zcut = step_down(points[..., 2], z0, 0.1 * radius) if pendant else 1.0
    fat = step_down(r, 0.85 * radius, 0.1 * radius)
    skin = bump((r - 0.9 * radius) / (0.08 * radius))

    fibro_centre = centre - np.array(
        [0.0, 0.0, 0.45 * radius if pendant else 0.0]
    )
    fibro_radius = (0.35 if pendant else 0.4) * radius
    fibro = step_down(
        np.linalg.norm(points - fibro_centre, axis=-1),
        fibro_radius,
        0.25 * fibro_radius,
    )
    x, y, z = np.moveaxis(points, -1, 0)
    distances = (
        np.hypot(x - 0.2 * radius, y - 0.15 * radius),
        np.hypot(
            y + 0.1 * radius,
            z - (z0 - 0.5 * radius if pendant else 0.2 * radius),
        ),
        np.hypot(
            x + 0.15 * radius,
            z - (z0 - 0.6 * radius if pendant else -0.1 * radius),
        ),
    )
    vessel = np.zeros(points.shape[:-1])
    for distance in distances:
        vessel = 1.0 - (1.0 - vessel) * (
            1.0 - bump(distance / (0.05 * radius))
        )
    vessel *= step_down(r, 0.8 * radius, 0.1 * radius)

    b = -0.041 * fat
    b = (1.0 - fibro) * b + 0.020 * fibro
    b = (1.0 - vessel) * b + 0.103 * vessel
    b = (1.0 - skin) * b + 0.174 * skin
    return b * zcut
