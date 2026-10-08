"""Synthetic tissue contrasts with compact C4 transitions.

Tissue values follow the historical Lucka-inspired example, not a segmented
clinical phantom. All joins have four continuous derivatives. Feature widths
are geometric parameters and must still be resolved by the discretization.
"""

from functools import lru_cache

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


@lru_cache(maxsize=8)
def vascular_geometry(variant="vascular", seed=7):
    """Return normalized branching centre lines and lobular ellipsoids."""
    if variant not in ("vascular", "dense"):
        raise ValueError("variant must be vascular or dense")
    rng = np.random.default_rng(seed)
    roots = 3 if variant == "vascular" else 5
    branches = []
    for angle in np.arange(roots) * (2 * np.pi / roots):
        start = np.array([0.42 * np.cos(angle), 0.36 * np.sin(angle), -0.12])
        pending = [(start, angle, 0)]
        while pending:
            start, heading, level = pending.pop()
            length = 0.23 * 0.84**level
            direction = np.array(
                [0.32 * np.cos(heading), 0.32 * np.sin(heading), -1.0]
            )
            end = start + length * direction
            middle = (start + end) / 2
            middle[:2] += rng.uniform(-0.035, 0.035, 2)
            radius = 0.045 * 0.8**level
            branches.append((start, middle, end, radius))
            if level < 3:
                for sign in (-1, 1):
                    pending.append((end, heading + sign * 1.15, level + 1))
    lobules = []
    for _ in range(18 if variant == "vascular" else 36):
        centre = rng.uniform([-0.4, -0.32, -0.65], [0.4, 0.32, -0.18])
        axes = rng.uniform(0.12, 0.26, 3)
        angle = rng.uniform(0, 2 * np.pi)
        lobules.append((centre, axes, angle))
    return tuple(branches), tuple(lobules)


def vascular_phantom(points, a=1.25, variant="vascular", seed=7):
    """C4 synthetic pendant breast with 45/75 curved, tapering branches.

    Coordinates are normalized by R=0.8a around z=0.55a. Overlapping
    ellipsoidal bumps form vessels without nearest-segment distance joins.
    This is a lossless scalar model, not a segmented anatomical phantom.
    """
    points = np.asarray(points, dtype=float)
    if a <= 0 or points.shape[-1:] != (3,):
        raise ValueError("positive half-width and (..., 3) points required")
    branches, lobules = vascular_geometry(variant, seed)
    fibro_shapes, vessel_shapes = [], []
    for centre, axes, angle in lobules:
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        fibro_shapes.append((centre, axes, rotation))
    t = np.linspace(0, 1, 6)[:, None]
    for start, middle, end, radius in branches:
        centres = (1 - t) ** 2 * start + 2 * t * (1 - t) * middle + t**2 * end
        tangents = 2 * ((1 - t) * (middle - start) + t * (end - middle))
        tangents /= np.linalg.norm(tangents, axis=-1, keepdims=True)
        normals = np.cross(tangents, [1, 0, 0])
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        rotations = np.stack(
            [normals, np.cross(tangents, normals), tangents], axis=-1
        )
        axes = np.array([radius, radius, np.linalg.norm(end - start) / 3])
        vessel_shapes.extend(
            (centre, axes, rotation)
            for centre, rotation in zip(centres, rotations)
        )
    shape = points.shape[:-1]
    normalized = (points.reshape(-1, 3) - [0, 0, 0.55 * a]) / (0.8 * a)
    result = np.zeros(len(normalized))
    for offset in range(0, len(normalized), 65536):
        pts = normalized[offset : offset + 65536]
        r = np.linalg.norm(pts / [0.92, 0.82, 1.0], axis=-1)
        envelope = step_down(r, 0.97, 0.10) * step_down(pts[:, 2], 0, 0.10)
        active = envelope > 0
        p = pts[active]
        fibro = np.zeros(len(p))
        vessels = np.zeros(len(p))
        for mask, shapes in ((fibro, fibro_shapes), (vessels, vessel_shapes)):
            for centre, axes, rotation in shapes:
                extent = np.abs(rotation) @ axes
                inside = np.all(np.abs(p - centre) < extent, axis=-1)
                local = (p[inside] - centre) @ rotation / axes
                profile = np.maximum(1 - np.sum(local**2, axis=-1), 0) ** 5
                mask[inside] = 1 - (1 - mask[inside]) * (1 - profile)
        fat = -0.041 + 0.006 * np.prod(np.cos(p * [11, 13, 9]), axis=-1)
        b = (1 - fibro) * fat + 0.020 * fibro
        b = (1 - vessels) * b + 0.103 * vessels
        skin = bump((r[active] - 0.9) / 0.04)
        b = (1 - skin) * b + 0.174 * skin
        values = np.zeros(len(pts))
        values[active] = b * envelope[active]
        result[offset : offset + len(pts)] = values
    return result.reshape(shape)
