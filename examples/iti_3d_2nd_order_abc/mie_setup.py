"""Strong-scattering single-sphere setup.

A homogeneous penetrable sphere (smoothed via tanh) at the origin.
The Mie series gives an exact analytic reference.

CONVENTION: ``EPS_SPHERE`` is the relative permittivity epsilon = n^2.
The Helmholtz equation we solve is Delta u + kappa^2 epsilon(x) u = 0.
Inside sphere epsilon = EPS_SPHERE, outside epsilon = 1.

Dimensionless numbers:
  size parameter  k * R = 4.0 * 0.5 = 2.0
  contrast        chi = epsilon - 1 = 1.0
  interior wavelength = 2 pi / (kappa sqrt(epsilon)) = 1.11
  exterior wavelength = pi / 2 ~ 1.57

Sensors at r = 1.0 (well outside sphere, well inside computational cube).
"""

import numpy as np

KAPPA = 4.0
SPHERE_RADIUS = 0.5
EPS_SPHERE = 2.0  # = n^2 (so refractive index = sqrt(2) ~ 1.414)
SPHERE_CENTER = np.array([0.0, 0.0, 0.0])

# Smoothing width for the tanh ramp (keeps the coefficient field C^infty).
# Keep small (4% of R) so the Mie hard-sphere series remains a good reference.
W_RAMP = 0.02

# Sensors
N_TX = 24
SENSOR_RADIUS = 1.0


def fibonacci_sphere(N, R=1.0):
    idx = np.arange(N) + 0.5
    z = 1.0 - 2.0 * idx / N
    phi = np.pi * (1 + 5**0.5) * idx
    x = np.sqrt(np.maximum(0.0, 1.0 - z * z)) * np.cos(phi)
    y = np.sqrt(np.maximum(0.0, 1.0 - z * z)) * np.sin(phi)
    pts = np.stack([x, y, z], axis=-1) * R
    return pts


def get_tx():
    return fibonacci_sphere(N_TX, R=SENSOR_RADIUS)


def epsilon_field(xg, yg, zg):
    """Smoothed epsilon = n^2 field. Returns 1 outside, EPS_SPHERE inside."""
    import jax.numpy as jnp

    r = jnp.sqrt(
        (xg - SPHERE_CENTER[0]) ** 2
        + (yg - SPHERE_CENTER[1]) ** 2
        + (zg - SPHERE_CENTER[2]) ** 2
    )
    mask = 0.5 * (1 - jnp.tanh((r - SPHERE_RADIUS) / W_RAMP))
    return 1.0 + (EPS_SPHERE - 1.0) * mask
