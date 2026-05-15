# 3D ItI + 2nd-order Engquist–Majda ABC (driver-side iteration)

This example exercises the 3D Impedance-to-Impedance (ItI) merge path on a
strong-scattering single-sphere problem. It compares a fixed 1st-order
Sommerfeld outer boundary condition against a driver-side **2nd-order
Engquist–Majda absorbing boundary condition (ABC)** layered on top via
fixed-point iteration.

## Problem

A penetrable sphere of radius `R = 0.5` with relative permittivity
`eps = 2.0` (refractive index `n = sqrt(2)`) embedded in a unit-permittivity
background. Wavenumber `kappa = 4`, so `kR = 2.0` and the Born scaling
`chi * (kR)^2 = 4` puts the problem firmly in the strong-scattering regime
(Born–1 fails). The sphere is regularized with a `tanh` ramp of width
`W = 0.02` to keep the HPS coefficient field `C^∞`.

Sources are point sources at 24 Fibonacci points on `r = 1.0`. The
measurement matrix `M[irx, itx]` records the scattered field at each receiver.
The reference is the exact Mie series for the hard sphere
(`scipy.special.spherical_jn` / `spherical_yn`).

## ABC formulation

The standard 1st-order Sommerfeld outflow trace on a cube face is

    ∂u/∂n − iκ u = 0

which in jaxhps's ItI machinery corresponds to `eta = −κ` and incoming
impedance data `g_in ≡ 0` (`bdry_data = 0`).

The 2nd-order Engquist–Majda ABC adds a tangential surface Laplacian term:

    ∂u/∂n − iκ u − (i / (2κ)) · Δ_τ u = 0

This reduces the angular reflection coefficient from `O(θ²)` to `O(θ⁴)`,
which materially cuts the absorbing-boundary error at the corners and
grazing angles of a cubic computational domain.

We do **not** bake the new ABC into the precomputed local solve operators.
Instead we solve the 1st-order Sommerfeld problem with a non-zero
`bdry_data` and iterate:

    bdry_data ← (i / (2κ)) · Δ_τ u_outer

`Δ_τ u_outer` is computed leaf-by-leaf on the outer faces using the
precomputed 3D Chebyshev differentiation operators
(`Δ_τ = D_yy + D_zz` on x-normal faces, etc.), interpolated from Cheby to
Gauss points to match the boundary data layout, and damped (`DAMP = 0.7`)
between iterations for stability.

Typically 5–8 iterations are enough; high-tangential modes (`k_τ²/(2κ) > 1`
at the Nyquist limit) eventually amplify, so the driver tracks the best
iterate against Mie rather than the last one.

## Results

For the parameters listed below, on a single 32 GB CPU box (16 threads):

| config (ItI)                  | rel err vs Mie | reciprocity | total time |
|:------------------------------|---------------:|------------:|-----------:|
| 1st-order Sommerfeld P=16 L=2 |          2.17% |       2.1e-2 |       715s |
| **2nd-order EM ABC P=16 L=2** |      **1.06%** |   **3.2e-3** |       576s |
| 2nd-order EM ABC P=12 L=2     |          1.91% |       5.7e-3 |       175s |

Both `P` and the iteration significantly improve accuracy; the 2nd-order
ABC iteration converges in ~8 inexpensive (~1 s each) re-solves once the
solver is built.

## Running

```bash
pip install -e .        # from repo root

cd examples/iti_3d_2nd_order_abc

# 1. analytic Mie reference (writes mie.npz next to script)
python mie_reference.py

# 2. 1st-order Sommerfeld baseline
P=12 L=2 ITI_L_PHYS=2.0 python run_iti_1st_order_abc.py

# 3. 2nd-order EM ABC iteration
P=12 L=2 ITI_L_PHYS=2.0 N_ITER=8 DAMP=0.7 python run_iti_2nd_order_abc.py

# higher accuracy (more memory):
P=16 L=2 ITI_L_PHYS=2.0 N_ITER=8 DAMP=0.6 python run_iti_2nd_order_abc.py
```

Environment knobs:
- `P`     — Chebyshev order in each cell (per axis). 12 is fast, 16 is accurate.
- `Q`     — Gauss order on boundary (defaults to 10).
- `L`     — octree depth (number of refinement levels). `L=2` ⇒ 64 leaves.
- `ITI_L_PHYS` — half-side of the cubic domain (cube spans `[-L_PHYS, L_PHYS]^3`).
- `N_ITER` — ABC fixed-point iterations.
- `DAMP`  — Picard damping coefficient.

## Files

| file                          | role                                                     |
|:------------------------------|:---------------------------------------------------------|
| `mie_setup.py`                | geometry, kappa, sensors                                  |
| `mie_reference.py`            | exact Mie series (`mie.npz`)                              |
| `run_iti_1st_order_abc.py`    | jaxhps-ItI driver with fixed 1st-order Sommerfeld         |
| `run_iti_2nd_order_abc.py`    | jaxhps-ItI driver + 2nd-order Engquist–Majda iteration    |

## Implementation notes

- The 2nd-order ABC is implemented **entirely in the driver** — no changes
  to `jaxhps` core. This matches how a user would layer a higher-order ABC
  on the existing 3D ItI solver.
- The same approach is applicable to the 2D ItI path; a 2D port would use
  the 1D tangential second derivative on each edge instead of `D_yy + D_zz`.
- Baking the 2nd-order BC directly into `precompute_QH_3D_ItI` / `precompute_G_3D_ItI`
  (so HPS solves it in one shot, no iteration) is a natural future
  feature. It would remove the high-tangential-mode drift entirely.
