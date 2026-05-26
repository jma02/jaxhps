# 3D ItI + 2nd-order Engquist–Majda ABC

This example validates the 3D Impedance-to-Impedance (ItI) merge path and
the **2nd-order Engquist–Majda absorbing boundary condition (ABC)**
update operator that a driver script layers on top of the precomputed
1st-order ItI solver. Both drivers solve the same constant-coefficient
Helmholtz problem with an analytic plane-wave reference.

## Problem

We solve the homogeneous Helmholtz equation

$$(\Delta + \kappa^2) u = 0 \quad \text{on} \quad \Omega = [-L_\text{phys}, L_\text{phys}]^3, \qquad \kappa = 4,$$

with the plane wave $u^*(x) = e^{i k \cdot x}$, $|k| = \kappa$, as the
manufactured solution. We deliberately pick a non-axis-aligned direction
$\hat{k} = (1, 2, 2) / 3$ so every cube face sees a non-trivial
$\hat{k} \cdot \hat{n}$ angle and the boundary traces are not trivially
zero.

The coefficient field is constant ($\varepsilon = 1$) — no smoothed
scatterer, no $\tanh$ transition layer — so any error in the recovered
solution is attributable to the discretization of $u^*$ or to the
boundary data itself, not to a non-smooth coefficient.

## 1st-order driver (`run_iti_1st_order_abc.py`)

Sets `bdry_data` to the analytic 1st-order Sommerfeld trace of $u^*$

$$g_1^*(x) = u^*_n(x) - i \kappa u^*(x) = i \kappa (\hat{k} \cdot \hat{n} - 1) u^*(x),$$

evaluated at every boundary Gauss point. A single `solve` should
recover $u^*$ at every interior Chebyshev point.

| $p$ | $q$ | $L$ | $\\|u_\text{computed} - u^*\\|_2 / \\|u^*\\|_2$ |
|----:|----:|----:|------------------------------------------------:|
| 12  | 10  | 2   |                                       $3 \times 10^{-12}$ |

The remaining $\sim 10^{-12}$ is the spectral truncation error of
$e^{i\kappa x}$ on a leaf of half-side $h = L_\text{phys} / 2^L = 0.25$
with $p = 12$, where $\kappa h \approx 1$. Higher $p$ pushes this lower
at the expected rate.

## 2nd-order driver (`run_iti_2nd_order_abc.py`)

The 2nd-order Engquist–Majda ABC equation

$$\partial_n u - i \kappa u - \frac{i}{2 \kappa} \Delta_\tau u = 0$$

is **not** satisfied by a plane wave at oblique incidence — a plane wave
is what gets reflected by the ABC at non-normal angles. To turn the
2nd-order ABC into something a plane wave can exactly satisfy, we add
the analytic residual

$$g_\text{base}(x) = -\frac{i \kappa}{2} (1 - \hat{k} \cdot \hat{n})^2 u^*(x)$$

as a manufactured boundary forcing. The forced ABC equation,
$\partial_n u - i \kappa u - (i / (2\kappa)) \Delta_\tau u = g_\text{base}$,
is then satisfied exactly by $u = u^*$.

In jaxhps's $\eta = -\kappa$ convention the impedance trace passed in
via `bdry_data` is $g(x) = u_n(x) - i \kappa u(x)$. Moving the
$\Delta_\tau$ term to the right-hand side gives the fixed-point
equation a driver-side iteration would aim to satisfy:

$$g = \frac{i}{2 \kappa} \Delta_\tau u \; + \; g_\text{base}. \quad (\ast)$$

The analytic fixed point of $(\ast)$ for $u = u^*$ is the 1st-order
Sommerfeld trace $g^*(x) = i \kappa (\hat{k} \cdot \hat{n} - 1) u^*(x)$
(one-line algebraic verification: substitute and use
$\Delta_\tau u^* = -\kappa^2 (1 - (\hat{k} \cdot \hat{n})^2) u^*$).

The driver does a one-shot fixed-point validation:

1. **Solve at the analytic fixed point.** Set `bdry_data = g_star`, run
   `solve`, and verify $u_\text{computed} = u^*$ at every interior
   Chebyshev point.
2. **Apply one iteration update.** Compute
   $\text{new\_bdry} = (i / (2\kappa)) \Delta_\tau u_\text{computed} + g_\text{base}$
   using the per-face surface-Laplacian + Cheby-to-Gauss interpolation
   + leaf-to-cube-face routing machinery that a Picard iteration would
   apply. Verify $\text{new\_bdry} = g^*$ at machine precision.

Step 2 is the unit test for the 2nd-order ABC update operator: every
piece of the iteration loop (3D Chebyshev $D_{xx}, D_{yy}, D_{zz}$,
$\Delta_\tau$ assembly per cube-face normal, $p^2 \to q^2$ barycentric
Lagrange Gauss-from-Cheby interpolation, and the leaf-index lookup that
routes each volume leaf's outgoing face into the right block of
`bdry_data`) gets exercised.

| $p$ | $q$ | $L$ | $\\|u_\text{computed} - u^*\\| / \\|u^*\\|$ | $\\|\text{new\\_bdry} - g^*\\| / \\|g^*\\|$ |
|----:|----:|----:|-------------------------------------------:|-----------------------------------------:|
| 12  | 10  | 2   |                          $3 \times 10^{-12}$ |                       $1.6 \times 10^{-10}$ |

The slight loss in step 2 vs. step 1 is consistent with applying
$\Delta_\tau$ (which has spectrum $\sim \kappa^2$) to a $u$ already at
$\sim 10^{-12}$ relative error: the result has the same absolute error
amplified by $\kappa^2$ in physical units, and divided by
$\\|g^*\\| \sim \kappa \\|u^*\\|$ gives the observed
$10^{-10}$.

### Why this isn't a Picard convergence test

The 2nd-order EM ABC iteration is well known to be unstable at large
$\kappa h$: $\Delta_\tau / (2 \kappa)$ amplifies tangential modes with
$k_\tau^2 / (2 \kappa) > 1$ faster than any uniform Picard relaxation can
damp them. For the constant-coefficient plane-wave problem these
high-tangential modes are populated *everywhere* on the boundary, so
running the iteration from `bdry_data = 0` diverges regardless of
relaxation parameter. The fixed-point verification above tests the same
$\Delta_\tau$ / Gauss / face machinery without depending on Picard
stability. A more robust scheme — e.g. solving $(\ast)$ directly with
GMRES, or applying a tangential low-pass filter to `bdry_data` between
iterations — would be needed to *find* the fixed point starting from a
poor initial guess; that is orthogonal to the validation of the
2nd-order ABC formula itself, which is what this example confirms.

## Running

```bash
pip install -e .        # from repo root

cd examples/iti_3d_2nd_order_abc

# 1st-order Sommerfeld, single solve with analytic plane-wave trace
P=12 Q=10 L=2 ITI_L_PHYS=1.0 python run_iti_1st_order_abc.py

# 2nd-order Engquist-Majda ABC, fixed-point verification
P=12 Q=10 L=2 ITI_L_PHYS=1.0 python run_iti_2nd_order_abc.py
```

Environment knobs:
- `P`           — Chebyshev order per axis on each leaf.
- `Q`           — Gauss order on boundary panels.
- `L`           — octree depth ($L = 2$ ⇒ 64 leaves on a cube).
- `ITI_L_PHYS`  — cube half-side; cube spans $[-L_\text{phys}, L_\text{phys}]^3$.

## Files

| file                          | role                                                                   |
|:------------------------------|:-----------------------------------------------------------------------|
| `planewave_setup.py`          | parameters and analytic helpers: $u^*$, $g_1^*$, $g_\text{base}$         |
| `run_iti_1st_order_abc.py`    | 1st-order Sommerfeld, single solve with analytic trace                  |
| `run_iti_2nd_order_abc.py`    | 2nd-order ABC fixed-point verification (solve + one iteration update)   |
