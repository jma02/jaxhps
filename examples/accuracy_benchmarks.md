# Field accuracy and matched-error benchmarks

## Scope

These drivers compare two discretizations of
`Delta u + kappa^2 (1-b) u = 0` with outgoing radiation and one unit plane
wave in the +z direction. The common box is `[-1.25,1.25]^3`.

- **HPS+BIE:** unmerged leaf ItI maps, dense exterior S/D quadrature,
  true-residual FGMRES, Jacobi or paired-face/coarse shifted preconditioning.
- **FFT volume integration:** a new JAX implementation of the
  Vico–Greengard–Ferrando truncated Green method
  ([JCP 323, 2016](https://doi.org/10.1016/j.jcp.2016.07.028)). This is an
  independent discretization, not a tuned external solver package. With
  `G=exp(ikr)/(4*pi*r)`, solve `(I+kappa^2 G b)u=u_inc` by FGMRES.
  The kernel is prepared on `(4n)^3` points, cropped to convolution weights
  on `(2n)^3`, then transformed for reuse. No absorbing layer is needed.

The FFT method requires the contrast/density to be resolved on a uniform
grid and vanish near its boundary. That is true geometrically for these
fixtures, but resolution must be checked. Neither solver is validated here
for discontinuous media or production ultrasound frequencies.

## Coefficients

The radial control is `b(r)=-0.4*(1-(r/0.6)^2)^4` for `r<0.6`, zero otherwise.
It is C³ at its support boundary. The partial-wave reference factors out
`r^ell` from each radial solution to avoid tiny initial amplitudes. Its ODE
tolerance, step size, starting radius, and multipole truncation must be
refined independently before assigning a reference uncertainty.

The synthetic tissue phantom is defined only in `lucka_phantom_3d.py` and
used by both discretizations and the historical Modal driver. The pendant
geometry has `R=0.8a`, `z0=0.55a`, skin, fat, a fibroglandular core, and three
vessels. Tissue values remain `(-0.041,0.020,0.103,0.174)`.

The transition is the regularized incomplete beta function `I_t(5,5)`, or
`t^5*(126-420t+540t^2-315t^3+70t^4)` on `[0,1]`. Its first four derivatives
vanish at both ends. Compact bumps use `max(1-t^2,0)^5`. Tissue overlays are
convex blends; vessel unions use `1-product(1-v_i)`, not `max` or thresholds.
All distance singularities occur in constant regions or polynomial functions
of squared distance. Products and compositions therefore preserve C⁴
regularity. This does not assert C-infinity smoothness or spectral convergence.
Thin vessels and transition bands still demand spatial refinement.

## Error and cost definitions

The error is a weighted relative discrete L² norm of the **scattered field**
on a sphere of radius 2.5: 12 Gauss–Legendre polar-cosine nodes times 24 uniform
azimuths. Weights are `w_theta * 2*pi/24`. It is an exterior receiver error,
not a volume error. Use the same targets/weights for every comparison.

For radial cases, the reference is the partial-wave field. For the phantom,
successive fine FFT grids estimate reference uncertainty; cross-method HPS
refinement supplies an independent check. An unverified finest grid is not
an exact solution. Do not claim a matched threshold tighter than the evidence
supports. Reject any solver configuration whose true residual exceeds `1e-8`.

Every case runs in a fresh child process on one H100. Timers synchronize
device results. JSON records:

- Setup and first solve, including their JIT effects; their sum is the cold
  measured solver cost. Python imports, process/container startup, and fixture
  loading are excluded and identified separately.
- Three zero-initial-guess repeated solves using the **same** setup and RHS
  after a converged first solve. A failed first solve is retained without
  spending more GPU time on ineligible repeats. Repeats measure repeated-solve
  cost, not varied-source or block throughput.
- Exterior target evaluation, timed separately, including its first compile.
- HPS exterior fixture generation is a separate CPU preprocessing cost,
  excluded from solver times.
- JAX live allocator peak and pool peak, plus host peak RSS. These are
  whole-case measurements through field evaluation; factor sizes are separate.
  They do not include every CUDA driver allocation or measure `nvidia-smi` peak.
- Raw fields, target coordinates/weights, true residual histories, iterations,
  versions, and GPU identity.

For matched-error tables, among converged tested configurations satisfying
the chosen error ceiling, select the least expensive by the stated cost
metric. Report cold and repeated costs separately; never compare equal node
counts and call that matched accuracy. Small frequency sweeps do not establish
asymptotic complexity or production feasibility.

## Reproduction

Use the repo environment and its pre-commit hooks. `modal_accuracy_3d.py` pins
the GPU runtime to JAX 0.6.2, NumPy 2.2.6, SciPy 1.15.3, Python 3.12, and four
host CPUs. It caps the app at one container and calls cases sequentially.

Generate exterior fixtures with `gen_SD_3D.py` in the fmm3dbie environment:

```sh
PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  /path/to/fmm-env/bin/python examples/gen_SD_3D.py \
  --a 1.25 --kappa 4 --q 8 --L 2 \
  --out data/examples/SD_3D/SD_k4_q8_L2_a1.25.npz
```

The radial suite needs `(kappa,q,L)` in `{4,8,12} ×
{(8,1),(10,1),(12,1),(6,2),(8,2)}`. The phantom suite also needs
`{4,8} × {(4,2),(6,1)}`. The smoke suite additionally needs `(4,4,1)`.
`PYTHONNOUSERSITE=1` prevents a user NumPy 2 installation from overriding
the NumPy 1 runtime used to compile the legacy Fortran extension.

```sh
.venv/bin/modal run examples/modal_accuracy_3d.py --suite smoke
.venv/bin/modal run examples/modal_accuracy_3d.py --suite radial
.venv/bin/modal run examples/modal_accuracy_3d.py --suite phantom
.venv/bin/modal run examples/modal_accuracy_3d.py --suite phantom-p
.venv/bin/python examples/reference_convergence_3d.py --out reference-controls.json
.venv/bin/python examples/analyze_accuracy_3d.py data/examples/accuracy \
  --reference-controls reference-controls.json --out accuracy-summary
# One case; use a separate output directory to avoid replacing prior files:
.venv/bin/modal run examples/modal_accuracy_3d.py --suite extra \
  --output data/examples/accuracy-extra \
  --case '--solver hps --kind phantom --kappa 4 --q 8 --L 2'
.venv/bin/python -m pytest tests/test_lucka_phantom_3d.py \
  tests/test_fft_volume_3d.py tests/test_scattering_reuse_3d.py -q
.venv/bin/ruff check
.venv/bin/ruff format --check .
```

The same per-case CLI runs on CPU for validation:
`python examples/accuracy_benchmark_3d.py --solver fft --n 32 --out case.json`.
GPU timings must not be inferred from CPU timings. These experiments use
plane waves, not point-source illumination or a receiver/source array model.

The analyzer requires independently measured radial reference controls.
It uses their largest new-reference change (excluding the legacy reference)
and the finest phantom FFT-grid change as uncertainty indicators. Missing
controls never count as zero uncertainty. A case is eligible only if all
four solves pass the true residual check, the reference change is at most
one tenth of the error ceiling, and field error plus that change stays below
the ceiling. These empirical changes are not rigorous error bounds.

The `phantom-p` suite refines interior order independently of exterior order:
`q=8, L=2, p=16,20`, at both phantom frequencies. It uses four-leaf batches
for the physical and shifted local factorizations. This bounds temporary
leaf-factorization storage without changing the discrete equations or the
retained dense exterior matrices. The default driver retains its original
unbatched behavior. Use `--start INDEX` to resume a suite after completed
cases; ensure the earlier Modal app has stopped before restarting it.

## Archived study

The [manuscript and data archive](https://github.com/jma02/jaxhps-devin-latex/pull/2)
contain all 77 tested configurations, including unconverged solves and process
timeouts, the reference controls, and generated comparisons. Matched-error
conclusions apply to these tested configurations and receiver fields; they do not
establish clinical-frequency accuracy or asymptotic scaling.
