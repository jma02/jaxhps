# Higher-frequency vascular phantom experiments

`vascular_phantom` adds asymmetry, smooth fat heterogeneity, overlapping
fibroglandular lobules, a skin shell, and curved vascular trees to the
synthetic pendant geometry. The `vascular` and `dense` variants have 62 and
108 branches, and 18 and 36 lobules, respectively. The random seed is fixed
at 7. Vessels start at radius 0.032 and narrow by 0.78 along every branch
over five generations, so the tips have radius 0.0092 (diameters 2.6 mm to
0.74 mm under the 10 cm reading); branch ends stay inside normalized radius
0.78, away from the skin blend. Overlapping compact C4 ellipsoids form the
vessels; products form smooth unions. The contrast
stays in `[-0.047, 0.174]` and vanishes near the computational box faces.

This adds geometric detail, not clinical validation. There is no segmented
anatomy, density variation, attenuation, chest wall outside the breast, or
transducer model. Sharp tissue boundaries are smoothed. Increasing frequency
does not cure inadequate sampling of the narrowest features.

The box is `[-1.25,1.25]^3`. The CLI specifies background wavelengths across
the **box**, `W = kappa * 2.5 / (2*pi)`. The breast occupies only part of the
box. To interpret it as a 10 cm box in water at 1500 m/s, W=10,20,40 means
150,300,600 kHz. Feature sizes scale with this optional physical mapping;
the computation itself is nondimensional.

The solver is the existing complex128 Vico–Greengard–Ferrando FFT volume
integral implementation, with unpreconditioned FGMRES, restart 30, at most
300 Arnoldi steps, and true relative residual tolerance 1e-8. It uses one
plane wave in the +z direction. No repeated solves are timed. Setup includes
phantom sampling, kernel preparation, and synchronization; solve and
receiver evaluation are timed separately with their first JIT compilation.
Process time includes imports and device startup. Memory is the JAX device
allocator peak and process host RSS, not total GPU driver memory.

Refinement compares the scattered fields at the same 512 equal-weight
Fibonacci receivers on radius 2.5. Use `norm(u_fine-u_coarse)/norm(u_fine)`.
Only converged, complete cases qualify. This is a sampled receiver metric,
not a volume error or a fully resolved spherical quadrature. Successive-grid
changes are evidence of convergence, not certified errors against an
independent reference. Never call a small algebraic residual field accuracy.

```sh
# Small CPU check
.venv/bin/python examples/vascular_benchmark_3d.py --waves 2 --n 16 --out local.json
# Short A100 check; wait until its app stops before starting refinement
.venv/bin/modal run examples/modal_vascular_3d.py --suite smoke --budget-seconds 180
# Fresh directory; one remote call, one GPU, sequential fresh child processes
.venv/bin/modal run examples/modal_vascular_3d.py --suite refinement \
  --output data/examples/vascular-refinement --budget-seconds 900 --case-seconds 150
# Optional denser-phantom 80-wavelength check, after reviewing 40-wavelength results
.venv/bin/modal run examples/modal_vascular_3d.py --suite high-frequency \
  --output data/examples/vascular-80 --budget-seconds 300 --case-seconds 120
```

The A100 has 80 GB. Each case is killed at its subprocess time limit; the
remaining suite is bounded by a 900-second work budget and a 1020-second
Modal function timeout. There are no retries, detached runs, deployments,
minimum warm containers, or parallel GPU calls. Results stream back after
each case; partial stage data and timeouts are retained. The idle scaledown
window is two seconds. Check `modal app list --json` after completion; if
interrupted, use `modal app stop APP_ID` for this app.
Budgets apply per invocation. When running several suites, subtract elapsed
GPU work from the total planned allowance before launching another suite.
The 80-wavelength grids give only 2.4–3.2 background points per wavelength;
field refinement must be checked before interpreting them as accurate.

Generate a PDF with geometry, field slices, all measured cases, and refinement
curves, plus a compact CSV/JSON table (no GPU needed):

```sh
.venv/bin/python examples/analyze_vascular_3d.py data/examples/vascular-refinement \
  --out vascular-report
```

## Memory-aware FFT implementation

Following the memory/data-movement emphasis of
[JaxHPS, section 4](https://arxiv.org/abs/2503.17535), the FFT implementation:

- Constructs the unchanged 4n Green quadrature in batches of frequency
  planes, cropping each spatial axis immediately after its inverse FFT.
- Passes the coefficient and convolution kernel as JIT arguments, avoiding
  embedding large device arrays in the compiled operator.
- Reuses the Arnoldi basis when no preconditioner is present, donates basis
  updates, and conjugates vectors instead of the full basis matrix.
- Samples the unchanged phantom ellipsoids on the GPU in bounded point
  tiles, retaining coefficients on device rather than copying them back.

These are implementation changes, not HPS subtree recomputation, adaptive
meshing, or a new FFT quadrature. Setup and each operator application remain
O(N log N); bounded-restart K-iteration solve work remains O(K N log N).

On one A100-SXM4-80GB, three fresh-process trials per implementation gave:

| Metric | Baseline | Optimized |
| --- | ---: | ---: |
| Median setup + first solve | 57.59 s | 12.90 s |
| Range, setup + first solve | 57.55–59.26 s | 12.86–13.03 s |
| Median phantom sampling, including compilation | 21.82 s | 5.15 s |
| Median first solve, including compilation | 34.38 s | 6.42 s |
| Peak live JAX allocation during setup | 76.63 GB | 15.48 GB |
| Peak live JAX allocation through solve | 76.63 GB | 42.23 GB |

The benchmark uses the same dense seed-7 phantom, 288³ grid, 80 wavelengths
across the box (1.25 mm for a 10 cm box), complex128, restart 30, and 1e-8
true residual tolerance. Every solve took 40 iterations; final residuals
were 9.59e-9. Relative differences were 1.92e-16 in the sampled coefficient,
3.21e-16 or less in the volume field and 5.03e-16 or less at 512 receivers.
This checks implementation equivalence,
not grid accuracy or thin-vessel recovery.

Timings include phantom sampling and initial compilation, but exclude
process/device startup and receiver evaluation (1.3–1.4 s).
The improvements are phantom sampling and cold compilation/data handling,
not a faster FFT primitive. Memory is peak *live JAX allocation*, not reserved or
driver memory: both processes used the same 95% preallocation setting.
Trials ran sequentially, alternating order, with the persistent compilation
cache disabled. No warmed repeated-solve timings enter this comparison;
the older 85-second measurement on a different host is not the baseline.

Reproduce with the pre-optimization research branch as the baseline:

```sh
.venv/bin/modal run examples/modal_fft_gpu_benchmark_3d.py \
  --baseline-ref devin/1792399026-phantom-inverse-fit \
  --output data/examples/fft-gpu-comparison
```

This runs six fresh processes sequentially on one GPU, saves per-case
JSON/logs, and stops automatically. Each process is capped at four minutes;
the function is capped at thirty minutes. Baseline FFT/Krylov sources come
from the requested git ref. Baseline sampling uses the CPU evaluator;
optimized sampling uses the device evaluator with shared, unchanged geometry.
