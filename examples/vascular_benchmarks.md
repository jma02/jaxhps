# Higher-frequency vascular phantom experiments

`vascular_phantom` adds asymmetry, smooth fat heterogeneity, overlapping
fibroglandular lobules, a skin shell, and curved binary vascular trees to the
synthetic pendant geometry. The `vascular` and `dense` variants have 45 and
75 branches, and 18 and 36 lobules, respectively. The random seed is fixed at
7. Branch radii taper across four generations. Overlapping compact C4
ellipsoids form the vessels; products form smooth unions. The contrast
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
CPU phantom sampling, kernel preparation, and synchronization; solve and
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
```

The A100 has 80 GB. Each case is killed at its subprocess time limit; the
remaining suite is bounded by a 900-second work budget and a 1020-second
Modal function timeout. There are no retries, detached runs, deployments,
minimum warm containers, or parallel GPU calls. Results stream back after
each case; partial stage data and timeouts are retained. The idle scaledown
window is two seconds. Check `modal app list --json` after completion; if
interrupted, use `modal app stop APP_ID` for this app.

Generate a PDF with geometry, field slices, all measured cases, and refinement
curves, plus a compact CSV/JSON table (no GPU needed):

```sh
.venv/bin/python examples/analyze_vascular_3d.py data/examples/vascular-refinement \
  --out vascular-report
```
