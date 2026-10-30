# Experiments — 2026-10-30

| Time (UTC) | Motivation | What Tried / Evidence | Correctness | Timing | Merge Decision |
| --- | --- | --- | --- | --- | --- |
| 04:11 | Test whether the proposed TT field-memory savings preserve the Born response at 288³. | One A100-SXM4-80GB; ranks 32/64/128; four-plane slabs; fresh processes for compressed-only measurements. | All adjoint checks pass. Rank 128 changes the full vascular contrast's Born signal by 0.256% and the dense-field CG image by 0.265%. All images still have about 99.93% contrast error. | Field preparation 455.54 s; rank-128 compression 34.79 s; warm eight-step CG 30.525 s versus 1.531 s uncompressed. | Retain as an opt-in experimental representation, not a default speed optimization or a vessel-recovery result. |

## Three-core tensor trains for matrix-free Born fields

### Measured outcome

**Rank 128 realizes the proposed approximately fivefold storage reduction, but
this first slab-wise implementation is about twenty times slower per CG solve.**
It is useful evidence for a memory/accuracy trade-off, not a drop-in performance
win. Ranks 32 and 64 save more storage but introduce much larger signal errors.

| Representation | Stored fields (GB) | Compression | Whole-vessel Born-signal error | Change in CG image | Warm eight-step CG (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncompressed | 53.509 | 1× | Reference | Reference | 1.531 |
| TT rank 32 | 0.702 | 76.2× | 11.166% | 27.123% | 5.059 |
| TT rank 64 | 2.725 | 19.6× | 5.547% | 8.963% | 10.782 |
| TT rank 128 | 10.735 | 4.98× | 0.256% | 0.265% | 30.525 |

Timings are means of three warm repetitions, on the same A100-SXM4-80GB with
JAX 0.6.2. GB means decimal bytes. Rank 128's forward/adjoint applications take
1.799/1.793 s individually. First eight-step CG calls, including compilation,
take 5.805/11.531/31.252 s at ranks 32/64/128. The four-plane slab size is a
conservative memory choice, **not a tuned optimum**. Larger slabs or direct
compressed contractions are potential follow-up optimizations, not measured
speedups.

| Rank | Compressed-only peak live JAX (GB) | Growing pool peak (GB) | TT-SVD setup (s) | Source field error | Receiver field error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 2.674 | 4.431 | 30.017 | 7.181% | 26.536% |
| 64 | 4.811 | 9.131 | 31.624 | 2.278% | 3.018% |
| 128 | 12.847 | 18.262 | 34.789 | 0.0719% | 0.0431% |

The field errors above are global relative Frobenius errors. Worst individual
source errors are 9.365%/5.248%/0.202%; worst receiver errors are
52.213%/5.077%/0.128%. Random-contrast forward errors are
25.594%/7.708%/0.208%, and random-data adjoint errors are
27.069%/3.600%/0.0754%. These differ from the actual vessel-signal errors;
field compression error alone is not an adequate accuracy test.

All real-adjoint identity errors are below 1.81e-6. Independently recomputed
normal residuals are 1.85e-6/1.76e-6/1.85e-6. All three satisfy the original
1e-4 convergence target after eight steps. Nevertheless, vessel-contrast
relative image errors are 99.9267%/99.9307%/99.9307%, compared with 99.9307%
uncompressed. **None demonstrates vessel recovery.** Rank 32's marginally lower
image error is not evidence of better recovery when it changes the reference
Born signal and CG image substantially.

Uncompressed preparation takes 455.54 s and peaks at 63.23 GB live JAX
allocation. The combined dense-cache-plus-compression process instead peaks
at **74.56 GB** by rank 128. Thus the 12.85 GB compressed-only application peak
does **not** mean this implementation can prepare the operator on a 16 GB GPU.
TT-SVD timing excludes field preparation, error evaluation, and writing cores;
rank-128 transfer/write adds 9.19 s. The complete sparse experiment still fits
one A100, but clinical-scale preparation is not demonstrated.

### Question and method

Can compression reduce the source/receiver field cache without materially
changing the vessel-induced measurements or the regularized Born image?
This is an accuracy/memory/runtime trade-off, not a proposed remedy for the
underdetermined inverse problem or the Born model's multiple-scattering error.

The target experiment matches the [uncompressed benchmark](experiments-2026-10-23.md):
288³, 100 mm box, 1.25 mm wavelength, 24 plane-wave sources, 256 receivers on the
lower hemispherical cap, complex64, and dense vascular phantom seed 7 with its
exact vessel-free background. Data are noiseless full-wave background-subtracted
measurements. The regularization is `0.001 * sum(abs(data)**2)` and the support
window is the breast, not a vessel mask. The dense baseline is rerun on the same
GPU allocation rather than using the earlier device's timing.

Each complex field is independently factorized by deterministic TT-SVD into
`(nx, r1)`, `(r1, ny, r2)`, and `(r2, nz)` cores. Both ranks are capped at a
prescribed value; no tolerance or optimal rank is inferred from the data.
The forward and real-adjoint actions reconstruct four x-planes at a time and
contract the slab without forming either all full fields or a dense Born matrix.
The existing CG algorithm is unchanged. Cores are dynamic JAX arguments, and
checkpointed slab loops support differentiation with respect to contrast/data.

### Interpretation of measurements

- Field errors are relative Frobenius errors, reported globally and for the
  worst individual source/receiver field.
- Vessel-signal error is `norm(B_TT delta - B delta) / norm(B delta)` for the
  actual full vascular contrast. This is not a bound over arbitrary images,
  a finest-branch-only check, or a worst-case operator norm.
- A breast-masked random contrast and a random complex measurement probe check
  another forward action and the real adjoint against uncompressed results.
- Image change is relative to the dense-field eight-step CG image. Vessel-image
  error is instead relative to the ground-truth contrast. Preserving a poor
  Born image does not establish anatomical recovery.
- The compressed normal-equation residual is recomputed independently after
  eight steps. Small residual only means the compressed inverse problem is solved.

Compression currently starts from the **entire dense cache**. It therefore does
not reduce preparation's memory requirement, and its factorization cost is added
to Helmholtz field preparation. A streaming solve-and-compress builder remains
necessary to scale setup beyond the dense cache's capacity.

Each compressed evaluation runs in a fresh Python process after the preparation
process exits. Its live-JAX high-water mark includes the TT cores, workspace,
CG vectors, and retained comparison fixtures, but no dense field cache. It is not
total driver memory or the minimum required GPU capacity. Preparation uses a
95% preallocated pool; evaluations use a growing pool. Their pool high-water
marks and live allocations must not be conflated.

### Reproduce

```sh
.venv/bin/modal run examples/modal_tensor_train_born_3d.py \
  --out /existing/directory/new-tt-result.json
```

One A100-80GB container, no retries, no deployment, at most 5,400 seconds of
subprocess work. Ranks default to 32, 64, and 128. No neural model is trained.
For a small end-to-end smoke case, add `--n 32 --sources 2 --receivers 8
--wavelength-mm 10 --ranks 4,8,16 --budget-seconds 240`.

Local CPU reproduction (each evaluation must use a separate process):

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/tensor_train_benchmark_3d.py \
  prepare --directory /existing/directory/tt-smoke --n 16 --sources 2 \
  --receivers 8 --wavelength-mm 20 --ranks 2 8 16
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/tensor_train_benchmark_3d.py \
  evaluate --directory /existing/directory/tt-smoke --rank 8
```

### Validation

Twenty-eight focused TT/inverse/FFT/sampling tests pass. These cover exact-rank
complex fields, rank-one and zero fields, rectangular grids, partial final slabs,
slabs larger than the volume, both precision modes, truncated-field agreement,
real adjoints, gradients through operator applications and CG, and equivalence
to the existing dense-field CG implementation. The 32³ A100 end-to-end smoke
passes at all three ranks; its accuracy is not target-resolution evidence.
Ruff, formatting, syntax compilation, and pre-commit pass; the project has no
configured static type-checker command.
