# Experiments — 2026-10-30

| Time (UTC) | Motivation | What Tried / Evidence | Correctness | Timing | Merge Decision |
| --- | --- | --- | --- | --- | --- |
| 04:11 | Test whether the proposed TT field-memory savings preserve the Born response at 288³. | One A100-SXM4-80GB; ranks 32/64/128; four-plane slabs; fresh processes for compressed-only measurements. | All adjoint checks pass. Rank 128 changes the full vascular contrast's Born signal by 0.256% and the dense-field CG image by 0.265%. All images still have about 99.93% contrast error. | Field preparation 455.54 s; rank-128 compression 34.79 s; warm eight-step CG 30.525 s versus 1.531 s uncompressed. | Retain as an opt-in experimental representation, not a default speed optimization or a vessel-recovery result. |
| 04:35 | Determine whether the slowdown is just four-plane slab overhead. | Repeat rank 128 on one A100 with slabs of 4/16/32/72 planes, each in a fresh process. | Signal and image errors unchanged; maximum adjoint identity error 2.14e-6; all normal residuals below 1.88e-6. | Best measured eight-step CG 11.425 s at 32 planes, versus 1.532 s uncompressed; peak live allocation 20.163 GB. | Larger slabs help but do not make this field-storage representation faster; preserve explicit slab-size control. |

## Three-core tensor trains for matrix-free Born fields

### Measured outcome

**Rank 128 realizes the proposed approximately fivefold storage reduction, but
this first slab-wise implementation is about twenty times slower per CG solve.**
It is useful evidence for a memory/accuracy trade-off, not a drop-in performance
win. Ranks 32 and 64 save more storage but introduce much larger signal errors.
The subsequent slab sweep below reduces the slowdown to **7.46 times**, at
higher application memory, without changing field-storage or approximation error.

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
compressed contractions were not tested in the first sweep. Larger-slab
measurements are now reported below; direct compressed contractions remain
untested.

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

### Rank-128 slab-size follow-up

On a second bounded single-A100 run, the dense preparation and rank-128
compression were repeated once. All four compressed evaluations used the same
cores, separate fresh processes, and three warm repetitions:

| Slab planes | Forward (s) | Adjoint (s) | Eight-step CG (s) | Peak live JAX (GB) | Peak pool (GB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1.798 | 1.793 | 30.518 | 12.847 | 18.262 |
| 16 | 0.822 | 0.839 | 14.125 | 15.982 | 18.262 |
| 32 | 0.660 | 0.682 | 11.425 | 20.163 | 35.442 |
| 72 | 0.786 | 0.812 | 13.612 | 30.614 | 52.622 |

The repeated dense-field reference took **1.532 s** for eight CG steps. Slab 32
is fastest among the tested sizes: **2.67 times faster than slab 4, but still
7.46 times slower than uncompressed CG**. Increasing to 72 planes raises both
runtime and memory; this is not a monotone tuning parameter. The default
remains four planes for low memory; callers can select 32 when the additional
workspace is acceptable. Field storage is 10.735 GB in every case.

All slab sizes retain 0.25562% vessel-signal error, 0.26550% CG-image change,
and approximately 99.9307% vessel-contrast error. Adjoint identity errors stay
below 2.14e-6; independently recomputed normal residuals stay below 1.88e-6.
The CPU benchmark also passed a non-dividing seven-plane slab smoke test, and
the 15 focused Born/TT tests passed again after the runner changes.

Reproduce the follow-up with:

```sh
.venv/bin/modal run examples/modal_tensor_train_born_3d.py \
  --ranks 128 --slab-sizes 4,16,32,72 --budget-seconds 2400 \
  --out /existing/directory/new-tt-slabs.json
```

### Why field compression is not automatically operator acceleration

This representation compresses stored fields, not the full Born matrix action.
After expanding a slab, it performs the same source/receiver contraction as the
uncompressed implementation. For equal internal rank `q`, the leading complex
multiply-accumulate counts per forward application are:

- Uncompressed contraction: `S R nx ny nz`.
- Additional TT expansion: `(S + R) nx ny (q² + q nz)`.

At 24 sources, 256 receivers, 288³, and rank 128, expansion adds **8.426 times**
the uncompressed contraction's arithmetic; the total is approximately **9.426
times**, excluding lower-order weighting/reduction work. This is an operation
count, not a runtime prediction. Four-plane slabs also require 72 passes over
the middle cores. Larger slabs reduce repeated core reads and improve matrix
shapes, but do not remove the expansion arithmetic.

A TT-matrix or another directly contracted structured operator could have a
different work count. It has not been implemented or benchmarked here. Applying
the separate field cores directly is not automatically cheaper: source and
receiver rank indices interact with an arbitrary dense contrast volume. No
claim that TT is inherently slower follows from this storage-first prototype.

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
to Helmholtz field preparation. A batched solve-and-compress builder remains
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

## Tucker, CP, and directly contracted receiver blocks

| Time (UTC) | Motivation | What Tried / Evidence | Correctness | Timing | Merge Decision |
| --- | --- | --- | --- | --- | --- |
| 06:43 | Test whether alternative spatial decompositions improve TT's storage/accuracy trade-off, and whether direct contractions avoid its expansion overhead. | Sequential Tucker/HOSVD, complex CP-ALS on Tucker cores, and local receiver SVDs; compare against dense fields and TT at 288³. Tucker 128 stores 4.945 GB with 0.342% whole-vessel signal error. CP 256 stores 0.495 GB but introduces 11.44% signal error. | All real-adjoint checks and 41 focused tests pass. Stabilized overcomplete CP fits and bounded its slab contraction. Every reconstructed image still has about 99.93% vessel-contrast error. | Field preparation ≈00:08; direct-block factorization 00:26. Per-solve timings below. | Retain as opt-in experiments. Tucker is the strongest tested storage/fidelity option; none is a compelling speed replacement for dense fields. No anatomical-recovery claim. |

### Matched problem and application results

Same 288³ grid, 100 mm box, 1.25 mm wavelength, dense vascular phantom seed 7,
24 plane waves, 256 receivers, lower-hemispherical aperture, complex64 fields,
and eight regularized CG iterations as the TT experiment above. Spatial tensor
methods use 32-plane slabs. Times below are **medians of three warm repetitions**,
not the means used in the earlier TT table. All storage is decimal GB and excludes
workspace. The dense reference is the uncompressed matrix-free field cache,
not an explicitly stored 1.17 TB Born matrix.

| Representation | Storage (GB) | Whole-vessel signal error | Random-image signal error | Adjoint application error | Eight-step CG (s) | CG / own dense baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense fields | 53.509 | Reference | Reference | Reference | 1.529–1.643 | 1.00 |
| TT 128 | 10.735 | 0.256% | 0.208% | 0.0754% | 12.263 | 7.46 |
| Tucker 64 | 0.711 | 6.569% | 10.112% | 4.744% | 7.134 | 4.34 |
| Tucker 128 | 4.945 | 0.342% | 0.270% | 0.0989% | 14.678 | 8.93 |
| CP 256, Tucker core 64³ | 0.495 | 11.441% | 19.513% | 10.477% | 16.291 | 10.61 |
| Direct blocks 4³, rank 32 | 35.163 | 1.991% | 1.015% | 2.323% | 1.474 | 0.964 |

TT/Tucker ran on an A100 80GB PCIe (dense median 1.643 s); blocks ran on an
A100-SXM4-80GB (1.529 s); CP ran on another SXM4 allocation (1.535 s).
Ratios use each allocation's own dense baseline. Cross-method wall-clock
comparisons remain approximate, not perfectly matched-device measurements.

Tucker 128 gives **10.8× less stored-field memory**, versus TT 128's 5.0×,
while maintaining sub-percent signal fidelity for these probes. Both still
reconstruct spatial slabs, so neither accelerates CG here. Direct blocks are
only 3.6% faster in this run, while adding 1.99% signal error and expensive setup.
The much smaller CP representation is inaccurate at the tested core/rank/budget;
this does not rule out different CP ranks, cores, initialization, or solvers.

### Field and inverse-problem checks

| Representation | Source-field error, global / worst | Receiver-field error, global / worst | Normal residual | Data residual | Change from dense CG image |
| --- | ---: | ---: | ---: | ---: | ---: |
| TT 128 | 0.0719% / 0.202% | 0.0431% / 0.128% | 1.86e-6 | 4.36e-5 | 0.265% |
| Tucker 64 | 2.949% / 5.272% | 4.053% / 6.514% | 1.71e-6 | 4.42e-5 | 11.533% |
| Tucker 128 | 0.0856% / 0.217% | 0.0647% / 0.153% | 1.84e-6 | 4.36e-5 | 0.336% |
| CP 256 | 5.766% / 7.290% | 9.339% / 11.910% | 1.80e-6 | 4.59e-5 | 21.093% |
| Direct blocks 4³, rank 32 | Exact | Not measured globally | 2.25e-6 | 4.36e-5 | 1.092% |

The maximum relative real-adjoint identity error is 1.58e-6. Application errors
above compare against the dense operator; adjoint identities instead test each
approximate operator against its own adjoint. Residuals are for the corresponding
compressed normal equation and data fit. **All contrast-image errors remain
99.928–99.931% relative to the vessel ground truth.** A small residual or small
change from the dense image therefore does not establish vessel recovery.

### Setup and memory

| Representation | Compression (s) | Compression peak live (GB) | Evaluation peak live (GB) | Evaluation peak pool (GB) |
| --- | ---: | ---: | ---: | ---: |
| TT 128 | 34.00 | 74.267 | 20.163 | 35.442 |
| Tucker 64 | 38.25 | 55.042 | 9.027 | 11.023 |
| Tucker 128 | 43.61 | 63.034 | 14.427 | 35.467 |
| CP 256 | 114.88 | 55.047 | 12.572 | 19.394 |
| Direct blocks 4³, rank 32 | 1536.64 | 76.441 | 43.003 | 77.309 |

Dense field preparation took 474.63 s for TT/Tucker, 455.57 s for blocks, and
456.24 s for CP, with a 63.231 GB live-allocation peak and 80.862 GB reserved
pool. Compression starts from that dense cache; **none of these builders removes
its memory requirement or preparation cost**. Compression peaks are sampled
before subsequent field-error validation, not peaks of the entire workflow.
The block timing includes streamed factor writes; other compression timings
exclude serialization and field-error checks. Evaluation uses fresh processes
without dense fields, and includes comparison fixtures and CG workspace.
These are JAX allocator high-water marks, not total driver usage or minimum GPU
capacity. In particular, 4.945 GB of Tucker factors does not mean a 5 GB GPU can
run this pipeline.

### What is compressed

- **Tucker:** sequential spatial HOSVD per stored source/receiver field, retaining
  three factor matrices and a core. This is field-cache compression, not a
  factorization of the full source–receiver Born matrix.
- **CP:** rank-256 complex CP-ALS on a rank-64 Tucker core, with 50 sweeps and two
  deterministic restarts. Trace-scaled ridge regularization and selection of the
  best finite iterate prevent unstable overcomplete fits. A complex64 rank-64
  fit to a separable 8×7×6 field produced NaNs before this fix and relative error
  8.84e-7 afterward. Earlier unstable CP measurements are superseded, not evidence
  about attainable CP accuracy. Explicit slab-local contractions also resolve
  the full-size validation OOM, which requested a 59.2 GB temporary allocation.
- **Direct blocks:** approximate reciprocal fields within each spatial block as
  `W[r,v] ≈ sum_a U[r,b,a] V[b,a,v]`, while retaining source samples exactly.
  Apply `d[s,r] = sum_(b,a) U[r,b,a] sum_(v in b) S[s,v] delta[v] V[b,a,v]`.
  No full receiver field is reconstructed during application. This is a directly
  contracted local receiver-mode factorization, not a full Born-matrix SVD.
  Compression streams factors to disk and keeps receiver reshaping inside JIT
  to avoid an extra 48.9 GB dense-cache copy.

At full resolution, three sampled **8³** blocks give these numerical ranks for
relative Frobenius tails (not worst-case operator bounds):

| Tail tolerance | Receiver block, 256×512 | Full source–receiver Born block, 6144×512 |
| --- | ---: | ---: |
| 10% | 50–56 | 220–238 |
| 1% | 85–93 | 340–355 |
| 0.1% | 118–128 | 442–445 |

The full local Born blocks are much less compressible than receiver blocks.
These three samples do not establish ranks across the whole volume, nor rule
out phase-aware hierarchical methods. A larger 6³/rank-48 block case was stopped
before completion after the 4³ result showed limited benefit and high setup
cost; no full-volume result is claimed for it.

No hierarchical Tucker, QTT, tensor-butterfly, phase-compensated interpolation,
FMM, or MLFMA implementation is included. Published Green-function and
translation-tensor compression results concern different operators and are not
evidence that this heterogeneous breast Born map has the same ranks or speedups.
There is no learned-recovery, training-memory, clinical-count, or multi-frequency
evaluation here. The next speed question is whether phase-aware/hierarchical
structure permits fast direct application, rather than further slab tuning.

### Reproduce the comparison

The default bounded run includes Tucker 128, direct blocks 4³/rank 32, and CP 256:

```sh
.venv/bin/modal run examples/modal_compression_benchmark_3d.py \
  --out /existing/directory/new-comparison.json
```

To include the additional TT/Tucker reference points:

```sh
.venv/bin/modal run examples/modal_compression_benchmark_3d.py \
  --cases tt-128,tucker-64,tucker-128,block-4-32,cp-256 \
  --out /existing/directory/new-expanded-comparison.json
```

Each allocation is capped at one A100-80GB, no retries, 5,400 seconds of subprocess
work, and fresh evaluation processes. A small interface smoke test can use
`--n 48 --wavelength-mm 7.5 --cases tucker-16,cp-32,block-4-16`.
Stages `prepare`, `compress`, `evaluate`, and `screen` are also available through
`examples/compression_benchmark_3d.py`; use a new process for each evaluation.
No neural network is trained.

Forty-one focused tests pass across compressed Born operators, TT, inverse/FFT,
and phantom sampling. Coverage includes complex64/128, rectangular grids,
non-dividing slabs/blocks, real adjoints, truncated operators, gradients through
applications and CG, complex known-rank CP fits, overcomplete CP stability, and
slab contraction with rank larger than spatial modes. Ruff, formatting, syntax
compilation, and pre-commit pass. No static type-checker is configured.

## Batched setup and directly contracted spectral receiver fields

**Best tested choices:** batched Tucker compression for compact storage; direct Fourier
contraction for faster breast-masked solves. The final Fourier batch sweep
reaches **1.095 s versus 1.529 s dense CG** (1.40× speedup), with **17.312 GB
stored versus 53.509 GB** and **0.256% whole-vessel Born-signal error**. This is
not uniform full-grid accuracy, and it does not improve vessel recovery.

| Time (UTC) | Motivation | What Tried / Evidence | Correctness | Timing | Merge Decision |
| --- | --- | --- | --- | --- | --- |
| 08:16 | Avoid the dense GPU field cache during setup and repeated receiver-field expansion during CG. | Solve one background field at a time; factorize each complete field independently with Tucker and collect Fourier coefficients in batches. Compare direct spectral applications against dense fields on the same A100-SXM4-80GB. | Setup peaks at 14.142 GB live GPU allocation. Tucker preserves its previous field/signal accuracy. Fourier truncation has much larger errors outside the reconstruction region than inside it. | Combined batched field generation, both representations, checks, and disk writes: 688.90 s. Four-source Fourier batches: 1.461 s at 1.424% vessel-signal error or 2.191 s at 0.256%; dense: 1.528 s. | Retain batched Tucker compression as the compact, accurate option; retain Fourier as an experimental speed/memory trade-off, not a uniformly accurate full-grid operator. |
| 08:30 | Reuse receiver coefficients across more sources per contraction, and check errors on the actual masked inverse problem. | Repeat the 0.3% tapered-field tail with source batches 4/8/12/24 and add masked-adjoint and masked-normal probes. Same A100-SXM4-80GB model, fresh dense reference. | Vessel/random masked signal errors 0.256%/0.299%; masked adjoint 0.297%; masked normal application 0.431%. Full-grid errors remain about 4.95%. | Best eight-step CG 1.095 s, versus 1.529 s dense, at 27.632 GB peak live allocation. | Prefer larger source batches when workspace permits; preserve the explicit mask-dependent accuracy limitation. |

### What changed

`make_background_field_solver` reuses the background solve for bounded batches.
The benchmark compresses each batch before generating the next. Tucker no
longer needs the complete GPU cache before compression; its application still
uses the existing slab reconstruction, so this change does not accelerate it.
Each complete field is factorized independently. Batching limits how many fields
are held in memory; it does not update a shared Tucker factorization as new data
arrive.

`SpectralBorn` instead uses a **shared spatial basis for all receivers**. It
transforms each source-weighted image, gathers retained modes, and contracts
with receiver coefficients. The adjoint scatters coefficients into a spectrum,
transforms back, and sums source contributions. It does not reconstruct a
receiver field during application. CG itself is unchanged.

Two bases were screened: orthonormal DCT and complex Fourier. For Fourier,
multiply receiver fields by a positive sine-squared product taper with floor
0.01, and divide source fields by the same taper. This is an exact rearrangement
before truncation. Select the smallest common mode set meeting an aggregate
**tapered-receiver** Frobenius-tail target. Dividing by the taper can amplify
truncation errors near the box boundary: this target is **not** a bound on the
original Born operator, nor on every image supported inside the breast.

At 48³, 4 sources, 24 receivers, and 10 mm wavelength, a 1% DCT tail stored
9.180 MB and gave 0.643% vessel-signal error with 8.802 ms CG. Fourier at the
same tail target stored 6.212 MB, gave 0.477% vessel-signal error, and took
3.046 ms (dense: 1.930 ms). Fourier's full-grid random-image error was 14.11%,
versus 0.897% for DCT. This motivated testing Fourier at full resolution with
both breast-masked and full-grid probes, rather than interpreting its spectral
tail as a global operator guarantee. DCT was not benchmarked at 288³.

### Matched 288³ comparison

100 mm box, 1.25 mm wavelength, 24 plane waves, 256 receivers, complex64.
Each application evaluation is a fresh process; times are medians of three
warm eight-step solves, using the same data, mask, and regularization 0.001.
GB denotes decimal bytes. The Fourier rows below use four-source batches.

| Representation | Storage (GB) | Vessel-signal error | Breast-masked random-image error | Full-grid random-image error | Eight-step CG (s) | Image change from dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense fields | 53.509 | Reference | Reference | Reference | 1.528 | Reference |
| Batched Tucker 128 | 4.945 | 0.342% | 0.270% | 0.099% | 13.803 | 0.336% |
| Fourier, 1% tapered-field tail | 11.001 | 1.424% | 1.268% | 8.221% | 1.461 | 1.384% |
| Fourier, 0.3% tapered-field tail | 17.312 | 0.256% | 0.299% | 4.942% | 2.191 | 0.341% |

The Fourier masks contain 3,125,834 and 6,201,432 modes, respectively (13.085%
and 25.961% of the spatial spectrum). Dense source fields remain uncompressed.
Two-source batches took 2.076 s and 3.473 s: larger batches avoid repeatedly
reading the same receiver coefficients. The loose setting is essentially at
dense speed, not a compelling measured speedup, and has greater signal error.

| Representation | Peak live evaluation (GB) | Allocator high-water mark (GB) | Full-grid adjoint-application error |
| --- | ---: | ---: | ---: |
| Dense fields | 59.338 | 80.862 | Reference |
| Batched Tucker 128 | 14.523 | 35.467 | 0.099% |
| Fourier, 1% tail, batch 4 | 13.772 | 17.180 | 8.203% |
| Fourier, 0.3% tail, batch 4 | 20.083 | 25.770 | 4.950% |

Full-grid adjoint errors include locations that the reconstruction mask removes.
They must not be described as sub-percent global operator accuracy. Real-adjoint
identity errors are at most 1.70e-6. Normal-equation residuals are at most
2.46e-6; data residuals are approximately 4.36e-5 **against each method's own
operator**. These residuals do not certify the compressed solution against the
original operator. Every image still has approximately **99.93% vessel-contrast
error**; this is a representation/capacity experiment, not vessel recovery.

### Setup memory and cost

The entire batched preparation, including phantom sampling and full-wave data
generation, peaked at **14.142 GB live JAX memory**, with a **30.065 GB allocator
high-water mark**. Together with the Tucker application measurement, this
demonstrates a compressed setup-to-solve path without the former 53.5 GB GPU
field cache. The dense evaluation above is a separate reference measurement,
not a prerequisite for compressed application.

All preparation/compressed processes were capped at 45% of the A100's available
GPU memory (38.303 GB allocator limit). The dense comparison alone used a 95%
preallocated pool; its reserved-memory number is therefore not its minimum
requirement. CUDA/context allocations are outside JAX live-memory accounting.
No smaller physical GPU or larger source/receiver configuration was tested.

The 688.90 s field-preparation interval includes both Tucker and Fourier builds,
field-error checks, transfers, and disk output, and is not a standalone Tucker
or Fourier setup time. Tucker compression/checks/writes consumed 52.17 s;
receiver FFT/energy accumulation consumed 1.08 s. Host-side Fourier mode
selection and compact-coefficient writes then took 21.95 s or 38.10 s.
Generating full-wave background/truth data took a separate 46.35/45.23 s.
All background-field residuals were below 1.72e-6.

This benchmark retains **dense fields and full receiver spectra on disk** for
the matched reference and mode selection (over 100 GB of intermediate files).
It does not establish low total host/disk usage. Tucker factors can be produced
one batch at a time; the current Fourier mode selection is a two-pass disk
procedure. Neither method eliminates the physical background solves.

Tucker source-field global/worst errors were 0.0856%/0.217%; receiver-field
errors were 0.0647%/0.153%. These match the preceding dense-cache construction.
Its larger allocation-pool figure is workspace/allocator overhead, not stored
factor size.

### Source-batch tuning and masked-operator checks

The second full-resolution allocation used the same geometry, data, 6,201,432
retained modes, and taper. It omitted Tucker compression during preparation and
added two diagnostics: `P B*` on a random data probe and `P B* B P` on a random
contrast, where `P` is the fixed breast reconstruction mask. These are sampled
application errors, not operator-norm estimates. The additional stored normal
probe adds approximately 0.096 GB to evaluation memory relative to the first run.

| Source batch | Warm eight-step CG (s) | Time / same-run dense | Peak live evaluation (GB) | Allocator high-water mark (GB) |
| --- | ---: | ---: | ---: | ---: |
| Dense reference | 1.529 | 1.000 | 59.433 | 80.862 |
| 4 | 2.182 | 1.427 | 20.179 | 25.770 |
| 8 | 1.421 | 0.930 | 21.708 | 25.770 |
| 12 | 1.198 | 0.783 | 23.236 | 38.303 |
| 24 | 1.095 | 0.716 | 27.632 | 38.303 |

For every batch, masked-adjoint error is **0.297%**, masked-normal application
error **0.431%**, and image change from dense approximately **0.341%**. The
full-grid random-image/adjoint errors remain **4.942%/4.950%**. Thus the speed
gain does not require relaxing the retained-mode tolerance, but the accuracy
case is specifically for the existing breast-masked problem. This is not a
drop-in guarantee for other contrast supports, phantoms, or frequencies.

The batch-24 real-adjoint identity error is 7.98e-6 and its normal-equation
residual 4.11e-6; both pass the existing numerical gates. All images retain
approximately 99.93% vessel-contrast error. No improved anatomical recovery or
network-training memory claim follows from these timings.

Batched field preparation without Tucker took 624.61 s, including dense and
spectral disk writes; compact coefficient selection/writes took 50.61 s. Setup
again peaked at 14.142 GB live / 30.065 GB allocated-pool memory. These timings
include different disk work from the earlier dense-cache-only setup and should
not be interpreted as eliminating setup cost.

### Reproduction and validation

```sh
.venv/bin/modal run examples/modal_spectral_born_3d.py \
  --basis fft --tolerances 0.01,0.003 --batches 2,4 --tucker-rank 128 \
  --out /existing/directory/new-spectral-comparison.json

.venv/bin/modal run examples/modal_spectral_born_3d.py \
  --basis fft --tolerances 0.003 --batches 4,8,12,24 \
  --out /existing/directory/new-spectral-batches.json
```

The runner uses one A100-80GB, sequential cases, no retries, and a 5,400-second
work budget. For a small test use `--n 48 --sources 4 --receivers 24
--wavelength-mm 10 --tucker-rank 24`. Standalone stages are available through
`examples/spectral_benchmark_3d.py`. No network is trained.

Sixty-seven focused CPU tests pass, including both bases, complex64/128,
rectangular grids, non-dividing source batches, real adjoints, gradients through
applications and CG, minimal mode selection, and batched/full-cache solver
equivalence. The exact tail-threshold/zero-mode cases additionally pass after
tightening mode selection. Small GPU comparisons and the full-resolution rows
above pass their numerical checks. DCT explicitly transforms real and imaginary
parts for compatibility with the GPU image's JAX 0.6.2. Ruff, formatting, syntax
compilation, and pre-commit pass; no static type-checker is configured.
