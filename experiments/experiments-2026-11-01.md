# Larger acquisitions with Fourier Born + CG

| Time | Motivation | What Tried / Evidence | Correctness | Timing | Merge Decision |
| --- | --- | --- | --- | --- | --- |
| 07:53 | Fourier receiver compression leaves room for more measurements; testing unused measurements distinguishes fitting from recovery. | Nested 24×256, 64×256, 64×512 acquisitions, three vascular phantoms, exact Born and full-wave data. Larger counts modestly improve held-out prediction but do not recover the fine vessels. | All forward residuals <5e-6 and CG normal-equation residuals <1e-4. Dense-field forward/adjoint references and unused nonlinear predictions checked. | One bounded A100; full experiment 00:45. | Keep the benchmark; no anatomical-recovery claim. |

## Controlled acquisition and inverse

288³ grid, 100 mm box, 1.25 mm wavelength, complex single precision. Tissue is
known exactly and fixed; only vascular seeds 7, 11 and 19 vary. Data are noiseless.
The geometry has 72 inward plane-wave directions and 576 lower-cap receivers
at 100 mm radius. Interleaved positions reserve 8 directions and 64 receivers
before inversion. The fitting subsets are 24×256, 64×256 and 64×512; all
reconstructions are tested on the same unused 8×64 pairs.

These are nested subsets of a **new shared geometry**, not the old standalone
24×256 coordinates. Matching acquisition counts does not make these plane
waves equivalent to a clinical transducer acquisition.

Exact Born data come from uncompressed fields. Receiver-field Fourier mode
selection uses only the 512 fitting receivers, with tapered-field tail 0.003
and taper floor 0.01. The common basis retains 6,216,147 modes. Field generation
uses batches of four; inverse applications use source batches of 16. The
unused receiver fields do not contribute to the basis.

CG uses the same breast support mask and regularization rule throughout:
`alpha = 0.001 * sum(abs(data)**2)`. It tries 8, 16, 32 and 64 steps until the
true regularized normal-equation residual is below 1e-4. The coefficient
changes with data norm; this is a fixed scaling rule, not a tuned acquisition
or regularization optimum.

## Computational capacity

| Fitting acquisition | Stored representation (GB) | Peak live evaluation (GB) | CG steps | Warm CG (s; range over both data types and three phantoms) |
| --- | ---: | ---: | ---: | ---: |
| 24×256 | 17.342 | 30.910 | 8 | 1.508–1.512 |
| 64×256 | 24.986 | 32.440 | 8 | 2.804–2.818 |
| 64×512 | 37.717 | 45.076 | 16 | 9.269–9.283 |

GB is decimal. Storage includes dense source fields, retained receiver
coefficients and indices, not a dense measurement-by-voxel matrix. Peak live
JAX memory includes evaluation workspace and loading peaks. The allocator
pre-reserves a 76.606 GB pool; this is not the live footprint. Held-out
evaluation is a separate process and peaks at 24.843 GB.

The benchmark prepares 72 source and 576 receiver fields in 1,673.52 s,
including dense-reference signals, adjoint/normal checks and disk writes;
coefficient selection/writing takes another 54.73 s. Setup peak live allocation
is 30.672 GB. This is **not** comparable to the old 7.7-minute field-only setup:
it includes more fields and extra reference work. The runner uses disk
memmaps, not a low-disk-storage algorithm.

Full-wave generation uses all 72 directions: 136.13 s for the background and
133.20–134.59 s per vascular phantom. These timings exclude Born preparation.

## Recovery and unused-measurement prediction

Full-wave results below are means over the three phantoms, not confidence
intervals. Image error is `||image - delta|| / ||delta||`; data residuals are
normalized by the background-subtracted vessel signal, not total measurements.
A zero vessel estimate has 100% image error and 100% unused-data residual.

| Fitting acquisition | Vessel-contrast error (%) | Fitted Born-model residual (%) | Unused nonlinear full-wave residual (%) |
| --- | ---: | ---: | ---: |
| 24×256 | 99.928 | 0.00451 | 99.311 |
| 64×256 | 99.816 | 0.01223 | 97.709 |
| 64×512 | 99.633 | 0.02348 | 96.879 |

At 64×512, individual full-wave image errors span 99.631–99.637%, unused
nonlinear residuals 96.425–97.351%, and recovered/true contrast norms
8.650–8.735%. More counts help, but the reconstructions remain very weak and
poorly predict unused measurements. Tiny fitted residuals are not vessel
recovery.

The exact-Born control also fails: image error spans 99.617–99.624% at 64×512,
with unused Born residual 96.350–97.497%. Thus nonlinear model mismatch alone
cannot explain the poor recovery. The full-wave/Born data mismatch is
18.502–19.051% in that acquisition, despite the much smaller fitted inverse
residual. Those are different metrics.

Held-out Born predictions use uncompressed fields. For each full-wave
reconstruction, a fresh nonlinear solve evaluates background + recovered
contrast on the unused pairs, subtracting the same known-background response.
No held-out data select modes, damping or CG iteration count.

## Numerical checks and limitations

For 64×512, relative Fourier errors against uncompressed references are
0.300% for a breast-masked random forward probe, 0.297% for the masked adjoint,
0.425% for the masked normal application, and 0.268–0.269% for vessel signals.
These are sampled errors, not operator-norm guarantees; the taper tolerance
is not an original-operator error bound. The largest real-adjoint identity
discrepancy is 1.87e-5 in complex single precision.

All wave solves pass the 5e-6 residual gate, and all 18 inverses pass 1e-4
normal residual. The small CPU/GPU runs and focused unit tests check acquisition
partitioning, dense contractions, real adjoints, spectral applications and
gradients. Exported fitted and held-out signals reproduce the reported
residuals. This is not an independent PDE discretization-accuracy check.

Noise, estimated background, multifrequency data, physical emitters, learned
corrections and clinical data remain untested. Compression and source-count
scaling are enabling infrastructure, not a solution to fine-vessel recovery.

## Reproduce

```sh
.venv/bin/modal run examples/modal_born_acquisition_3d.py \
  --smoke --out /existing/directory/acquisition-smoke.tar.gz
.venv/bin/modal run examples/modal_born_acquisition_3d.py \
  --budget-seconds 5400 --out /existing/directory/acquisition-288.tar.gz
.venv/bin/pytest tests/test_born_acquisition_3d.py tests/test_spectral_born.py
```

The wrapper runs one A100-80GB, sequential stages, no retries, and a bounded
90-minute work budget. It returns metrics and compact figure arrays. Direct
`prepare`, `evaluate` and `heldout` stages are available in
`examples/born_acquisition_3d.py`. The completed app and all GPU containers
are stopped.
