# Experiments — 2026-10-22

| Time | Motivation | What Tried / Evidence | Correctness | Timing | Merge Decision |
| --- | --- | --- | --- | --- | --- |
| 02:46 | Establish the complete larger-grid pilot cost before considering 20,000 cases. | Original sphere pipeline, 128³, 5 mm, 24 sources, 256 receivers, eight training/two test cases, one fresh epoch. JAX reported 9.40 GB of captured constants during Born lowering. | Forward residuals below 1e-6. One epoch is a smoke test, not scientific reconstruction evidence: test contrast error was 91% Born and 182% corrected. | Data preparation 00:04 (233.4 s); first Born call 00:02 (143.57 s); epoch completed at 247.8 s. | Use only to identify bottlenecks; do not update proposal results. |
| 02:51 | Avoid compilation constants and repeated source/receiver geometry work. | Explicit Born pytree arguments, compiled CG loop, and receiver batches shared across sources. Same-A100 fresh-process comparison gave 1.480 → 1.261 s per warm case. | Measurement and Born-image comparisons passed; true residuals unchanged. | Cold preparation plus first case 00:02 → 00:00 (120.64 → 23.56 s). | Refine: runtime conjugation of large Born fields increased warm CG cost. |
| 02:56 | The real adjoint can conjugate the small data array instead of cached fields. | Rearranged `Re(conj(S) * (D @ conj(R)))` as `Re(S * (conj(D) @ R))`. Repeated the matched baseline/optimized comparison; details below. | 61 targeted tests passed; eight inverse tests re-run after the final adjoint change. GPU equivalence checks passed. | Cold preparation plus first case 00:02 → 00:00 (113.46 → 22.56 s); warm mean 1.472 → 1.196 s. | Retain in PR #12, not merged. No large dataset launched. |
| 03:04 | Batching independent FFT source solves might reduce launch overhead and improve GPU utilization. | Tested two-source `lax.map` batches against the current optimized pipeline, with identical ten cases and compiled Born CG in both fresh processes. Warm forward/Born mean increased 1.193 → 1.309 s, about 9.7% slower. | Maximum total-measurement relative difference 5.61e-3 exceeded the 2e-5 gate, despite algebraic residuals below 7e-7. Born-image comparison was not reached. | Preparation plus first case 00:00 → 00:00 (24.22 → 26.38 s). | Reject. No batched-source change retained; do not weaken the comparison gate. |
| 03:06 | Fixed acquisition geometry might be reusable across thousands of phantoms rather than recomputed each time. | Cached all incident plane waves and receiver Green rows, then used a single highest-precision receiver matrix product. Warm forward/Born mean 1.183 → 1.168 s; cached arrays alone require an additional 4.70 GB. | Total-measurement gate passed, but background-subtracted relative difference 2.565e-5 exceeded the unchanged 2e-5 gate. Algebraic residuals were identical; Born-image gate was not reached. | Preparation plus first case 00:00 → 00:00 (20.54 → 20.58 s). | Reject: about 1.2% warm savings does not justify the storage cost or failed equivalence gate. Restore the existing pipeline. |

## Matched final measurement

One NVIDIA A100 80 GB PCIe container, sequential fresh Python processes, JAX 0.6.2, compilation cache disabled. Both paths use the same ten seed-0 spherical-inclusion phantoms and a fixed known breast background; 128³ grid, 5 mm wavelength, 24 plane-wave sources, 256 receivers, complex64, GMRES tolerance 1e-6, eight CG steps and damping 0.001. The original closed-over, unrolled CG calculation is retained in the benchmark, not in production. The baseline forward/Born builder is loaded from the preceding FFT optimization branch.

| Metric | Baseline | Optimized |
| --- | ---: | ---: |
| Preparation + first forward/Born case | 113.46 s | 22.56 s |
| Warm forward mean, cases 1–9 | 1.310 s | 1.039 s |
| Warm eight-step Born mean | 0.162 s | 0.158 s |
| Warm forward + Born mean | 1.472 s | 1.196 s |
| Warm forward + Born range | 1.414–1.535 s | 1.136–1.259 s |
| Peak live JAX GPU allocation | 10.075 GB | 5.973 GB |
| Maximum full-wave relative residual | 6.779e-7 | 6.779e-7 |

Warm compute time fell **18.7%** (1.23× throughput); cold preparation plus first case fell **5.03×**. The cold speedup is not the large-dataset throughput improvement. Peak live allocation fell **40.7%**; this is not total driver-reserved GPU memory. These are one final paired comparison, with nine timed warm cases per path, not a multi-host performance study.

Maximum relative differences across the ten paired cases were **6.86e-7** for total scattered measurements, **1.18e-5** for background-subtracted data, and **1.34e-5** for Born images. Prespecified relative thresholds were 2e-5, 2e-5, and 1e-4 respectively. Both paths check the true wave-equation residual; background source/reciprocal preparation also rejects residuals above 5e-6. Equivalence does not establish discretization accuracy at this resolution or vessel recovery.

### Scale estimate, not a completed dataset

20,000 × 1.196 s is approximately **6.65 A100-hours for forward/Born compute**, versus 8.18 hours before this pass. This excludes host phantom construction, transfers, dataset writing, and all network training. It applies only to this resolution/acquisition configuration and **shared known background**, not independently varying breast tissue backgrounds that require new cached fields.

At 128³, two float32 volumes per case (truth and Born image) plus complex64 measurements need approximately **337 GB uncompressed** for 20,000 cases. The current demonstration still materializes the full dataset in RAM/device memory and repeatedly rewrites its archive; it must be replaced by chunked generation/storage and minibatch loading before a 20,000-case run. Training throughput remains unmeasured at that scale. No such run was launched and no proposal results were changed.

### Reproduce

```sh
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET .venv/bin/modal run --profile jma02 \
  examples/modal_breast_pipeline_benchmark_3d.py \
  --baseline-ref devin/1792632847-fft-gpu-optimization \
  --out /path/to/new-output-directory
```

The runner caps concurrency at one A100 container, executes baseline then optimized in fresh processes, and stops after the comparison. All pilot and benchmark apps were confirmed stopped. Raw timing and equivalence JSON/logs were retained with the session artifacts.

## Additional FFT-only optimization pass — rejected candidates

Both new paired trials used one **NVIDIA A100-SXM4-80GB**, JAX 0.6.2, compilation caching disabled, and sequential fresh Python processes. Their baseline was the already-optimized PR #12 implementation, not the original unrolled CG baseline above. The temporary benchmark used the same compiled eight-step Born reconstruction in both paths. Grid, wavelength, seed-0 phantoms, fixed known background, 24 sources, 256 receivers, complex64, GMRES tolerance, damping, and numerical gates were unchanged. Warm means exclude case 0 and cover cases 1–9.

| Trial | Baseline warm forward / Born | Candidate warm forward / Born | Baseline / candidate maximum algebraic residual |
| --- | ---: | ---: | ---: |
| Two-source batching | 1.037 / 0.156 s | 1.153 / 0.156 s | 6.779e-7 / 6.618e-7 |
| Full fixed-geometry cache | 1.027 / 0.156 s | 1.012 / 0.156 s | 6.779e-7 / 6.779e-7 |

The source-batching result illustrates why a small algebraic residual is not sufficient to certify equivalent measurements. The discrepancy's cause was not isolated, so this is not evidence for a particular precision or FFT-library failure. The cached variant also changed the reduction organization; its failed background-subtracted-data gate was not waived as rounding error. Because the scripts rejected candidates before writing final result JSON, candidate peak live allocation and complete Born-image differences were not recorded. The 4.70 GB cache figure is the calculated storage of `(24+256)*128^3` complex64 values, **not a measured peak**.

The three-configuration CPU prototype suite passed 66 selected tests, but those tests did not override the failed GPU comparisons. Both prototypes and their temporary benchmark changes were removed. The retained solver and production benchmark are unchanged; there is **no additional speedup claim**, and the earlier approximately 1.2 s/case fixed-background estimate is unchanged. Raw per-case timing/error logs remain in the session artifacts. Both new Modal apps were confirmed stopped with zero tasks. No HPS code, proposal outputs, network training, or 20,000-case dataset was changed or launched.
