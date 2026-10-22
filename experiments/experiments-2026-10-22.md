# Experiments — 2026-10-22

| Time | Motivation | What Tried / Evidence | Correctness | Timing | Merge Decision |
| --- | --- | --- | --- | --- | --- |
| 02:46 | Establish the complete larger-grid pilot cost before considering 20,000 cases. | Original sphere pipeline, 128³, 5 mm, 24 sources, 256 receivers, eight training/two test cases, one fresh epoch. JAX reported 9.40 GB of captured constants during Born lowering. | Forward residuals below 1e-6. One epoch is a smoke test, not scientific reconstruction evidence: test contrast error was 91% Born and 182% corrected. | Data preparation 00:04 (233.4 s); first Born call 00:02 (143.57 s); epoch completed at 247.8 s. | Use only to identify bottlenecks; do not update proposal results. |
| 02:51 | Avoid compilation constants and repeated source/receiver geometry work. | Explicit Born pytree arguments, compiled CG loop, and receiver batches shared across sources. Same-A100 fresh-process comparison gave 1.480 → 1.261 s per warm case. | Measurement and Born-image comparisons passed; true residuals unchanged. | Cold preparation plus first case 00:02 → 00:00 (120.64 → 23.56 s). | Refine: runtime conjugation of large Born fields increased warm CG cost. |
| 02:56 | The real adjoint can conjugate the small data array instead of cached fields. | Rearranged `Re(conj(S) * (D @ conj(R)))` as `Re(S * (conj(D) @ R))`. Repeated the matched baseline/optimized comparison; details below. | 61 targeted tests passed; eight inverse tests re-run after the final adjoint change. GPU equivalence checks passed. | Cold preparation plus first case 00:02 → 00:00 (113.46 → 22.56 s); warm mean 1.472 → 1.196 s. | Retain in PR #12, not merged. No large dataset launched. |

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
