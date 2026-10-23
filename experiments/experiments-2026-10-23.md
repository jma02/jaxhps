# Experiments — 2026-10-23

| Time | Motivation | What Tried / Evidence | Correctness | Timing | Merge Decision |
| --- | --- | --- | --- | --- | --- |
| 06:13 | Replace the estimated 288³ Born-CG memory/runtime with a measurement at the proposed wavelength. | One A100 at a time, 288³, 1.25 mm, 24 sources, 256 receivers, complex64, dense seed-7 vascular phantom. The first attempt could not allocate the receiver fields under the default growing 75% JAX pool. A fresh process with a 95% preallocated pool completed without changing solver code, precision, or resolution. | Background/truth full-wave residuals 1.30e-6/1.94e-6; cached fields passed the existing 5e-6 residual gate. Real-adjoint error 3.37e-7. Eight CG steps reached normal residual 2.45e-6. Vessel-contrast image error was nevertheless 99.93%. | Successful field preparation 00:07 (464.64 s); warm eight-step CG 00:00 (1.592 s mean); entire benchmark 00:09 (577.82 s). | Retain the reproducible capacity benchmark. Computational feasibility, not successful vessel recovery. No solver change, compression, network training, or proposal edit. |

## Full-resolution Born CG capacity

The successful run used one **NVIDIA A100 80GB PCIe**, JAX 0.6.2, and a fresh
Python process with compilation caching disabled. The failed allocator trial
used an A100-SXM4-80GB; these are not a controlled timing comparison. Both apps
stopped with zero tasks after their runs.

The box is 100 mm wide, sampled at 288³ (23,887,872 voxels, 0.3472 mm spacing).
The dense vascular phantom is unchanged. Its exact vessel-free tissue is the
known reference background. Twenty-four inward plane-wave directions and 256
receivers use a lower hemispherical Fibonacci cap, radius 100 mm and upper
cutoff z=-4 mm. Data are noiseless full-wave measurements minus the full-wave
background measurements, at a 1.25 mm wavelength (80 wavelengths across the
box). The reconstruction window is the binary breast support, not a vessel
mask. The regularization is the existing `0.001 * sum(abs(data)**2)`.

### Measured runtime and memory

| Quantity | Measurement |
| --- | ---: |
| Stored source and reciprocal fields | 53.509 GB |
| Peak live JAX device allocation, all phases | **63.231 GB** (58.889 GiB) |
| JAX preallocated pool | 80.862 GB |
| Peak sampled driver usage, including unused reserved pool | 81.341 GB |
| Field preparation, including compilation and residual certification | **464.643 s** (7.74 min) |
| First eight-step CG call, including compilation | 3.126 s |
| Three warm eight-step CG calls | 1.5931 / 1.5916 / 1.5920 s |
| Warm mean | **1.5923 s** |
| Background forward calculation, including first-solve compilation | 46.980 s |
| Full-phantom forward calculation | 46.469 s |
| Complete benchmark, including checks | 577.819 s |

GB means decimal bytes. Live allocation is not total driver usage. The A100
80GB accommodates the reported pool; its nominal capacity must not be confused
with 80 billion bytes. Driver usage was sampled approximately every two seconds;
JAX supplied its own live-allocation high-water mark.

The initial 58.7 GB estimate omitted parts of the actual setup/workspace peak.
The default growing allocator had a 63.839 GB limit and failed to allocate the
48.922 GB receiver-field array after earlier smaller allocations. A larger
contiguous preallocated pool worked. This does **not** measure the minimum pool
size needed, nor establish that an additional network will fit. Reproduction
uses `XLA_PYTHON_CLIENT_PREALLOCATE=true` and
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`.

### Convergence is not recovery

| Check | Result |
| --- | ---: |
| Maximum background full-wave relative residual | 1.302e-6 |
| Maximum phantom full-wave relative residual | 1.939e-6 |
| Cached background fields | All passed existing finite/5e-6 gate |
| Real-adjoint inner-product relative error | 3.370e-7 |
| True regularized normal-equation residual after eight CG steps | **2.454e-6** |
| Born reconstructed-data relative residual | 4.359e-5 |
| Born model mismatch evaluated at the true vessel contrast | 18.926% |
| Reconstructed vessel-contrast relative L2 error | **99.931%** |

The residual target was 1e-4, so the benchmark stopped its 8/16/32/64/128-step
sequence after eight steps. The normal residual was independently recomputed
from the output, not taken from CG's recursive residual. The binary support
window makes the returned masked image equal to the internal CG iterate.

The image error compares the reconstructed perturbation with the true
vessel-induced perturbation, **not** the total breast including its known
background. A very small measurement or normal-equation residual does not imply
a correct image in this underdetermined problem. This single case establishes
that matrix-free CG can run and converge at the proposed scale, not that plain
Born inversion or a learned correction recovers the branches. No network was
trained, and noise, unfamiliar backgrounds, vessel-width accuracy, and grid
refinement remain untested.

Field preparation can be amortized only while background, wavelength, and
acquisition geometry remain fixed. Neither tensor compression nor host streaming
was used, and no dense Born or normal matrix was formed.

### Reproduce

```sh
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET .venv/bin/modal run --profile jma02 \
  examples/modal_born_capacity_3d.py --out /existing/directory/new-result.json
```

The runner permits one A100 container, no retries, and at most 1,800 seconds of
subprocess work. It saves partial results on failure, monitors driver usage, and
does not deploy a service. The local CLI also supports small CPU smoke cases.
Eleven existing inverse/sampler regression tests passed, including agreement of
CG with dense normal equations. A 32³, 25 mm, 3-source/7-receiver CPU smoke case
passed the full benchmark and reached normal residual 1.68e-7 in 16 steps.
