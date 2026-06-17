r"""Generate a free-space 3D Helmholtz multistatic scattering dataset.

Each sample is a random penetrable scatterer made of ``n in {1, 2, 3}`` smooth
radial bumps placed at random positions in free space (no skin layer):

    b(x) = sum_k  A_k (1 - (|x - c_k| / R_k)^2)^4  1_{|x - c_k| < R_k},

so the medium has squared index ``n(x)^2 = 1 - b(x)`` and the scattered field
solves ``Lap u^s + kappa^2 (1 - b) u^s = kappa^2 b u^inc``.  We illuminate with
``n_tx`` plane waves whose directions are spread on a Fibonacci sphere and
record the scattered field at ``n_rx`` receivers on a sphere of radius ``rho``
(also Fibonacci-spread), giving a complex multistatic matrix
``M[j, i] = u^s(w_i, r_j)`` of shape ``(n_rx, n_tx)`` per sample.

The HPS+BIE forward solver (:func:`wave_scattering_utils_3D.
solve_scattering_bie_3D_cartesian`) reuses a precomputed single/double-layer
matrix file (``--npz``); the cube size ``a`` and wavenumber ``kappa`` are read
from it.  Scatterers are constrained to ``|c_k| + R_k <= c_max`` so their
support stays inside the cube with margin.

Output is written as Apache Parquet shards under ``<out>/data/`` together with
``metadata.json`` (sensor geometry + global config) and ``README.md`` (a
Hugging Face dataset card).  Generation is sharded, seeded, and resumable: a
shard already present on disk is skipped, and every sample's scatterer
configuration is a deterministic function of ``(seed, sample_index)``.

Examples
--------
Generate 20k samples in shards of 250, 128 tx / 128 rx::

    python examples/scattering_dataset_3d.py \
        --npz data/examples/SD_3D/SD_k4_q8_L2_a1.25.npz \
        --n_samples 20000 --shard_size 250 --n_tx 128 --n_rx 128 \
        --out ~/scattering_ds

Push an already-generated folder to the Hub (token via ``$HF_TOKEN``)::

    python examples/scattering_dataset_3d.py --out ~/scattering_ds \
        --push_to_hub <user>/<dataset> --push_only
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from wave_scattering_utils_3D import (  # noqa: E402
    build_cartesian_ctx,
    eval_uscat_offsurface_3D,
    load_SD_matrices_3D,
    solve_scattering_bie_3D_cartesian,
)


def fibonacci_sphere(n):
    """``n`` roughly-uniform unit vectors on S^2 (Fibonacci spiral)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ],
        axis=-1,
    )


def make_sensors(n_tx, n_rx, rho):
    """Plane-wave transmitter directions and receiver points on radius ``rho``."""
    tx_dirs = fibonacci_sphere(n_tx)
    rx_pts = rho * fibonacci_sphere(n_rx)
    return tx_dirs, rx_pts


def sample_config(rng, n_max, R_range, A_range, c_max):
    """Draw one scatterer configuration.

    Returns ``(n, centers, radii, amps)`` with ``centers`` of shape ``(n, 3)``.
    Each bump is sampled uniformly in the ball of radius ``c_max - R`` so its
    support ``|x - c| < R`` stays within ``|x| < c_max``.
    """
    n = int(rng.integers(1, n_max + 1))
    radii = rng.uniform(R_range[0], R_range[1], size=n)
    amps = rng.uniform(A_range[0], A_range[1], size=n)
    centers = np.zeros((n, 3))
    for k in range(n):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        rmax = max(c_max - radii[k], 0.0)
        rad = rmax * rng.uniform() ** (1.0 / 3.0)
        centers[k] = rad * d
    return n, centers, radii, amps


def make_b(centers, radii, amps):
    """Cartesian potential ``b(pts)`` summing C^3 polynomial bumps."""

    def b(pts):
        pts = np.asarray(pts)
        out = np.zeros(pts.shape[:-1])
        for c, R, A in zip(centers, radii, amps):
            r = np.linalg.norm(pts - c, axis=-1)
            rho = np.where(r < R, r / R, 1.0)
            out = out + np.where(r < R, A * (1.0 - rho * rho) ** 4, 0.0)
        return out

    return b


def solve_sample(sd, centers, radii, amps, tx_dirs, rx_pts, p, ctx=None):
    """Forward solve + off-surface eval -> multistatic matrix ``(n_rx, n_tx)``.

    ``ctx`` is an optional cache from
    :func:`wave_scattering_utils_3D.build_cartesian_ctx`; passing the same
    ``ctx`` across samples amortizes the ``b``-independent setup.
    """
    import jax.numpy as jnp

    b = make_b(centers, radii, amps)
    out = solve_scattering_bie_3D_cartesian(sd, b, tx_dirs, p=p, ctx=ctx)
    M = np.asarray(
        eval_uscat_offsurface_3D(
            target_pts=jnp.asarray(rx_pts),
            src_pts=jnp.asarray(out["boundary_points"]),
            src_normals=jnp.asarray(out["normals"]),
            src_weights=jnp.asarray(out["sdp"]["wts"]),
            uscat_b=jnp.asarray(out["uscat_b"]),
            uscat_dn_b=jnp.asarray(out["uscat_dn_b"]),
            k=float(sd["kappa"]),
        )
    )
    return M


def pad(arr, n_max, width=None):
    """Pad leading axis to ``n_max`` with NaN (for variable scatterer count)."""
    arr = np.asarray(arr, dtype=np.float64)
    shape = (n_max,) if width is None else (n_max, width)
    out = np.full(shape, np.nan)
    out[: arr.shape[0]] = arr
    return out


def write_shard(path, rows, n_max, n_tx, n_rx):
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = {
        "n_scatterers": pa.array([r["n"] for r in rows], pa.int32()),
        "centers": pa.array(
            [pad(r["centers"], n_max, 3).reshape(-1).tolist() for r in rows],
            pa.list_(pa.float32(), n_max * 3),
        ),
        "radii": pa.array(
            [pad(r["radii"], n_max).tolist() for r in rows],
            pa.list_(pa.float32(), n_max),
        ),
        "amps": pa.array(
            [pad(r["amps"], n_max).tolist() for r in rows],
            pa.list_(pa.float32(), n_max),
        ),
        "u_real": pa.array(
            [
                np.real(r["M"]).astype(np.float32).reshape(-1).tolist()
                for r in rows
            ],
            pa.list_(pa.float32(), n_rx * n_tx),
        ),
        "u_imag": pa.array(
            [
                np.imag(r["M"]).astype(np.float32).reshape(-1).tolist()
                for r in rows
            ],
            pa.list_(pa.float32(), n_rx * n_tx),
        ),
        "sample_index": pa.array([r["idx"] for r in rows], pa.int64()),
    }
    pq.write_table(pa.table(cols), path)


def write_metadata(out, sd, args, tx_dirs, rx_pts):
    meta = dict(
        kappa=float(sd["kappa"]),
        cube_half_width=float(sd["a"]),
        q=int(sd["q"]),
        L=int(sd["L"]),
        interior_order_p=args.p if args.p is not None else int(sd["q"]) + 4,
        n_tx=args.n_tx,
        n_rx=args.n_rx,
        rho=args.rho,
        matrix_shape=[args.n_rx, args.n_tx],
        matrix_layout="M[j_rx, i_tx] = u^s(w_i, r_j), row-major flatten",
        n_scatterers_max=args.n_max,
        R_range=args.R_range,
        A_range=args.A_range,
        c_max=args.c_max,
        bump="A*(1-(r/R)^2)^4 for r<R (C^3, compact support)",
        index_convention="n(x)^2 = 1 - b(x)",
        seed=args.seed,
        n_samples=args.n_samples,
        tx_dirs=tx_dirs.tolist(),
        rx_pts=rx_pts.tolist(),
    )
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def write_card(out, meta):
    card = f"""---
license: mit
task_categories:
  - other
tags:
  - helmholtz
  - inverse-scattering
  - pde
  - physics
pretty_name: Free-space 3D Helmholtz multistatic scattering
size_categories:
  - 10K<n<100K
---

# Free-space 3D Helmholtz multistatic scattering

Synthetic dataset of multistatic far-/near-field scattering matrices for
penetrable scatterers in 3D free space, generated with the jaxhps HPS+BIE
forward solver (`examples/scattering_dataset_3d.py`).

## Physics

Each sample is a sum of `n in {{1,2,3}}` smooth radial bumps

```
b(x) = sum_k A_k (1 - (|x - c_k|/R_k)^2)^4  for |x - c_k| < R_k,
```

with squared refractive index `n(x)^2 = 1 - b(x)`.  The scattered field solves

```
Lap u^s + kappa^2 (1 - b) u^s = kappa^2 b u^inc,   u^inc = exp(i kappa w . x),
```

with the Sommerfeld radiation condition (`kappa = {meta["kappa"]}`).

## Geometry

- Cube half-width `a = {meta["cube_half_width"]}` (HPS domain `[-a, a]^3`),
  boundary order `q = {meta["q"]}`, octree depth `L = {meta["L"]}`,
  interior Chebyshev order `p = {meta["interior_order_p"]}`.
- `{meta["n_tx"]}` plane-wave transmitter directions (Fibonacci sphere).
- `{meta["n_rx"]}` receivers on a sphere of radius `rho = {meta["rho"]}`
  (Fibonacci sphere).
- Scatterer support is constrained to `|c_k| + R_k <= {meta["c_max"]}`.

The exact transmitter directions and receiver points are in `metadata.json`
(`tx_dirs`, `rx_pts`).

## Fields

| column | shape | description |
|---|---|---|
| `n_scatterers` | scalar | number of bumps (1-3) |
| `centers` | {meta["n_scatterers_max"]}x3 (flattened, NaN-padded) | bump centers `c_k` |
| `radii` | {meta["n_scatterers_max"]} (NaN-padded) | bump radii `R_k` |
| `amps` | {meta["n_scatterers_max"]} (NaN-padded) | bump amplitudes `A_k` |
| `u_real`, `u_imag` | {meta["n_rx"]}*{meta["n_tx"]} (flattened) | `Re/Im` of `M[j_rx, i_tx]` |
| `sample_index` | scalar | deterministic seed index |

## Loading

```python
import json
import numpy as np
from datasets import load_dataset

ds = load_dataset("REPO_ID", split="train")
meta = json.load(open("metadata.json"))  # or hf_hub_download
nrx, ntx = meta["matrix_shape"]

ex = ds[0]
M = (np.array(ex["u_real"]) + 1j * np.array(ex["u_imag"])).reshape(nrx, ntx)
centers = np.array(ex["centers"]).reshape(meta["n_scatterers_max"], 3)
```

Generated by [jaxhps](https://github.com/jma02/jaxhps).
"""
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(card)


def push_to_hub(out, repo_id, token):
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=out, repo_id=repo_id, repo_type="dataset")
    print(f"pushed {out} -> https://huggingface.co/datasets/{repo_id}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--npz", default="data/examples/SD_3D/SD_k4_q8_L2_a1.25.npz"
    )
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--n_samples", type=int, default=20000)
    ap.add_argument("--shard_size", type=int, default=250)
    ap.add_argument("--n_tx", type=int, default=128)
    ap.add_argument("--n_rx", type=int, default=128)
    ap.add_argument("--rho", type=float, default=2.5)
    ap.add_argument("--n_max", type=int, default=3, help="max scatterers")
    ap.add_argument("--R_range", type=float, nargs=2, default=[0.15, 0.35])
    ap.add_argument("--A_range", type=float, nargs=2, default=[-0.5, -0.1])
    ap.add_argument("--c_max", type=float, default=1.1)
    ap.add_argument("--p", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push_to_hub", default=None, help="HF dataset repo id")
    ap.add_argument(
        "--push_only",
        action="store_true",
        help="skip generation; just push --out to the Hub",
    )
    args = ap.parse_args()

    data_dir = os.path.join(args.out, "data")
    os.makedirs(data_dir, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    if args.push_only:
        if not args.push_to_hub:
            raise SystemExit("--push_only requires --push_to_hub")
        push_to_hub(args.out, args.push_to_hub, token)
        return

    sd = load_SD_matrices_3D(args.npz)
    tx_dirs, rx_pts = make_sensors(args.n_tx, args.n_rx, args.rho)
    if args.rho <= sd["a"] * np.sqrt(3.0):
        print(
            f"warning: rho={args.rho} close to cube corner "
            f"(a*sqrt(3)={sd['a'] * np.sqrt(3.0):.3f})"
        )

    meta = write_metadata(args.out, sd, args, tx_dirs, rx_pts)
    write_card(args.out, meta)

    ctx = build_cartesian_ctx(sd, tx_dirs, p=args.p)

    n_shards = (args.n_samples + args.shard_size - 1) // args.shard_size
    print(
        f"generating {args.n_samples} samples in {n_shards} shards "
        f"(kappa={sd['kappa']}, a={sd['a']}, n_tx={args.n_tx}, "
        f"n_rx={args.n_rx})"
    )

    for s in range(n_shards):
        path = os.path.join(
            data_dir, f"train-{s:05d}-of-{n_shards:05d}.parquet"
        )
        if os.path.exists(path):
            print(f"shard {s} exists, skipping")
            continue
        lo = s * args.shard_size
        hi = min(lo + args.shard_size, args.n_samples)
        rows = []
        t0 = time.time()
        for idx in range(lo, hi):
            rng = np.random.default_rng([args.seed, idx])
            n, centers, radii, amps = sample_config(
                rng, args.n_max, args.R_range, args.A_range, args.c_max
            )
            M = solve_sample(
                sd, centers, radii, amps, tx_dirs, rx_pts, args.p, ctx=ctx
            )
            rows.append(
                dict(
                    n=n,
                    centers=centers,
                    radii=radii,
                    amps=amps,
                    M=M,
                    idx=idx,
                )
            )
            dt = time.time() - t0
            done = idx - lo + 1
            print(
                f"  shard {s} sample {done}/{hi - lo} "
                f"(global {idx}) n={n} | {dt / done:.1f}s/sample",
                flush=True,
            )
        write_shard(path, rows, args.n_max, args.n_tx, args.n_rx)
        print(f"wrote {path} ({len(rows)} samples)")

    if args.push_to_hub:
        push_to_hub(args.out, args.push_to_hub, token)


if __name__ == "__main__":
    main()
