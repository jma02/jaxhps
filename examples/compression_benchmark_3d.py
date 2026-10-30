"""Matched-reference decomposition comparison; run each stage separately."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from born_capacity_benchmark_3d import run as baseline_run
from breast_spheres_learn_3d import reconstruct_born
from compressed_born import (
    BlockBorn,
    CPFields,
    TuckerFields,
    compress_receiver_blocks,
    compress_spatial,
    to_blocks,
)
from phantom_inverse_3d import BackgroundBorn
from tensor_train_benchmark_3d import evaluate_operator, field_errors
from tensor_train_born import SlabBorn, TensorTrainFields, compress_born


def prepare(args: argparse.Namespace) -> None:
    operator, data, window, delta = baseline_run(
        argparse.Namespace(
            n=args.n,
            sources=args.sources,
            receivers=args.receivers,
            wavelength_mm=args.wavelength_mm,
            out=args.directory / "baseline.json",
        )
    )
    probe = jax.random.normal(
        jax.random.PRNGKey(2), data.shape, dtype=data.dtype
    )
    random_image = window * jax.random.normal(
        jax.random.PRNGKey(3), delta.shape, dtype=delta.dtype
    )
    np.savez(
        args.directory / "reference.npz",
        **jax.device_get(
            dict(
                data=data,
                window=window,
                delta=delta,
                image=reconstruct_born(operator, data, window, 8, 0.001),
                vessel_data=operator(delta),
                probe=probe,
                adjoint=operator.adjoint(probe),
                random_image=random_image,
                random_data=operator(random_image),
            )
        ),
    )
    for name in ("sources", "reciprocal_fields"):
        np.save(
            args.directory / f"{name}.npy", np.asarray(getattr(operator, name))
        )
    (args.directory / "shape.json").write_text(json.dumps(operator.shape))


def load_dense(directory: Path) -> BackgroundBorn:
    shape = tuple(json.loads((directory / "shape.json").read_text()))
    return BackgroundBorn(
        *[
            jnp.asarray(np.load(directory / f"{name}.npy", mmap_mode="r"))
            for name in ("sources", "reciprocal_fields")
        ],
        shape,
        np.dtype("float32"),
    )


def compress(args: argparse.Namespace) -> None:
    jax.config.update("jax_enable_x64", False)
    operator = load_dense(args.directory)
    jax.block_until_ready(operator)
    start = time.perf_counter()
    if args.method == "block":
        sources = to_blocks(
            operator.sources.reshape((-1, *operator.shape)), args.block_size
        )
        count = sources.shape[0]
        rank = min(
            args.rank, operator.reciprocal_fields.shape[0], args.block_size**3
        )
        np.save(
            args.directory / f"{args.label}-sources.npy", np.asarray(sources)
        )
        storage_bytes = sources.nbytes
        del sources
        outputs = [
            np.lib.format.open_memmap(
                args.directory / f"{args.label}-{name}.npy",
                mode="w+",
                dtype=np.complex64,
                shape=shape,
            )
            for name, shape in (
                ("basis", (operator.reciprocal_fields.shape[0], count, rank)),
                ("core", (count, rank, args.block_size**3)),
            )
        ]
        storage_bytes += sum(x.nbytes for x in outputs)
        for first in range(0, count, 1024):
            last = min(first + 1024, count)
            factors = compress_receiver_blocks(
                operator.reciprocal_fields,
                jnp.arange(first, last),
                operator.shape,
                rank,
                args.block_size,
            )
            outputs[0][:, first:last] = np.asarray(factors[0]).swapaxes(0, 1)
            outputs[1][first:last] = np.asarray(factors[1])
            if first % (16 * 1024) == 0:
                print(
                    f"Compressed receiver blocks: {last}/{count}", flush=True
                )
        for output in outputs:
            output.flush()
    elif args.method == "tt":
        compressed = compress_born(operator, args.rank, args.slab_size)
    else:
        compressed = compress_spatial(
            operator,
            args.method,
            args.rank,
            args.slab_size,
            args.core_rank,
            args.iterations,
        )
    if args.method != "block":
        jax.block_until_ready(compressed)
        storage_bytes = compressed.nbytes
    compression_seconds = time.perf_counter() - start
    row = dict(
        method=args.method,
        rank=args.rank,
        block_size=args.block_size,
        slab_size=args.slab_size,
        core_rank=args.core_rank,
        iterations=args.iterations,
        compression_seconds=compression_seconds,
        includes_streamed_writes=args.method == "block",
        storage_bytes=storage_bytes,
        setup_memory=jax.devices()[0].memory_stats(),
    )
    if args.method != "block":
        row["field_errors"] = {}
        for name in ("sources", "reciprocal_fields"):
            per_field, global_error = field_errors(
                getattr(operator, name),
                getattr(compressed, name),
                args.slab_size,
            )
            row["field_errors"][name] = dict(
                global_relative_error=float(global_error),
                max_relative_error=float(jnp.max(per_field)),
            )
        np.savez(
            args.directory / f"{args.label}.npz",
            *[np.asarray(x) for x in jax.tree_util.tree_leaves(compressed)],
        )
    (args.directory / f"compression-{args.label}.json").write_text(
        json.dumps(row, indent=2)
    )
    print(json.dumps(row), flush=True)


def evaluate(args: argparse.Namespace) -> None:
    jax.config.update("jax_enable_x64", False)
    shape = tuple(json.loads((args.directory / "shape.json").read_text()))
    if args.method == "block":
        arrays = [
            jnp.asarray(
                np.load(
                    args.directory / f"{args.label}-{name}.npy", mmap_mode="r"
                )
            )
            for name in ("sources", "basis", "core")
        ]
        operator = BlockBorn(
            *arrays, shape, np.dtype("float32"), args.block_size
        )
    else:
        with np.load(args.directory / f"{args.label}.npz") as stored:
            arrays = [
                jnp.asarray(stored[f"arr_{i}"])
                for i in range(len(stored.files))
            ]
        constructor = {
            "tucker": TuckerFields,
            "cp": CPFields,
            "tt": TensorTrainFields,
        }[args.method]
        half = len(arrays) // 2
        operator = SlabBorn(
            constructor(*arrays[:half]),
            constructor(*arrays[half:]),
            shape,
            np.dtype("float32"),
            args.slab_size,
        )
    args.report_name = f"evaluation-{args.label}.json"
    evaluate_operator(operator, args)


def screen(args: argparse.Namespace) -> None:
    """Inspect actual receiver and Born block ranks, not assumed TT ranks."""
    jax.config.update("jax_enable_x64", False)
    shape = tuple(json.loads((args.directory / "shape.json").read_text()))
    arrays = [
        np.load(args.directory / f"{name}.npy", mmap_mode="r").reshape(
            (-1, *shape)
        )
        for name in ("sources", "reciprocal_fields")
    ]
    rows = []
    # Centre, off-axis interior, and near-surface blocks.
    for fraction in ((0.5, 0.5, 0.5), (0.35, 0.7, 0.55), (0.5, 0.88, 0.5)):
        b = args.block_size
        origin = tuple(min(int(n * f), n - b) for n, f in zip(shape, fraction))
        fields = [
            jnp.asarray(
                np.array(a[(slice(None), *[slice(i, i + b) for i in origin])])
            ).reshape(a.shape[0], -1)
            for a in arrays
        ]
        for name, matrix in (
            ("receiver", fields[1]),
            (
                "born",
                (fields[0][:, None, :] * fields[1][None, :, :]).reshape(
                    -1, b**3
                ),
            ),
        ):
            singular = np.asarray(jnp.linalg.svd(matrix, compute_uv=False))
            energy = np.cumsum(singular.astype(float)[::-1] ** 2)[::-1]
            tails = np.sqrt(np.r_[energy, 0] / max(energy[0], 1e-30))
            rows.append(
                dict(
                    block_size=b,
                    origin=origin,
                    matrix=name,
                    shape=matrix.shape,
                    relative_frobenius_tails={
                        str(r): float(tails[min(r, len(singular))])
                        for r in (8, 16, 32, 64, 96, 128, 256)
                    },
                    ranks={
                        str(tol): int(np.flatnonzero(tails <= tol)[0])
                        for tol in (0.1, 0.01, 0.001)
                    },
                )
            )
    (args.directory / f"screen-block-{args.block_size}.json").write_text(
        json.dumps(rows, indent=2)
    )
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "compress", "evaluate", "screen")
    )
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--n", type=int, default=288)
    parser.add_argument("--sources", type=int, default=24)
    parser.add_argument("--receivers", type=int, default=256)
    parser.add_argument("--wavelength-mm", type=float, default=1.25)
    parser.add_argument(
        "--method", choices=("tt", "tucker", "cp", "block"), default="tucker"
    )
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--core-rank", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--slab-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--label", default="compressed")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    {
        "prepare": prepare,
        "compress": compress,
        "evaluate": evaluate,
        "screen": screen,
    }[args.stage](args)
