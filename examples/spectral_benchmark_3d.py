"""Generate fields in batches and benchmark a direct spectral Born map.

The dense disk copy is retained only to measure an uncompressed reference in
a separate process. No full source/receiver field cache is built on the GPU
during batched preparation or spectral application.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from born_capacity_benchmark_3d import capacity_problem
from breast_spheres_learn_3d import reconstruct_born
from compression_benchmark_3d import load_dense
from compressed_born import TuckerFields, tucker_fields
from phantom_inverse_3d import make_background_field_solver, make_forward
from spectral_born import (
    SpectralBorn,
    cosine_transform,
    spectral_indices,
    spectral_taper,
)
from tensor_train_benchmark_3d import evaluate_operator, field_errors
from tensor_train_born import SlabBorn


def prepare(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    points, kappa, directions, receivers, truth, background, window = (
        capacity_problem(
            args.n, args.wavelength_mm, args.sources, args.receivers
        )
    )
    row = dict(
        n=args.n,
        sources=args.sources,
        receivers=args.receivers,
        wavelength_mm=args.wavelength_mm,
        device=jax.devices()[0].device_kind,
        field_batch_size=args.field_batch_size,
        basis=args.basis,
        taper_floor=args.taper_floor if args.basis == "fft" else 1.0,
        stages=[],
        completed=False,
    )

    def checkpoint(stage: str) -> None:
        row["stages"].append(
            dict(
                stage=stage,
                elapsed_seconds=time.perf_counter() - started,
                memory=jax.devices()[0].memory_stats(),
            )
        )
        (args.directory / "preparation.json").write_text(
            json.dumps(row, indent=2)
        )
        print(json.dumps(row["stages"][-1]), flush=True)

    checkpoint("phantom_ready")
    forward = make_forward(points, kappa, directions, receivers, 1e-6)
    for name, coefficient in (("background", background), ("truth", truth)):
        start = time.perf_counter()
        measured, residuals = forward(coefficient)
        jax.block_until_ready(measured)
        residual = float(jnp.max(residuals))
        if not np.isfinite(residual) or residual > 5e-6:
            raise RuntimeError(f"{name} forward residual {residual}")
        row[f"{name}_forward_seconds"] = time.perf_counter() - start
        row[f"{name}_forward_residual"] = residual
        if name == "background":
            reference = measured
        else:
            data = measured - reference
    delta = truth - background
    np.savez(
        args.directory / "reference.npz",
        **jax.device_get(dict(data=data, delta=delta, window=window)),
    )
    shape = tuple(background.shape)
    (args.directory / "shape.json").write_text(json.dumps(shape))
    del (
        forward,
        truth,
        delta,
        data,
        reference,
        measured,
        residuals,
        coefficient,
        window,
    )
    jax.clear_caches()
    gc.collect()
    checkpoint("data_ready")

    start = time.perf_counter()
    solve = make_background_field_solver(points, kappa, background)
    energy = jnp.zeros(args.n**3, dtype=jnp.float32)
    taper = (
        spectral_taper(shape, args.taper_floor)
        if args.basis == "fft"
        else jnp.ones(shape)
    )
    transform = jax.jit(
        lambda x, taper: (
            cosine_transform(x.reshape(-1, *shape))
            if args.basis == "dct"
            else jnp.fft.fftn(
                x.reshape(-1, *shape) * taper, axes=(-3, -2, -1), norm="ortho"
            )
        ).reshape(x.shape)
    )
    spectral = np.lib.format.open_memmap(
        args.directory / "receiver-spectrum.npy",
        mode="w+",
        dtype=np.complex64,
        shape=(args.receivers, args.n**3),
    )
    norms = []
    max_residual = 0.0
    transform_seconds = 0.0
    tucker_seconds = 0.0
    tucker_storage = 0
    tucker_errors = {}
    scaled_sources = np.lib.format.open_memmap(
        args.directory / "scaled-sources.npy",
        mode="w+",
        dtype=np.complex64,
        shape=(args.sources, args.n**3),
    )
    for name, positions, reciprocal in (
        ("sources", directions, False),
        ("reciprocal_fields", receivers, True),
    ):
        tucker_outputs = {}
        squared_error, squared_norm, worst_error = 0.0, 0.0, 0.0
        dense = np.lib.format.open_memmap(
            args.directory / f"{name}.npy",
            mode="w+",
            dtype=np.complex64,
            shape=(len(positions), args.n**3),
        )
        for first in range(0, len(positions), args.field_batch_size):
            last = min(first + args.field_batch_size, len(positions))
            fields, residuals = solve(positions[first:last], reciprocal)
            max_residual = max(max_residual, float(jnp.max(residuals)))
            dense[first:last] = np.asarray(fields)
            if args.tucker_rank:
                t = time.perf_counter()
                compressed = tucker_fields(fields, shape, args.tucker_rank)
                for factor in ("first", "second", "third", "core"):
                    values = np.asarray(getattr(compressed, factor))
                    if factor not in tucker_outputs:
                        tucker_outputs[factor] = np.lib.format.open_memmap(
                            args.directory / f"tucker-{name}-{factor}.npy",
                            mode="w+",
                            dtype=values.dtype,
                            shape=(len(positions), *values.shape[1:]),
                        )
                        tucker_storage += tucker_outputs[factor].nbytes
                    tucker_outputs[factor][first:last] = values
                per_field, global_error = field_errors(fields, compressed, 32)
                norm = float(jnp.sum(jnp.abs(fields) ** 2))
                squared_norm += norm
                squared_error += float(global_error) ** 2 * norm
                worst_error = max(worst_error, float(jnp.max(per_field)))
                tucker_seconds += time.perf_counter() - t
                del compressed, values
            if reciprocal:
                t = time.perf_counter()
                transformed = transform(fields, taper).block_until_ready()
                energy = (
                    energy + jnp.sum(jnp.abs(transformed) ** 2, axis=0)
                ).block_until_ready()
                norms.extend(
                    np.asarray(
                        jnp.sum(jnp.abs(transformed) ** 2, axis=1)
                    ).tolist()
                )
                transform_seconds += time.perf_counter() - t
                spectral[first:last] = np.asarray(transformed)
                del transformed
            else:
                scaled_sources[first:last] = np.asarray(fields / taper.ravel())
            del fields, residuals
            if first % 16 == 0:
                print(f"{name}: {last}/{len(positions)}", flush=True)
        dense.flush()
        for output in tucker_outputs.values():
            output.flush()
        tucker_errors[name] = dict(
            global_relative_error=float(
                np.sqrt(squared_error / max(squared_norm, 1e-30))
            ),
            max_relative_error=worst_error,
        )
        del dense
        checkpoint(f"{name}_prepared")
    spectral.flush()
    scaled_sources.flush()
    np.save(args.directory / "spectral-energy.npy", np.asarray(energy))
    np.save(args.directory / "spectral-norms.npy", np.asarray(norms))
    row.update(
        field_preparation_and_disk_seconds=time.perf_counter() - start,
        spectral_transform_seconds=transform_seconds,
        maximum_field_residual=max_residual,
        completed=True,
    )
    if args.tucker_rank:
        (args.directory / "compression-batched-tucker.json").write_text(
            json.dumps(
                dict(
                    rank=args.tucker_rank,
                    compression_validation_and_write_seconds=tucker_seconds,
                    storage_bytes=tucker_storage,
                    field_errors=tucker_errors,
                ),
                indent=2,
            )
        )
    checkpoint("finished")


def dense_reference(args: argparse.Namespace) -> None:
    operator = load_dense(args.directory)
    with np.load(args.directory / "reference.npz") as stored:
        reference = {key: jnp.asarray(stored[key]) for key in stored.files}
    data, delta, window = (
        reference[key] for key in ("data", "delta", "window")
    )
    probe = jax.random.normal(
        jax.random.PRNGKey(2), data.shape, dtype=data.dtype
    )
    random_image = window * jax.random.normal(
        jax.random.PRNGKey(3), delta.shape, dtype=delta.dtype
    )
    reference.update(
        image=reconstruct_born(operator, data, window, 8, 0.001),
        vessel_data=operator(delta),
        probe=probe,
        adjoint=operator.adjoint(probe),
        random_image=random_image,
        random_data=operator(random_image),
    )
    full_random = jax.random.normal(
        jax.random.PRNGKey(19), delta.shape, dtype=delta.dtype
    )
    reference.update(
        full_random_image=full_random, full_random_data=operator(full_random)
    )
    reference["normal_random"] = window * operator.adjoint(
        reference["random_data"]
    )
    np.savez(args.directory / "reference.npz", **jax.device_get(reference))
    args.method = "dense"
    args.report_name = "evaluation-dense.json"
    del reference, data, delta, window, probe, random_image
    gc.collect()
    evaluate_operator(operator, args)


def compress(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    energy = np.load(args.directory / "spectral-energy.npy")
    indices = spectral_indices(energy, args.tolerance)
    spectrum = np.load(args.directory / "receiver-spectrum.npy", mmap_mode="r")
    output = np.lib.format.open_memmap(
        args.directory / f"{args.label}-coefficients.npy",
        mode="w+",
        dtype=spectrum.dtype,
        shape=(len(spectrum), len(indices)),
    )
    norms = np.load(args.directory / "spectral-norms.npy")
    kept = []
    for index, field in enumerate(spectrum):
        values = field[indices]
        output[index] = values
        kept.append(np.sum(np.abs(values.astype(np.complex128)) ** 2))
    output.flush()
    np.save(args.directory / f"{args.label}-indices.npy", indices)
    discarded = np.maximum(norms - kept, 0)
    sources = np.load(args.directory / "sources.npy", mmap_mode="r")
    row = dict(
        method=f"spectral-{args.basis}",
        taper_floor=args.taper_floor if args.basis == "fft" else 1.0,
        receiver_field_tolerance=args.tolerance,
        retained_modes=len(indices),
        mode_fraction=len(indices) / len(energy),
        storage_bytes=sources.nbytes + output.nbytes + indices.nbytes,
        compression_seconds=time.perf_counter() - start,
        includes_disk_writes=True,
        transformed_receiver_global_relative_error=float(
            np.sqrt(np.sum(discarded) / np.sum(norms))
        ),
        transformed_receiver_max_relative_error=float(
            np.sqrt(np.max(discarded / np.maximum(norms, 1e-30)))
        ),
    )
    (args.directory / f"compression-{args.label}.json").write_text(
        json.dumps(row, indent=2)
    )
    print(json.dumps(row), flush=True)


def evaluate(args: argparse.Namespace) -> None:
    shape = tuple(json.loads((args.directory / "shape.json").read_text()))
    operator = SpectralBorn(
        *[
            jnp.asarray(np.load(args.directory / name, mmap_mode="r"))
            for name in (
                "scaled-sources.npy",
                f"{args.label}-coefficients.npy",
                f"{args.label}-indices.npy",
            )
        ],
        shape,
        np.dtype("float32"),
        args.source_batch_size,
        args.basis,
    )
    args.rank = len(operator.indices)
    args.method = f"spectral-{args.basis}"
    args.report_name = (
        f"evaluation-{args.label}-batch-{args.source_batch_size}.json"
    )
    evaluate_operator(operator, args)


def evaluate_tucker(args: argparse.Namespace) -> None:
    shape = tuple(json.loads((args.directory / "shape.json").read_text()))
    fields = [
        TuckerFields(
            *[
                jnp.asarray(
                    np.load(
                        args.directory / f"tucker-{name}-{factor}.npy",
                        mmap_mode="r",
                    )
                )
                for factor in ("first", "second", "third", "core")
            ]
        )
        for name in ("sources", "reciprocal_fields")
    ]
    operator = SlabBorn(*fields, shape, np.dtype("float32"), 32)
    args.method = "batched-tucker"
    args.rank = args.tucker_rank
    args.slab_size = 32
    args.report_name = "evaluation-batched-tucker.json"
    evaluate_operator(operator, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "dense", "compress", "evaluate", "tucker")
    )
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--n", type=int, default=288)
    parser.add_argument("--sources", type=int, default=24)
    parser.add_argument("--receivers", type=int, default=256)
    parser.add_argument("--wavelength-mm", type=float, default=1.25)
    parser.add_argument("--field-batch-size", type=int, default=1)
    parser.add_argument("--source-batch-size", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=0.003)
    parser.add_argument("--basis", choices=("dct", "fft"), default="fft")
    parser.add_argument("--taper-floor", type=float, default=0.01)
    parser.add_argument("--tucker-rank", type=int, default=0)
    parser.add_argument("--label", default="spectral")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.rank, args.slab_size = 0, 0
    args.directory.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", False)
    {
        "prepare": prepare,
        "dense": dense_reference,
        "compress": compress,
        "evaluate": evaluate,
        "tucker": evaluate_tucker,
    }[args.stage](args)
