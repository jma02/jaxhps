"""Compare TT Born fields against the full-resolution cached-field baseline.

Run prepare once, then evaluate each rank in a fresh process. This separates
compression-setup memory (which includes the dense cache) from application
memory. Field error, vessel-signal error, and CG-image error are distinct.
"""

import argparse
import gc
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from born_capacity_benchmark_3d import run as baseline_run
from breast_spheres_learn_3d import reconstruct_born
from tensor_train_born import SlabBorn, TensorTrainFields, compress_born


def relative_error(actual, reference) -> float:
    return float(
        jnp.linalg.norm(actual - reference)
        / jnp.maximum(jnp.linalg.norm(reference), 1e-30)
    )


@partial(jax.jit, static_argnames=("slab_size",))
def field_errors(dense, compressed, slab_size):
    nx, ny, nz = compressed.shape
    padding = (-nx) % slab_size
    dense = jnp.pad(
        dense.reshape(-1, nx, ny, nz), ((0, 0), (0, padding), (0, 0), (0, 0))
    )

    def step(index, errors):
        reference = jax.lax.dynamic_slice_in_dim(
            dense, index * slab_size, slab_size, 1
        ).reshape(dense.shape[0], -1)
        return errors + jnp.sum(
            jnp.abs(compressed.slab(index, slab_size) - reference) ** 2, axis=1
        )

    errors = jax.lax.fori_loop(
        0,
        (nx + padding) // slab_size,
        step,
        jnp.zeros(dense.shape[0], dtype=dense.real.dtype),
    )
    norms = jnp.sum(jnp.abs(dense) ** 2, axis=(1, 2, 3))
    return jnp.sqrt(errors / jnp.maximum(norms, 1e-30)), jnp.sqrt(
        jnp.sum(errors) / jnp.maximum(jnp.sum(norms), 1e-30)
    )


def prepare(args) -> None:
    operator, data, window, delta = baseline_run(
        argparse.Namespace(
            n=args.n,
            wavelength_mm=args.wavelength_mm,
            sources=args.sources,
            receivers=args.receivers,
            out=args.directory / "baseline.json",
        )
    )
    image = reconstruct_born(operator, data, window, 8, 0.001)
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
                image=image,
                vessel_data=operator(delta),
                probe=probe,
                adjoint=operator.adjoint(probe),
                random_image=random_image,
                random_data=operator(random_image),
            )
        ),
    )
    del image, probe, random_image
    jax.clear_caches()
    gc.collect()
    rows = []
    for rank in args.ranks:
        print(f"TT compression starting: rank={rank}", flush=True)
        start = time.perf_counter()
        compressed = compress_born(operator, rank, args.slab_size)
        jax.block_until_ready(compressed)
        row = dict(
            rank=rank,
            compression_seconds=time.perf_counter() - start,
            field_storage_bytes=compressed.nbytes,
            field_errors={},
        )
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
        row["setup_memory"] = jax.devices()[0].memory_stats()
        start = time.perf_counter()
        np.savez(
            args.directory / f"rank-{rank}.npz",
            **{
                f"{group}_{core}": np.asarray(
                    getattr(getattr(compressed, group), core)
                )
                for group in ("sources", "reciprocal_fields")
                for core in ("first", "middle", "last")
            },
        )
        row["write_seconds"] = time.perf_counter() - start
        rows.append(row)
        (args.directory / "compression.json").write_text(
            json.dumps(rows, indent=2)
        )
        print(json.dumps(row), flush=True)
        del compressed
        jax.clear_caches()
        gc.collect()


def evaluate(args) -> None:
    jax.config.update("jax_enable_x64", False)
    rank = args.rank
    with np.load(args.directory / f"rank-{rank}.npz") as stored:
        fields = [
            TensorTrainFields(
                *(
                    jnp.asarray(stored[f"{group}_{core}"])
                    for core in ("first", "middle", "last")
                )
            )
            for group in ("sources", "reciprocal_fields")
        ]
    operator = SlabBorn(
        *fields, fields[0].shape, np.dtype("float32"), args.slab_size
    )
    evaluate_operator(operator, args)


def evaluate_operator(operator, args) -> None:
    with np.load(args.directory / "reference.npz") as stored:
        reference = {key: jnp.asarray(stored[key]) for key in stored.files}
    jax.block_until_ready((operator, reference))
    row = dict(
        rank=args.rank,
        method=getattr(args, "method", "tt"),
        slab_size=args.slab_size,
        field_storage_bytes=operator.nbytes,
        device=jax.devices()[0].device_kind,
        jax=jax.__version__,
        completed=False,
        stages=[],
    )

    def checkpoint(stage):
        row["stages"].append(
            dict(stage=stage, memory=jax.devices()[0].memory_stats())
        )
        (
            args.directory
            / getattr(
                args,
                "report_name",
                f"evaluation-{args.rank}-slab-{args.slab_size}.json",
            )
        ).write_text(json.dumps(row, indent=2))
        print(json.dumps(row), flush=True)

    checkpoint("loaded_compressed_only")
    data, delta, window = (reference[k] for k in ("data", "delta", "window"))
    rhs = window * operator.adjoint(data)
    vessel_data = operator(delta)
    adjoint = operator.adjoint(reference["probe"])
    row["vessel_signal_relative_error"] = relative_error(
        vessel_data, reference["vessel_data"]
    )
    row["random_image_signal_relative_error"] = relative_error(
        operator(reference["random_image"]), reference["random_data"]
    )
    row["adjoint_application_relative_error"] = relative_error(
        adjoint, reference["adjoint"]
    )
    lhs = float(jnp.real(jnp.vdot(vessel_data, reference["probe"])))
    right = float(jnp.vdot(delta, adjoint))
    row["adjoint_identity_relative_error"] = abs(lhs - right) / max(
        abs(lhs), abs(right), 1e-30
    )
    if (
        not np.isfinite(row["adjoint_identity_relative_error"])
        or row["adjoint_identity_relative_error"] > 1e-4
    ):
        raise RuntimeError("Real-adjoint identity failed")
    checkpoint("operator_checked")
    for name, call in (
        ("forward", lambda: operator(delta)),
        ("adjoint", lambda: operator.adjoint(data)),
        (
            "eight_step_cg",
            lambda: reconstruct_born(operator, data, window, 8, 0.001),
        ),
    ):
        start = time.perf_counter()
        jax.block_until_ready(call())
        row[f"{name}_first_call_seconds"] = time.perf_counter() - start
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            jax.block_until_ready(call())
            timings.append(time.perf_counter() - start)
        row[f"{name}_warm_seconds"] = timings
        checkpoint(f"{name}_timed")
    image = reconstruct_born(operator, data, window, 8, 0.001)
    predicted = operator(image)
    residual = (
        window * operator.adjoint(predicted)
        + 0.001 * jnp.sum(jnp.abs(data) ** 2) * image
        - rhs
    )
    row.update(
        normal_relative_residual=float(
            jnp.linalg.norm(residual) / jnp.linalg.norm(rhs)
        ),
        data_relative_residual=relative_error(predicted, data),
        contrast_relative_error=relative_error(image, delta),
        image_change_relative_to_dense=relative_error(
            image, reference["image"]
        ),
        completed=True,
    )
    if not all(
        np.isfinite(row[key])
        for key in (
            "normal_relative_residual",
            "data_relative_residual",
            "contrast_relative_error",
            "image_change_relative_to_dense",
        )
    ):
        raise RuntimeError("Nonfinite reconstruction")
    checkpoint("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--n", type=int, default=288)
    parser.add_argument("--sources", type=int, default=24)
    parser.add_argument("--receivers", type=int, default=256)
    parser.add_argument("--wavelength-mm", type=float, default=1.25)
    parser.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--slab-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(args)
    else:
        evaluate(args)
