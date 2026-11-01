"""Nested acquisition study with exact Born data and unused source/receiver pairs.

The background is fixed and known. Fourier coefficients are fitted using only
the acquisition receivers; held-out predictions use uncompressed fields.
"""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from born_capacity_benchmark_3d import capacity_problem
from breast_spheres_learn_3d import reconstruct_born
from lucka_phantom_3d import vascular_ellipsoids
from phantom_inverse_3d import (
    BackgroundBorn,
    make_background_field_solver,
    make_forward,
)
from spectral_born import SpectralBorn, spectral_indices, spectral_taper
from vascular_phantom_device import _sample


def acquisition_indices(count: int, held_out: int) -> np.ndarray:
    """Put fitting positions first, followed by interleaved unused positions."""
    total = count + held_out
    held = ((np.arange(held_out) + 0.5) * total / held_out).astype(int)
    fit = np.setdiff1d(np.arange(total), held)
    return np.concatenate((fit, held))


def subset_indices(total: int, count: int) -> np.ndarray:
    return (np.arange(count) * total // count).astype(int)


@jax.jit
def dense_signals(
    sources: jax.Array, receivers: jax.Array, images: jax.Array
) -> jax.Array:
    return jax.lax.map(
        lambda source: (images * source) @ receivers.T, sources
    ).transpose(1, 0, 2)


@jax.jit
def dense_adjoint(
    sources: jax.Array, receivers: jax.Array, data: jax.Array
) -> jax.Array:
    def step(i: int, image: jax.Array) -> jax.Array:
        return image + jnp.real(sources[i] * (data[i].conj() @ receivers))

    return jax.lax.fori_loop(
        0, len(sources), step, jnp.zeros(sources.shape[1], jnp.float32)
    )


def relative_error(actual: Any, expected: Any) -> float:
    return float(
        jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)
    )


def checkpoint(row: dict, stage: str, out: Path) -> None:
    entry = dict(stage=stage, memory=jax.devices()[0].memory_stats())
    row.setdefault("stages", []).append(entry)
    out.write_text(json.dumps(row, indent=2))
    print(json.dumps(entry), flush=True)


def prepare(args: argparse.Namespace) -> None:
    out = args.directory
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    row = dict(
        n=args.n,
        sources=args.sources,
        receivers=args.receivers,
        held_sources=args.held_sources,
        held_receivers=args.held_receivers,
        wavelength_mm=args.wavelength_mm,
        seeds=args.seeds,
        box_mm=100,
        device=jax.devices()[0].device_kind,
        background="known vessel-free tissue, seed 7",
        geometry="plane-wave sources and lower-cap receivers, radius 100 mm",
        noise=0,
        tolerance=args.tolerance,
        taper_floor=0.01,
        completed=False,
    )
    points, kappa, directions, targets, truth, background, window = (
        capacity_problem(
            args.n,
            args.wavelength_mm,
            args.sources + args.held_sources,
            args.receivers + args.held_receivers,
        )
    )
    directions = directions[
        acquisition_indices(args.sources, args.held_sources)
    ]
    targets = targets[acquisition_indices(args.receivers, args.held_receivers)]
    originals = []
    for seed in args.seeds:
        if seed == 7:
            originals.append(np.asarray(truth))
            continue
        jax.config.update("jax_enable_x64", True)
        families = (
            vascular_ellipsoids("dense", 7)[0],
            vascular_ellipsoids("dense", seed)[1],
        )
        fibro, vessels = (
            tuple(
                jnp.asarray(np.stack(values))
                for values in zip(
                    *(shape for group in family for shape in group)
                )
            )
            for family in families
        )
        originals.append(
            np.asarray(
                _sample(jnp.asarray(points), 1.25, fibro, vessels, 8192),
                dtype=np.float32,
            )
        )
        del fibro, vessels
        jax.config.update("jax_enable_x64", False)
    del truth
    background_cpu = np.asarray(background)
    deltas = np.stack([x - background_cpu for x in originals])
    for case, delta in enumerate(deltas):
        np.save(out / f"delta-{case}.npy", delta)
    np.save(out / "background.npy", background_cpu)
    np.save(out / "window.npy", np.asarray(window))
    np.savez(
        out / "geometry.npz",
        axis_mm=points[:, 0, 0, 0] * 40,
        directions=directions,
        receivers_mm=targets * 40,
    )
    forward = make_forward(points, kappa, directions, targets, 1e-6)
    fullwave = []
    row["forward"] = []
    for name, coefficient in [
        ("background", background_cpu),
        *list(zip(args.seeds, originals)),
    ]:
        start = time.perf_counter()
        data, residuals = forward(jnp.asarray(coefficient))
        data = np.asarray(data)
        residual = float(jnp.max(residuals))
        if not np.isfinite(residual) or residual > 5e-6:
            raise RuntimeError(f"Full-wave residual: {residual}")
        row["forward"].append(
            dict(
                case=name,
                seconds=time.perf_counter() - start,
                residual=residual,
            )
        )
        fullwave.append(data)
        checkpoint(row, f"fullwave_{name}", out / "prepare.json")
        del data, coefficient, residuals
    np.savez(
        out / "fullwave.npz",
        data=np.stack(fullwave[1:]) - fullwave[0],
        reference=fullwave[0],
    )
    del forward, originals
    jax.clear_caches()
    gc.collect()

    start = time.perf_counter()
    solve = make_background_field_solver(points, kappa, background)
    source_file = np.lib.format.open_memmap(
        out / "sources.npy",
        mode="w+",
        dtype=np.complex64,
        shape=(len(directions), args.n**3),
    )
    for first in range(0, len(directions), args.field_batch):
        last = min(first + args.field_batch, len(directions))
        fields, _ = solve(directions[first:last], False)
        source_file[first:last] = np.asarray(fields)
        print(f"Source fields: {last}/{len(directions)}", flush=True)
    source_file.flush()
    sources = jnp.asarray(source_file)
    taper = spectral_taper((args.n, args.n, args.n), 0.01)
    scaled = np.lib.format.open_memmap(
        out / "scaled-sources.npy",
        mode="w+",
        dtype=np.complex64,
        shape=(args.sources, args.n**3),
    )
    for i in range(args.sources):
        scaled[i] = np.asarray(sources[i] / taper.ravel())
    scaled.flush()
    spectrum = np.lib.format.open_memmap(
        out / "spectrum.npy",
        mode="w+",
        dtype=np.complex64,
        shape=(args.receivers, args.n**3),
    )
    held_fields = np.lib.format.open_memmap(
        out / "held-receivers.npy",
        mode="w+",
        dtype=np.complex64,
        shape=(args.held_receivers, args.n**3),
    )
    random_image = window * jax.random.normal(
        jax.random.PRNGKey(3), window.shape
    )
    images = jnp.asarray(
        np.concatenate((deltas, np.asarray(random_image)[None])).reshape(
            len(deltas) + 1, -1
        )
    )
    probe = jax.random.normal(
        jax.random.PRNGKey(2),
        (args.sources, args.receivers),
        dtype=jnp.complex64,
    )
    signals = np.empty(
        (len(images), len(directions), len(targets)), np.complex64
    )
    energy = jnp.zeros(args.n**3, jnp.float32)
    adjoint = jnp.zeros(args.n**3, jnp.float32)
    normal = jnp.zeros_like(adjoint)
    for first in range(0, len(targets), args.field_batch):
        last = min(first + args.field_batch, len(targets))
        fields, _ = solve(targets[first:last], True)
        values = dense_signals(sources, fields, images)
        signals[:, :, first:last] = np.asarray(values)
        fit_last = min(last, args.receivers)
        if first < fit_last:
            count = fit_last - first
            transformed = jnp.fft.fftn(
                fields[:count].reshape(-1, *background.shape) * taper,
                axes=(-3, -2, -1),
                norm="ortho",
            ).reshape(count, -1)
            spectrum[first:fit_last] = np.asarray(transformed)
            energy += jnp.sum(jnp.abs(transformed) ** 2, axis=0)
            adjoint += dense_adjoint(
                sources[: args.sources],
                fields[:count],
                probe[:, first:fit_last],
            )
            normal += dense_adjoint(
                sources[: args.sources],
                fields[:count],
                values[-1, : args.sources, :count],
            )
        if last > args.receivers:
            held_first = max(first, args.receivers)
            held_fields[
                held_first - args.receivers : last - args.receivers
            ] = np.asarray(fields[held_first - first :])
        if first % 32 == 0 or last == len(targets):
            jax.block_until_ready((normal, adjoint, energy))
            print(f"Receiver fields: {last}/{len(targets)}", flush=True)
    spectrum.flush()
    held_fields.flush()
    row["field_preparation_validation_seconds"] = time.perf_counter() - start
    np.savez(
        out / "dense-reference.npz", signals=signals, probe=np.asarray(probe)
    )
    for name, value in (
        ("random-image", random_image),
        ("adjoint", window.ravel() * adjoint),
        ("normal", window.ravel() * normal),
    ):
        np.save(
            out / f"{name}.npy", np.asarray(value).reshape(background.shape)
        )
    indices = spectral_indices(np.asarray(energy), args.tolerance)
    np.save(out / "indices.npy", indices)
    checkpoint(row, "fields_ready", out / "prepare.json")
    start = time.perf_counter()
    coefficients = np.lib.format.open_memmap(
        out / "coefficients.npy",
        mode="w+",
        dtype=np.complex64,
        shape=(args.receivers, len(indices)),
    )
    for i, field in enumerate(spectrum):
        coefficients[i] = field[indices]
    coefficients.flush()
    row.update(
        compression_seconds=time.perf_counter() - start,
        retained_modes=len(indices),
        field_storage_bytes=scaled.nbytes
        + coefficients.nbytes
        + indices.nbytes,
        seconds=time.perf_counter() - started,
        completed=True,
    )
    checkpoint(row, "finished", out / "prepare.json")


def converged_inverse(
    operator: SpectralBorn,
    data: jax.Array,
    window: jax.Array,
) -> tuple[jax.Array, dict]:
    rhs = window * operator.adjoint(data)
    penalty = 0.001 * jnp.sum(jnp.abs(data) ** 2)
    start = time.perf_counter()
    for iterations in (8, 16, 32, 64):
        image = reconstruct_born(
            operator, data, window, iterations, 0.001
        ).block_until_ready()
        residual = (
            window * operator.adjoint(operator(image)) + penalty * image - rhs
        )
        error = float(jnp.linalg.norm(residual) / jnp.linalg.norm(rhs))
        if np.isfinite(error) and error < 1e-4:
            cold_seconds = time.perf_counter() - start
            start = time.perf_counter()
            image = reconstruct_born(
                operator, data, window, iterations, 0.001
            ).block_until_ready()
            return image, dict(
                iterations=iterations,
                normal_residual=error,
                cold_search_seconds=cold_seconds,
                warm_cg_seconds=time.perf_counter() - start,
                penalty=float(penalty),
            )
    raise RuntimeError(f"CG did not converge: {error}")


def evaluate(args: argparse.Namespace) -> None:
    out = args.directory
    setup = json.loads((out / "prepare.json").read_text())
    s = subset_indices(setup["sources"], args.sources)
    r = subset_indices(setup["receivers"], args.receivers)
    shape = (setup["n"],) * 3
    operator = SpectralBorn(
        jnp.asarray(np.load(out / "scaled-sources.npy", mmap_mode="r")[s]),
        jnp.asarray(np.load(out / "coefficients.npy", mmap_mode="r")[r]),
        jnp.asarray(np.load(out / "indices.npy")),
        shape,
        np.dtype("float32"),
        args.source_batch,
        "fft",
    )
    window = jnp.asarray(np.load(out / "window.npy"))
    row = dict(
        sources=args.sources,
        receivers=args.receivers,
        field_storage_bytes=operator.nbytes,
        damping=0.001,
        cases=[],
        completed=False,
    )
    label = f"{args.sources}-{args.receivers}"
    report = out / f"evaluation-{label}.json"
    checkpoint(row, "loaded_operator", report)
    with np.load(out / "dense-reference.npz") as stored:
        signals = stored["signals"][:, s][:, :, r]
        probe = jnp.asarray(stored["probe"][np.ix_(s, r)])
    random_image = jnp.asarray(np.load(out / "random-image.npy"))
    random_data = operator(random_image)
    row["masked_random_forward_error"] = relative_error(
        random_data, signals[-1]
    )
    if (
        args.sources == setup["sources"]
        and args.receivers == setup["receivers"]
    ):
        row["masked_adjoint_error"] = relative_error(
            window * operator.adjoint(probe),
            jnp.asarray(np.load(out / "adjoint.npy")),
        )
        row["masked_normal_error"] = relative_error(
            window * operator.adjoint(random_data),
            jnp.asarray(np.load(out / "normal.npy")),
        )
    lhs = jnp.real(jnp.vdot(random_data, probe))
    rhs = jnp.vdot(random_image, operator.adjoint(probe))
    row["adjoint_identity_error"] = float(
        jnp.abs(lhs - rhs)
        / jnp.maximum(jnp.maximum(jnp.abs(lhs), jnp.abs(rhs)), 1e-20)
    )
    checkpoint(row, "operator_checked", report)
    with np.load(out / "fullwave.npz") as stored:
        fullwave = stored["data"][:, s][:, :, r]
    arrays = {}
    for case, seed in enumerate(setup["seeds"]):
        delta = jnp.asarray(np.load(out / f"delta-{case}.npy"))
        truth = np.asarray(delta)
        indices = [
            int(
                np.argmax(
                    np.sum(truth**2, axis=tuple(j for j in range(3) if j != a))
                )
            )
            for a in range(3)
        ]
        case_row = dict(
            seed=seed,
            slice_indices=indices,
            born_model_error=relative_error(signals[case], fullwave[case]),
            vessel_signal_error=relative_error(operator(delta), signals[case]),
        )
        arrays[f"case{case}_delta"] = np.stack(
            [np.take(truth, i, axis=a) for a, i in enumerate(indices)]
        )
        arrays[f"case{case}_mip_delta"] = np.max(truth, axis=2)
        for kind, measured in (
            ("born", signals[case]),
            ("fullwave", fullwave[case]),
        ):
            data = jnp.asarray(measured)
            image, metric = converged_inverse(operator, data, window)
            predicted = operator(image)
            metric.update(
                data_residual=relative_error(predicted, data),
                contrast_error=relative_error(image, delta),
                norm_ratio=float(
                    jnp.linalg.norm(image) / jnp.linalg.norm(delta)
                ),
                peak_abs=float(jnp.max(jnp.abs(image))),
            )
            case_row[kind] = metric
            image_cpu = np.asarray(image)
            if not np.isfinite(image_cpu).all():
                raise RuntimeError("Nonfinite reconstruction")
            np.save(out / f"image-{label}-{case}-{kind}.npy", image_cpu)
            arrays[f"case{case}_{kind}"] = np.stack(
                [np.take(image_cpu, i, axis=a) for a, i in enumerate(indices)]
            )
            arrays[f"case{case}_mip_{kind}"] = np.max(
                np.abs(image_cpu), axis=2
            )
            arrays[f"case{case}_data_{kind}"] = measured
            arrays[f"case{case}_predicted_{kind}"] = np.asarray(predicted)
            checkpoint(row, f"inverse_{seed}_{kind}", report)
            print(json.dumps(metric), flush=True)
        row["cases"].append(case_row)
    np.savez_compressed(out / f"{label}-slices.npz", **arrays)
    row["completed"] = True
    checkpoint(row, "finished", report)


def heldout(args: argparse.Namespace) -> None:
    out = args.directory
    setup = json.loads((out / "prepare.json").read_text())
    s, r = setup["sources"], setup["receivers"]
    shape = (setup["n"],) * 3
    operator = BackgroundBorn(
        jnp.asarray(np.load(out / "sources.npy", mmap_mode="r")[s:]),
        jnp.asarray(np.load(out / "held-receivers.npy", mmap_mode="r")),
        shape,
        np.dtype("float32"),
    )
    with np.load(out / "dense-reference.npz") as stored:
        born = stored["signals"][:-1, s:, r:]
    with np.load(out / "fullwave.npz") as stored:
        fullwave = stored["data"][:, s:, r:]
        reference = stored["reference"][s:, r:]
    with np.load(out / "geometry.npz") as geometry:
        axis = geometry["axis_mm"] / 40
        points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)
        forward = make_forward(
            points,
            2 * np.pi / (setup["wavelength_mm"] / 40),
            geometry["directions"][s:],
            geometry["receivers_mm"][r:] / 40,
        )
    background = jnp.asarray(np.load(out / "background.npy"))
    for path in sorted(out.glob("evaluation-*.json")):
        row = json.loads(path.read_text())
        label = f"{row['sources']}-{row['receivers']}"
        arrays = {}
        for case, case_row in enumerate(row["cases"]):
            for kind, measured in (
                ("born", born[case]),
                ("fullwave", fullwave[case]),
            ):
                image = jnp.asarray(
                    np.load(out / f"image-{label}-{case}-{kind}.npy")
                )
                predicted = operator(image)
                case_row[kind]["heldout_born_residual"] = relative_error(
                    predicted, measured
                )
                arrays[f"case{case}_{kind}_data"] = measured
                arrays[f"case{case}_{kind}_born_prediction"] = np.asarray(
                    predicted
                )
                if kind == "fullwave":
                    predicted, residuals = forward(background + image)
                    residual = float(jnp.max(residuals))
                    if not np.isfinite(residual) or residual > 5e-6:
                        raise RuntimeError(
                            f"Held-out nonlinear solve residual: {residual}"
                        )
                    predicted = predicted - reference
                    case_row[kind]["heldout_fullwave_residual"] = (
                        relative_error(predicted, measured)
                    )
                    case_row[kind]["heldout_forward_residual"] = residual
                    arrays[f"case{case}_{kind}_fullwave_prediction"] = (
                        np.asarray(predicted)
                    )
            checkpoint(row, f"heldout_{case_row['seed']}", path)
        np.savez_compressed(out / f"{label}-held-slices.npz", **arrays)
        row["heldout_completed"] = True
        checkpoint(row, "heldout_finished", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "evaluate", "heldout"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--n", type=int, default=288)
    parser.add_argument("--sources", type=int, default=64)
    parser.add_argument("--receivers", type=int, default=512)
    parser.add_argument("--held-sources", type=int, default=8)
    parser.add_argument("--held-receivers", type=int, default=64)
    parser.add_argument(
        "--seeds",
        type=lambda x: [int(s) for s in x.split(",")],
        default=[7, 11, 19],
    )
    parser.add_argument("--wavelength-mm", type=float, default=1.25)
    parser.add_argument("--field-batch", type=int, default=4)
    parser.add_argument("--source-batch", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=0.003)
    args = parser.parse_args()
    if (
        min(
            args.n,
            args.sources,
            args.receivers,
            args.held_sources,
            args.held_receivers,
            args.field_batch,
            args.source_batch,
        )
        < 1
    ):
        parser.error("counts must be positive")
    if args.stage == "evaluate":
        setup = json.loads((args.directory / "prepare.json").read_text())
        if (
            args.sources > setup["sources"]
            or args.receivers > setup["receivers"]
        ):
            parser.error("evaluation counts exceed prepared acquisition")
    jax.config.update("jax_enable_x64", False)
    {"prepare": prepare, "evaluate": evaluate, "heldout": heldout}[args.stage](
        args
    )
