"""Plot geometry, field refinement, and A100 measurements without GPU work."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

from lucka_phantom_3d import vascular_geometry, vascular_phantom


def summarize(raw):
    rows = []
    previous = {}
    for data in sorted(raw, key=lambda r: (r["variant"], r["waves"], r["n"])):
        memory = data.get("device_memory") or {}
        stats = data.get("solve") or {}
        complete = (
            not data.get("failed")
            and not data.get("skipped")
            and data.get("phase") == "complete"
            and data.get("converged")
            and "field_real" in data
            and stats.get("final_rel_res", 1) <= 1e-8
        )
        row = dict(
            variant=data["variant"],
            waves=data["waves"],
            n=data["n"],
            device=data.get("environment", {}).get("device"),
            ppw=data["n"] / data["waves"],
            status="converged"
            if complete
            else (
                "skipped"
                if data.get("skipped")
                else "failed/timeout"
                if data.get("failed")
                else "unconverged"
            ),
            residual=stats.get("final_rel_res"),
            iterations=stats.get("n_iter"),
            setup_s=data.get("setup_seconds"),
            solve_s=data.get("solve_seconds"),
            evaluation_s=data.get("evaluation_seconds"),
            process_s=data.get("process_seconds"),
            peak_device_gib=memory["peak_bytes_in_use"] / 2**30
            if "peak_bytes_in_use" in memory
            else None,
            receiver_change=None,
            previous_n=None,
        )
        if complete:
            field = np.asarray(data["field_real"]) + 1j * np.asarray(
                data["field_imag"]
            )
            key = (data["variant"], data["waves"])
            if key in previous:
                old = previous[key]
                for parameter in ("seed", "a", "kappa"):
                    if (
                        old["parameters"][parameter]
                        != data["parameters"][parameter]
                    ):
                        raise ValueError(f"incompatible {parameter}")
                if not np.array_equal(old["receivers"], data["receivers"]):
                    raise ValueError("receiver sets differ")
                old_field = np.asarray(old["field_real"]) + 1j * np.asarray(
                    old["field_imag"]
                )
                norm = np.linalg.norm(field)
                if norm == 0 or not np.isfinite(norm):
                    raise ValueError(
                        "reference field norm must be finite and positive"
                    )
                row["receiver_change"] = float(
                    np.linalg.norm(field - old_field) / norm
                )
                row["previous_n"] = old["n"]
            previous[key] = data
        rows.append(row)
    return rows


def geometry_figure():
    fig = plt.figure(figsize=(12, 7), layout="constrained")
    axis = np.linspace(-1.1, 1.1, 401)
    x, z = np.meshgrid(axis, axis)
    norm = TwoSlopeNorm(vmin=-0.047, vcenter=0, vmax=0.174)
    for index, variant in enumerate(("vascular", "dense")):
        for column, fixed_y in enumerate((0, 0.3)):
            ax = fig.add_subplot(2, 3, index * 3 + column + 1)
            points = np.stack([x, np.full_like(x, fixed_y), z], axis=-1)
            b = vascular_phantom(points, variant=variant)
            im = ax.imshow(
                b,
                origin="lower",
                extent=[-1.1, 1.1, -1.1, 1.1],
                cmap="RdBu_r",
                norm=norm,
            )
            ax.set(
                title=f"{variant.capitalize()}: y={fixed_y:g}",
                xlabel="x",
                ylabel="z",
            )
        ax = fig.add_subplot(2, 3, index * 3 + 3, projection="3d")
        branches, lobules = vascular_geometry(variant)
        for start, middle, end, radius in branches:
            t = np.linspace(0, 1, 30)[:, None]
            curve = (
                (1 - t) ** 2 * start + 2 * t * (1 - t) * middle + t**2 * end
            )
            curve[:, 2] += 0.6875
            ax.plot(
                *curve.T, color="#a12a32", linewidth=radius * 65, alpha=0.85
            )
        centres = np.array([c for c, _, _ in lobules]) + [0, 0, 0.6875]
        ax.scatter(*centres.T, color="#197e89", s=12, alpha=0.4)
        ax.set(
            title=f"{len(branches)} branches / {len(lobules)} lobules",
            xlabel="x",
            ylabel="y",
            zlabel="z",
        )
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(22, -65)
    fig.colorbar(
        im, ax=fig.axes[:], shrink=0.7, label="Contrast b = 1 − (c₀/c)²"
    )
    fig.suptitle(
        "Synthetic pendant breast: curved vessels and lobular tissue",
        fontsize=16,
    )
    return fig


def cost_figure(rows):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    colors = {10: "#0072B2", 20: "#D55E00", 40: "#009E73", 80: "#882255"}
    for variant, style in [("vascular", "-o"), ("dense", "--s")]:
        for waves in sorted({r["waves"] for r in rows}):
            subset = [
                r
                for r in rows
                if r["variant"] == variant
                and r["waves"] == waves
                and r["status"] == "converged"
            ]
            for ax, metric in zip(
                axes, ["receiver_change", "process_s", "peak_device_gib"]
            ):
                valid = [r for r in subset if r[metric] is not None]
                if valid:
                    ax.plot(
                        [r["n"] for r in valid],
                        [r[metric] for r in valid],
                        style,
                        color=colors.get(waves, "#555555"),
                        label=f"{variant}, {waves}λ",
                    )
            axes[0].set_yscale("log")
    for ax, title, ylabel in zip(
        axes,
        [
            "Successive-grid field change",
            "Complete process cost",
            "Device allocator peak",
        ],
        ["Relative change at 512 receivers", "Seconds", "GiB"],
    ):
        ax.set(title=title, xlabel="Points per box edge n", ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "A100-80GB · complex128 · one plane wave · fixed receiver samples",
        fontsize=13,
    )
    return fig


def value(x, spec=".2f"):
    return "—" if x is None else format(x, spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = [json.loads(p.read_text()) for p in args.input.glob("*.json")]
    if not raw:
        raise ValueError("no benchmark results")
    rows = summarize(raw)
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2))
    with (args.out / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    with PdfPages(args.out / "vascular-a100-report.pdf") as pdf:
        fig = geometry_figure()
        fig.savefig(args.out / "geometry.png", dpi=170)
        pdf.savefig(fig)
        plt.close(fig)
        fig = cost_figure(rows)
        fig.savefig(args.out / "refinement.png", dpi=170)
        pdf.savefig(fig)
        plt.close(fig)
        for offset in range(0, len(rows), 18):
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.axis("off")
            table_rows = [
                [
                    r["variant"],
                    r["waves"],
                    r["n"],
                    r["status"],
                    value(r["receiver_change"], ".2e"),
                    value(r["residual"], ".1e"),
                    r["iterations"],
                    value(r["setup_s"]),
                    value(r["solve_s"]),
                    value(r["evaluation_s"]),
                    value(r["peak_device_gib"]),
                ]
                for r in rows[offset : offset + 18]
            ]
            table = ax.table(
                cellText=table_rows,
                colLabels=[
                    "Phantom",
                    "Box λ",
                    "n",
                    "Status",
                    "Field Δ",
                    "Residual",
                    "Iters",
                    "Setup s",
                    "Solve s",
                    "Eval s",
                    "GiB",
                ],
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.6)
            ax.set_title("Measured A100 results", fontsize=16)
            fig.text(
                0.07,
                0.04,
                "Field Δ compares successive converged grids; the finest grid is not an independent reference.\n"
                "512 fixed receiver samples do not establish a full-aperture or volume error. First-call JIT costs are included.\n"
                "Procedural C⁴ tissue, lossless scalar physics, one plane wave: no claim of anatomical or clinical validation.\n"
                "GiB is the JAX allocator peak, not total VRAM. Device models: "
                + ", ".join(
                    sorted({r["device"] for r in rows if r["device"]})
                ),
                fontsize=8,
            )
            pdf.savefig(fig)
            plt.close(fig)
        completed = [
            r
            for r in raw
            if r.get("converged")
            and r.get("phase") == "complete"
            and "slice_abs" in r
        ]
        if completed:
            chosen = max(
                completed,
                key=lambda r: (r["waves"], r["n"], r["variant"] == "dense"),
            )
            fig, axes = plt.subplots(
                1, 2, figsize=(12, 5), layout="constrained"
            )
            for ax, key, title, cmap in zip(
                axes,
                ["slice_real", "slice_abs"],
                ["Real total field", "Total field magnitude"],
                ["RdBu_r", "viridis"],
            ):
                im = ax.imshow(
                    np.array(chosen[key]).T,
                    origin="lower",
                    extent=[-1.25, 1.25, -1.25, 1.25],
                    cmap=cmap,
                )
                ax.set(title=title, xlabel="x", ylabel="z")
                fig.colorbar(im, ax=ax, shrink=0.8)
            fig.suptitle(
                f"{chosen['variant'].capitalize()} phantom · {chosen['waves']} wavelengths across box · n={chosen['n']}"
            )
            fig.savefig(args.out / "field.png", dpi=170)
            pdf.savefig(fig)
            plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
