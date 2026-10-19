"""Plot whole-breast full-wave reconstructions against the phantom."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads((args.out / "results.json").read_text())
    stages = results["stages"]
    fig, axes = plt.subplots(
        2, 1 + len(stages), figsize=(3.4 * (1 + len(stages)), 6.4)
    )
    final = np.load(args.out / f"stage-{stages[-1]['waves']:g}.npz")
    axis, truth, core = final["axis"], final["truth"], final["core"]
    mid = truth.shape[1] // 2
    extent = [axis[0] * 80, axis[-1] * 80, axis[0] * 80, axis[-1] * 80]
    vmax = float(truth.max())
    panels = [("Phantom", truth)]
    for stage in stages:
        image = np.load(args.out / f"stage-{stage['waves']:g}.npz")[
            "reconstruction"
        ]
        if image.shape != truth.shape:
            image = zoom(image, truth.shape[0] / image.shape[0], order=0)
        panels.append(
            (
                f"{stage['wavelength_mm']:.0f} mm wavelength, {stage['n']}³\n"
                f"image error {100 * stage['relative_image_error']:.1f}%, "
                f"core Dice {stage['core_dice']:.2f}",
                image,
            )
        )
    for column, (title, image) in enumerate(panels):
        axes[0, column].imshow(
            image[:, mid, :].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=extent,
        )
        axes[0, column].set_title(title, fontsize=9)
        axes[1, column].imshow(
            np.max(np.where(core, image, 0), axis=1).T,
            origin="lower",
            cmap="magma",
            vmin=0,
            vmax=vmax,
            extent=extent,
        )
        axes[0, column].set_xticks([])
        axes[1, column].set_xticks([])
        axes[0, column].set_yticks([])
        axes[1, column].set_yticks([])
    axes[0, 0].set_ylabel("Mid-coronal slice (x–z, mm)")
    axes[1, 0].set_ylabel("Vessel-core maximum projection")
    fig.suptitle(
        "Whole-breast full-wave fit from exterior measurements, starting from water",
        fontsize=11,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.out / f"breast.{suffix}", dpi=160)
    for stage in stages:
        print(
            f"{stage['wavelength_mm']:.0f} mm n={stage['n']} evals={stage['evaluations']} "
            f"image_error={stage['relative_image_error']:.3f} dice={stage['core_dice']:.3f} "
            f"precision={stage['core_precision']:.3f} recall={stage['core_recall']:.3f} "
            f"data_error={stage['relative_data_error']:.4f} seconds={stage['seconds']:.0f}"
        )


if __name__ == "__main__":
    main()
