"""Plot held-out sphere-inclusion reconstructions from breast_spheres_learn_3d."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(out: Path, count: int) -> None:
    results = json.loads((out / "results.json").read_text())
    arrays = np.load(out / "images.npz")
    truth, born, corrected = (
        arrays["truth"],
        arrays["born"],
        arrays["corrected"],
    )
    fits = arrays["fits"]
    count = min(count, len(truth))
    columns = ["Truth", "Linearized Born", "Network-corrected"]
    if len(fits):
        columns.append("Full-wave fit")
    test = results["test"]
    titles = [
        "inclusion contrast",
        f"Dice {test['born']['sphere_dice']:.2f}",
        f"Dice {test['corrected']['sphere_dice']:.2f}",
    ]
    if len(fits):
        titles.append(
            f"Dice {np.mean([f['sphere_dice'] for f in results['fits']]):.2f}"
        )
    fig, axes = plt.subplots(
        count, len(columns), figsize=(2.6 * len(columns), 2.4 * count)
    )
    axes = np.atleast_2d(axes)
    limit = float(np.abs(truth).max())
    for row in range(count):
        images = [truth[row], born[row], corrected[row]]
        if len(fits):
            images.append(
                fits[row]
                if row < len(fits)
                else np.full_like(truth[row], np.nan)
            )
        centre = np.unravel_index(np.argmin(truth[row]), truth[row].shape)
        for col, image in enumerate(images):
            ax = axes[row, col]
            ax.imshow(
                image[:, :, centre[2]].T,
                cmap="RdBu",
                vmin=-limit,
                vmax=limit,
                origin="lower",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{columns[col]}\n{titles[col]}", fontsize=9)
    params = results["parameters"]
    fig.suptitle(
        f"Held-out samples, {params['wavelength_mm']:g} mm wavelength, "
        f"{params['n']}³ grid, {params['train']} training samples",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "spheres.png", dpi=150)
    fig.savefig(out / "spheres.pdf")
    print(json.dumps(test, indent=1))
    for record in results["fits"]:
        print("fit", json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    main(args.out, args.count)
