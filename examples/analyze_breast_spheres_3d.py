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
    window = arrays["window"]
    count = min(count, len(truth))
    columns = ["Truth", "Linearized Born", "Network-corrected"]
    if len(fits):
        columns.append("Full-wave fit")
    test = results["test"]
    titles = [
        "inclusion contrast",
        f"mean Dice {test['born']['sphere_dice']:.2f} (n={len(truth)})",
        f"mean Dice {test['corrected']['sphere_dice']:.2f} (n={len(truth)})",
    ]
    if len(fits):
        titles.append(
            f"mean Dice {np.mean([f['sphere_dice'] for f in results['fits']]):.2f} "
            f"(n={len(fits)})"
        )
    fig, axes = plt.subplots(
        count,
        len(columns),
        figsize=(2.9 * len(columns), 2.5 * count),
        layout="constrained",
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
            if col == 3 and row >= len(fits):
                ax.text(
                    0.5, 0.5, "Not fitted", ha="center", transform=ax.transAxes
                )
            ax.imshow(
                image[:, :, centre[2]].T,
                cmap="RdBu",
                vmin=-limit,
                vmax=limit,
                origin="lower",
            )
            ax.contour(
                window[:, :, centre[2]].T,
                levels=[0.5],
                colors=["0.45"],
                linewidths=0.6,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"Held-out {row + 1}")
            if row == 0:
                ax.set_title(f"{columns[col]}\n{titles[col]}", fontsize=9)
    params = results["parameters"]
    fig.suptitle(
        f"Held-out samples, {params['wavelength_mm']:g} mm wavelength, "
        f"{params['n']}³ grid, {params['train']} training samples\n"
        "Noiseless data; known skin/tissue background. "
        "Slices pass through each sample's strongest inclusion.",
        fontsize=10,
    )
    fig.colorbar(
        axes[0, 0].images[0],
        ax=axes.ravel().tolist(),
        shrink=0.65,
        label="Inclusion contrast (background subtracted)",
    )
    fig.savefig(out / "spheres.png", dpi=150)
    fig.savefig(out / "spheres.pdf")
    plt.close(fig)

    history = results["history"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    axes[0].semilogy(
        [record["epoch"] + 1 for record in history],
        [record["loss"] for record in history],
        color="#0072B2",
    )
    axes[0].set(
        xlabel="Epoch",
        ylabel="Relative squared image error",
        title="Training loss (mean over batches)",
    )
    errors = [
        np.linalg.norm((image - truth).reshape(len(truth), -1), axis=1)
        / np.linalg.norm(truth.reshape(len(truth), -1), axis=1)
        for image in (born, corrected)
    ]
    axes[1].scatter(*errors, color="#0072B2", alpha=0.75)
    limit = float(np.max(errors)) * 1.05
    axes[1].plot([0, limit], [0, limit], color="0.5", linestyle="--")
    axes[1].set(
        xlim=(0, limit),
        ylim=(0, limit),
        xlabel="Born relative image error",
        ylabel="Network relative image error",
        title=f"All {len(truth)} held-out phantoms (below line = improved)",
        aspect="equal",
    )
    fig.savefig(out / "training.png", dpi=150)
    fig.savefig(out / "training.pdf")
    plt.close(fig)
    print(json.dumps(test, indent=1))
    for record in results["fits"]:
        print("fit", json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    main(args.out, args.count)
