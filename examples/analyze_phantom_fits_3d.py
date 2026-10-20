"""Plot independent full-wave fits of held-out vascular crops."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main(run: Path, count: int) -> None:
    results = json.loads((run / "results.json").read_text())
    saved = np.load(run / "reconstructions.npz")
    truth, images = saved["truth"], saved["reconstructions"][-1]
    interior = saved["window"] > 0.99
    dice = np.array([c["stages"][-1]["core_dice"] for c in results["crops"]])
    order = np.argsort(dice)
    shown = order[np.linspace(0, len(order) - 1, count).astype(int)]
    fig = plt.figure(figsize=(1.9 * count + 3.2, 4.1), constrained_layout=True)
    grid = fig.add_gridspec(
        2, count + 2, width_ratios=[1] * count + [0.15, 1.6]
    )
    vmax = float(np.max(truth * interior))
    for column, index in enumerate(shown):
        for row, (label, volume) in enumerate(
            (("Target", truth), ("Full-wave fit", images))
        ):
            axis = fig.add_subplot(grid[row, column])
            axis.imshow(
                np.max(volume[index] * interior, axis=2).T,
                cmap="magma",
                vmin=0,
                vmax=vmax,
                origin="lower",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(label)
            if row == 1:
                axis.set_xlabel(f"Dice {dice[index]:.2f}")
    histogram = fig.add_subplot(grid[:, -1])
    histogram.hist(dice, bins=np.linspace(0, 1, 11), color="#7a1f5c")
    histogram.set_xlabel("vessel-core Dice")
    histogram.set_ylabel(f"held-out crops (of {len(dice)})")
    histogram.set_title(f"median {np.median(dice):.2f}", fontsize=10)
    fig.suptitle(
        "Independent full-wave fits on held-out crops, "
        f"{16 / results['parameters']['waves'][-1]:.1f} mm wavelength",
        fontsize=11,
    )
    fig.savefig(run / "fits.png", dpi=160)
    fig.savefig(run / "fits.pdf")
    for waves, scores in results["summary"].items():
        print(
            f"{waves:>4s} waves "
            + " ".join(f"{k}={v:.3f}" for k, v in scores.items())
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()
    main(args.run, args.count)
