"""Plot held-out Born and corrected reconstructions from the crop dataset."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main(run: Path, count: int) -> None:
    results = json.loads((run / "results.json").read_text())
    images = np.load(run / "test-images.npz")
    truth, born, corrected = (
        images[k] for k in ("truth", "born", "corrected")
    )
    interior = images["window"] > 0.99
    errors = np.linalg.norm(
        (corrected - truth).reshape(len(truth), -1), axis=1
    ) / np.linalg.norm(truth.reshape(len(truth), -1), axis=1)
    order = np.argsort(errors)
    shown = order[np.linspace(0, len(order) - 1, count).astype(int)]
    rows = ("Target", "Born approximation", "Learned correction")
    fig, axes = plt.subplots(
        3, count, figsize=(1.9 * count, 5.8), constrained_layout=True
    )
    vmax = float(np.max(truth[shown] * interior))
    for column, index in enumerate(shown):
        for row, volume in enumerate((truth, born, corrected)):
            axis = axes[row, column]
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
                axis.set_ylabel(rows[row])
        axes[2, column].set_xlabel(f"error {100 * errors[index]:.0f}%")
    test = results["test"]
    fig.suptitle(
        "Held-out crops: vessel-core Dice "
        f"{test['born']['core_dice']:.2f} (Born) vs "
        f"{test['corrected']['core_dice']:.2f} (corrected)",
        fontsize=11,
    )
    fig.savefig(run / "heldout.png", dpi=160)
    fig.savefig(run / "heldout.pdf")
    for split in ("train", "test"):
        for method in ("born", "corrected"):
            scores = results[split][method]
            print(
                f"{split:5s} {method:9s} "
                + " ".join(f"{k}={v:.3f}" for k, v in scores.items())
            )
    print(
        f"test corrected_measurement_error={test['corrected_measurement_error']:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()
    main(args.run, args.count)
