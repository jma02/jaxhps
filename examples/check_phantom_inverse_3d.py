"""Check an existing fit on unused directions and independently refined grids."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import map_coordinates

from phantom_inverse_3d import crop, make_forward, sphere


def check(fitted: Path, out: Path, waves: float, grids: list[int]) -> None:
    saved = np.load(fitted)
    reconstruction = saved["reconstruction"]
    n_fit = reconstruction.shape[0]
    directions = sphere(9, 0.617)
    receivers = 0.36 * sphere(128, 0.893)
    rows = []
    fields = []
    for n in grids:
        points, _, truth = crop(n)
        coordinates = np.moveaxis((points + 0.2) * n_fit / 0.4 - 0.5, -1, 0)
        interpolated = map_coordinates(
            reconstruction, coordinates, order=1, mode="constant"
        )
        forward = make_forward(
            points, np.pi * waves / 0.2, directions, receivers, 1e-6
        )
        observed, truth_residual = forward(jnp.asarray(truth))
        prediction, fit_residual = forward(jnp.asarray(interpolated))
        maximum = float(
            jnp.maximum(jnp.max(truth_residual), jnp.max(fit_residual))
        )
        if not np.isfinite(maximum) or maximum > 5e-6:
            raise RuntimeError(f"Validation solve residual {maximum}")
        data = np.asarray(observed)
        fields.append(data)
        row = dict(
            n=n,
            relative_holdout_error=float(
                jnp.linalg.norm(prediction - observed)
                / jnp.linalg.norm(observed)
            ),
            max_solve_residual=maximum,
        )
        if len(fields) > 1:
            row["truth_change_from_previous_grid"] = float(
                np.linalg.norm(fields[-1] - fields[-2])
                / np.linalg.norm(fields[-1])
            )
        rows.append(row)
        print(json.dumps(row), flush=True)
        jax.clear_caches()
    out.write_text(
        json.dumps(
            dict(waves=waves, sources=9, receivers=128, grids=rows), indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--waves", type=float, default=12)
    parser.add_argument("--grids", type=int, nargs="+", default=[64, 96, 128])
    args = parser.parse_args()
    check(args.fit, args.out, args.waves, args.grids)
