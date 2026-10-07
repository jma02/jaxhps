"""Refine each radial-reference control independently of either PDE solver."""

import argparse
import json
from pathlib import Path

import numpy as np

from accuracy_benchmark_3d import radial, relative_error, targets
from mie_3d import mie_scattered_field
from radial_reference_3d import radial_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kappa", nargs="+", type=float, default=[4, 8, 12])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    xyz, weights = targets()
    direction = np.array([0.0, 0.0, 1.0])
    variants = {
        "multipole_48": dict(ell_max=48),
        "ode_tight": dict(rtol=1e-12, atol=1e-14),
        "origin_half": dict(origin_fraction=5e-6),
        "step_half": dict(step_fraction=0.0125),
        "all_tight": dict(
            ell_max=48,
            rtol=1e-12,
            atol=1e-14,
            origin_fraction=5e-6,
            step_fraction=0.0125,
        ),
    }
    rows = []
    for kappa in args.kappa:
        reference = radial_reference(xyz, direction, kappa, radial, 0.6)
        changes = {
            name: relative_error(
                radial_reference(xyz, direction, kappa, radial, 0.6, **kwargs),
                reference,
                weights,
            )
            for name, kwargs in variants.items()
        }
        changes["legacy_reference"] = relative_error(
            mie_scattered_field(xyz, direction, kappa, radial, 0.6),
            reference,
            weights,
        )
        rows.append(dict(kappa=kappa, changes=changes))
        print(rows[-1], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
