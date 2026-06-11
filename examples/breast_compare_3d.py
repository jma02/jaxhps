"""Compare two breast-phantom measurement matrices entry-wise.

Usage:
    python examples/breast_compare_3d.py breast_umeas_hps.npz breast_umeas_ngsolve.npz
"""

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz_a")
    ap.add_argument("npz_b")
    args = ap.parse_args()

    da, db = np.load(args.npz_a), np.load(args.npz_b)
    ua, ub = da["umeas"], db["umeas"]
    if ua.shape != ub.shape:
        raise SystemExit(f"shape mismatch: {ua.shape} vs {ub.shape}")
    if not np.allclose(da["sensors"], db["sensors"]):
        raise SystemExit("sensor locations differ between the two files")

    diff = ua - ub
    rel_fro = np.linalg.norm(diff) / np.linalg.norm(ub)
    rel_max = np.max(np.abs(diff)) / np.max(np.abs(ub))
    print(f"umeas shape          : {ua.shape}")
    print(
        f"||A||_F, ||B||_F     : {np.linalg.norm(ua):.6e}, {np.linalg.norm(ub):.6e}"
    )
    print(f"rel Frobenius error  : {rel_fro:.6e}")
    print(f"rel max-entry error  : {rel_max:.6e}")


if __name__ == "__main__":
    main()
