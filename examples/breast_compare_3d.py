"""Compare two breast-phantom measurement matrices entry-wise.

Prints relative errors and (optionally, with ``--plot``) writes figures of
the full measurement matrices and the near-field pattern of a single
transmitter over the sensor cap, for both solvers and their difference.

Usage:
    python examples/breast_compare_3d.py breast_umeas_hps.npz breast_umeas_ngsolve.npz \
        --plot breast_cmp --tx 0
"""

import argparse

import numpy as np


def plot_matrices(ua, ub, names, fname):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    vmax = float(max(np.abs(ua).max(), np.abs(ub).max()))
    for ax, m, title, vm in zip(
        axes,
        [np.abs(ua), np.abs(ub), np.abs(ua - ub)],
        [f"|umeas| {names[0]}", f"|umeas| {names[1]}", "|difference|"],
        [vmax, vmax, None],
    ):
        im = ax.imshow(m, origin="lower", cmap="viridis", vmin=0, vmax=vm)
        ax.set_title(title)
        ax.set_xlabel("tx index")
        ax.set_ylabel("rx index")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(fname, dpi=130)
    print(f"wrote {fname}")


def plot_nearfield_tx(ua, ub, sensors, itx, names, fname):
    """Scatter the per-receiver near field for transmitter ``itx`` on the cap.

    The cap ``|x| = R, y >= offset`` is shown in its ``(x, z)`` projection;
    the transmitter location is marked with a star.
    """
    import matplotlib.pyplot as plt

    fa, fb = ua[:, itx], ub[:, itx]
    fields = [np.real(fa), np.real(fb), np.abs(fa - fb)]
    titles = [
        rf"$\Re\,u^s$ at rx, {names[0]} (tx {itx})",
        rf"$\Re\,u^s$ at rx, {names[1]} (tx {itx})",
        "|difference|",
    ]
    vmax = float(max(np.abs(fields[0]).max(), np.abs(fields[1]).max()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for j, (ax, f, title) in enumerate(zip(axes, fields, titles)):
        if j < 2:
            kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        else:
            kw = dict(cmap="viridis", vmin=0)
        sc = ax.scatter(sensors[:, 0], sensors[:, 2], c=f, s=36, **kw)
        ax.plot(
            sensors[itx, 0],
            sensors[itx, 2],
            "k*",
            markersize=14,
            markerfacecolor="yellow",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.85)
    fig.savefig(fname, dpi=130)
    print(f"wrote {fname}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz_a")
    ap.add_argument("npz_b")
    ap.add_argument(
        "--plot",
        default=None,
        help="prefix for output figures (writes <prefix>_matrices.png and"
        " <prefix>_tx<i>.png)",
    )
    ap.add_argument(
        "--tx",
        type=int,
        nargs="*",
        default=[0],
        help="transmitter indices for the near-field pattern plots",
    )
    ap.add_argument(
        "--names",
        nargs=2,
        default=["HPS+BIE", "ngsolve"],
        help="labels for the two files in plot titles",
    )
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

    if args.plot is not None:
        plot_matrices(ua, ub, args.names, f"{args.plot}_matrices.png")
        sensors = da["sensors"]
        for itx in args.tx:
            plot_nearfield_tx(
                ua, ub, sensors, itx, args.names, f"{args.plot}_tx{itx}.png"
            )


if __name__ == "__main__":
    main()
