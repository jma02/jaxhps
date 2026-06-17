r"""Static plotly report for the free-space scattering dataset.

Renders, for the dataset produced by ``scattering_dataset_3d.py`` /
``modal_scattering_dataset_3d.py``:

  * the discretization and sampling parameters,
  * the transmitter/receiver arrangement,
  * example scatterer geometries (bump supports and a slice of ``n(x)``),
  * example near-field multistatic matrices ``|U|``, and
  * the radial ``z = 0`` spherical-scatterer sanity test (sorted sine waves).

Inputs are local files pulled from the Hub repo plus the ``--npz_out`` dump of
``sphere_radial_scattering_3d.py``.

Usage:
    python examples/scattering_dataset_site.py \
        --metadata site_data/metadata.json \
        --shards site_data/train-0000000-0000050.parquet ... \
        --radial site_assets/sphere_radial.npz \
        --out site/index.html
"""

import argparse
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def load_rows(shard_paths):
    """Read parquet shards into a list of per-sample dicts."""
    import pyarrow.parquet as pq

    rows = []
    for p in shard_paths:
        t = pq.read_table(p)
        d = t.to_pydict()
        n_tx = int(np.sqrt(len(d["u_real"][0])))
        for i in range(t.num_rows):
            ns = int(d["n_scatterers"][i])
            c = np.array(d["centers"][i], dtype=float).reshape(-1, 3)[:ns]
            r = np.array(d["radii"][i], dtype=float)[:ns]
            a = np.array(d["amps"][i], dtype=float)[:ns]
            M = (
                np.array(d["u_real"][i], dtype=float)
                + 1j * np.array(d["u_imag"][i], dtype=float)
            ).reshape(n_tx, n_tx)
            rows.append(
                dict(
                    n=ns,
                    centers=c,
                    radii=r,
                    amps=a,
                    M=M,
                    idx=int(d["sample_index"][i]),
                )
            )
    return rows


def b_of_x(pts, centers, radii, amps):
    """Volume potential ``b(x) = sum_k A_k (1-(|x-c_k|/R_k)^2)^4`` (compact)."""
    b = np.zeros(pts.shape[:-1])
    for c, R, A in zip(centers, radii, amps):
        d = np.linalg.norm(pts - c, axis=-1) / R
        m = d < 1.0
        b[m] += A * (1.0 - d[m] ** 2) ** 4
    return b


def cube_edges(a):
    """Line traces for the wireframe of ``[-a, a]^3``."""
    v = np.array(
        [[sx, sy, sz] for sx in (-a, a) for sy in (-a, a) for sz in (-a, a)],
        dtype=float,
    )
    E = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]
    xs, ys, zs = [], [], []
    for i, j in E:
        xs += [v[i, 0], v[j, 0], None]
        ys += [v[i, 1], v[j, 1], None]
        zs += [v[i, 2], v[j, 2], None]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color="#444", width=2),
        name=f"cube |x_i| <= {a:g}",
        hoverinfo="name",
    )


def fig_sensors(tx_dirs, rx_pts, rho, a):
    fig = go.Figure()
    fig.add_trace(cube_edges(a))
    fig.add_trace(
        go.Scatter3d(
            x=rx_pts[:, 0],
            y=rx_pts[:, 1],
            z=rx_pts[:, 2],
            mode="markers",
            marker=dict(size=3, color="#1f77b4"),
            name=f"receivers ({len(rx_pts)}), |x| = {rho:g}",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=tx_dirs[:, 0],
            y=tx_dirs[:, 1],
            z=tx_dirs[:, 2],
            mode="markers",
            marker=dict(size=3, color="#d62728"),
            name=f"transmitter directions ({len(tx_dirs)}), |w| = 1",
        )
    )
    fig.update_layout(
        scene=dict(aspectmode="data"),
        legend=dict(orientation="h", yanchor="bottom", y=0.0),
        margin=dict(l=0, r=0, t=10, b=0),
        height=620,
    )
    return fig


def sphere_surf(c, R, n=24):
    th = np.linspace(0, np.pi, n)
    ph = np.linspace(0, 2 * np.pi, 2 * n)
    th, ph = np.meshgrid(th, ph)
    x = c[0] + R * np.sin(th) * np.cos(ph)
    y = c[1] + R * np.sin(th) * np.sin(ph)
    z = c[2] + R * np.cos(th)
    return x, y, z


def fig_phantom(sample, a):
    """Support boundaries ``|x-c_k| = R_k`` for one sample, inside the cube."""
    c, r, A = sample["centers"], sample["radii"], sample["amps"]
    fig = go.Figure()
    fig.add_trace(cube_edges(a))
    for k, (ck, Rk, Ak) in enumerate(zip(c, r, A)):
        xs, ys, zs = sphere_surf(ck, Rk)
        fig.add_trace(
            go.Surface(
                x=xs,
                y=ys,
                z=zs,
                opacity=0.6,
                showscale=False,
                colorscale=[[0, "#b22222"], [1, "#b22222"]],
                name=f"bump {k + 1}: A = {Ak:.2f}, R = {Rk:.2f}",
                hoverinfo="name",
                showlegend=True,
            )
        )
    fig.update_layout(
        scene=dict(aspectmode="data"),
        legend=dict(orientation="h", yanchor="bottom", y=0.0),
        margin=dict(l=0, r=0, t=10, b=0),
        height=560,
    )
    return fig


def fig_slice(sample, a):
    """Slice of ``n(x)`` through the plane of the strongest bump."""
    c, r, A = sample["centers"], sample["radii"], sample["amps"]
    z0 = float(c[int(np.argmax(np.abs(A)))][2])
    ng = 400
    g = np.linspace(-a, a, ng)
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([X, Y, np.full_like(X, z0)], axis=-1)
    n = np.sqrt(np.maximum(1.0 - b_of_x(pts, c, r, A), 0.0))
    fig = go.Figure(
        go.Heatmap(
            x=g, y=g, z=n.T, colorscale="Viridis", colorbar=dict(title="n(x)")
        )
    )
    fig.update_layout(
        title=f"z = {z0:.2f} slice",
        xaxis_title="x",
        yaxis_title="y",
        yaxis=dict(scaleanchor="x"),
        margin=dict(l=40, r=10, t=30, b=40),
        height=460,
    )
    return fig


def fig_matrix_grid(samples):
    """Grid of ``|U|`` heatmaps for several samples."""
    ncol = 4
    nrow = (len(samples) + ncol - 1) // ncol
    titles = [
        f"n={s['n']}, max|A|={np.abs(s['amps']).max():.2f}" for s in samples
    ]
    fig = make_subplots(
        rows=nrow,
        cols=ncol,
        subplot_titles=titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.09,
    )
    for k, s in enumerate(samples):
        rr, cc = k // ncol + 1, k % ncol + 1
        fig.add_trace(
            go.Heatmap(
                z=np.abs(s["M"]), colorscale="Viridis", showscale=False
            ),
            row=rr,
            col=cc,
        )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=240 * nrow)
    return fig


def fig_radial_geometry(rx, R_bump, a):
    fig = go.Figure()
    th = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(
        go.Scatter(
            x=R_bump * np.cos(th),
            y=R_bump * np.sin(th),
            fill="toself",
            fillcolor="rgba(120,120,120,0.4)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"scatterer (r < {R_bump:g})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[-a, a, a, -a, -a],
            y=[-a, -a, a, a, -a],
            mode="lines",
            line=dict(color="#444", dash="dash"),
            name=f"cube |x_i| <= {a:g}",
        )
    )
    for r in rx:
        fig.add_trace(
            go.Scatter(
                x=[0, r[0]],
                y=[0, r[1]],
                mode="lines",
                line=dict(color="rgba(31,119,180,0.35)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=rx[:, 0],
            y=rx[:, 1],
            mode="markers",
            marker=dict(size=6, color="#1f77b4"),
            name=f"tx/rx (rho = {np.linalg.norm(rx[0]):.2f})",
        )
    )
    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        yaxis=dict(scaleanchor="x"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
        margin=dict(l=40, r=10, t=10, b=40),
        height=560,
    )
    return fig


def fig_radial_sorted(M, g, mie, angles, residual, rel_mie):
    deg = np.degrees(angles)
    N = M.shape[0]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Aligned receiver traces (all {N} tx), residual {residual:.1e}",
            f"Sorted pattern g(gamma) vs Mie (rel. L2 {rel_mie:.1e})",
        ),
    )
    for i in range(N):
        fig.add_trace(
            go.Scatter(
                x=deg,
                y=np.real(np.roll(M[:, i], -i)),
                mode="lines",
                line=dict(color="rgba(150,150,150,0.5)", width=1),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=deg,
            y=np.real(g),
            mode="lines",
            line=dict(color="#d62728", width=3),
            name="mean Re g",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=deg,
            y=np.real(g),
            mode="lines",
            line=dict(color="#1f77b4", width=3),
            name="HPS Re g",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=deg,
            y=np.imag(g),
            mode="lines",
            line=dict(color="#ff7f0e", width=3),
            name="HPS Im g",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=deg,
            y=np.real(mie),
            mode="lines",
            line=dict(color="black", width=1.5, dash="dash"),
            name="Mie Re g",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=deg,
            y=np.imag(mie),
            mode="lines",
            line=dict(color="black", width=1.5, dash="dot"),
            name="Mie Im g",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="angular separation gamma (deg)")
    fig.update_layout(margin=dict(l=40, r=10, t=40, b=40), height=440)
    return fig


def fig_radial_matrix(M):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Re u^s", "|u^s|"),
        horizontal_spacing=0.13,
    )
    vm = float(np.abs(np.real(M)).max())
    fig.add_trace(
        go.Heatmap(
            z=np.real(M),
            zmin=-vm,
            zmax=vm,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(x=0.43, len=0.9, thickness=14),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=np.abs(M),
            zmin=0,
            colorscale="Viridis",
            colorbar=dict(x=1.005, len=0.9, thickness=14),
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="transmitter index (angle)")
    fig.update_yaxes(title_text="receiver index (angle)", autorange="reversed")
    fig.update_layout(margin=dict(l=40, r=10, t=40, b=40), height=420)
    return fig


def circulant_residual(M):
    N = M.shape[0]
    aligned = np.stack([np.roll(M[:, i], -i) for i in range(N)], axis=0)
    g = aligned.mean(axis=0)
    return g, float(np.max(np.abs(aligned - g[None, :])))


HEAD = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Free-space Helmholtz scattering dataset</title>
<script src="{PLOTLY_CDN}"></script>
<script>MathJax = {{tex: {{inlineMath: [['$', '$']]}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 1100px;
       margin: 2em auto; padding: 0 1em; color: #222; line-height: 1.5; }}
h1 {{ font-size: 1.6em; }} h2 {{ font-size: 1.25em; margin-top: 2em; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
td, th {{ border: 1px solid #999; padding: 0.3em 0.8em; text-align: left; }}
.fig {{ margin: 1em 0; }} code {{ font-size: 0.95em; }}
pre {{ background: #f6f6f6; padding: 0.8em; overflow-x: auto; }}
</style></head><body>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--radial", default=None)
    ap.add_argument("--repo", default="jma02/helmholtz-scattering-3d")
    ap.add_argument(
        "--n_generated",
        type=int,
        default=None,
        help="number of samples generated so far (for the prose)",
    )
    ap.add_argument("--out", default="dataset_site/index.html")
    args = ap.parse_args()

    meta = json.load(open(args.metadata))
    tx = np.array(meta["tx_dirs"])
    rx = np.array(meta["rx_pts"])
    a = meta["cube_half_width"]
    rows = load_rows(args.shards)

    def div(fig):
        return fig.to_html(full_html=False, include_plotlyjs=False)

    parts = [HEAD]
    ngen = args.n_generated if args.n_generated is not None else len(rows)
    parts.append(f"""
<h1>A free-space Helmholtz scattering dataset</h1>
<p>Multistatic far-/near-field data for the penetrable Helmholtz problem
$\\Delta u + \\kappa^2 n(x)\\,u = 0$ in $\\mathbb{{R}}^3$, $u = u^i + u^s$ with
$u^s$ Sommerfeld-radiating, at fixed wavenumber $\\kappa = {meta["kappa"]:g}$.
The contrast $b = 1 - n^2$ is a random superposition of
$n_s \\in \\{{1,2,3\\}}$ smooth, compactly supported bumps
$b(x) = \\sum_{{k=1}}^{{n_s}} A_k\\,(1 - (|x-c_k|/R_k)^2)^4\\,
\\mathbf 1_{{|x-c_k| < R_k}}$ ($C^3$, so resolvable by the spectral
collocation), with $A_k \\in [{meta["A_range"][0]:g}, {meta["A_range"][1]:g}]$,
$R_k \\in [{meta["R_range"][0]:g}, {meta["R_range"][1]:g}]$ and centers drawn
uniformly in the ball $|c_k| + R_k \\le {meta["c_max"]:g}$. No skin layer or
background structure is imposed; the medium is $n \\equiv 1$ outside the bumps.</p>

<p>The forward map is the HPS (ItI) volume solver coupled to an exterior
boundary integral equation enforcing radiation
(<a href="https://github.com/jma02/jaxhps/pull/3">jaxhps PR&nbsp;#3</a>;
cf. <a href="https://arxiv.org/abs/2503.17535">arXiv:2503.17535</a> and
<a href="https://arxiv.org/abs/1308.5998">arXiv:1308.5998</a>). For each of
$n_{{tx}} = {meta["n_tx"]}$ plane-wave directions $w_i$ on a Fibonacci lattice
of $S^2$ we record $u^s$ at $n_{{rx}} = {meta["n_rx"]}$ receivers $r_j$ on the
sphere $|x| = \\rho = {meta["rho"]:g}$, giving a complex matrix
$U_{{ij}} = u^s(w_i, r_j) \\in \\mathbb{{C}}^{{{meta["n_rx"]}\\times{meta["n_tx"]}}}$
per sample. Each forward solve batches all {meta["n_tx"]} incident directions
as simultaneous right-hand sides.</p>

<p><b>Dataset:</b>
<a href="https://huggingface.co/datasets/{args.repo}">{args.repo}</a>
(parquet, HF <code>datasets</code>). {ngen} samples shown here.</p>
""")

    parts.append(f"""
<h2>Parameters</h2>
<table>
<tr><th>quantity</th><th>symbol</th><th>value</th></tr>
<tr><td>wavenumber</td><td>$\\kappa$</td><td>{meta["kappa"]:g}</td></tr>
<tr><td>HPS cube half-width</td><td>$a$</td><td>{meta["cube_half_width"]:g}</td></tr>
<tr><td>Gauss nodes / leaf (SD)</td><td>$q$</td><td>{meta["q"]}</td></tr>
<tr><td>interior Chebyshev order</td><td>$p$</td><td>{meta["interior_order_p"]}</td></tr>
<tr><td>octree levels</td><td>$L$</td><td>{meta["L"]}</td></tr>
<tr><td>transmitters / receivers</td><td>$n_{{tx}}, n_{{rx}}$</td>
    <td>{meta["n_tx"]}, {meta["n_rx"]}</td></tr>
<tr><td>receiver radius</td><td>$\\rho$</td><td>{meta["rho"]:g}</td></tr>
<tr><td>scatterers / sample</td><td>$n_s$</td>
    <td>$\\{{1,2,3\\}}$</td></tr>
<tr><td>bump amplitude</td><td>$A_k$</td>
    <td>$[{meta["A_range"][0]:g}, {meta["A_range"][1]:g}]$</td></tr>
<tr><td>bump radius</td><td>$R_k$</td>
    <td>$[{meta["R_range"][0]:g}, {meta["R_range"][1]:g}]$</td></tr>
<tr><td>center constraint</td><td></td>
    <td>$|c_k| + R_k \\le {meta["c_max"]:g}$</td></tr>
<tr><td>index convention</td><td></td><td>$n(x)^2 = 1 - b(x)$</td></tr>
</table>
<p>The single/double-layer operators are precomputed once at
$(\\kappa, a, q, L) = ({meta["kappa"]:g}, {meta["cube_half_width"]:g},
{meta["q"]}, {meta["L"]})$ and reused across all samples; only the volume
right-hand side changes per sample.</p>
""")

    parts.append("""
<h2>Sensor arrangement</h2>
<p>Transmitter directions $w_i \\in S^2$ (red) and receivers $r_j$ on
$|x| = \\rho$ (blue); the scatterer support lies in the cube (grey).</p>
""")
    parts.append(
        '<div class="fig">'
        + div(fig_sensors(tx, rx, meta["rho"], a))
        + "</div>"
    )

    # representative sample: most scatterers, then largest support
    rep = max(rows, key=lambda s: (s["n"], float(np.sum(s["radii"]))))
    parts.append(f"""
<h2>Example geometry</h2>
<p>A sample with $n_s = {rep["n"]}$: the bump supports
$|x - c_k| < R_k$ inside the cube, and a slice of $n(x)$ through the
strongest bump. Penetrable inclusions ($A_k < 0$, so $n > 1$).</p>
""")
    parts.append('<div class="fig">' + div(fig_phantom(rep, a)) + "</div>")
    parts.append('<div class="fig">' + div(fig_slice(rep, a)) + "</div>")

    # matrix grid: up to 8 samples spanning scatterer counts
    pick = sorted(rows, key=lambda s: (s["n"], -np.abs(s["amps"]).max()))
    sel = pick[:: max(len(pick) // 8, 1)][:8]
    parts.append("""
<h2>Near-field data</h2>
<p>Magnitude $|U_{ij}|$ of the multistatic matrix for several samples
(rows: receivers; columns: transmitters). Structure reflects scatterer
number, size and position.</p>
""")
    parts.append('<div class="fig">' + div(fig_matrix_grid(sel)) + "</div>")

    if args.radial:
        d = np.load(args.radial)
        M = d["M"]
        angles = d["angles"]
        mie = d["mie"]
        g, residual = circulant_residual(M)
        rel_mie = float(np.linalg.norm(g - mie) / np.linalg.norm(mie))
        R_bump = float(d["R_bump"])
        parts.append(f"""
<h2>Validation: radial scatterer on the $z = 0$ slice</h2>
<p>A sanity check of the same solver. A single spherically symmetric bump
$b(r) = {float(d["A_bump"]):g}\\,(1 - (r/{R_bump:g})^2)^4$ sits at the origin.
In the $z = 0$ plane we take $N = {M.shape[0]}$ in-plane plane-wave
transmitters $w_i = (\\cos\\alpha_i, \\sin\\alpha_i, 0)$ and receivers
$r_j = \\rho(\\cos\\alpha_j, \\sin\\alpha_j, 0)$ at the same angles
$\\alpha_i = 2\\pi i / N$, $\\rho = {float(d["rho"]):g}$.</p>
<p>Rotational invariance about the $z$ axis forces
$u^s(w_i, r_j) = g(\\alpha_j - \\alpha_i)$: sorted by angle the matrix is
circulant, and every receiver trace is the same sinusoid $g$ shifted by its
angle &mdash; the plane-wave sine waves. The aligned traces collapse with
residual ${residual:.1e}$ (vs $|u^s|_{{\\max}} = {np.abs(M).max():.1e}$), and
the recovered $g(\\gamma)$ matches the Mie series to relative $L^2$ error
${rel_mie:.1e}$. The lobe count grows with $\\kappa R$; here
$\\kappa R \\approx {meta["kappa"] * R_bump:.2f}$.</p>
""")
        parts.append(
            '<div class="fig">'
            + div(fig_radial_geometry(d["rx"], R_bump, a))
            + "</div>"
        )
        parts.append(
            '<div class="fig">'
            + div(fig_radial_sorted(M, g, mie, angles, residual, rel_mie))
            + "</div>"
        )
        parts.append(
            '<div class="fig">' + div(fig_radial_matrix(M)) + "</div>"
        )

    parts.append(f"""
<h2>Access</h2>
<p>Each row is an $(X, Y)$ pair: $X$ = near field
(<code>u_real</code>, <code>u_imag</code>, each a length-{
        meta["n_rx"] * meta["n_tx"]
    } vector, row-major
$U_{{ij}} = u^s(w_i, r_j)$); $Y$ = geometry
(<code>n_scatterers</code>, and NaN-padded <code>centers</code>,
<code>radii</code>, <code>amps</code>). Sensor geometry is in
<code>metadata.json</code>.</p>
<pre><code>from datasets import load_dataset
import numpy as np
ds = load_dataset("{args.repo}", split="train")
r = ds[0]
U = (np.array(r["u_real"]) + 1j*np.array(r["u_imag"])).reshape(
    {meta["n_rx"]}, {meta["n_tx"]})</code></pre>
</body></html>
""")

    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("".join(parts))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
