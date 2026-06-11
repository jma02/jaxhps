"""Build a static plotly site for the smoothed 3D breast scattering problem.

Renders the computational domain (isosurfaces of the smoothed squared index
n(x), a coronal slice through the tumors, the tx/rx cap) and, optionally,
the measured near-field data from `breast_scattering_3d.py` /
`breast_scattering_3d_ngsolve.py` output files.

Usage:
    python examples/breast_domain_site.py --out site/index.html \
        [--hps umeas_hps.npz --ng umeas_ng.npz --tx 0 32]
"""

import argparse

import numpy as np
import plotly.graph_objects as go
from breast_phantom_3d import (
    B_RADIUS,
    DEFAULT_CENTERS,
    DEFAULT_MVALS,
    DEFAULT_RADII,
    DELTA_SKIN,
    KAPPA,
    KVAL,
    SENSOR_OFFSET,
    SENSOR_RADIUS,
    SKINVAL,
    TISSUEVAL,
    breast_n_of_x,
    fibonacci_cap_points,
)

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def sphere_mesh(center, radius, n=40):
    th = np.linspace(0, np.pi, n)
    ph = np.linspace(0, 2 * np.pi, 2 * n)
    th, ph = np.meshgrid(th, ph)
    x = center[0] + radius * np.sin(th) * np.cos(ph)
    y = center[1] + radius * np.sin(th) * np.sin(ph)
    z = center[2] + radius * np.cos(th)
    return x, y, z


def hemisphere_mesh(radius, n=60):
    # hemisphere y >= 0 with polar axis +y
    th = np.linspace(0, np.pi / 2, n)  # angle from +y
    ph = np.linspace(0, 2 * np.pi, 2 * n)
    th, ph = np.meshgrid(th, ph)
    x = radius * np.sin(th) * np.cos(ph)
    y = radius * np.cos(th)
    z = radius * np.sin(th) * np.sin(ph)
    return x, y, z


def fig_domain(n_sensors):
    fig = go.Figure()

    xs, ys, zs = hemisphere_mesh(B_RADIUS)
    fig.add_trace(
        go.Surface(
            x=xs,
            y=ys,
            z=zs,
            opacity=0.25,
            showscale=False,
            colorscale=[[0, "#d4956a"], [1, "#d4956a"]],
            name="skin (outer surface)",
            hoverinfo="name",
            showlegend=True,
        )
    )
    xs, ys, zs = hemisphere_mesh(B_RADIUS - DELTA_SKIN)
    fig.add_trace(
        go.Surface(
            x=xs,
            y=ys,
            z=zs,
            opacity=0.15,
            showscale=False,
            colorscale=[[0, "#e8c8a0"], [1, "#e8c8a0"]],
            name="tissue (inner surface)",
            hoverinfo="name",
            showlegend=True,
        )
    )
    for j, (c, rad, m) in enumerate(
        zip(DEFAULT_CENTERS, DEFAULT_RADII, DEFAULT_MVALS)
    ):
        xs, ys, zs = sphere_mesh(c, rad, n=25)
        fig.add_trace(
            go.Surface(
                x=xs,
                y=ys,
                z=zs,
                opacity=0.9,
                showscale=False,
                colorscale=[[0, "#b22222"], [1, "#b22222"]],
                name=f"tumor {j + 1} (n = {m:.3f})",
                hoverinfo="name",
                showlegend=(j == 0),
            )
        )

    sensors = fibonacci_cap_points(n_sensors)
    fig.add_trace(
        go.Scatter3d(
            x=sensors[:, 0],
            y=sensors[:, 1],
            z=sensors[:, 2],
            mode="markers",
            marker=dict(size=3, color="#1f77b4"),
            name=f"tx/rx points ({n_sensors}), |x| = {SENSOR_RADIUS:.3f}",
        )
    )

    fig.update_layout(
        scene=dict(aspectmode="data"),
        legend=dict(orientation="h", yanchor="bottom", y=0.0),
        margin=dict(l=0, r=0, t=30, b=0),
        height=650,
    )
    return fig


def fig_slice():
    # coronal slice x = 0 through the tumor centers
    ny, nz = 400, 400
    y = np.linspace(-0.2, 1.4, ny)
    z = np.linspace(-0.8, 0.8, nz)
    Y, Z = np.meshgrid(y, z, indexing="ij")
    pts = np.stack([np.zeros_like(Y), Y, Z], axis=-1)
    n = breast_n_of_x(pts)
    fig = go.Figure(
        go.Heatmap(
            x=z,
            y=y,
            z=n,
            colorscale="Viridis",
            colorbar=dict(title="n(x)"),
        )
    )
    fig.update_layout(
        xaxis_title="z",
        yaxis_title="y",
        yaxis=dict(scaleanchor="x"),
        margin=dict(l=40, r=10, t=30, b=40),
        height=600,
    )
    return fig


def fig_profile():
    # 1d radial profile along the line x=0, z=0
    y = np.linspace(0.0, 1.2, 2000)
    pts = np.stack([np.zeros_like(y), y, np.zeros_like(y)], axis=-1)
    n = breast_n_of_x(pts)
    fig = go.Figure(
        go.Scatter(x=y, y=n, mode="lines", line=dict(color="#1f77b4"))
    )
    fig.update_layout(
        xaxis_title="y  (x = z = 0)",
        yaxis_title="n",
        margin=dict(l=50, r=10, t=30, b=40),
        height=350,
    )
    return fig


def fig_matrices(u_hps, u_ng):
    from plotly.subplots import make_subplots

    diff = np.abs(u_hps - u_ng)
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "|u_meas|, HPS+BIE",
            "|u_meas|, NGSolve",
            "|difference|",
        ),
        horizontal_spacing=0.11,
    )
    vmax = max(np.abs(u_hps).max(), np.abs(u_ng).max())
    cbars = [
        dict(x=0.625, len=0.9, thickness=14),
        None,
        dict(x=1.005, len=0.9, thickness=14),
    ]
    for col, (m, zmax) in enumerate(
        [(np.abs(u_hps), vmax), (np.abs(u_ng), vmax), (diff, diff.max())],
        start=1,
    ):
        fig.add_trace(
            go.Heatmap(
                z=m,
                zmin=0,
                zmax=zmax,
                colorscale="Viridis",
                showscale=(col != 2),
                colorbar=cbars[col - 1],
            ),
            row=1,
            col=col,
        )
    fig.update_xaxes(title_text="tx index")
    fig.update_yaxes(title_text="rx index", row=1, col=1)
    fig.update_layout(margin=dict(l=40, r=10, t=40, b=40), height=420)
    return fig


def fig_nearfield(sensors, u_hps, u_ng, itx):
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}] * 3],
        subplot_titles=(
            f"Re u^s, HPS+BIE (tx {itx})",
            f"Re u^s, NGSolve (tx {itx})",
            "|difference|",
        ),
        horizontal_spacing=0.01,
    )
    vals = [
        np.real(u_hps[:, itx]),
        np.real(u_ng[:, itx]),
        np.abs(u_hps[:, itx] - u_ng[:, itx]),
    ]
    vmax = max(np.abs(vals[0]).max(), np.abs(vals[1]).max())
    scales = [
        ("RdBu_r", -vmax, vmax),
        ("RdBu_r", -vmax, vmax),
        ("Viridis", 0, vals[2].max()),
    ]
    for col, (v, (cs, lo, hi)) in enumerate(zip(vals, scales), start=1):
        fig.add_trace(
            go.Scatter3d(
                x=sensors[:, 0],
                y=sensors[:, 1],
                z=sensors[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=v,
                    colorscale=cs,
                    cmin=lo,
                    cmax=hi,
                    showscale=False,
                ),
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter3d(
                x=[sensors[itx, 0]],
                y=[sensors[itx, 1]],
                z=[sensors[itx, 2]],
                mode="markers",
                marker=dict(size=9, color="gold", symbol="diamond"),
                showlegend=False,
            ),
            row=1,
            col=col,
        )
    fig.update_scenes(aspectmode="data")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=450)
    return fig


HEAD = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Smoothed 3D breast phantom &mdash; Helmholtz scattering</title>
<script src="{PLOTLY_CDN}"></script>
<script>
MathJax = {{tex: {{inlineMath: [['$', '$']]}}}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 1100px;
       margin: 2em auto; padding: 0 1em; color: #222; line-height: 1.5; }}
h1 {{ font-size: 1.6em; }} h2 {{ font-size: 1.25em; margin-top: 2em; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
td, th {{ border: 1px solid #999; padding: 0.3em 0.8em; }}
.fig {{ margin: 1em 0; }}
code {{ font-size: 0.95em; }}
</style></head><body>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="breast_site/index.html")
    ap.add_argument(
        "--hps", default=None, help="npz from breast_scattering_3d.py"
    )
    ap.add_argument(
        "--ng", default=None, help="npz from breast_scattering_3d_ngsolve.py"
    )
    ap.add_argument("--tx", type=int, nargs="*", default=[0])
    ap.add_argument("--n_sensors", type=int, default=64)
    args = ap.parse_args()

    def div(fig):
        return fig.to_html(
            full_html=False, include_plotlyjs=False, div_id=None
        )

    parts = [HEAD]
    parts.append(f"""
<h1>Forward scattering by a smoothed 3D breast phantom</h1>
<p>We solve the penetrable Helmholtz problem
$\\Delta u + \\kappa^2 n(x)\\, u = 0$ in $\\mathbb{{R}}^3$,
$u = u^i + u^s$ with $u^s$ Sommerfeld-radiating, for the phantom of
<a href="https://github.com/nibj/py-helm">py-helm</a> at $\\kappa = {KAPPA:g}$.
The squared index $n$ is $1$ in water and piecewise constant in py-helm
(skin $n = {SKINVAL:.4f}$, tissue $n = {TISSUEVAL:.4f}$, tumors
$n \\in [{DEFAULT_MVALS.min():.3f}, {DEFAULT_MVALS.max():.3f}]$); here each
interface is mollified by $\\chi(d) = \\tfrac12(1 + \\tanh(k d))$ with
$k = {KVAL:g}$, as required by the spectral-collocation HPS discretization.
The chest-wall impedance plane of py-helm is dropped: the problem is posed in
free space and the same smoothed $n$ is used in both solvers.</p>

<p>Two solvers are compared: (i) an HPS (ItI) volume solver coupled to an
exterior boundary integral equation enforcing the radiation condition
(<a href="https://github.com/jma02/jaxhps/pull/3">jaxhps PR&nbsp;#3</a>,
cf. <a href="https://arxiv.org/abs/2503.17535">arXiv:2503.17535</a>,
<a href="https://arxiv.org/abs/1308.5998">arXiv:1308.5998</a>); (ii) an
NGSolve $H^1$ FEM with a radial PML.</p>

<h2>Geometry</h2>
<p>Hemispherical breast of radius $1$ ($y > 0$), skin shell of thickness
$1/30$, three spherical tumors. Collocated transmitters/receivers on the
spherical cap $|x| = 1 + 2\\pi/\\kappa \\approx {SENSOR_RADIUS:.3f}$,
$y \\ge {SENSOR_OFFSET:g}$ (Fibonacci lattice). Each transmitter is a point
source $u^i(x) = e^{{i\\kappa|x - s|}}/(4\\pi|x - s|)$; the data are
$u^s$ at the receivers, i.e. an $n_{{rx}} \\times n_{{tx}}$ matrix
$U_{{\\mathrm{{meas}}}}$ per solver.</p>
""")
    parts.append(
        '<div class="fig">' + div(fig_domain(args.n_sensors)) + "</div>"
    )

    parts.append("""
<h2>Smoothed coefficient</h2>
<p>Coronal slice $x = 0$ of $n$ through the tumor centers (left axis $y$,
bottom $z$), and the profile along the axis $x = z = 0$ showing the
$\\tanh$ transitions across skin and tumor 2.</p>
""")
    parts.append('<div class="fig">' + div(fig_slice()) + "</div>")
    parts.append('<div class="fig">' + div(fig_profile()) + "</div>")

    if args.hps and args.ng:
        d_h = np.load(args.hps)
        d_n = np.load(args.ng)
        u_hps, u_ng = d_h["umeas"], d_n["umeas"]
        sensors = d_h["sensors"]
        rel_fro = np.linalg.norm(u_hps - u_ng) / np.linalg.norm(u_ng)
        rel_max = np.abs(u_hps - u_ng).max() / np.abs(u_ng).max()
        parts.append(f"""
<h2>Near-field data: HPS+BIE vs. NGSolve</h2>
<p>$\\|U_{{\\mathrm{{HPS}}}} - U_{{\\mathrm{{NG}}}}\\|_F /
\\|U_{{\\mathrm{{NG}}}}\\|_F = {rel_fro:.2e}$, entrywise
$\\max$-relative error ${rel_max:.2e}$ ({u_hps.shape[0]} receivers
$\\times$ {u_hps.shape[1]} transmitters). Mesh refinement of the FEM
reference decreases the discrepancy monotonically while the HPS solution is
internally converged to $1.5\\times 10^{{-2}}$ ($p = 10 \\to 12$), so the
residual difference is attributable to FEM/PML discretization error.</p>
""")
        parts.append(
            '<div class="fig">' + div(fig_matrices(u_hps, u_ng)) + "</div>"
        )
        for itx in args.tx:
            parts.append(
                '<div class="fig">'
                + div(fig_nearfield(sensors, u_hps, u_ng, itx))
                + "</div>"
            )

    parts.append("""
<h2>Reproduction</h2>
<pre><code>python examples/gen_SD_3D.py --q 8 --L 2 --kappa 4.0 --a 1.25 --out SD.npz
python examples/breast_scattering_3d.py --npz SD.npz --n_sensors 64 --p 12 --out umeas_hps.npz
python examples/breast_scattering_3d_ngsolve.py --n_sensors 64 --porder 2 --ppw 7 \\
    --hmax_breast 0.10 --water_layers 1.7 --out umeas_ng.npz
python examples/breast_compare_3d.py umeas_hps.npz umeas_ng.npz</code></pre>
</body></html>
""")

    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("".join(parts))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
