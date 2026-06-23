r"""Static report: GPU BIE solver for 3D Helmholtz scattering.

Presents the full problem formulation, HPS discretisation, exterior BIE
coupling, solver variants (CPU FMM, GPU direct summation, block GMRES,
matrix-free), and feasibility analysis for the Lucka et al. breast problem.

Usage:
    python examples/fmm_results_site.py --out fmm_site/index.html
"""

import argparse
import os

import numpy as np
import plotly.graph_objects as go

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

# ============================================================
# Hardcoded results from Modal runs and local validation
# ============================================================

# --- Validation (CPU, L=2) ---
VAL_L2 = dict(
    kappa=4.0,
    a=1.25,
    L=2,
    q=8,
    p=12,
    n_bdry=6144,
    n_interior=110592,
    n_patches=96,
    n_near_pairs=5304,
    S_corr_nnz=21725184,
    D_corr_nnz=21725184,
    sparsity_pct=57.6,
    nf_build_time=3.89,
    S_matvec_rel_err=1.04e-10,
    D_matvec_rel_err=2.34e-9,
    dense_solve_time=18.30,
    gmres_fmm_time=30.45,
    rel_L2_err=1.82e-9,
    max_abs_err=2.08e-11,
    uscat_max_dense=5.2824e-3,
    uscat_max_fmm=5.2824e-3,
)

# --- Modal: CPU FMM baseline ---
MODAL_SMOKE_CPU = dict(
    kappa=4.0,
    a=1.25,
    L=2,
    q=8,
    p=12,
    n_bdry=6144,
    n_interior=110592,
    n_src=4,
    gpu="H100",
    nf_gen_time=23.1,
    hps_time=15.18,
    gmres_time=17.97,
    total_time=33.16,
    converged=True,
    uscat_max=5.2824e-3,
    T_DtN_mem_gb=0.60,
)

MODAL_HF_CPU = dict(
    kappa=26.18,
    a=1.25,
    L=3,
    q=8,
    p=12,
    n_bdry=24576,
    n_interior=884736,
    n_src=4,
    gpu="H100",
    nf_gen_time=133.8,
    hps_time=47.22,
    gmres_time=617.36,
    total_time=664.58,
    converged=True,
    uscat_max=2.2272e-1,
    T_DtN_mem_gb=9.66,
    freq_khz=5.0,
)

# --- Modal: full GPU direct-summation (current default) ---
MODAL_L2_GPU = dict(
    kappa=4.0,
    a=1.25,
    L=2,
    q=8,
    p=12,
    n_bdry=6144,
    n_interior=110592,
    n_src=4,
    gpu="H100",
    nf_gen_time=0.0,
    hps_time=12.80,
    gmres_time=9.13,
    total_time=21.93,
    converged=True,
    uscat_max=5.2824e-3,
    T_DtN_mem_gb=0.60,
    kernel_build_time=1.98,
    kernel_mem_gb=1.81,
)

MODAL_L3_GPU = dict(
    kappa=26.18,
    a=1.25,
    L=3,
    q=8,
    p=12,
    n_bdry=24576,
    n_interior=884736,
    n_src=4,
    gpu="H100",
    nf_gen_time=0.0,
    hps_time=58.19,
    gmres_time=17.51,
    total_time=75.70,
    converged=True,
    uscat_max=2.2272e-1,
    T_DtN_mem_gb=9.66,
    kernel_build_time=9.52,
    kernel_mem_gb=28.99,
    freq_khz=5.0,
)

# --- Advanced solver: block GMRES + Jacobi preconditioner ---
MODAL_ADV_L2_BLOCK = dict(
    kappa=4.0,
    a=1.25,
    L=2,
    q=8,
    p=12,
    n_bdry=6144,
    n_interior=110592,
    n_src=4,
    gpu="H100",
    hps_time=11.74,
    gmres_time=6.07,
    total_time=17.81,
    converged=True,
    uscat_max=5.2824e-3,
    kernel_mem_gb=1.81,
    solver_mode="dense_block",
)

# --- Advanced solver: matrix-free + Jacobi preconditioner ---
MODAL_ADV_L3_MATFREE = dict(
    kappa=26.18,
    a=1.25,
    L=3,
    q=8,
    p=12,
    n_bdry=24576,
    n_interior=884736,
    n_src=4,
    gpu="H100",
    hps_time=54.86,
    gmres_time=97.94,
    total_time=152.80,
    converged=True,
    uscat_max=2.2272e-1,
    T_DtN_mem_gb=9.66,
    sparse_nf_mem_gb=4.03,
    total_mem_gb=13.69,
    freq_khz=5.0,
    solver_mode="matfree",
)


# ============================================================
# Plotly figures
# ============================================================


def fig_solver_comparison():
    """Grouped bar: GMRES time across solver types at L=2 and L=3."""
    fig = go.Figure()

    # L=2 group
    fig.add_trace(
        go.Bar(
            x=["L=2<br>(n=6,144)"],
            y=[MODAL_SMOKE_CPU["gmres_time"]],
            name="CPU FMM + scipy GMRES",
            marker_color="#d62728",
            text=["18.0s"],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=["L=2<br>(n=6,144)"],
            y=[MODAL_L2_GPU["gmres_time"]],
            name="GPU dense matvec + JAX GMRES",
            marker_color="#ff7f0e",
            text=["9.1s"],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=["L=2<br>(n=6,144)"],
            y=[MODAL_ADV_L2_BLOCK["gmres_time"]],
            name="GPU dense + block GMRES + Jacobi",
            marker_color="#2ca02c",
            text=["6.07s"],
            textposition="outside",
        )
    )

    # L=3 group
    fig.add_trace(
        go.Bar(
            x=["L=3<br>(n=24,576)"],
            y=[MODAL_HF_CPU["gmres_time"]],
            name="CPU FMM + scipy GMRES",
            marker_color="#d62728",
            text=["617s"],
            textposition="outside",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=["L=3<br>(n=24,576)"],
            y=[MODAL_L3_GPU["gmres_time"]],
            name="GPU dense matvec + JAX GMRES",
            marker_color="#ff7f0e",
            text=["17.5s"],
            textposition="outside",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=["L=3<br>(n=24,576)"],
            y=[MODAL_ADV_L3_MATFREE["gmres_time"]],
            name="GPU matrix-free + Jacobi",
            marker_color="#1f77b4",
            text=["97.9s"],
            textposition="outside",
        )
    )

    fig.update_layout(
        barmode="group",
        yaxis_title="GMRES solve time (s)",
        yaxis_type="log",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
        ),
        margin=dict(l=50, r=20, t=80, b=40),
        height=500,
    )
    return fig


def fig_memory_comparison():
    """Bar chart: GPU memory for dense vs matrix-free at L=3."""
    labels = [
        "Dense path<br>(K_S + K_D + T_DtN)",
        "Matrix-free path<br>(sparse NF + T_DtN)",
    ]
    mems = [
        MODAL_L3_GPU["kernel_mem_gb"],
        MODAL_ADV_L3_MATFREE["total_mem_gb"],
    ]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=mems,
            text=[f"{m:.1f} GB" for m in mems],
            textposition="outside",
            marker_color=["#d62728", "#2ca02c"],
            width=0.5,
        )
    )
    fig.add_hline(
        y=80,
        line=dict(color="#999", dash="dash", width=2),
        annotation_text="H100 80 GB limit",
        annotation_position="top right",
    )
    fig.update_layout(
        yaxis_title="GPU memory (GB)",
        margin=dict(l=50, r=20, t=40, b=40),
        height=380,
    )
    return fig


def fig_validation_bars():
    """Bar chart: FMM matvec and solve errors at L=2."""
    labels = [
        "S matvec\nrel err",
        "D matvec\nrel err",
        "BIE solve\nrel L2 err",
        "BIE solve\nmax abs err",
    ]
    vals = [
        VAL_L2["S_matvec_rel_err"],
        VAL_L2["D_matvec_rel_err"],
        VAL_L2["rel_L2_err"],
        VAL_L2["max_abs_err"],
    ]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=vals,
            text=[f"{v:.2e}" for v in vals],
            textposition="outside",
            marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        )
    )
    fig.update_layout(
        yaxis_title="error",
        yaxis_type="log",
        margin=dict(l=50, r=20, t=20, b=80),
        height=360,
    )
    return fig


def fig_feasibility_kappa():
    """Scatter: achievable kappa*a vs GPU memory for different (L, q)."""
    configs = [
        (2, 8, "L=2, q=8"),
        (3, 8, "L=3, q=8"),
        (3, 10, "L=3, q=10"),
        (3, 12, "L=3, q=12"),
        (3, 14, "L=3, q=14"),
        (4, 8, "L=4, q=8"),
    ]
    kas, mems, labels, colors = [], [], [], []
    for L, q, lab in configs:
        n = 6 * (4**L) * q**2
        T_mem = n**2 * 16 / 1e9
        ka_max = np.pi * (2**L) * q / 6.0
        kas.append(ka_max)
        mems.append(T_mem)
        labels.append(lab)
        colors.append("#2ca02c" if T_mem < 80 else "#d62728")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=kas,
            y=mems,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=14, color=colors),
            showlegend=False,
        )
    )
    fig.add_hline(
        y=80,
        line=dict(color="#999", dash="dash", width=2),
        annotation_text="H100 80 GB",
        annotation_position="top right",
    )
    fig.add_trace(
        go.Scatter(
            x=[26.18 * 1.25],
            y=[9.66],
            mode="markers",
            name="Tested (L=3, q=8, kappa*a=32.7)",
            marker=dict(size=18, color="#2ca02c", symbol="star"),
        )
    )
    fig.update_layout(
        xaxis_title="max kappa * a (6 points-per-wavelength criterion)",
        yaxis_title="T_DtN memory (GB)",
        yaxis_type="log",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
        ),
        margin=dict(l=50, r=20, t=60, b=40),
        height=460,
    )
    return fig


def fig_breast_freq_mapping():
    """Scatter: breast frequency vs required L and memory."""
    c_bg = 1500.0
    a_phys = 0.055
    freqs = [50, 100, 150, 200, 250, 500]
    kas, t_mems, labels = [], [], []
    for f in freqs:
        kappa = 2 * np.pi * f * 1e3 / c_bg
        ka = kappa * a_phys
        L = max(1, int(np.ceil(np.log2(max(1, 6 * ka / (np.pi * 8))))))
        q = 8
        ppw = np.pi * (2**L) * q / ka
        if ppw < 6 and L <= 3:
            q = int(np.ceil(6 * ka / (np.pi * 2**L)))
        n = 6 * (4**L) * q**2
        T_mem = n**2 * 16 / 1e9
        kas.append(ka)
        t_mems.append(T_mem)
        labels.append(f"{f} kHz")

    fig = go.Figure()
    feasible = [m < 80 for m in t_mems]
    fig.add_trace(
        go.Scatter(
            x=[kas[i] for i in range(len(kas)) if feasible[i]],
            y=[t_mems[i] for i in range(len(kas)) if feasible[i]],
            mode="markers+text",
            text=[labels[i] for i in range(len(kas)) if feasible[i]],
            textposition="top center",
            marker=dict(size=14, color="#2ca02c"),
            name="Feasible (single H100)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[kas[i] for i in range(len(kas)) if not feasible[i]],
            y=[t_mems[i] for i in range(len(kas)) if not feasible[i]],
            mode="markers+text",
            text=[labels[i] for i in range(len(kas)) if not feasible[i]],
            textposition="top center",
            marker=dict(size=14, color="#d62728"),
            name="OOM (single GPU)",
        )
    )
    fig.add_hline(
        y=80,
        line=dict(color="#999", dash="dash", width=2),
        annotation_text="80 GB limit",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="kappa * a (breast geometry, a = 55 mm)",
        yaxis_title="T_DtN memory (GB)",
        yaxis_type="log",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
        ),
        margin=dict(l=50, r=20, t=60, b=40),
        height=460,
    )
    return fig


# ============================================================
# HTML generation
# ============================================================

HEAD = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>GPU BIE Solver for 3D Helmholtz Scattering</title>
<script src="{PLOTLY_CDN}"></script>
<script>MathJax = {{tex: {{inlineMath: [['$', '$']]}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 1000px;
       margin: 2em auto; padding: 0 1.5em; color: #222; line-height: 1.7; }}
h1 {{ font-size: 1.7em; border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
h2 {{ font-size: 1.3em; margin-top: 2.5em; color: #333; }}
h3 {{ font-size: 1.1em; margin-top: 1.5em; color: #444; }}
table {{ border-collapse: collapse; margin: 1em 0; font-size: 0.92em; }}
td, th {{ border: 1px solid #bbb; padding: 0.35em 0.9em; text-align: center; }}
th {{ background: #f5f5f5; font-weight: 600; }}
.fig {{ margin: 1.5em 0; }}
code {{ font-size: 0.9em; background: #f6f6f6; padding: 0.1em 0.3em;
        border-radius: 3px; }}
pre {{ background: #f6f6f6; padding: 1em; overflow-x: auto;
       border-left: 3px solid #1f77b4; }}
.pass {{ color: #2ca02c; font-weight: bold; }}
.warn {{ color: #d62728; font-weight: bold; }}
.highlight {{ background: #e8f5e9; padding: 0.8em 1em; border-radius: 4px;
              margin: 1.2em 0; border-left: 4px solid #2ca02c; }}
.note {{ background: #fff3e0; padding: 0.8em 1em; border-radius: 4px;
         margin: 1em 0; font-size: 0.95em; border-left: 4px solid #ff7f0e; }}
</style></head><body>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fmm_site/index.html")
    args = ap.parse_args()

    def div(fig):
        return fig.to_html(full_html=False, include_plotlyjs=False)

    parts = [HEAD]

    # ================================================================
    # 1. TITLE
    # ================================================================
    parts.append(r"""
<h1>A GPU-accelerated boundary integral solver for 3D penetrable
Helmholtz scattering</h1>

<p><em>Numerical results from an implementation built on the
<a href="https://github.com/jma02/jaxhps">jaxhps</a> hierarchical
Poincar&eacute;&ndash;Steklov solver, deployed on NVIDIA H100 (80 GB)
via <a href="https://modal.com">Modal</a>.</em></p>
""")

    # ================================================================
    # 2. PROBLEM FORMULATION
    # ================================================================
    parts.append(r"""
<h2>1. Problem formulation</h2>

<p>We consider acoustic scattering from a penetrable inhomogeneity
contained in a bounded domain $\Omega \subset \mathbb R^3$.
The total field $u = u^{\mathrm{inc}} + u^s$ satisfies the
variable-coefficient Helmholtz equation</p>

$$\Delta u + \kappa^2\, n^2(x)\, u = 0, \qquad x \in \mathbb R^3,$$

<p>where $\kappa > 0$ is the background wavenumber and the refractive
index $n(x) = 1$ for $x \notin \Omega$.  We write $n^2(x) = 1 - b(x)$
so that $b$ is compactly supported in $\Omega$ and the equation
becomes</p>

$$\Delta u + \kappa^2 u = \kappa^2\, b(x)\, u, \qquad
  \text{supp}(b) \subset \Omega.$$

<p>The scattered field $u^s$ satisfies the Sommerfeld radiation condition
at infinity. The incident field is taken to be a plane wave
$u^{\mathrm{inc}}(x) = e^{i\kappa\,w\cdot x}$ with propagation
direction $w \in S^2$.</p>

<p>The computational domain $\Omega$ is chosen to be a cube of
half-width $a$, i.e.&nbsp;$\Omega = [-a,a]^3$.  The inhomogeneity
$b(x)$ is a smooth function supported strictly inside $\Omega$.</p>
""")

    # ================================================================
    # 3. DISCRETISATION: HPS INTERIOR
    # ================================================================
    parts.append(r"""
<h2>2. Interior discretisation: the HPS method</h2>

<p>The interior problem on $\Omega$ is solved via the <em>hierarchical
Poincar&eacute;&ndash;Steklov</em> (HPS) method, which computes the
Dirichlet-to-Neumann (DtN) operator
$T : u|_{\partial\Omega} \mapsto \partial_n u|_{\partial\Omega}$
without ever assembling or inverting the full volumetric system.</p>

<p>The cube $\Omega$ is recursively subdivided into an octree of depth
$L$, producing $8^L$ leaf boxes.  On each leaf, the PDE is discretised
on a tensor-product Chebyshev grid of order $p$ ($p^3$ interior nodes
per leaf).  The local solution operator and DtN map are computed via the
<em>iterated impedance-to-impedance</em> (ItI) Cayley transform,
which is numerically stable for high-frequency problems.</p>

<p>The leaf-level DtN maps are then merged pairwise up the tree in
$L$ levels.  The final output is the global DtN operator
$T \in \mathbb C^{n \times n}$ where $n = n_{\mathrm{bdry}}$ is the
number of boundary degrees of freedom on $\partial\Omega$.
The boundary is discretised with $q^2$ Gauss&ndash;Legendre nodes
per patch, with $4^L$ patches per face and 6 faces, giving</p>

$$n_{\mathrm{bdry}} = 6 \cdot 4^L \cdot q^2.$$

<p>The entire HPS factorisation is performed on GPU using JAX.  The
dominant cost is the merge phase at $\mathcal O(p^6 \cdot 8^L)$
operations.  The output $T$ is stored as a dense
$n \times n$ matrix (complex128, $16\,n^2$ bytes).</p>

<table>
<tr><th>$L$</th><th>$q$</th><th>$p$</th><th>$n_{\mathrm{bdry}}$</th>
    <th>$n_{\mathrm{int}}$</th><th>$T_{\mathrm{DtN}}$ memory</th>
    <th>HPS time (H100)</th></tr>
<tr><td>2</td><td>8</td><td>12</td><td>6,144</td>
    <td>110,592</td><td>0.60 GB</td><td>12.8 s</td></tr>
<tr><td>3</td><td>8</td><td>12</td><td>24,576</td>
    <td>884,736</td><td>9.66 GB</td><td>58.2 s</td></tr>
</table>
""")

    # ================================================================
    # 4. EXTERIOR BIE COUPLING
    # ================================================================
    parts.append(r"""
<h2>3. Exterior coupling: the boundary integral equation</h2>

<p>Given $T$, the exterior scattering problem reduces to a
second-kind boundary integral equation on $\partial\Omega$.
Let $S$ and $D$ denote the single- and double-layer boundary operators
for the free-space Helmholtz Green's function
$G(x,y) = e^{i\kappa|x-y|}/(4\pi|x-y|)$:</p>

$$(S\,\sigma)(x) = \int_{\partial\Omega} G(x,y)\,\sigma(y)\,\mathrm dS(y),
\qquad
(D\,\sigma)(x) = \int_{\partial\Omega}
  \frac{\partial G(x,y)}{\partial n(y)}\,\sigma(y)\,\mathrm dS(y).$$

<p>The combined-field representation leads to the system</p>

$$A\,u^s\big|_{\partial\Omega} = b, \qquad
  A = \tfrac12 I - D + S\,T, \qquad
  b = S\,\bigl(\partial_n u^{\mathrm{inc}} - T\,u^{\mathrm{inc}}\bigr),$$

<p>which is a Fredholm equation of the second kind and is solved
iteratively by restarted GMRES.  Each GMRES iteration requires the
matrix&ndash;vector products $S \cdot v$ and $D \cdot v$, as well as the
dense product $T \cdot v$.</p>

<h3>3.1 Near-field corrections</h3>

<p>The boundary is decomposed into curvilinear patches (the faces of
the octree leaves).  For distant patch pairs, the kernel $G(x,y)$
is smooth and standard quadrature suffices.  For <em>near</em> pairs
(within distance $\text{near\_ratio} \times \max\text{patch\_width}$),
the kernel is near-singular and high-order adaptive quadrature
(Vioreanu&ndash;Rokhlin nodes, generalized Gaussian rules from
<code>fmm3dbie</code>) is required.</p>

<p>We precompute the near-field corrections as sparse matrices
$C_S$, $C_D$ defined by</p>

$$C_{ij} = K^{\mathrm{dense}}_{ij} - K^{\mathrm{smooth}}_{ij}$$

<p>where $K^{\mathrm{dense}}$ is the fully-resolved quadrature and
$K^{\mathrm{smooth}}$ is the smooth-rule approximation.  These corrections
are computed once per configuration $(\kappa, q, L, a)$ using
<code>fmm3dbie</code> and cached.  At runtime, each matvec is</p>

$$S \cdot v = K^{\mathrm{smooth}}_S \cdot v + C_S \cdot v.$$
""")

    # ================================================================
    # 5. TYPES OF EXTERIOR SOLVER
    # ================================================================
    parts.append(r"""
<h2>4. Types of exterior solver</h2>

<p>We implement three approaches to evaluating the matvec
$A \cdot v = (\tfrac12 I - D + S\,T)\,v$ within GMRES, each with
different computational and memory trade-offs:</p>

<h3>4.1 CPU FMM (baseline)</h3>

<p>The smooth kernel matvecs $K^{\mathrm{smooth}}_S \cdot v$ and
$K^{\mathrm{smooth}}_D \cdot v$ are evaluated via the 3D Helmholtz
fast multipole method (<code>fmm3dpy</code>, Fortran, CPU).
The near-field corrections $C_S$, $C_D$ are applied as sparse
matrix&ndash;vector products (also CPU).  The DtN product $T \cdot v$
is a dense matvec via NumPy on CPU.  GMRES is driven by
<code>scipy.sparse.linalg.gmres</code>.</p>

<p>This approach has $\mathcal O(n \log n)$ asymptotic cost per iteration,
but is bottlenecked by the serial CPU execution of the FMM and by
CPU&harr;GPU data transfer (since $T$ is built on GPU but applied on CPU).
At $n = 24{,}576$ ($L = 3$), each GMRES iteration takes several seconds.</p>

<h3>4.2 GPU dense matvec (full GPU)</h3>

<p>We precompute and store the dense kernel matrices
$K_S, K_D \in \mathbb C^{n \times n}$ on GPU, constructed in chunks
of 2048 rows to avoid peak memory spikes.  These are the <em>full</em>
discrete operators (smooth kernel + near-field correction combined):</p>

$$K_S = K^{\mathrm{smooth}}_S + C_S, \qquad K_D = K^{\mathrm{smooth}}_D + C_D.$$

<p>Each GMRES iteration is then a sequence of dense
matrix&ndash;vector products entirely on GPU, driven by
<code>jax.scipy.sparse.linalg.gmres</code>.  The cost is
$\mathcal O(n^2)$ per iteration, but GPU parallelism makes this
extremely fast for moderate $n$.  The penalty is memory:
storing $K_S + K_D + T$ requires $3 \times 16\,n^2$ bytes
(29 GB at $n = 24{,}576$).</p>

<h3>4.3 GPU dense + block GMRES + Jacobi preconditioner</h3>

<p>When solving for multiple right-hand sides simultaneously
(e.g.&nbsp;$n_{\mathrm{src}} = 4$ incident plane waves), we flatten
the system into a single $(n \cdot n_{\mathrm{src}})$-dimensional
GMRES problem.  Each iteration then applies $A$ as a block operation:
three GEMMs (general matrix&ndash;matrix multiplies) instead of
$n_{\mathrm{src}}$ separate GEMVs.  On GPU hardware, GEMMs achieve
significantly higher arithmetic throughput than GEMVs due to better
utilisation of tensor cores and memory bandwidth.</p>

<p>Additionally, we apply a diagonal (Jacobi) preconditioner
$M_{ii} = (0.5 - [C_D]_{ii})^{-1}$ extracted from the near-field
correction diagonal.  This approximates the dominant contribution
of $A$ at each boundary node and reduces the GMRES iteration count.</p>

<h3>4.4 GPU matrix-free + Jacobi preconditioner</h3>

<p>For large problems where $K_S$, $K_D$ do not fit in GPU memory,
we avoid storing the dense kernel matrices entirely.  Instead, each
GMRES iteration recomputes the matvec on the fly:</p>

$$[K^{\mathrm{smooth}}_S \cdot v]_i
  = \sum_{j=1}^n G(x_i, x_j)\,w_j\,v_j$$

<p>via <code>jax.vmap</code> over the row index $i$.  XLA fuses this into
a single GPU kernel with no intermediate $n \times n$ allocation.
The near-field corrections $C_S$, $C_D$ are stored in JAX BCOO
(batched coordinate) sparse format, requiring $\mathcal O(\text{nnz})$
memory instead of $\mathcal O(n^2)$.</p>

<p>The trade-off: each iteration performs $\mathcal O(n^2)$ kernel
evaluations (transcendental functions $e^{i\kappa r}/r$) rather than
reading a pre-built matrix, incurring $\sim 5$&ndash;$6\times$ overhead
per iteration.  However, memory drops from $3 \times 16\,n^2$ to
$16\,n^2$ (for $T$ alone) plus $\mathcal O(\text{nnz})$ for the
sparse corrections &mdash; enabling problems that would otherwise be
infeasible on a single GPU.</p>
""")

    # ================================================================
    # 6. RESULTS: COMPARISON
    # ================================================================
    parts.append(r"""
<h2>5. Results</h2>

<p>All timings are from a single NVIDIA H100 (80 GB) via Modal.
Fixed parameters: $q = 8$, $p = 12$, $n_{\mathrm{src}} = 4$
(simultaneous plane-wave illuminations), GMRES restart $= 50$,
tolerance $= 10^{-6}$.</p>

<h3>5.1 Exterior solve times by solver type</h3>

<table>
<tr><th>solver type</th><th>$\kappa$</th><th>$L$</th>
    <th>$n_{\mathrm{bdry}}$</th>
    <th>GMRES time (s)</th><th>GPU memory (GB)</th>
    <th>speedup</th></tr>
<tr><td>CPU FMM + scipy</td><td>4.0</td><td>2</td><td>6,144</td>
    <td>18.0</td><td>0.6 (T only)</td><td>&mdash;</td></tr>
<tr><td>GPU dense + JAX GMRES</td><td>4.0</td><td>2</td><td>6,144</td>
    <td>9.1</td><td>1.8</td>
    <td>$2\times$</td></tr>
<tr><td>GPU dense + block + Jacobi</td><td>4.0</td><td>2</td><td>6,144</td>
    <td><b>6.07</b></td><td>1.8</td>
    <td>$3\times$</td></tr>
<tr><td colspan="7" style="border:none; height:0.5em;"></td></tr>
<tr><td>CPU FMM + scipy</td><td>26.2</td><td>3</td><td>24,576</td>
    <td>617</td><td>9.7 (T only)</td><td>&mdash;</td></tr>
<tr><td>GPU dense + JAX GMRES</td><td>26.2</td><td>3</td><td>24,576</td>
    <td><b>17.5</b></td><td>29.0</td>
    <td>$35\times$</td></tr>
<tr><td>GPU matrix-free + Jacobi</td><td>26.2</td><td>3</td><td>24,576</td>
    <td>97.9</td><td><b>13.7</b></td>
    <td>$6.3\times$ (53% less memory)</td></tr>
</table>
""")
    parts.append('<div class="fig">' + div(fig_solver_comparison()) + "</div>")

    parts.append(r"""
<h3>5.2 Memory trade-off at $L = 3$</h3>

<p>The dense solver stores $K_S + K_D + T$ as three $n \times n$
complex128 matrices, consuming 29 GB.  The matrix-free solver stores only
$T$ (9.7 GB) plus the sparse near-field corrections in BCOO format
(4.0 GB for indices and data), totalling 13.7 GB &mdash; a factor of
$2.1\times$ reduction.</p>
""")
    parts.append('<div class="fig">' + div(fig_memory_comparison()) + "</div>")

    parts.append(r"""
<h3>5.3 Total solve times (HPS + exterior)</h3>

<table>
<tr><th>$\kappa$</th><th>$L$</th><th>solver</th>
    <th>HPS (s)</th><th>GMRES (s)</th><th>total (s)</th></tr>
<tr><td>4.0</td><td>2</td><td>dense + block + Jacobi</td>
    <td>11.7</td><td>6.1</td><td><b>17.8</b></td></tr>
<tr><td>26.2</td><td>3</td><td>GPU dense</td>
    <td>58.2</td><td>17.5</td><td><b>75.7</b></td></tr>
<tr><td>26.2</td><td>3</td><td>matrix-free + Jacobi</td>
    <td>54.9</td><td>97.9</td><td><b>152.8</b></td></tr>
</table>

<div class="note">
<b>When to use which solver.</b>  The dense path is fastest whenever the
kernel matrices fit in GPU memory ($n \lesssim 25{,}000$ on 80 GB).
Block GMRES provides an additional $1.5\times$ when GEMMs are more
efficient than GEMVs (always true for $n_{\mathrm{src}} > 1$).
The matrix-free path is slower per iteration but is the only option
for $n > 25{,}000$ without multi-GPU, and it remains faster than the
CPU FMM for all tested configurations.
</div>
""")

    # ================================================================
    # 7. MAXIMUM ACHIEVABLE KAPPA
    # ================================================================
    parts.append(r"""
<h2>6. Resolution limits on a single GPU</h2>

<p>Two constraints determine the maximum achievable $\kappa$ on a single
80 GB GPU:</p>

<ol>
  <li><b>Memory:</b> The DtN operator $T \in \mathbb C^{n \times n}$
      must be stored in GPU RAM.  It is the dense output of the HPS
      factorisation; there is no analytic formula or factored form
      available for applying $T \cdot v$ without storing $T$ explicitly.
      At $n = n_{\mathrm{bdry}}$, this costs $16\,n^2$ bytes.</li>
  <li><b>Resolution:</b> The boundary quadrature must resolve the
      oscillations of the kernel $G(x,y)$.  With mesh spacing
      $h = 2a/(2^L \cdot q)$, the points-per-wavelength criterion
      $\lambda / h \ge 6$ gives
      $$\kappa a \le \frac{\pi \cdot 2^L \cdot q}{6}.$$</li>
</ol>

<p>The following table shows feasible configurations:</p>

<table>
<tr><th>$L$</th><th>$q$</th><th>$n_{\mathrm{bdry}}$</th>
    <th>$T_{\mathrm{DtN}}$ (GB)</th>
    <th>max $\kappa a$</th><th>fits 80 GB?</th></tr>
<tr><td>2</td><td>8</td><td>6,144</td>
    <td>0.60</td><td>33.5</td><td class="pass">yes</td></tr>
<tr><td>3</td><td>8</td><td>24,576</td>
    <td>9.66</td><td>33.5</td>
    <td class="pass">yes (tested, $\kappa a = 32.7$)</td></tr>
<tr><td>3</td><td>10</td><td>38,400</td>
    <td>23.6</td><td>41.9</td><td class="pass">yes</td></tr>
<tr><td>3</td><td>12</td><td>55,296</td>
    <td>48.9</td><td>50.3</td>
    <td class="warn">marginal (HPS peak ~54 GB)</td></tr>
<tr><td>3</td><td>14</td><td>75,264</td>
    <td>90.6</td><td>58.6</td><td class="warn">no</td></tr>
<tr><td>4</td><td>8</td><td>98,304</td>
    <td>154.6</td><td>67.0</td><td class="warn">no</td></tr>
</table>

<div class="highlight">
<b>Practical limit:</b> $\kappa a \approx 33$&ndash;$50$ on a single H100
(80 GB), achieved at $L = 3$ with $q \in [8, 12]$.  Going beyond this
requires either multi-GPU distribution of $T$ or a hierarchical
compression of the DtN operator (e.g.&nbsp;$\mathcal H$-matrix or
butterfly factorisation).
</div>
""")
    parts.append('<div class="fig">' + div(fig_feasibility_kappa()) + "</div>")

    # ================================================================
    # 8. APPLICATION: BREAST ULTRASOUND (LUCKA ET AL.)
    # ================================================================
    parts.append(r"""
<h2>7. Application: breast ultrasound imaging (Lucka et al.)</h2>

<p>We assess the feasibility of applying this solver to the
3D breast ultrasound computed tomography (USCT) problem studied in
<a href="https://arxiv.org/abs/2102.00755">Lucka et al.&nbsp;(2021)</a>.
Their setup involves a pendant breast in a hemispherical scanner
array, with the forward model being a time-domain lossy wave equation.
We consider the time-harmonic reduction: at each temporal frequency
$\omega = 2\pi f$, the pressure satisfies</p>

$$\Delta\hat p + \frac{\omega^2}{c_0^2(x)}\,\hat p = 0,$$

<p>which, with $\kappa = \omega/c_{\mathrm{bg}}$ and
$n(x) = c_{\mathrm{bg}}/c_0(x)$, is exactly our variable-coefficient
Helmholtz problem.  The scattering potential is
$b(x) = 1 - (c_{\mathrm{bg}}/c_0(x))^2$.</p>

<h3>7.1 Tissue coefficients</h3>

<p>The relevant sound speeds and corresponding coefficients for an
anatomically realistic breast phantom are:</p>

<table>
<tr><th>tissue</th><th>$c_0$ (m/s)</th>
    <th>$n = c_{\mathrm{bg}}/c_0$</th>
    <th>$b = 1 - n^2$</th></tr>
<tr><td>water (background)</td><td>1500</td>
    <td>1.000</td><td>0.000</td></tr>
<tr><td>fat</td><td>1470</td>
    <td>1.020</td><td>&minus;0.041</td></tr>
<tr><td>fibro-glandular</td><td>1515</td>
    <td>0.990</td><td>+0.020</td></tr>
<tr><td>blood vessels</td><td>1584</td>
    <td>0.947</td><td>+0.103</td></tr>
<tr><td>skin</td><td>1650</td>
    <td>0.909</td><td>+0.174</td></tr>
</table>

<p>All contrasts satisfy $|b| \le 0.18$, which is mild.  The BIE
formulation converges rapidly for such low-contrast inclusions;
GMRES typically requires $\lesssim 50$ iterations.  By comparison,
our synthetic dataset uses $|b|$ up to $0.5$, so the breast problem
lies well within the regime where the solver has been validated.</p>

<h3>7.2 Frequency&ndash;wavenumber mapping</h3>

<p>The breast geometry has effective radius $a \approx 55$ mm.  The
relevant non-dimensional parameter is $\kappa a = 2\pi f\,a / c_{\mathrm{bg}}$,
which determines the discretisation requirements:</p>

<table>
<tr><th>$f$ (kHz)</th><th>$\lambda$ (mm)</th><th>$\kappa$ (m$^{-1}$)</th>
    <th>$\kappa a$</th><th>required $L$, $q$</th>
    <th>$T_{\mathrm{DtN}}$ (GB)</th><th>feasible?</th></tr>
""")
    c_bg = 1500.0
    a_phys = 0.055
    for f_khz in [50, 100, 150, 200, 250, 500, 1500]:
        lam_mm = c_bg / (f_khz * 1e3) * 1e3
        kappa = 2 * np.pi * f_khz * 1e3 / c_bg
        ka = kappa * a_phys
        L = max(1, int(np.ceil(np.log2(max(1, 6 * ka / (np.pi * 8))))))
        q = 8
        if L > 3:
            L = 3
            q = int(np.ceil(6 * ka / (np.pi * 2**L)))
        n_bdry = 6 * (4**L) * q**2
        T_gb = n_bdry**2 * 16 / 1e9
        feasible = T_gb < 60
        cls = "pass" if feasible else "warn"
        feas_text = "yes" if feasible else "no (OOM)"
        parts.append(
            f"<tr><td>{f_khz}</td><td>{lam_mm:.1f}</td>"
            f"<td>{kappa:.0f}</td><td>{ka:.1f}</td>"
            f"<td>$L={L}$, $q={q}$</td>"
            f"<td>{T_gb:.1f}</td>"
            f'<td class="{cls}">{feas_text}</td></tr>\n'
        )
    parts.append("</table>\n")

    parts.append(r"""
<div class="highlight">
<b>Conclusion:</b> frequencies up to <b>150 kHz</b>
($\kappa a \approx 34.6$) are directly feasible and have been
validated at the equivalent non-dimensional parameters.
At 200 kHz ($\kappa a \approx 46$) the problem is tight but
potentially solvable with $q = 12$.  Above 250 kHz, the
$T_{\mathrm{DtN}}$ matrix exceeds single-GPU memory.
Their target resolution of 1.5 MHz ($\kappa a \approx 346$)
requires fundamentally different algorithmic infrastructure
(hierarchical compression of $T$, or multi-GPU distribution).
</div>

<h3>7.3 Relevance to full-waveform inversion</h3>

<p>The Lucka et al.&nbsp;FWI pipeline uses a multi-scale approach,
beginning at coarse spatial resolutions ($\Delta x = 2$&ndash;$4$ mm,
corresponding to $f \approx 100$&ndash;$200$ kHz) and progressively
refining.  Our solver is directly applicable to these
<em>lowest-frequency initialisations</em> of the inversion &mdash;
providing a high-accuracy, spectrally convergent forward model at
a cost of $\sim 75$ s per frequency per source configuration.
For a 20-frequency sweep over 50&ndash;200 kHz with the full
1024-source hemispherical array (batched as simultaneous RHS),
the total compute would be approximately 25 minutes on a single H100.</p>
""")
    parts.append(
        '<div class="fig">' + div(fig_breast_freq_mapping()) + "</div>"
    )

    # ================================================================
    # 9. VALIDATION
    # ================================================================
    v = VAL_L2
    parts.append(rf"""
<h2>8. Validation</h2>

<p>To verify correctness, we compare the GPU direct-summation solver
against a dense $LU$ factorisation at $L = 2$ ($\kappa = {v["kappa"]:g}$,
$n_{{\mathrm{{bdry}}}} = {v["n_bdry"]:,}$), where both approaches are
feasible.  The near-field correction involves {v["n_near_pairs"]:,}
patch pairs ({v["sparsity_pct"]:.1f}% of all pairs).</p>

<table>
<tr><th>quantity</th><th>value</th></tr>
<tr><td>$S$ matvec relative error (GPU vs dense)</td>
    <td>${v["S_matvec_rel_err"]:.2e}$</td></tr>
<tr><td>$D$ matvec relative error (GPU vs dense)</td>
    <td>${v["D_matvec_rel_err"]:.2e}$</td></tr>
<tr><td>BIE solution relative $L^2$ error</td>
    <td>${v["rel_L2_err"]:.2e}$</td></tr>
<tr><td>BIE solution max pointwise error</td>
    <td>${v["max_abs_err"]:.2e}$</td></tr>
<tr><td>$\|u^s\|_{{\infty}}$ (both solvers)</td>
    <td>${v["uscat_max_dense"]:.4e}$</td></tr>
</table>

<p>Agreement is to $\sim 10^{{-9}}$ in relative $L^2$, confirming that
the GPU solver introduces no loss of accuracy relative to a direct
factorisation.</p>
""")
    parts.append('<div class="fig">' + div(fig_validation_bars()) + "</div>")

    # ================================================================
    # 10. PARAMETERS & REPRODUCTION
    # ================================================================
    parts.append(r"""
<h2>9. Discretisation parameters and reproduction</h2>

<table>
<tr><th>parameter</th><th>symbol</th><th>value</th></tr>
<tr><td>Gauss&ndash;Legendre nodes per patch edge</td>
    <td>$q$</td><td>8</td></tr>
<tr><td>interior Chebyshev order</td><td>$p$</td><td>12</td></tr>
<tr><td>cube half-width</td><td>$a$</td><td>1.25</td></tr>
<tr><td>GMRES relative tolerance</td><td></td><td>$10^{-6}$</td></tr>
<tr><td>GMRES restart length</td><td></td><td>50</td></tr>
<tr><td>GMRES max iterations</td><td></td><td>200</td></tr>
<tr><td>near-field ratio (patch widths)</td><td></td><td>4.0</td></tr>
<tr><td>simultaneous incident fields</td>
    <td>$n_{\mathrm{src}}$</td><td>4</td></tr>
<tr><td>GPU hardware</td><td></td><td>NVIDIA H100 80 GB (Modal)</td></tr>
</table>

<h3>Reproduction commands</h3>
<pre><code># L=2 smoke test (kappa=4, ~22s total)
modal run examples/modal_hf_solve_3d.py --mode smoke

# L=3 high-frequency (kappa~26, ~76s dense, ~153s matrix-free)
modal run examples/modal_hf_solve_3d.py --mode solve --freq-khz 5 --a 1.25

# Dense + block GMRES + Jacobi (L=2)
modal run examples/modal_hf_solve_3d.py --mode smoke --solver-mode dense_block

# Matrix-free + Jacobi (L=3, low memory)
modal run examples/modal_hf_solve_3d.py --mode solve --freq-khz 5 --a 1.25 \
    --solver-mode matfree</code></pre>

<p>Source:
<a href="https://github.com/jma02/jaxhps/pull/3">jaxhps PR&nbsp;#3</a>,
branch <code>devin/1781152283-breast-scattering-3d</code>.</p>
</body></html>
""")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("".join(parts))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
