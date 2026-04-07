import jax
import jax.numpy as jnp
from jaxhps import Domain, DiscretizationNode3D, PDEProblem, build_solver, solve
from jaxhps._interpolation_methods import vmapped_interp_to_point_3D
from jaxhps._grid_creation_3D import get_all_uniform_leaves_3D
import logging
from timeit import default_timer
import plotly.graph_objects as go
import plotly.express as px

# Set logging to INFO
logging.basicConfig(level=logging.INFO)

def sigmoid_ramp(dist, width=0.03):
    return 0.5 * (1.0 - jnp.tanh(dist / width))

def interp_point_cloud(domain, samples, pts):
    """Interpolate HPS solution to an arbitrary point cloud instead of a meshgrid."""
    leaves = get_all_uniform_leaves_3D(domain.root, domain.L)
    corners_lst = [
        jnp.array([[node.xmin, node.ymin, node.zmin],
                   [node.xmax, node.ymax, node.zmax]])
        for node in leaves
    ]
    corners_iter = jnp.stack(corners_lst)
    
    # Find which leaf each point belongs to
    satisfies_xmin = pts[:, 0, None] >= corners_iter[None, :, 0, 0]
    satisfies_xmax = pts[:, 0, None] <= corners_iter[None, :, 1, 0]
    satisfies_ymin = pts[:, 1, None] >= corners_iter[None, :, 0, 1]
    satisfies_ymax = pts[:, 1, None] <= corners_iter[None, :, 1, 1]
    satisfies_zmin = pts[:, 2, None] >= corners_iter[None, :, 0, 2]
    satisfies_zmax = pts[:, 2, None] <= corners_iter[None, :, 1, 2]
    
    all_bools = satisfies_xmin & satisfies_xmax & satisfies_ymin & satisfies_ymax & satisfies_zmin & satisfies_zmax
    patch_idx = jnp.argmax(all_bools, axis=1)
    
    corners_for_vmap = corners_iter[patch_idx]
    
    # To avoid OOM (1024 pts x 1728 nodes x 1024 sources = 28GB), 
    # we interpolate one source at a time
    M = []
    for i in range(samples.shape[-1]):
        f_for_vmap = samples[patch_idx, :, i]
        vals = vmapped_interp_to_point_3D(
            pts[:, 0], pts[:, 1], pts[:, 2],
            corners_for_vmap, f_for_vmap, domain.p
        )
        M.append(vals)
        
    return jnp.stack(M, axis=-1)


def exact_pml_imaging_demo():
    print("=== Exact 3D PML: Multi-Source Imaging on a Spherical Bowl ===")
    
    # -------------------------------------------------------------------
    # 1. Configuration
    # -------------------------------------------------------------------
    kappa = 12.0
    L = 1.0
    L_total = 2.0 
    delta_pml = L_total - L
    p, q = 12, 10
    
    root = DiscretizationNode3D(xmin=-L_total, xmax=L_total, ymin=-L_total, ymax=L_total, zmin=-L_total, zmax=L_total)
    domain = Domain(p=p, q=q, root=root, L=2)
    pts = domain.interior_points
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    
    # -------------------------------------------------------------------
    # 2. Refractive Index n(x) (Breast Geometry)
    # -------------------------------------------------------------------
    # Breast centered at z=-1.0 (Chest Wall), radius 0.8, pointing up into z > -1.0
    r_breast = 0.8
    z_chest = -1.0
    dist_breast = jnp.sqrt(x**2 + y**2 + (z - z_chest)**2) - r_breast
    mask_breast = sigmoid_ramp(dist_breast) * sigmoid_ramp(z_chest - z)
    
    # Inclusion (Tumor) offset from center
    r_tumor = 0.15
    dist_tumor = jnp.sqrt((x-0.2)**2 + (y-0.1)**2 + (z+0.5)**2) - r_tumor
    mask_tumor = sigmoid_ramp(dist_tumor, width=0.02)
    
    n_field = 1.0 + (0.2 + 0.02j) * mask_breast + (0.3 + 0.08j) * mask_tumor
    
    # -------------------------------------------------------------------
    # 3. PML Coefficients
    # -------------------------------------------------------------------
    sigma_max = 20.0
    def sigma_f(t): return sigma_max * (jnp.maximum(0.0, jnp.abs(t) - L) / delta_pml)**3
    
    s_x, s_y, s_z = sigma_f(x), sigma_f(y), sigma_f(z)
    d1, d2, d3 = 1.0 + 1j*s_x/kappa, 1.0 + 1j*s_y/kappa, 1.0 + 1j*s_z/kappa
    
    A11, A22, A33 = (d2 * d3) / d1, (d1 * d3) / d2, (d1 * d2) / d3
    I_coeffs = kappa**2 * n_field * (d1 * d2 * d3)

    # -------------------------------------------------------------------
    # 4. Multi-Source Definition (Spherical Bowl)
    # -------------------------------------------------------------------
    n_sources = 1024
    print(f"Generating {n_sources} transceivers on a spherical bowl...")
    
    # Use a Fibonacci lattice to evenly distribute sensors on a hemisphere
    indices = jnp.arange(0, n_sources, dtype=float)
    golden_ratio = (1.0 + 5.0**0.5) / 2.0
    
    # Bowl is centered at (0, 0, -1.0), pointing up. Radius = 0.9 (just outside breast)
    # We want points from the base of the breast (z=-1.0) up to the tip (z=-0.1)
    r_bowl = 0.9
    z_rel = r_bowl * (1.0 - indices / (n_sources - 1)) # From 0.9 down to 0
    r_xy = jnp.sqrt(r_bowl**2 - z_rel**2)
    theta = 2.0 * jnp.pi * indices / golden_ratio
    
    sx, sy, sz = r_xy * jnp.cos(theta), r_xy * jnp.sin(theta), z_chest + z_rel
    sensor_pos = jnp.stack([sx, sy, sz], axis=-1)
    
    def get_source_term(pos):
        r = jnp.sqrt((x-pos[0])**2 + (y-pos[1])**2 + (z-pos[2])**2 + 1e-6)
        u_inc = jnp.exp(1j * kappa * r) / (4.0 * jnp.pi * r)
        return kappa**2 * (1.0 - n_field) * u_inc
    
    sources = jax.vmap(get_source_term, in_axes=0, out_axes=-1)(sensor_pos)

    # -------------------------------------------------------------------
    # 5. Build and Solve
    # -------------------------------------------------------------------
    problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=A11, D_yy_coefficients=A22, D_zz_coefficients=A33,
        I_coefficients=I_coeffs, source=sources
    )
    
    print("Building Direct Solver...")
    t0 = default_timer()
    build_solver(problem)
    print(f"   Built in {default_timer()-t0:.2f}s")
    
    print(f"Solving for {n_sources} scattered fields simultaneously...")
    def zero_fn(pts): return jnp.zeros((pts.shape[0], n_sources), dtype=jnp.complex128)
    bdry_data = domain.get_adaptive_boundary_data_lst(zero_fn)
    u_scat_all = solve(problem, bdry_data) 
    u_scat_all.block_until_ready()
    print(f"   Solved in {default_timer()-t0:.2f}s")
    
    # -------------------------------------------------------------------
    # 6. Measurement Matrix & Reciprocity Check
    # -------------------------------------------------------------------
    print(f"Interpolating to {n_sources} receiver locations on the bowl...")
    M = interp_point_cloud(domain, u_scat_all, sensor_pos)
    M = jnp.squeeze(M)  # Ensure it is exactly (1024, 1024)
    
    reciprocity_error = jnp.linalg.norm(M - M.T) / jnp.linalg.norm(M)
    print(f"   Reciprocity Error ||M - M^T|| / ||M||: {reciprocity_error:.6e}")
    
    # -------------------------------------------------------------------
    # 7. Visualization: Coefficients & Sensors
    # -------------------------------------------------------------------
    print("Generating Coefficient and Data visualizations...")
    n_vis = 50
    lin = jnp.linspace(-L_total, L_total, n_vis)
    X, Y, Z = jnp.meshgrid(lin, lin, lin, indexing='ij')
    n_vol, _ = domain.interp_from_interior_points(n_field, lin, lin, lin)
    
    # Plot Breast, Chest Wall, and the Sensor Point Cloud
    fig_n = go.Figure(data=[
        go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=jnp.real(n_vol).flatten(), isomin=1.05, isomax=1.4,
            opacity=0.3, surface_count=15, colorscale='Viridis',
            name="Tissue Index"
        ),
        go.Surface(
            x=lin, y=lin, z=jnp.full((n_vis, n_vis), z_chest),
            colorscale=[[0, 'gray'], [1, 'gray']], opacity=0.4, showscale=False,
            name="Chest Wall"
        ),
        go.Scatter3d(
            x=sx, y=sy, z=sz, mode='markers',
            marker=dict(size=3, color='red'), name='Transceivers'
        )
    ])
    fig_n.update_layout(title="Breast Geometry & Spherical Sensor Array")
    fig_n.write_html("coef_index_n.html")
    
    # Plot Measurement Matrix
    fig_m = px.imshow(jnp.abs(M), title="1024-Pair Measurement Matrix |M| (Spherical Array)",
                      labels=dict(x="Source", y="Receiver"), color_continuous_scale='Inferno')
    fig_m.write_html("imaging_measurement_matrix.html")
    
    print("Done! Check files:")
    print("- coef_index_n.html (Shows the spherical bowl of sensors)")
    print("- imaging_measurement_matrix.html")

if __name__ == "__main__":
    exact_pml_imaging_demo()
