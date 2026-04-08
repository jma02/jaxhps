import jax
import jax.numpy as jnp
import numpy as np
from jaxhps import Domain, DiscretizationNode3D, PDEProblem, build_solver, solve
from jaxhps._interpolation_methods import vmapped_interp_to_point_3D
from jaxhps._grid_creation_3D import get_all_uniform_leaves_3D
import logging
from timeit import default_timer
import plotly.graph_objects as go
import plotly.express as px

# Set logging to INFO
logging.basicConfig(level=logging.INFO)

def generate_ngsolve_sensors(kappa, r_breast, z_chest, target_n=500):
    import netgen.csg as csg
    from ngsolve import Mesh

    lam_water = 2 * np.pi / kappa
    sensor_radius = r_breast + lam_water
    sensor_offset = 0.1 * r_breast
    
    def rx_tx(rad, offset, hmax):
        geo = csg.CSGeometry()
        sphere = csg.Sphere(csg.Pnt(0,0,0), rad)
        base = csg.Plane(csg.Pnt(0,offset,0), csg.Vec(0,-1,0))
        cap = sphere * base
        geo.AddSurface(sphere, cap)
        return Mesh(geo.GenerateMesh(maxh=hmax))
    
    sensor_area = 2 * np.pi * sensor_radius * (sensor_radius - sensor_offset)
    repl = np.sqrt(sensor_area)
    h_sensor = repl / np.sqrt(5 * target_n)
    
    mesh = rx_tx(sensor_radius, sensor_offset, h_sensor)
    for _ in range(5):
        if mesh.nv > 1.2 * target_n:
            h_sensor *= 1.2
        elif mesh.nv < 0.9 * target_n:
            h_sensor *= 0.9
        else:
            break
        mesh = rx_tx(sensor_radius, sensor_offset, h_sensor)
        
    pnts = mesh.ngmesh.Points()
    coords = np.array([[p.p[0], p.p[2], p.p[1] + z_chest] for p in pnts])
    return jnp.array(coords)

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
    print("=== Exact 3D PML: Multi-Source Imaging ===")
    
    # -------------------------------------------------------------------
    # 1. Configuration (Matching py-helm exactly)
    # -------------------------------------------------------------------
    kappa = 16.0             # Increased wavenumber per paper
    r_breast = 1.0           # Physical breast radius
    lam_water = 2.0 * jnp.pi / kappa
    
    # In py-helm: pmlmin = b_radius + 2*lam_water = 1.0 + 0.785 = 1.785
    # So we set our physical domain L to 1.8 to contain everything.
    L = 1.8
    L_total = 2.0            # PML outer boundary
    delta_pml = L_total - L  # 0.2
    
    # Since the domain is larger and kappa is higher, we need adequate resolution
    # L_tree=2 means leaves are size 1.0. With kappa=16, lambda is ~0.39.
    # We need p >= 12 for good spectral accuracy.
    p, q = 12, 10
    
    root = DiscretizationNode3D(xmin=-L_total, xmax=L_total, ymin=-L_total, ymax=L_total, zmin=-L_total, zmax=L_total)
    domain = Domain(p=p, q=q, root=root, L=2)
    pts = domain.interior_points
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    
    # -------------------------------------------------------------------
    # 2. Refractive Index n(x) (Breast Geometry from py-helm)
    # -------------------------------------------------------------------
    z_chest = -1.0
    dist_from_center = jnp.sqrt(x**2 + y**2 + (z - z_chest)**2)
    
    # py-helm properties
    n_water = 1.0
    n_tissue = (1524.0 / 1485.0)**2  # ~1.053
    n_skin = (1524.0 / 1610.0)**2    # ~0.896
    n_tumor = 1.2                    # ~ (1524/1500)^2 * random_factor
    
    delta_skin = r_breast / 30.0     # ~0.033
    
    # Masks
    mask_outer = sigmoid_ramp(dist_from_center - r_breast) * sigmoid_ramp(z_chest - z)
    mask_inner = sigmoid_ramp(dist_from_center - (r_breast - delta_skin)) * sigmoid_ramp(z_chest - z)
    mask_skin = mask_outer - mask_inner
    
    # 3 Tumors as defined in test_params.py
    # cen[:,0]=(0, b_radius/2, -b_radius/4), rad=0.1
    # cen[:,1]=(0, b_radius/2, 0), rad=0.1
    # cen[:,2]=(0, b_radius/2, b_radius/4), rad=0.05
    # (Mapping y->z, z->y to match our orientation)
    mask_t1 = sigmoid_ramp(jnp.sqrt(x**2 + (y + 0.25)**2 + (z - (-0.5))**2) - 0.1, width=0.02)
    mask_t2 = sigmoid_ramp(jnp.sqrt(x**2 + y**2 + (z - (-0.5))**2) - 0.1, width=0.02)
    mask_t3 = sigmoid_ramp(jnp.sqrt(x**2 + (y - 0.25)**2 + (z - (-0.5))**2) - 0.05, width=0.02)
    
    mask_tumors = jnp.maximum(jnp.maximum(mask_t1, mask_t2), mask_t3)
    
    n_field = n_water + (n_skin - n_water)*mask_skin + (n_tissue - n_water)*mask_inner + (n_tumor - n_tissue)*mask_tumors
    
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
    # 4. Multi-Source Definition (Spherical Bowl matching py-helm)
    # -------------------------------------------------------------------
    target_n = 500 # Slightly reduced from 1024 for speed, but matches their exact default
    print(f"Generating ~{target_n} transceivers on a spherical bowl using NGSolve/Netgen...")
    sensor_pos = generate_ngsolve_sensors(kappa, r_breast, z_chest, target_n=target_n)
    n_sources = sensor_pos.shape[0]
    print(f"Actual generated transceivers: {n_sources}")
    
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
    M = jnp.squeeze(M)
    
    reciprocity_error = jnp.linalg.norm(M - M.T) / jnp.linalg.norm(M)
    print(f"   Reciprocity Error ||M - M^T|| / ||M||: {reciprocity_error:.6e}")
    
    # -------------------------------------------------------------------
    # 7. Visualization: Coefficients & Sensors
    # -------------------------------------------------------------------
    print("Generating Coefficient and Data visualizations...")
    n_vis = 60
    lin = jnp.linspace(-L_total, L_total, n_vis)
    X, Y, Z = jnp.meshgrid(lin, lin, lin, indexing='ij')
    n_vol, _ = domain.interp_from_interior_points(n_field, lin, lin, lin)
    
    sx, sy, sz = sensor_pos[:, 0], sensor_pos[:, 1], sensor_pos[:, 2]
    
    fig_n = go.Figure(data=[
        go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=jnp.real(n_vol).flatten(), isomin=0.8, isomax=1.25,
            opacity=0.3, surface_count=20, colorscale='Viridis',
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
    fig_n.update_layout(title=f"Breast Anatomy (n) & Sensor Array (kappa={kappa})")
    fig_n.write_html("coef_index_n.html")
    
    fig_m = px.imshow(jnp.abs(M), title=f"{n_sources}-Pair Measurement Matrix |M|",
                      labels=dict(x="Source", y="Receiver"), color_continuous_scale='Inferno')
    fig_m.write_html("imaging_measurement_matrix.html")
    
    print("Done!")

if __name__ == "__main__":
    exact_pml_imaging_demo()
