import jax
import jax.numpy as jnp
from jaxhps import Domain, DiscretizationNode3D, PDEProblem, build_solver, solve
import logging
from timeit import default_timer
import plotly.express as px

# Set logging to INFO
logging.basicConfig(level=logging.INFO)

def sigmoid_ramp(dist, width=0.03):
    return 0.5 * (1.0 - jnp.tanh(dist / width))

def farfield_imaging_demo():
    print("=== 3D PML: Farfield Scattering Matrix ===")
    
    # 1. Configuration
    kappa = 12.0
    L = 1.0
    L_total = 2.0 
    delta_pml = L_total - L
    p, q = 10, 8
    
    root = DiscretizationNode3D(xmin=-L_total, xmax=L_total, ymin=-L_total, ymax=L_total, zmin=-L_total, zmax=L_total)
    domain = Domain(p=p, q=q, root=root, L=2)
    pts = domain.interior_points
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    
    # 2. Anatomy
    r_breast = 0.8
    z_chest = -1.0
    dist_breast = jnp.sqrt(x**2 + y**2 + (z - z_chest)**2) - r_breast
    mask_breast = sigmoid_ramp(dist_breast) * sigmoid_ramp(z_chest - z)
    
    r_tumor = 0.15
    dist_tumor = jnp.sqrt((x-0.2)**2 + (y-0.1)**2 + (z+0.5)**2) - r_tumor
    mask_tumor = sigmoid_ramp(dist_tumor, width=0.02)
    
    n_field = 1.0 + (0.2 + 0.02j) * mask_breast + (0.3 + 0.08j) * mask_tumor
    
    # 3. PML Coefficients
    sigma_max = 20.0
    def sigma_f(t): return sigma_max * (jnp.maximum(0.0, jnp.abs(t) - L) / delta_pml)**3
    
    s_x, s_y, s_z = sigma_f(x), sigma_f(y), sigma_f(z)
    d1, d2, d3 = 1.0 + 1j*s_x/kappa, 1.0 + 1j*s_y/kappa, 1.0 + 1j*s_z/kappa
    
    A11, A22, A33 = (d2 * d3) / d1, (d1 * d3) / d2, (d1 * d2) / d3
    I_coeffs = kappa**2 * n_field * (d1 * d2 * d3)

    # 4. Plane Wave Sources
    n_angles = 128
    print(f"Generating {n_angles} plane wave incident fields...")
    theta_inc = jnp.linspace(0, 2*jnp.pi, n_angles, endpoint=False)
    
    # Incident directions in xy plane
    d_x = jnp.cos(theta_inc)
    d_y = jnp.sin(theta_inc)
    d_z = jnp.zeros_like(theta_inc)
    d_vecs = jnp.stack([d_x, d_y, d_z], axis=-1) # (128, 3)
    
    def get_source_term(d_vec):
        phase = kappa * (x*d_vec[0] + y*d_vec[1] + z*d_vec[2])
        u_inc = jnp.exp(1j * phase)
        return kappa**2 * (1.0 - n_field) * u_inc
        
    sources = jax.vmap(get_source_term, in_axes=0, out_axes=-1)(d_vecs)

    # 5. Build and Solve
    problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=A11, D_yy_coefficients=A22, D_zz_coefficients=A33,
        I_coefficients=I_coeffs, source=sources
    )
    
    print("Building Direct Solver...")
    t0 = default_timer()
    build_solver(problem)
    print(f"   Built in {default_timer()-t0:.2f}s")
    
    print(f"Solving for {n_angles} scattered fields simultaneously...")
    def zero_fn(pts): return jnp.zeros((pts.shape[0], n_angles), dtype=jnp.complex128)
    bdry_data = domain.get_adaptive_boundary_data_lst(zero_fn)
    t0 = default_timer()
    u_scat_all = solve(problem, bdry_data) 
    u_scat_all.block_until_ready()
    print(f"   Solved in {default_timer()-t0:.2f}s")
    
    # 6. Volume Integration for Farfield Pattern
    print("Computing Farfield Pattern via Volume Integral (OOM-safe loop)...")
    n_grid = 60
    lin = jnp.linspace(-L, L, n_grid)
    dx = lin[1] - lin[0]
    X, Y, Z = jnp.meshgrid(lin, lin, lin, indexing='ij')
    
    n_grid_val, _ = domain.interp_from_interior_points(n_field, lin, lin, lin)
    scattering_potential = n_grid_val - 1.0 # (n_grid, n_grid, n_grid)
    
    theta_obs = theta_inc
    obs_x = jnp.cos(theta_obs)
    obs_y = jnp.sin(theta_obs)
    obs_z = jnp.zeros_like(theta_obs)
    obs_vecs = jnp.stack([obs_x, obs_y, obs_z], axis=-1)
    
    farfield_matrix = []
    
    t0 = default_timer()
    for i in range(n_angles):
        # 1. Get the i-th scattered field and incident direction
        u_scat_i = u_scat_all[..., i]
        d_vec = d_vecs[i]
        
        # 2. Interpolate scattered field to grid
        u_scat_grid_i, _ = domain.interp_from_interior_points(u_scat_i, lin, lin, lin)
        
        # 3. Compute incident field on grid
        phase_grid_i = kappa * (X * d_vec[0] + Y * d_vec[1] + Z * d_vec[2])
        u_inc_grid_i = jnp.exp(1j * phase_grid_i)
        
        # 4. Total field on grid
        u_tot_grid_i = u_scat_grid_i + u_inc_grid_i
        
        # 5. Integrate against all observation directions
        def compute_obs(obs_vec):
            phase_obs = -kappa * (X*obs_vec[0] + Y*obs_vec[1] + Z*obs_vec[2])
            obs_kernel = jnp.exp(1j * phase_obs)
            integrand = obs_kernel * scattering_potential * u_tot_grid_i
            integral = jnp.sum(integrand) * (dx**3)
            return (kappa**2 / (4.0 * jnp.pi)) * integral
            
        row = jax.vmap(compute_obs)(obs_vecs)
        farfield_matrix.append(row)
        
        if (i+1) % 32 == 0:
            print(f"   Processed {i+1}/{n_angles} sources...")
            
    farfield_matrix = jnp.stack(farfield_matrix, axis=1) # (n_obs, n_inc)
    print(f"   Farfield integration done in {default_timer()-t0:.2f}s")
    
    # 7. Visualization
    print("Generating Farfield visualization...")
    # Roll the matrix to align with standard scattering visualizations (e.g. diagonal dominance)
    fig = px.imshow(jnp.abs(farfield_matrix), 
                    title="Farfield Scattering Pattern |u_inf(theta_obs, theta_inc)|",
                    labels=dict(x="Incident Angle", y="Observation Angle"),
                    color_continuous_scale='Magma')
    fig.write_html("farfield_matrix.html")
    print("Done! Check 'farfield_matrix.html'")

if __name__ == "__main__":
    farfield_imaging_demo()
