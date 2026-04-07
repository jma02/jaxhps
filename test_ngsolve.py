import numpy as np

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
    
    print(f"Generating Netgen mesh, target vertices = {target_n}")
    mesh = rx_tx(sensor_radius, sensor_offset, h_sensor)
    for i in range(5):
        print(f"Iteration {i}: nv={mesh.nv}")
        if mesh.nv > 1.2 * target_n:
            h_sensor *= 1.2
        elif mesh.nv < 0.9 * target_n:
            h_sensor *= 0.9
        else:
            break
        mesh = rx_tx(sensor_radius, sensor_offset, h_sensor)
        
    pnts = mesh.ngmesh.Points()
    print(f"Final vertices = {len(pnts)}")
    coords = np.array([[p.p[0], p.p[2], p.p[1] + z_chest] for p in pnts])
    return coords

if __name__ == "__main__":
    coords = generate_ngsolve_sensors(kappa=12.0, r_breast=0.8, z_chest=-1.0, target_n=1024)
    print("Coords shape:", coords.shape)
