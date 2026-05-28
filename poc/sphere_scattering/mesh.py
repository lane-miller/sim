import gmsh
import os

# Parameters
RADIUS = 0.025       # meters
MESH_SIZE = 343.0 / 10000 / 10    # element size — ~6 elements per wavelength at 10 kHz
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
MESH_FILE = os.path.join(OUTPUT_DIR, "sphere.msh")

def generate_sphere_mesh(radius=RADIUS, mesh_size=MESH_SIZE, output_path=MESH_FILE, gui=False):
    gmsh.initialize()
    gmsh.model.add("sphere")

    # Geometry
    gmsh.model.occ.addSphere(0, 0, 0, radius)
    gmsh.model.occ.synchronize()

    # Mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    # Surface mesh only (BEM needs surface, not volume)
    gmsh.model.mesh.generate(2)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gmsh.write(output_path)
    print(f"Mesh written to {output_path}")

    if gui:
        gmsh.fltk.run()  # opens the Gmsh GUI to inspect

    gmsh.finalize()

if __name__ == "__main__":
    generate_sphere_mesh(gui=False)