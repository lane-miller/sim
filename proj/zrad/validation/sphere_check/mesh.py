"""Generate uniform sphere surface meshes for the pulsating-sphere BEM check."""

import argparse
import os

import gmsh
import numpy as np

A = 0.01            # sphere radius [m]
C = 343.0           # speed of sound [m/s]
RHO = 1.21          # air density [kg/m^3]
U0 = 1.0            # uniform radial surface velocity [m/s]
KA_VALUES = [0.05, 0.1, 0.5, 1.0, 5.0]

MESH_DIR = os.path.join(os.path.dirname(__file__), "meshes")


def frequency_hz(ka):
    k = ka / A
    return k * C / (2.0 * np.pi)


def mesh_path(ka):
    tag = f"{ka:g}".replace(".", "p")
    return os.path.join(MESH_DIR, f"sphere_ka{tag}.msh")


def _count_triangles():
    """Return number of 3-node triangle elements on the surface."""
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
    return sum(len(tags) for etype, tags in zip(elem_types, elem_tags) if etype == 2)


def generate_mesh(ka, gui=False):
    """Build one closed sphere surface mesh for the given ka."""
    k = ka / A
    f = frequency_hz(ka)
    wavelength = 2.0 * np.pi / k
    h_wavelength = wavelength / 6.0
    # When λ/6 exceeds the sphere, Gmsh fails to triangulate the surface (1D edges only).
    h_geom = A / 6.0
    h = min(h_wavelength, h_geom)

    gmsh.initialize()
    gmsh.model.add(f"sphere_ka{ka:g}")

    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, A)
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.model.mesh.generate(2)

    n_tri = _count_triangles()
    if n_tri == 0:
        gmsh.finalize()
        raise RuntimeError(
            f"No surface triangles for ka={ka:g} (h={h:.6f} m, a={A:.4f} m). "
            "Characteristic length must be smaller than the sphere diameter."
        )

    out_path = mesh_path(ka)
    os.makedirs(MESH_DIR, exist_ok=True)
    gmsh.write(out_path)

    print(f"ka={ka:g}  f={f:.1f} Hz  a={A:.4f} m  h={h:.6f} m")
    if h < h_wavelength:
        print(f"  (h capped from λ/6={h_wavelength:.6f} m by a/6={h_geom:.6f} m)")
    print(f"  Elements : {n_tri}")
    print(f"  Written  : {out_path}")

    if gui:
        gmsh.fltk.run()

    gmsh.finalize()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate pulsating-sphere BEM check meshes.")
    parser.add_argument(
        "--ka",
        type=float,
        nargs="*",
        default=None,
        help="ka values to mesh (default: all five validation points)",
    )
    parser.add_argument("--gui", action="store_true", help="Open Gmsh GUI after meshing.")
    args = parser.parse_args()

    ka_list = args.ka if args.ka is not None else KA_VALUES
    for ka in ka_list:
        generate_mesh(ka, gui=args.gui)


if __name__ == "__main__":
    main()
