"""In-house 2-D outlines and 3-D meshes (v0.3 step 3)."""

from boxrad.geom.geometry import Geometry, RadiatorGeom, build_geometry
from boxrad.geom.mesh import TriMesh, box_mesh, cylinder_mesh, sphere_mesh
from boxrad.geom.poly2d import Outline

__all__ = [
    "Geometry",
    "Outline",
    "RadiatorGeom",
    "TriMesh",
    "box_mesh",
    "build_geometry",
    "cylinder_mesh",
    "sphere_mesh",
]