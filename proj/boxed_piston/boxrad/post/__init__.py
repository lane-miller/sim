"""Post-processing: Kirchhoff–Helmholtz and related reductions.

Import submodules directly (``boxrad.post.kirchhoff``, ``boxrad.post.zrad``) to
avoid pulling optional dependencies at package import time.
"""

from .kirchhoff import (
    SurfacePatches,
    box_surface_patches,
    dpdn_from_vn,
    face_p_from_cells,
    far_field_at_radius,
    huygens_box_patches,
    huygens_open_patches,
    kh_far_field,
    kh_pressure,
    kh_pressure_loop,
)

__all__ = [
    "SurfacePatches",
    "box_surface_patches",
    "huygens_box_patches",
    "huygens_open_patches",
    "dpdn_from_vn",
    "face_p_from_cells",
    "far_field_at_radius",
    "kh_far_field",
    "kh_pressure",
    "kh_pressure_loop",
]