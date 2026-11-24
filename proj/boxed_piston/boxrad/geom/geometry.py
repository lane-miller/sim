"""Geometry builder: box (numpy) and parametric (manifold union)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from boxrad.config import Config, derive, test_source_active
from boxrad.frames import Pose
from boxrad.geom.mesh import (
    TriMesh,
    box_mesh,
    concat_meshes,
    cylinder_mesh,
    inside,
    parts_disjoint,
    polygonal_cylinder_volume,
    ray_mesh_nearest,
    sphere_mesh,
    union_meshes,
    weld,
)
from boxrad.geom.poly2d import from_shape_spec
from boxrad.geomconfig import GeometrySpec, PlaneSpec, PrimitiveSpec, RadiatorSpec, ShapeSpec

_F64 = np.float64
_PRECISION_METHOD = "to_mesh64"


@dataclass
class RadiatorGeom:
    name: str
    pose: Pose
    kind: Literal["flat"] = "flat"
    outline: Any = None
    vertices_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    triangles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))
    area: float = 0.0
    motion: str = "axial"
    clearance_min: float = math.inf
    clearance_median: float = math.inf


@dataclass
class Geometry:
    solid: TriMesh
    radiators: list[RadiatorGeom]
    planes: tuple[Any, ...]
    provenance: dict[str, Any]

    def inside(self, pts: np.ndarray) -> np.ndarray:
        return inside(self.solid, pts)


def _box_center(cfg: Config, derived) -> np.ndarray:
    xc = float(cfg.radiator.x_c)
    yc = float(cfg.radiator.y_c)
    return np.array([-xc, -yc, -0.5 * derived.realized_D], dtype=_F64)


def _legacy_shape_spec(rad) -> ShapeSpec | None:
    from boxrad.geomconfig import _legacy_shape

    return _legacy_shape(rad)


def _radiator_geom(spec: RadiatorSpec, *, tol: float) -> RadiatorGeom:
    if spec.shape is None or spec.pose is None:
        raise ValueError("radiator needs shape and pose")
    outline = from_shape_spec(spec.shape, tol=1e-6)
    from boxrad.geom import poly2d

    verts2d, tris = poly2d.triangulate(outline)
    local = np.column_stack((verts2d, np.zeros(verts2d.shape[0], dtype=_F64)))
    world = spec.pose.to_world(local)
    return RadiatorGeom(
        name=spec.name,
        pose=spec.pose,
        outline=outline,
        vertices_world=world,
        triangles=tris,
        area=outline.area,
        motion=spec.motion,
    )


def from_box(cfg: Config, *, tol: float = 1e-5) -> Geometry:
    if cfg.box is None:
        raise ValueError("box geometry requires box section")
    derived = derive(cfg)
    W, H, D = derived.realized_W, derived.realized_H, derived.realized_D
    center = _box_center(cfg, derived)
    mesh = box_mesh([W, H, D]).transformed(np.eye(3, dtype=_F64), center)
    spec = cfg.geometry_spec
    if spec is None:
        raise ValueError("missing geometry_spec")
    radiators = [_radiator_geom(rd, tol=tol) for rd in spec.radiators]
    return Geometry(
        solid=mesh,
        radiators=radiators,
        planes=spec.environment.planes,
        provenance={
            "builder": "box",
            "tol": tol,
            "part_count": 0,
            "precision": "numpy",
        },
    )


def _primitive_mesh(part: PrimitiveSpec, tol: float) -> TriMesh:
    p = part.params
    if part.kind == "box":
        m = box_mesh([p["sx"], p["sy"], p["sz"]])
    elif part.kind == "cylinder":
        m = cylinder_mesh(p["d"], p["length"], tol)
    elif part.kind == "sphere":
        m = sphere_mesh(p["d"], tol)
    else:
        raise ValueError(f"unknown primitive {part.kind!r}")
    return m.transformed(part.pose.R, part.pose.position)


def _empty_solid_mesh() -> TriMesh:
    return TriMesh(
        np.zeros((0, 3), dtype=_F64),
        np.zeros((0, 3), dtype=np.int64),
    )


def from_parametric(cfg: Config, *, tol: float = 1e-5) -> Geometry:
    spec = cfg.geometry_spec
    if spec is None or spec.tag != "parametric":
        raise ValueError("expected parametric geometry")
    parts = [p for p in spec.parts if isinstance(p, PrimitiveSpec)]
    if not parts:
        if not test_source_active(cfg):
            raise ValueError("parametric geometry requires at least one part")
        solid = _empty_solid_mesh()
        precision = "empty"
    else:
        meshes = [_primitive_mesh(p, tol) for p in parts]
        if len(meshes) == 1:
            solid = meshes[0]
            precision = "analytic"
        elif parts_disjoint(meshes):
            cat = concat_meshes(meshes)
            solid = weld(cat.vertices, cat.faces)
            precision = "concat"
        else:
            united = union_meshes(meshes)
            solid = weld(united.vertices, united.faces)
            precision = _PRECISION_METHOD
        if not solid.is_closed():
            raise ValueError("parametric solid is not closed")
        if solid.volume <= 0.0:
            raise ValueError("parametric solid has non-positive volume")
    radiators = [_radiator_geom(rd, tol=tol) for rd in spec.radiators]
    return Geometry(
        solid=solid,
        radiators=radiators,
        planes=spec.environment.planes,
        provenance={
            "builder": "parametric",
            "tol": tol,
            "part_count": len(parts),
            "precision": precision,
        },
    )


def seating_check(
    geometry: Geometry,
    radiator: RadiatorGeom,
    *,
    eps: float = 1e-4,
    n: int = 200,
) -> None:
    ring = radiator.outline.outer
    n_ring = min(ring.shape[0], max(n // 4, 8))
    idx = np.linspace(0, ring.shape[0] - 1, n_ring, dtype=int)
    boundary = ring[idx]
    x0, y0 = radiator.outline.centroid
    half = 0.45 * np.sqrt(radiator.area)
    gx = np.linspace(x0 - half, x0 + half, int(np.sqrt(n)))
    gy = np.linspace(y0 - half, y0 + half, int(np.sqrt(n)))
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    grid = grid[radiator.outline.contains(grid)]
    samples = np.vstack((boundary, grid))[:n]
    n_hat = radiator.pose.R[:, 2]
    n_hat = n_hat / np.linalg.norm(n_hat)
    fails = 0
    first_fail = None
    for q in samples:
        p = radiator.pose.to_world(np.array([q[0], q[1], 0.0], dtype=_F64))
        if any(
            abs(float(p["xyz".index(pl.axis)]) - float(pl.offset)) <= 1e-9
            for pl in (geometry.planes or ())
        ):
            continue
        pin = geometry.inside(p - eps * n_hat)
        pout = geometry.inside(p + eps * n_hat)
        if not pin or pout:
            fails += 1
            if first_fail is None:
                first_fail = p
    if fails:
        raise ValueError(
            f"radiator {radiator.name!r} must sit on a flat face of the solid; "
            f"it overhangs or is not on the surface ({fails} failures, "
            f"first at {first_fail.tolist()})"
        )


def _radiator_sample_points(radiator: RadiatorGeom, *, n: int = 64) -> np.ndarray:
    ring = radiator.outline.outer
    n_ring = min(ring.shape[0], max(n // 4, 8))
    idx = np.linspace(0, ring.shape[0] - 1, n_ring, dtype=int)
    boundary = ring[idx]
    x0, y0 = radiator.outline.centroid
    half = 0.45 * np.sqrt(radiator.area)
    nx = max(int(np.sqrt(n)), 2)
    gx = np.linspace(x0 - half, x0 + half, nx)
    gy = np.linspace(y0 - half, y0 + half, nx)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    grid = grid[radiator.outline.contains(grid)]
    samples = np.vstack((boundary, grid))[:n]
    return samples


def _ray_plane_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    plane: PlaneSpec,
) -> float:
    d = direction / np.linalg.norm(direction)
    axis = {"x": 0, "y": 1, "z": 2}[plane.axis]
    if abs(d[axis]) < 1e-12:
        return math.inf
    t_hit = (plane.offset - origin[axis]) / d[axis]
    if t_hit <= 1e-9:
        return math.inf
    return float(t_hit)


def radiator_clearance(
    geometry: Geometry,
    radiator: RadiatorGeom,
    *,
    n: int = 64,
) -> tuple[float, float]:
    samples = _radiator_sample_points(radiator, n=n)
    n_hat = radiator.pose.R[:, 2]
    n_hat = n_hat / np.linalg.norm(n_hat)
    offset = 1e-7
    dists: list[float] = []
    for q in samples:
        p = radiator.pose.to_world(np.array([q[0], q[1], 0.0], dtype=_F64))
        o = p + offset * n_hat
        t_mesh = ray_mesh_nearest(o, n_hat, geometry.solid)
        t_plane = math.inf
        for pl in geometry.planes:
            t_pl = _ray_plane_hit(o, n_hat, pl)
            t_plane = min(t_plane, t_pl)
        dist = min(t_mesh, t_plane)
        if dist < math.inf:
            dist += offset
        dists.append(dist)
    arr = np.asarray(dists, dtype=_F64)
    cmin = float(np.min(arr))
    cmed = float(np.median(arr))
    radiator.clearance_min = cmin
    radiator.clearance_median = cmed
    return cmin, cmed


def clearance_warnings(
    geometry: Geometry,
    h: float,
    *,
    min_cells: int = 5,
) -> list[str]:
    warnings: list[str] = []
    for rd in geometry.radiators:
        cmin, _ = radiator_clearance(geometry, rd, n=64)
        if not np.isfinite(cmin):
            continue
        cells = cmin / h
        if cells < min_cells:
            h_need = cmin / min_cells
            warnings.append(
                f"{rd.name}: clearance {cmin:g} m, {cells:.1f} cells, "
                f"h needed {h_need:g} m"
            )
    return warnings


def build_geometry(cfg: Config, tol: float = 1e-5) -> Geometry | None:
    tag = cfg.geometry
    if tag == "baffle":
        return None
    if tag == "step":
        raise NotImplementedError("STEP ingest lands in v0.3 step 11")
    if tag == "box":
        geom = from_box(cfg, tol=tol)
    elif tag == "parametric":
        geom = from_parametric(cfg, tol=tol)
    else:
        raise ValueError(f"unsupported geometry {tag!r}")
    for rd in geom.radiators:
        seating_check(geom, rd)
    return geom


def polygon_area_exact(spec: ShapeSpec, *, tol: float = 1e-6) -> float:
    return from_shape_spec(spec, tol=tol).area


def cylinder_volume_exact(d: float, length: float, tol: float) -> float:
    r = 0.5 * d
    n = max(16, int(math.ceil(math.pi / math.acos(max(-1.0, min(1.0, 1.0 - tol / r))))))
    return polygonal_cylinder_volume(n, r, length)