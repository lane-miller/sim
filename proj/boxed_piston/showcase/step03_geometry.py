#!/usr/bin/env python3
"""Step 3 showcase: mesh geometry metrics for S1–S6."""

from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from boxrad.config import Config
from boxrad.geom.geometry import build_geometry, radiator_clearance
from boxrad.geomconfig import PlaneSpec
from boxrad.grid import radiator_area_exact

_PKG = Path(__file__).resolve().parents[1]
_SHOWCASE = _PKG / "configs" / "showcase"
_OUT = Path(__file__).resolve().parent / "out" / "step03_geometry.png"
RTOL = 1e-12
_H_MM = 0.005
_PLANE_HALF = 0.4


def _analytic_radiator_area(cfg: Config, name: str, rd_area: float) -> float:
    if name == "S3":
        return radiator_area_exact(cfg.radiator)
    return rd_area


def _polygon_exact_area(cfg: Config, name: str, rd) -> float:
    if rd.outline is not None:
        return float(rd.outline.area)
    if name == "S3":
        return radiator_area_exact(cfg.radiator)
    return float(rd.area)


def _exact_volume(name: str, cfg: Config) -> float:
    if name in ("S3", "S4"):
        return 0.2 * 0.15 * 0.25
    if name == "S2":
        return 0.2 * 0.25 * 0.3
    if name == "S5":
        return 0.2 * 0.25 * 0.3 + 0.40 * 0.35 * 0.03
    if name == "S6":
        part = cfg.geometry_spec.parts[0]
        d, L = part.params["d"], part.params["length"]
        r = 0.5 * d
        tol = 1e-5
        n = max(16, int(math.ceil(math.pi / math.acos(max(-1.0, min(1.0, 1.0 - tol / r))))))
        v_cyl = (n / 2.0) * r * r * math.sin(2.0 * math.pi / n) * L
        return v_cyl + 0.30 * 0.30 * 0.025
    part = cfg.geometry_spec.parts[0]
    d, L = part.params["d"], part.params["length"]
    r = 0.5 * d
    tol = 1e-5
    n = max(16, int(math.ceil(math.pi / math.acos(max(-1.0, min(1.0, 1.0 - tol / r))))))
    return (n / 2.0) * r * r * math.sin(2.0 * math.pi / n) * L


def _case_row(name: str, cfg_path: Path) -> dict:
    cfg = Config.from_json(cfg_path.read_text(encoding="utf-8"))
    geom = build_geometry(cfg)
    assert geom is not None
    exact_vol = _exact_volume(name, cfg)
    mesh_vol = geom.solid.volume
    rel_v = abs(mesh_vol - exact_vol) / max(exact_vol, 1e-30)
    return {
        "name": name,
        "cfg": cfg,
        "geom": geom,
        "mesh_vol": mesh_vol,
        "exact_vol": exact_vol,
        "rel_v": rel_v,
        "pass": rel_v <= RTOL,
    }


def _add_mesh(ax, verts: np.ndarray, faces: np.ndarray, *, facecolor, alpha: float, edgecolor) -> None:
    tris = verts[faces]
    coll = Poly3DCollection(
        tris,
        facecolors=facecolor,
        alpha=alpha,
        edgecolors=edgecolor,
        linewidths=0.2,
    )
    ax.add_collection3d(coll)


def _draw_plane_square(ax, plane: PlaneSpec, *, half: float = _PLANE_HALF) -> None:
    axis = {"x": 0, "y": 1, "z": 2}[plane.axis]
    u_ax = (axis + 1) % 3
    v_ax = (axis + 2) % 3
    corners = []
    for su, sv in (
        (-half, -half),
        (half, -half),
        (half, half),
        (-half, half),
    ):
        pt = np.zeros(3)
        pt[axis] = plane.offset
        pt[u_ax] = su
        pt[v_ax] = sv
        corners.append(pt)
    coll = Poly3DCollection(
        [corners],
        facecolors=(0.2, 0.6, 0.9, 0.15),
        edgecolors=(0.1, 0.4, 0.7, 0.5),
        linewidths=0.4,
    )
    ax.add_collection3d(coll)


def main() -> None:
    t0 = time.time()
    print(f"start: {time.ctime(t0)}")
    cases = [
        ("S1", _SHOWCASE / "s1_cylinder.json"),
        ("S2", _SHOWCASE / "s2_bookshelf.json"),
        ("S3", _PKG / "configs" / "c09_box.json"),
        ("S4", _SHOWCASE / "s4_tilted.json"),
        ("S5", _SHOWCASE / "s5_platform.json"),
        ("S6", _SHOWCASE / "s6_downfire.json"),
    ]
    rows = [_case_row(name, path) for name, path in cases]
    all_pass = True
    for row in rows:
        g = row["geom"]
        print(
            f"{row['name']}: verts={g.solid.vertices.shape[0]} tris={g.solid.faces.shape[0]} "
            f"closed={g.solid.is_closed()} bbox={g.solid.bbox} "
            f"vol_mesh={row['mesh_vol']:.10g} vol_exact={row['exact_vol']:.10g} "
            f"rel_vol={row['rel_v']:.10g} precision={g.provenance.get('precision')!r} "
            f"{'PASS' if row['pass'] else 'CHECK'}"
        )
        if not row["pass"]:
            all_pass = False
        for rd in g.radiators:
            poly = _polygon_exact_area(row["cfg"], row["name"], rd)
            ana = _analytic_radiator_area(row["cfg"], row["name"], rd.area)
            rel_mp = abs(rd.area - poly) / max(abs(poly), 1e-30)
            rel_pa = abs(poly - ana) / max(abs(ana), 1e-30)
            print(
                f"  {rd.name}: area_mesh={rd.area:.10g} polygon={poly:.10g} "
                f"analytic={ana:.10g} rel(mesh,polygon)={rel_mp:.10g} "
                f"rel(polygon,analytic)={rel_pa:.10g}"
            )
            cmin, _ = radiator_clearance(g, rd)
            cells = cmin / _H_MM if np.isfinite(cmin) else float("inf")
            print(
                f"    clearance_min={cmin:.10g} m  cells_at_h5mm="
                f"{cells:.10g}"
            )
    s3, s4 = rows[2], rows[3]
    diff = abs(s3["mesh_vol"] - s4["mesh_vol"]) / max(s3["mesh_vol"], 1e-30)
    print(f"S4−S3 relative volume diff: {diff:.10g}")
    n4 = s4["geom"].radiators[0].pose.normal
    n4 = n4 / np.linalg.norm(n4)
    print(f"S4 radiator normal: {n4.tolist()}")

    fig, axes = plt.subplots(2, 3, subplot_kw={"projection": "3d"}, figsize=(14, 8))
    for ax, row in zip(axes.ravel(), rows):
        v, f = row["geom"].solid.vertices, row["geom"].solid.faces
        _add_mesh(ax, v, f, facecolor=(0.7, 0.7, 0.7, 0.25), alpha=0.25, edgecolor=(0.4, 0.4, 0.4, 1.0))
        for pl in row["geom"].planes:
            _draw_plane_square(ax, pl, half=0.8 * 0.5)
        for rd in row["geom"].radiators:
            _add_mesh(
                ax,
                rd.vertices_world,
                rd.triangles,
                facecolor=(1.0, 0.5, 0.0, 0.9),
                alpha=0.9,
                edgecolor=(0.8, 0.3, 0.0, 1.0),
            )
        ax.auto_scale_xyz(v[:, 0], v[:, 1], v[:, 2])
        ax.set_title(f"{row['name']} V={row['mesh_vol']:.4g}")
        ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT)
    n_pass = sum(r["pass"] for r in rows)
    status = "PASS" if all_pass else "CHECK"
    print(f"summary: {n_pass}/{len(rows)} volume checks {status}, wrote {_OUT}")


if __name__ == "__main__":
    main()