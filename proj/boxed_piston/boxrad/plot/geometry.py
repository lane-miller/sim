"""Voxel / mesh geometry plot from config alone (v0.3 step 12)."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from boxrad.config import Config, test_source_active
from boxrad.cutcell import wetted_wall_arrays
from boxrad.geom.geometry import Geometry, build_geometry
from boxrad.geomrun import _derive_geom_with_cfl_resolution
from boxrad.geomop import build_geom_op_cache
from boxrad.geomrec import huygens_patches_planes
from boxrad.recorder import resolve_huygens_offset

_F64 = np.float64
_STATE_SOLID = 0
_STATE_FLUID = 1
_STATE_CUT = 2
_MAX_SCATTER = 20_000


def triangle_plane_segments(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    axis: int,
    offset: float,
    tol: float | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Triangle–plane intersection segments as pairs of 3-vectors."""
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    verts = np.asarray(vertices, dtype=_F64)
    tris = np.asarray(triangles, dtype=np.int64)
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    scale = float(np.linalg.norm(hi - lo))
    if tol is None:
        tol = 1e-12 * scale
    for tri in tris:
        pts = verts[tri]
        d = pts[:, axis] - offset
        if np.all(np.abs(d) <= tol):
            continue
        side = np.sign(d)
        side[np.abs(d) <= tol] = 0
        if (side > 0).all() or (side < 0).all():
            continue
        edge_pts: list[np.ndarray] = []
        for e in ((0, 1), (1, 2), (2, 0)):
            a, b = pts[e[0]], pts[e[1]]
            da, db = a[axis] - offset, b[axis] - offset
            if abs(da) <= tol:
                edge_pts.append(a.copy())
                continue
            if abs(db) <= tol:
                edge_pts.append(b.copy())
                continue
            if da * db < 0:
                t = da / (da - db)
                edge_pts.append(a + t * (b - a))
        if len(edge_pts) >= 2:
            p0 = edge_pts[0]
            p1 = max(edge_pts[1:], key=lambda q: float(np.linalg.norm(q - p0)))
            if float(np.linalg.norm(p1 - p0)) <= tol:
                continue
            segs.append((p0, p1))
    return segs


def _default_slice_point(cfg: Config, geometry: Geometry, derived) -> tuple[float, float, float]:
    if geometry.radiators:
        rd = geometry.radiators[0]
        c = rd.pose.to_world(
            np.array([rd.outline.centroid[0], rd.outline.centroid[1], 0.0], dtype=_F64)
        )
        return float(c[0]), float(c[1]), float(c[2])
    if test_source_active(cfg) and cfg.numerics.test_source is not None:
        p = cfg.numerics.test_source.position
        return float(p[0]), float(p[1]), float(p[2])
    if geometry.solid.vertices.shape[0]:
        xlo, ylo, zlo, xhi, yhi, zhi = geometry.solid.bbox
        return (
            0.5 * (xlo + xhi),
            0.5 * (ylo + yhi),
            0.5 * (zlo + zhi),
        )
    return 0.0, 0.0, 0.0


def _parse_slice(text: str | None) -> tuple[float, float, float] | None:
    if text is None:
        return None
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError("slice must be x,y,z in meters")
    return parts[0], parts[1], parts[2]


def prepare_geometry_plot(
    cfg: Config,
    *,
    h_override: float | None = None,
) -> dict[str, Any]:
    """Build geometry, grid, fractions, and floor once (no stepping)."""
    work = cfg
    if h_override is not None:
        work = copy.deepcopy(cfg)
        work.numerics.h_override = float(h_override)
    geometry = build_geometry(work)
    if geometry is None:
        raise ValueError("prepare_geometry_plot requires parametric geometry")
    derived, gg, _, _ = _derive_geom_with_cfl_resolution(work, geometry)
    cache = build_geom_op_cache(work, derived, gg, geometry)
    fractions = cache.fractions
    beta = fractions.beta
    beta_tilde = cache.beta_tilde
    cut = (beta > 0.0) & (beta < 1.0)
    floored = cut & (beta_tilde > beta + 1e-15)
    return {
        "cfg": work,
        "geometry": geometry,
        "derived": derived,
        "gg": gg,
        "fractions": fractions,
        "beta_tilde": beta_tilde,
        "floored": floored,
        "floor_stats": cache.floor_stats,
        "cache": cache,
    }


def _oriented_slice(arr: np.ndarray, plane: str, index: int) -> np.ndarray:
    """2-D slice with rows along the panel vertical axis, cols along horizontal."""
    if plane == "x":
        return np.asarray(arr[index, :, :]).T
    if plane == "y":
        return np.asarray(arr[:, index, :]).T
    return np.asarray(arr[:, :, index]).T


def _panel_extent_mm(gg, plane: str) -> tuple[float, float, float, float]:
    h = gg.h
    x0 = gg.x0
    if plane == "x":
        n_horiz, n_vert = gg.ny, gg.nz
        ih, iv = 1, 2
    elif plane == "y":
        n_horiz, n_vert = gg.nx, gg.nz
        ih, iv = 0, 2
    else:
        n_horiz, n_vert = gg.nx, gg.ny
        ih, iv = 0, 1
    h_lo = float(x0[ih]) * 1e3
    h_hi = float(x0[ih] + n_horiz * h) * 1e3
    v_lo = float(x0[iv]) * 1e3
    v_hi = float(x0[iv] + n_vert * h) * 1e3
    return h_lo, h_hi, v_lo, v_hi


def _marker_xy_mm(rows: np.ndarray, cols: np.ndarray, gg, plane: str) -> tuple[np.ndarray, np.ndarray]:
    """Cell-center (horizontal mm, vertical mm) for oriented panel row/col indices."""
    h = gg.h
    x0 = gg.x0
    if plane == "x":
        xm = (x0[1] + (cols + 0.5) * h) * 1e3
        ym = (x0[2] + (rows + 0.5) * h) * 1e3
    elif plane == "y":
        xm = (x0[0] + (cols + 0.5) * h) * 1e3
        ym = (x0[2] + (rows + 0.5) * h) * 1e3
    else:
        xm = (x0[0] + (cols + 0.5) * h) * 1e3
        ym = (x0[1] + (rows + 0.5) * h) * 1e3
    return xm, ym


def _slice_rgb_from_oriented(
    st: np.ndarray,
    bt: np.ndarray,
    fl: np.ndarray,
    rm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nrow, ncol = st.shape
    img = np.ones((nrow, ncol, 3), dtype=np.float64)
    solid = st == _STATE_SOLID
    fluid = (st == _STATE_FLUID) & ~rm
    cut = st == _STATE_CUT
    img[solid] = (0.25, 0.25, 0.25)
    img[fluid] = (1.0, 1.0, 1.0)
    if np.any(cut):
        v = np.clip(bt[cut], 0.0, 1.0)
        cmap = plt.get_cmap("viridis")
        img[cut] = cmap(v)[:, :3]
    return img, fl, rm, bt


def slice_image(
    state: np.ndarray,
    beta_tilde: np.ndarray,
    floored: np.ndarray,
    rad_mask: np.ndarray,
    plane: str,
    index: int,
    gg,
) -> dict[str, Any]:
    """Oriented 2-D slice for one orthogonal panel (rows=vertical, cols=horizontal)."""
    st = _oriented_slice(state, plane, index)
    bt = _oriented_slice(beta_tilde, plane, index)
    fl = _oriented_slice(floored, plane, index)
    rm = _oriented_slice(rad_mask, plane, index)
    img, fl, rm, bt = _slice_rgb_from_oriented(st, bt, fl, rm)
    return {
        "state": st,
        "beta_tilde": bt,
        "floored": fl,
        "radiator": rm,
        "image": img,
        "extent_mm": _panel_extent_mm(gg, plane),
        "plane": plane,
    }


def _format_slice_offset_mm(offset_m: float) -> str:
    mm = float(offset_m) * 1e3
    if abs(mm) < 0.005:
        mm = 0.0
    return f"{mm:.2f}"


def _cell_index_for_point(gg, xyz: tuple[float, float, float]) -> tuple[int, int, int]:
    h = gg.h
    x0 = gg.x0
    return (
        int(math.floor((xyz[0] - float(x0[0])) / h)),
        int(math.floor((xyz[1] - float(x0[1])) / h)),
        int(math.floor((xyz[2] - float(x0[2])) / h)),
    )


def _radiator_mask(fractions, nx: int, ny: int, nz: int) -> np.ndarray:
    mask = np.zeros((nx, ny, nz), dtype=bool)
    h2 = 1.0  # only compare a_c > 0
    for j, cells in enumerate(fractions.radiator_cell):
        areas = fractions.radiator_a[j]
        for flat, a in zip(cells, areas):
            if float(a) <= 0.0:
                continue
            i = int(flat) // (ny * nz)
            rem = int(flat) % (ny * nz)
            jj = rem // nz
            k = rem % nz
            mask[i, jj, k] = True
        _ = h2
    return mask


def _draw_cpml_huygens(ax, plane: str, gg, derived, rigid: dict[str, bool]) -> None:
    h = gg.h
    x0 = gg.x0
    slabs = gg.cpml_slabs
    i0, i1 = slabs.i_inner.lo, slabs.i_inner.hi
    j0, j1 = slabs.j_inner.lo, slabs.j_inner.hi
    k0, k1 = slabs.k_inner.lo, slabs.k_inner.hi
    cpml_lo = x0 + np.array([i0, j0, k0], dtype=_F64) * h
    cpml_hi = x0 + np.array([i1, j1, k1], dtype=_F64) * h

    offset = resolve_huygens_offset(gg.cfg)
    gap_cpml = (derived.gap_cells + derived.cpml_cells) * h
    lo_s = np.asarray(x0, dtype=_F64) + gap_cpml * (
        np.array(
            [
                0.0 if rigid.get("x-", False) else 1.0,
                0.0 if rigid.get("y-", False) else 1.0,
                0.0 if rigid.get("z-", False) else 1.0,
            ],
            dtype=_F64,
        )
    )
    hi_s = lo_s + np.array([derived.realized_W, derived.realized_H, derived.realized_D], dtype=_F64)
    domain_hi = x0 + np.array([gg.nx, gg.ny, gg.nz], dtype=_F64) * h
    patches = huygens_patches_planes(
        lo_s, hi_s, h, offset, rigid, x0, domain_hi
    )
    hx_lo = lo_s[0] - offset * h if not rigid.get("x-", False) else lo_s[0]
    hx_hi = hi_s[0] + offset * h if not rigid.get("x+", False) else hi_s[0]
    hy_lo = lo_s[1] - offset * h if not rigid.get("y-", False) else lo_s[1]
    hy_hi = hi_s[1] + offset * h if not rigid.get("y+", False) else hi_s[1]
    hz_lo = lo_s[2] - offset * h if not rigid.get("z-", False) else lo_s[2]
    hz_hi = hi_s[2] + offset * h if not rigid.get("z+", False) else hi_s[2]
    _ = patches

    def _rect_x(ax_, y0, y1, z0, z1, style: str) -> None:
        ax_.plot(
            [y0 * 1e3, y1 * 1e3, y1 * 1e3, y0 * 1e3, y0 * 1e3],
            [z0 * 1e3, z0 * 1e3, z1 * 1e3, z1 * 1e3, z0 * 1e3],
            linestyle=style,
            color="k",
            linewidth=0.8,
        )

    def _rect_y(ax_, x0m, x1m, z0, z1, style: str) -> None:
        ax_.plot(
            [x0m * 1e3, x1m * 1e3, x1m * 1e3, x0m * 1e3, x0m * 1e3],
            [z0 * 1e3, z0 * 1e3, z1 * 1e3, z1 * 1e3, z0 * 1e3],
            linestyle=style,
            color="k",
            linewidth=0.8,
        )

    def _rect_z(ax_, x0m, x1m, y0m, y1m, style: str) -> None:
        ax_.plot(
            [x0m * 1e3, x1m * 1e3, x1m * 1e3, x0m * 1e3, x0m * 1e3],
            [y0m * 1e3, y0m * 1e3, y1m * 1e3, y1m * 1e3, y0m * 1e3],
            linestyle=style,
            color="k",
            linewidth=0.8,
        )

    if plane == "x":
        _rect_x(ax, cpml_lo[1], cpml_hi[1], cpml_lo[2], cpml_hi[2], "--")
        _rect_x(ax, hy_lo, hy_hi, hz_lo, hz_hi, ":")
    elif plane == "y":
        _rect_y(ax, cpml_lo[0], cpml_hi[0], cpml_lo[2], cpml_hi[2], "--")
        _rect_y(ax, hx_lo, hx_hi, hz_lo, hz_hi, ":")
    else:
        _rect_z(ax, cpml_lo[0], cpml_hi[0], cpml_lo[1], cpml_hi[1], "--")
        _rect_z(ax, hx_lo, hx_hi, hy_lo, hy_hi, ":")


def plot_geometry_from_config(
    cfg: Config | str | Path,
    *,
    out: Path | None = None,
    slice_xyz: tuple[float, float, float] | None = None,
    h_override: float | None = None,
    show: bool = False,
    name: str | None = None,
) -> plt.Figure:
    if isinstance(cfg, (str, Path)):
        path = Path(cfg)
        cfg = Config.from_json(path.read_text(encoding="utf-8"))
        cfg_name = path.stem
    else:
        cfg_name = name if name is not None else "config"

    ctx = prepare_geometry_plot(cfg, h_override=h_override)
    geometry = ctx["geometry"]
    derived = ctx["derived"]
    gg = ctx["gg"]
    fractions = ctx["fractions"]
    beta_tilde = ctx["beta_tilde"]
    floored = ctx["floored"]
    floor_stats = ctx["floor_stats"]

    if slice_xyz is None:
        slice_xyz = _default_slice_point(ctx["cfg"], geometry, derived)
    ii, jj, kk = _cell_index_for_point(gg, slice_xyz)
    ii = int(np.clip(ii, 0, gg.nx - 1))
    jj = int(np.clip(jj, 0, gg.ny - 1))
    kk = int(np.clip(kk, 0, gg.nz - 1))

    rad_mask = _radiator_mask(fractions, gg.nx, gg.ny, gg.nz)
    state = fractions.state
    rigid = dict(gg.rigid_sides or {})
    h = gg.h
    x0 = gg.x0
    verts = geometry.solid.vertices
    tris = geometry.solid.faces

    fig = plt.figure(figsize=(12, 10))
    slice_axes = [
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3),
    ]
    slice_specs = [
        ("x", ii, slice_xyz[0], f"x = {_format_slice_offset_mm(slice_xyz[0])} mm"),
        ("y", jj, slice_xyz[1], f"y = {_format_slice_offset_mm(slice_xyz[1])} mm"),
        ("z", kk, slice_xyz[2], f"z = {_format_slice_offset_mm(slice_xyz[2])} mm"),
    ]
    axis_map = {"x": 0, "y": 1, "z": 2}

    for ax, (plane, idx, offset_m, title) in zip(slice_axes, slice_specs):
        panel = slice_image(state, beta_tilde, floored, rad_mask, plane, idx, gg)
        img = panel["image"]
        fl = panel["floored"]
        rm = panel["radiator"]
        ext = list(panel["extent_mm"])
        ax.imshow(img, origin="lower", aspect="equal", extent=ext)
        rows, cols = np.where(fl)
        if cols.size:
            px, py = _marker_xy_mm(rows, cols, gg, plane)
            ax.plot(px, py, "r.", markersize=2)
        rows, cols = np.where(rm)
        if cols.size:
            px, py = _marker_xy_mm(rows, cols, gg, plane)
            ax.plot(px, py, "b.", markersize=1.5, alpha=0.6)
        segs = triangle_plane_segments(verts, tris, axis=axis_map[plane], offset=offset_m)
        for p0, p1 in segs:
            if plane == "x":
                ax.plot([p0[1] * 1e3, p1[1] * 1e3], [p0[2] * 1e3, p1[2] * 1e3], "k-", lw=0.5)
            elif plane == "y":
                ax.plot([p0[0] * 1e3, p1[0] * 1e3], [p0[2] * 1e3, p1[2] * 1e3], "k-", lw=0.5)
            else:
                ax.plot([p0[0] * 1e3, p1[0] * 1e3], [p0[1] * 1e3, p1[1] * 1e3], "k-", lw=0.5)
        _draw_cpml_huygens(ax, plane, gg, derived, rigid)
        if plane == "x":
            ax.set_xlabel("y (mm)")
            ax.set_ylabel("z (mm)")
        elif plane == "y":
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("z (mm)")
        else:
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
        ax.set_title(title)

    ax3d = fig.add_subplot(2, 2, 4, projection="3d")
    cells, areas, centroids, _ = wetted_wall_arrays(fractions, beta_tilde, h)
    n_pts = int(centroids.shape[0])
    stride = max(1, int(math.ceil(n_pts / _MAX_SCATTER)))
    pts = centroids[::stride]
    n_show = pts.shape[0]
    bt_show = beta_tilde.reshape(-1)[cells[::stride]] if cells.size else np.array([])
    floor_show = floored.reshape(-1)[cells[::stride]] if cells.size else np.array([], dtype=bool)
    rad_show = rad_mask.reshape(-1)[cells[::stride]] if cells.size else np.array([], dtype=bool)
    colors = np.zeros((n_show, 3))
    cmap = plt.get_cmap("viridis")
    if n_show:
        colors[~floor_show & ~rad_show] = cmap(np.clip(bt_show[~floor_show & ~rad_show], 0, 1))[:, :3]
        colors[floor_show] = (1.0, 0.0, 0.0)
        colors[rad_show] = (0.0, 0.0, 1.0)
    ax3d.scatter(pts[:, 0] * 1e3, pts[:, 1] * 1e3, pts[:, 2] * 1e3, c=colors, s=1, linewidths=0)
    ax3d.set_xlabel("x (mm)")
    ax3d.set_ylabel("z (mm)")
    ax3d.set_zlabel("y (mm)")
    ax3d.set_title(f"wetted wall points (n={n_show} of {n_pts})")

    fr = fractions.stats
    n_cut = fr.get("n_cut_cells", 0)
    n_floor = floor_stats.get("n_floored", 0)
    deh = floor_stats.get("delta_eq_over_h", 0.0)
    cfl = derived.cfl_safety_resolved
    fig.suptitle(
        f"{cfg_name}  h={h * 1e3:.4g} mm  cells={derived.total_cells:,}  "
        f"n_cut={n_cut}  n_floored={n_floor}  δ_eq/h={deh:.4g}  cfl={cfl:.4g}"
    )
    fig.tight_layout()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


plot_geometry = plot_geometry_from_config