"""Geometry visualization for boxrad configs and results."""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from boxrad.config import Config, derive, format_realized_box_dims
from boxrad.grid import Grid, radiator_area_exact, radiator_outline_loops, _radiator_face_weights
from boxrad.post.kirchhoff import huygens_box_patches, huygens_open_patches
from boxrad.recorder import resolve_huygens_offset

_ORIGIN_NOTE = "origin = radiator center, +z radiating"
_MESH_LINE_CAP = 200
_WEIGHT_EPS = 1e-6
_TITLE_WRAP = 50
_LIMIT_MARGIN_FRAC = 0.05


class _Bounds3D:
    def __init__(self) -> None:
        self.xmin = math.inf
        self.xmax = -math.inf
        self.ymin = math.inf
        self.ymax = -math.inf
        self.zmin = math.inf
        self.zmax = -math.inf

    def include(self, x: float, y: float, z: float) -> None:
        self.xmin = min(self.xmin, float(x))
        self.xmax = max(self.xmax, float(x))
        self.ymin = min(self.ymin, float(y))
        self.ymax = max(self.ymax, float(y))
        self.zmin = min(self.zmin, float(z))
        self.zmax = max(self.zmax, float(z))

    def include_arrays(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
    ) -> None:
        for x, y, z in zip(xs.ravel(), ys.ravel(), zs.ravel(), strict=True):
            self.include(float(x), float(y), float(z))

    def limits(self, *, margin_frac: float = _LIMIT_MARGIN_FRAC) -> tuple[float, float, float]:
        def _lim(lo: float, hi: float) -> tuple[float, float]:
            if not math.isfinite(lo) or not math.isfinite(hi):
                return -0.05, 0.05
            if hi < lo:
                lo, hi = hi, lo
            span = hi - lo
            if span < 1e-15:
                pad = max(abs(lo), abs(hi), 0.05) * margin_frac
                return lo - pad, hi + pad
            pad = span * margin_frac
            return lo - pad, hi + pad

        xlo, xhi = _lim(self.xmin, self.xmax)
        ylo, yhi = _lim(self.ymin, self.ymax)
        zlo, zhi = _lim(self.zmin, self.zmax)
        return (xlo, xhi), (ylo, yhi), (zlo, zhi)


def _load_source(source: Config | str | Path | Any) -> tuple[Config, object]:
    from boxrad.result import Result

    if isinstance(source, Result):
        return source.config, source.derived
    if isinstance(source, Config):
        return source, derive(source)
    path = Path(source)
    if path.suffix.lower() in (".h5", ".hdf5"):
        res = Result(path)
        return res.config, res.derived
    return Config.from_json(path.read_text(encoding="utf-8")), None


def _face_indices(grid: Grid) -> tuple[int, int, int, int]:
    assert grid.i_box is not None and grid.j_box is not None
    i0, i1 = grid.i_box, grid.i_box + grid.nbx
    j0, j1 = grid.j_box, grid.j_box + grid.nby
    return i0, i1, j0, j1


def _face_bounds(grid: Grid) -> tuple[float, float, float, float, float]:
    """Front-face axis-aligned limits in the grid frame (radiator center at origin)."""
    i0, i1, j0, j1 = _face_indices(grid)
    x0 = float(grid.x_vx[i0])
    x1 = float(grid.x_vx[i1])
    y0 = float(grid.y_vy[j0])
    y1 = float(grid.y_vy[j1])
    z_front = float(grid.z_vz[grid.k_face_z]) if grid.k_face_z is not None else 0.0
    return x0, x1, y0, y1, z_front


def _radiator_outline_xy(cfg: Config) -> list[np.ndarray]:
    """Radiator outline in the grid frame (shape centered at the radiator origin)."""
    return radiator_outline_loops(cfg.radiator)


def _face_weight_plane(cfg: Config, grid: Grid, h: float) -> np.ndarray:
    i0, i1, j0, j1 = _face_indices(grid)
    w_plane = _radiator_face_weights(
        cfg.radiator,
        np.asarray(grid.x_p[i0:i1], dtype=np.float64),
        np.asarray(grid.y_p[j0:j1], dtype=np.float64),
        h,
        grid.radiator_supersample,
    )
    return np.asarray(w_plane, dtype=np.float64)


def _weight_centroid_xy(w_plane: np.ndarray, x_c: np.ndarray, y_c: np.ndarray) -> tuple[float, float]:
    ws = float(np.sum(w_plane))
    if ws <= 0.0:
        return 0.0, 0.0
    cx = float(np.sum(w_plane * x_c[:, np.newaxis]) / ws)
    cy = float(np.sum(w_plane * y_c[np.newaxis, :]) / ws)
    return cx, cy


def _outline_centroid_xy(outlines: list[np.ndarray]) -> tuple[float, float]:
    """Area centroid of closed outline loop(s) in the radiator-centered frame."""
    cx_acc = 0.0
    cy_acc = 0.0
    a_acc = 0.0
    for loop in outlines:
        pts = loop
        if pts.shape[0] > 1 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        if pts.shape[0] < 3:
            cx_acc += float(np.sum(pts[:, 0]))
            cy_acc += float(np.sum(pts[:, 1]))
            a_acc += float(pts.shape[0])
            continue
        x = pts[:, 0]
        y = pts[:, 1]
        x1 = np.roll(x, -1)
        y1 = np.roll(y, -1)
        cross = x * y1 - x1 * y
        a = 0.5 * float(np.sum(cross))
        if abs(a) < 1e-30:
            cx_acc += float(np.sum(x))
            cy_acc += float(np.sum(y))
            a_acc += float(x.size)
            continue
        cx = float(np.sum((x + x1) * cross) / (6.0 * a))
        cy = float(np.sum((y + y1) * cross) / (6.0 * a))
        cx_acc += a * cx
        cy_acc += a * cy
        a_acc += a
    if abs(a_acc) < 1e-30:
        return 0.0, 0.0
    return cx_acc / a_acc, cy_acc / a_acc


def _wrap_title(text: str, *, width: int = _TITLE_WRAP) -> str:
    return textwrap.fill(text, width=width)


def _draw_yee_mesh(
    ax,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    title_stride_note: str = "",
) -> int:
    """Draw Yee cell boundaries on the front face; return line stride used."""
    nx = x_edges.size - 1
    ny = y_edges.size - 1
    stride = 1
    if max(nx, ny) > _MESH_LINE_CAP:
        stride = int(math.ceil(max(nx, ny) / _MESH_LINE_CAP))
    for i in range(0, x_edges.size, stride):
        ax.axvline(float(x_edges[i]), color="0.45", lw=0.3, alpha=0.55, zorder=3)
    for j in range(0, y_edges.size, stride):
        ax.axhline(float(y_edges[j]), color="0.45", lw=0.3, alpha=0.55, zorder=3)
    if stride > 1 and title_stride_note == "":
        return stride
    return stride


def plot_face_mesh(
    source: Config | str | Path | Any,
    ax=None,
    *,
    show: bool = True,
    out: str | Path | None = None,
    colorbar_pad: float | None = None,
    **mesh_kwargs: Any,
) -> plt.Figure:
    """Plot the front face: per-cell radiator area weights on the Yee grid.

    Shows the weight ``w ∈ [0, 1]`` of each ``h × h`` front-face cell, the exact
    radiator outline (red), the face outline, the radiator center (origin) and
    the face center. The title lists ``h``, the cell count, ``Σw``,
    ``S_grid = h² Σw`` and the exact area ``S``. Mesh lines are thinned above
    200 cells per side.

    Args:
        source: ``Config``, :class:`~boxrad.result.Result`, or a path to a JSON
            config or ``.h5``/``.hdf5`` result.
        ax: Existing Matplotlib axes to draw into; a new figure is created if None.
        show: Call ``plt.show()`` (only when this function created the figure).
        out: If given, save the figure as PNG to this path (150 dpi).
        colorbar_pad: Optional ``pad`` passed to ``fig.colorbar``.
        **mesh_kwargs: Extra keyword arguments for ``ax.pcolormesh``.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: Unknown radiator shape.
    """
    cfg, derived_in = _load_source(source)
    derived = derived_in if derived_in is not None else derive(cfg)
    grid = Grid(cfg, derived)
    h = float(derived.h)
    i0, i1, j0, j1 = _face_indices(grid)
    x0, x1, y0, y1, _ = _face_bounds(grid)
    x_c = np.asarray(grid.x_p[i0:i1], dtype=np.float64)
    y_c = np.asarray(grid.y_p[j0:j1], dtype=np.float64)
    nx = x_c.size
    ny = y_c.size
    x_e = x_c[0] - 0.5 * h + h * np.arange(nx + 1, dtype=np.float64)
    y_e = y_c[0] - 0.5 * h + h * np.arange(ny + 1, dtype=np.float64)

    w_plane = _face_weight_plane(cfg, grid, h)
    n_cells = int(np.count_nonzero(w_plane > _WEIGHT_EPS))
    sum_w = float(np.sum(w_plane))
    s_grid = float(sum_w * h * h)
    s_exact = float(radiator_area_exact(cfg.radiator))

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6.5, 6), layout="constrained")
        ax = fig.add_subplot(1, 1, 1)
        created_fig = True
    else:
        fig = ax.figure

    c = w_plane.T
    assert c.shape == (len(y_e) - 1, len(x_e) - 1), (
        f"pcolormesh C shape {c.shape} != (ny, nx)=({len(y_e) - 1}, {len(x_e) - 1})"
    )
    mesh = ax.pcolormesh(
        x_e,
        y_e,
        c,
        shading="flat",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        edgecolors="none",
        zorder=2,
        **mesh_kwargs,
    )
    cbar_kw: dict[str, Any] = {"shrink": 0.8, "label": "radiator weight"}
    if colorbar_pad is not None:
        cbar_kw["pad"] = colorbar_pad
    fig.colorbar(mesh, ax=ax, **cbar_kw)

    stride = _draw_yee_mesh(ax, x_e, y_e)
    stride_note = f"; grid stride {stride}" if stride > 1 else ""

    outlines = _radiator_outline_xy(cfg)
    for loop in outlines:
        ax.plot(loop[:, 0], loop[:, 1], "r-", lw=2, zorder=4)
    ax.plot(
        [x0, x1, x1, x0, x0],
        [y0, y0, y1, y1, y0],
        "k-",
        lw=1.2,
        zorder=3,
    )

    ax.scatter([0.0], [0.0], c="k", s=28, zorder=5, label="radiator center")
    face_cx = 0.5 * (x0 + x1)
    face_cy = 0.5 * (y0 + y1)
    ax.scatter(
        [face_cx],
        [face_cy],
        marker="+",
        c="k",
        s=80,
        lw=1.2,
        zorder=5,
        label="face center",
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_xlim(x0 - h, x1 + h)
    ax.set_ylim(y0 - h, y1 + h)
    title = (
        f"front face, h = {h * 1e3:.3g} mm: {n_cells} cells "
        f"(Σw = {sum_w:.4g}), "
        f"S_grid = {s_grid * 1e4:.4g} cm², S = {s_exact * 1e4:.4g} cm²"
        f"{stride_note}"
    )
    ax.set_title(_wrap_title(title))

    if out is not None:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    if show and created_fig:
        plt.show()
    return fig


def _box_3d_verts(grid: Grid) -> tuple[list, tuple[float, float, float, float, float, float]]:
    i0, i1, j0, j1 = _face_indices(grid)
    kb = int(grid.k_box)
    kb1 = kb + int(grid.nbz)
    x0, x1 = float(grid.x_vx[i0]), float(grid.x_vx[i1])
    y0, y1 = float(grid.y_vy[j0]), float(grid.y_vy[j1])
    z0, z1 = float(grid.z_vz[kb]), float(grid.z_vz[kb1])
    verts = [
        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],
        [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
        [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
        [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]],
        [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],
    ]
    return verts, (x0, x1, y0, y1, z0, z1)


def plot_geometry(
    source: Config | str | Path | Any,
    ax=None,
    *,
    show_huygens: bool = False,
    show_probes: bool = True,
    out: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Two-panel geometry figure: 3-D view (left) and front-face mesh (right).

    The 3-D panel shows the realized (h-snapped) box, or the baffle plane, the
    radiator outline, the origin and the ``+z`` radiating direction (green
    arrow), plus optional Huygens patch centers and probes. The right panel is
    :func:`plot_face_mesh`. Coordinates are in m with the origin at the
    radiator center.

    Args:
        source: ``Config``, :class:`~boxrad.result.Result`, or a path to a JSON
            config or ``.h5``/``.hdf5`` result.
        ax: Unused; a new figure is always created.
        show_huygens: Overlay Huygens-surface patch centers (orange).
        show_probes: Mark ``output.probes`` targets (magenta).
        out: If given, save the figure as PNG to this path (150 dpi).
        show: Call ``plt.show()`` (blocks in interactive back ends).

    Returns:
        The Matplotlib figure.
    """
    cfg, derived_in = _load_source(source)
    derived = derived_in if derived_in is not None else derive(cfg)
    grid = Grid(cfg, derived)
    h = float(derived.h)
    W, H, D = derived.realized_W, derived.realized_H, derived.realized_D
    f_min, f_max = cfg.band.f_min, float(derived.f_max)
    x0, x1, y0, y1, z_front = _face_bounds(grid)

    fig = plt.figure(figsize=(13, 6), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    ax_face = fig.add_subplot(gs[0, 1])
    bounds = _Bounds3D()

    if cfg.geometry == "box":
        verts, (xb0, xb1, yb0, yb1, zb0, zb1) = _box_3d_verts(grid)
        for corners in verts:
            for x_a, y_a, z_a in corners:
                bounds.include(x_a, y_a, z_a)
        box_mesh = Poly3DCollection(verts, alpha=0.12, facecolor="steelblue", edgecolor="k")
        ax3.add_collection3d(box_mesh)
        edges = [
            (xb0, yb0, zb0, xb1, yb0, zb0),
            (xb0, yb1, zb0, xb1, yb1, zb0),
            (xb1, yb0, zb0, xb1, yb1, zb0),
            (xb0, yb0, zb0, xb0, yb1, zb0),
            (xb0, yb0, zb1, xb1, yb0, zb1),
            (xb0, yb1, zb1, xb1, yb1, zb1),
            (xb1, yb0, zb1, xb1, yb1, zb1),
            (xb0, yb0, zb1, xb0, yb1, zb1),
            (xb0, yb0, zb0, xb0, yb0, zb1),
            (xb1, yb0, zb0, xb1, yb0, zb1),
            (xb0, yb1, zb0, xb0, yb1, zb1),
            (xb1, yb1, zb0, xb1, yb1, zb1),
        ]
        for x_a, y_a, z_a, x_b, y_b, z_b in edges:
            ax3.plot([x_a, x_b], [y_a, y_b], [z_a, z_b], "k-", lw=0.8, alpha=0.5)
        span_x, span_y, span_z = xb1 - xb0, yb1 - yb0, zb1 - zb0
    else:
        for x_a, y_a in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            bounds.include(x_a, y_a, 0.0)
        xx, yy = np.meshgrid([x0, x1], [y0, y1], indexing="xy")
        zz = np.zeros_like(xx)
        ax3.plot_surface(xx, yy, zz, alpha=0.15, color="gray")
        span_x, span_y, span_z = x1 - x0, y1 - y0, h

    if show_huygens:
        off = resolve_huygens_offset(cfg)
        depth = 0.0 if cfg.geometry == "baffle" else D
        assert grid.k_box is not None
        z_back = float(grid.z_vz[int(grid.k_box)])
        origin_hy = (x0, y0, z_back if cfg.geometry == "box" else 0.0)
        if cfg.geometry == "baffle":
            hy = huygens_open_patches(W, H, depth, h, origin_hy, offset_cells=off)
        else:
            hy = huygens_box_patches(W, H, D, h, origin_hy, offset_cells=off)
        bounds.include_arrays(hy.centers[:, 0], hy.centers[:, 1], hy.centers[:, 2])
        ax3.scatter(
            hy.centers[:, 0],
            hy.centers[:, 1],
            hy.centers[:, 2],
            s=4,
            c="orange",
            alpha=0.5,
            depthshade=False,
        )

    if show_probes and cfg.output.probes:
        for row in cfg.output.probes:
            bounds.include(row[0], row[1], row[2])
            ax3.scatter([row[0]], [row[1]], [row[2]], c="magenta", s=30)

    outlines = _radiator_outline_xy(cfg)
    for loop in outlines:
        bounds.include_arrays(loop[:, 0], loop[:, 1], np.full(loop.shape[0], z_front))
        ax3.plot(loop[:, 0], loop[:, 1], np.full(loop.shape[0], z_front), "r-", lw=2)

    bounds.include(0.0, 0.0, z_front)
    ax3.scatter([0.0], [0.0], [z_front], c="k", s=24)
    arrow_len = 0.15 * max(span_x, span_y, span_z, 0.05)
    bounds.include(0.0, 0.0, z_front + arrow_len)
    ax3.quiver(
        0.0,
        0.0,
        z_front,
        0.0,
        0.0,
        arrow_len,
        color="green",
        arrow_length_ratio=0.2,
    )
    (xlim, ylim, zlim) = bounds.limits()
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_zlim(zlim)
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]
    dz = zlim[1] - zlim[0]
    ax3.set_xlabel("x [m]", labelpad=8)
    ax3.set_ylabel("y [m]", labelpad=8)
    ax3.set_zlabel("z [m]", labelpad=8)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_major_locator(ticker.MaxNLocator(4))
    ax3.set_box_aspect((dx, dy, dz))

    plot_face_mesh(source, ax=ax_face, show=False, colorbar_pad=0.02)

    rad = cfg.radiator
    dim_s = "×".join(f"{x:.4g}" for x in rad.dims)
    box_line = format_realized_box_dims(cfg, derived)
    title = (
        f"{box_line}; "
        f"radiator {rad.shape} {dim_s} m @ ({rad.x_c:g},{rad.y_c:g}); "
        f"h={h*1e3:.3g} mm; band {f_min:g}–{f_max:g} Hz; {_ORIGIN_NOTE}"
    )
    fig.suptitle(_wrap_title(title, width=72))

    if out is not None:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def main(argv: list[str] | None = None) -> int:
    """``boxrad-plot CONFIG_OR_H5 [-o PNG] [--no-show] [--huygens] [--face]``.

    Args:
        argv: Argument list without the program name (default ``sys.argv[1:]``).

    Returns:
        Exit code 0.
    """
    parser = argparse.ArgumentParser(description="Plot boxrad geometry from config or HDF5")
    parser.add_argument("config_or_h5", type=Path, help="JSON config or result HDF5")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Also write PNG to this path")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive window (use with -o to save PNG only)",
    )
    parser.add_argument("--huygens", action="store_true", help="Overlay Huygens patch centers")
    parser.add_argument("--face", action="store_true", help="Show only the front-face mesh panel")
    parser.add_argument(
        "--geometry",
        action="store_true",
        help="Parametric voxel/mesh view from JSON config (no HDF5 required)",
    )
    parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help="Slice point x,y,z in meters (default: radiator 0 / test source / bbox center)",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=None,
        dest="h_override",
        help="Override numerics.h_override for a quick coarse/fine look",
    )
    args = parser.parse_args(argv)
    show = not args.no_show
    if args.geometry:
        from boxrad.plot.geometry import _parse_slice, plot_geometry_from_config

        slice_xyz = _parse_slice(args.slice)
        plot_geometry_from_config(
            args.config_or_h5,
            out=args.output,
            slice_xyz=slice_xyz,
            h_override=args.h_override,
            show=show,
        )
    elif args.face:
        plot_face_mesh(args.config_or_h5, out=args.output, show=show)
    else:
        plot_geometry(
            args.config_or_h5,
            show_huygens=args.huygens,
            out=args.output,
            show=show,
        )
    if args.output is not None:
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())