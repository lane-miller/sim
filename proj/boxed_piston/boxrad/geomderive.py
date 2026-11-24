"""Parametric-geometry grid derivation (v0.3 step 6b1); mirrors :func:`boxrad.config.derive`."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from boxrad.config import (
    CPML_SIDES_ALL,
    DEFAULT_GAP_CELLS,
    F_MAX_CAP_HZ,
    SQRT3,
    Config,
    Derived,
    _memory_bytes,
    _radiator_D,
    provisional_geom_cfl,
    pulse_params,
    resolve_decimation,
    test_source_active,
)
from boxrad.geom.geometry import Geometry
from boxrad.cutcell import _side_key
from boxrad.geomconfig import PlaneSpec
from boxrad.grid import SlabRange, build_cpml_slabs

if TYPE_CHECKING:
    from boxrad.geomconfig import GeometrySpec, ShapeSpec

_F64 = np.float64
_AXIS = ("x", "y", "z")


@dataclass(frozen=True)
class GeomGrid:
    """Structured grid for parametric runs (duck-types :class:`~boxrad.grid.Grid` for fields/CPML)."""

    nx: int
    ny: int
    nz: int
    h: float
    cpml: int
    x0: np.ndarray
    cpml_slabs: object
    cfg: Config
    rigid_sides: dict[str, bool] | None = None
    planes: tuple = ()
    h_nominal: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "x0", np.asarray(self.x0, dtype=_F64).reshape(3))
        if self.rigid_sides is None:
            object.__setattr__(self, "rigid_sides", {})
        if self.h_nominal <= 0.0:
            object.__setattr__(self, "h_nominal", float(self.h))


def cut_grid(gg: GeomGrid):
    """Cut-cell background grid aligned to *gg*."""
    from boxrad.cutcell import Grid as CutGrid

    return CutGrid(
        x0=gg.x0,
        h=gg.h,
        shape=(gg.nx, gg.ny, gg.nz),
        rigid_sides=dict(gg.rigid_sides or {}),
        h_nominal=float(gg.h_nominal),
    )


def _shape_D(shape: ShapeSpec) -> float:
    if shape.kind == "circle":
        return float(shape.params["d"])
    if shape.kind in ("rect", "racetrack"):
        return min(float(shape.params["w"]), float(shape.params["h"]))
    if shape.kind == "ellipse":
        return 2.0 * min(float(shape.params["a"]), float(shape.params["b"]))
    raise ValueError(f"unsupported radiator shape kind {shape.kind!r}")


def _radiator_D_min(spec: GeometrySpec) -> float:
    dims = [_shape_D(rd.shape) for rd in spec.radiators if rd.shape is not None]
    if not dims:
        raise ValueError("no radiator shapes in geometry_spec")
    return min(dims)


def _snap_bbox(
    lo: np.ndarray,
    hi: np.ndarray,
    h: float,
    anchors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lo_s = np.empty(3, dtype=_F64)
    hi_s = np.empty(3, dtype=_F64)
    for a in range(3):
        anchor = float(anchors[a])
        lo_s[a] = anchor + math.floor((float(lo[a]) - anchor) / h + 1e-9) * h
        hi_s[a] = anchor + math.ceil((float(hi[a]) - anchor) / h - 1e-9) * h
    return lo_s, hi_s


def _anchors_axis(planes: tuple[PlaneSpec, ...], n_hat: np.ndarray, pos0: np.ndarray, lo_a: float, axis: int) -> list[float]:
    anc: list[float] = []
    for pl in planes:
        if pl.axis == _AXIS[axis]:
            anc.append(float(pl.offset))
    if abs(float(n_hat[axis])) > 1.0 - 1e-12:
        anc.append(float(pos0[axis]))
    if not anc:
        anc.append(float(lo_a))
    return sorted(set(anc))


def _shrink_h_isotropic(h: float, planes: tuple[PlaneSpec, ...], n_hat: np.ndarray, pos0: np.ndarray, lo: np.ndarray) -> float:
    h_use = float(h)
    for axis in range(3):
        anc = _anchors_axis(planes, n_hat, pos0, float(lo[axis]), axis)
        if len(anc) == 2:
            d = abs(anc[1] - anc[0])
            if d > 0.0:
                ncell = int(math.ceil(d / h_use - 1e-9))
                h_use = min(h_use, d / max(ncell, 1))
    return h_use


def _plane_crossing_errors(
    geometry: Geometry,
    planes: tuple[PlaneSpec, ...],
) -> None:
    xlo, ylo, zlo, xhi, yhi, zhi = geometry.solid.bbox
    lo = np.array([xlo, ylo, zlo], dtype=_F64)
    hi = np.array([xhi, yhi, zhi], dtype=_F64)
    l_bbox = float(np.max(hi - lo))
    eps = 1e-9 * max(1.0, l_bbox)
    for pl in planes:
        axis = _AXIS.index(pl.axis)
        off = float(pl.offset)
        if pl.side == "-":
            if float(lo[axis]) < off - eps:
                raise ValueError(
                    f"solid crosses rigid plane {pl.axis}- at offset {off} by more than tolerance"
                )
        elif float(hi[axis]) > off + eps:
            raise ValueError(
                f"solid crosses rigid plane {pl.axis}+ at offset {off} by more than tolerance"
            )


def _plane_offset(planes: tuple[PlaneSpec, ...], axis: int, side: str) -> float | None:
    for pl in planes:
        if pl.axis == _AXIS[axis] and pl.side == side:
            return float(pl.offset)
    return None


def _cpml_slabs_for_planes(
    nx: int,
    ny: int,
    nz: int,
    cpml_cells: int,
    rigid_sides: dict[str, bool],
) -> object:
    slabs = build_cpml_slabs(nx, ny, nz, cpml_cells)
    i_lo = 0 if rigid_sides.get("x-", False) else cpml_cells
    i_hi = nx if rigid_sides.get("x+", False) else nx - cpml_cells
    j_lo = 0 if rigid_sides.get("y-", False) else cpml_cells
    j_hi = ny if rigid_sides.get("y+", False) else ny - cpml_cells
    k_lo = 0 if rigid_sides.get("z-", False) else cpml_cells
    k_hi = nz if rigid_sides.get("z+", False) else nz - cpml_cells
    return replace(
        slabs,
        i_inner=SlabRange(i_lo, i_hi),
        j_inner=SlabRange(j_lo, j_hi),
        k_inner=SlabRange(k_lo, k_hi),
    )


def derive_geom(
    cfg: Config,
    geometry: Geometry,
    *,
    cfl_safety: float | None = None,
) -> tuple[Derived, GeomGrid]:
    """Compute derived parameters and a :class:`GeomGrid` from solid bbox and radiator rules."""
    from boxrad.geomconfig import check_geometry_v03_guards

    check_geometry_v03_guards(cfg)
    spec = cfg.geometry_spec
    if spec is None:
        raise ValueError("derive_geom requires geometry_spec")

    c = cfg.air.c
    num = cfg.numerics
    gap_cells = DEFAULT_GAP_CELLS if num.gap_cells is None else num.gap_cells
    cpml_cells = num.cpml_cells

    if spec.radiators:
        D_min = _radiator_D_min(spec)
    else:
        D_min = _radiator_D(cfg.radiator)
    f_ka10 = 10.0 * c / (math.pi * D_min)
    f_max = cfg.band.f_max if cfg.band.f_max is not None else min(F_MAX_CAP_HZ, f_ka10)
    lambda_min = c / f_max

    if num.h_override is not None:
        h_nominal = float(num.h_override)
    else:
        h_nominal = min(lambda_min / num.ppw, D_min / num.cells_across_radiator)

    planes = spec.environment.planes
    if geometry.solid.faces.shape[0] > 0:
        _plane_crossing_errors(geometry, planes)

    ts_pos: np.ndarray | None = None
    if test_source_active(cfg):
        assert cfg.numerics.test_source is not None
        ts_pos = np.asarray(cfg.numerics.test_source.position, dtype=_F64).reshape(3)
    if geometry.solid.faces.shape[0] == 0:
        if ts_pos is None:
            raise ValueError("empty solid requires numerics.test_source")
        lo = ts_pos - h_nominal
        hi = ts_pos + h_nominal
    else:
        xlo, ylo, zlo, xhi, yhi, zhi = geometry.solid.bbox
        lo = np.array([xlo, ylo, zlo], dtype=_F64)
        hi = np.array([xhi, yhi, zhi], dtype=_F64)
        if ts_pos is not None:
            lo = np.minimum(lo, ts_pos)
            hi = np.maximum(hi, ts_pos)

    if geometry.radiators:
        n_hat = geometry.radiators[0].pose.R[:, 2]
        n_hat = n_hat / np.linalg.norm(n_hat)
        pos0 = geometry.radiators[0].pose.position.astype(_F64)
    elif ts_pos is not None:
        pos0 = ts_pos.copy()
        n_hat = np.array([0.0, 0.0, 1.0], dtype=_F64)
    else:
        raise ValueError("parametric derive requires a radiator or numerics.test_source")

    h = _shrink_h_isotropic(h_nominal, planes, n_hat, pos0, lo)

    anchors = np.empty(3, dtype=_F64)
    for a in range(3):
        anc = _anchors_axis(planes, n_hat, pos0, float(lo[a]), a)
        anchors[a] = anc[0]
        if (
            not geometry.radiators
            and ts_pos is not None
            and _plane_offset(planes, a, "-") is None
            and _plane_offset(planes, a, "+") is None
        ):
            # test source at a cell center: faces at ts ± h/2
            anchors[a] = float(ts_pos[a]) - 0.5 * h

    lo_s, hi_s = _snap_bbox(lo, hi, h, anchors)

    rigid_sides: dict[str, bool] = {f"{pl.axis}{pl.side}": True for pl in planes}

    lo_d = lo_s.copy()
    hi_d = hi_s.copy()
    for a in range(3):
        neg_key = _side_key(a, "-")
        pos_key = _side_key(a, "+")
        if rigid_sides.get(neg_key, False):
            plane_off = _plane_offset(planes, a, "-")
            if plane_off is None:
                raise AssertionError(f"missing plane offset for rigid {neg_key}")
            gap = float(lo_s[a]) - plane_off
            if gap > 1e-12:
                n_gap = gap / h
                if abs(n_gap - round(n_gap)) > 1e-9:
                    raise AssertionError(
                        f"plane-to-bbox gap not integer cells on {_AXIS[a]}"
                    )
            lo_d[a] = plane_off
        if rigid_sides.get(pos_key, False):
            plane_off = _plane_offset(planes, a, "+")
            if plane_off is None:
                raise AssertionError(f"missing plane offset for rigid {pos_key}")
            gap = plane_off - float(hi_s[a])
            if gap > 1e-12:
                n_gap = gap / h
                if abs(n_gap - round(n_gap)) > 1e-9:
                    raise AssertionError(
                        f"bbox-to-plane gap not integer cells on {_AXIS[a]}"
                    )
            hi_d[a] = plane_off

    nbx = max(1, round((hi_d[0] - lo_d[0]) / h))
    nby = max(1, round((hi_d[1] - lo_d[1]) / h))
    nbz = max(1, round((hi_d[2] - lo_d[2]) / h))
    rw, rh, rd = nbx * h, nby * h, nbz * h
    gap_cpml = gap_cells + cpml_cells
    pad_lo = np.empty(3, dtype=np.int64)
    pad_hi = np.empty(3, dtype=np.int64)
    for a in range(3):
        pad_lo[a] = 0 if rigid_sides.get(_side_key(a, "-"), False) else gap_cpml
        pad_hi[a] = 0 if rigid_sides.get(_side_key(a, "+"), False) else gap_cpml

    nx = nbx + int(pad_lo[0]) + int(pad_hi[0])
    ny = nby + int(pad_lo[1]) + int(pad_hi[1])
    nz = nbz + int(pad_lo[2]) + int(pad_hi[2])

    cpml_sides = list(CPML_SIDES_ALL)
    for pl in planes:
        idx = _AXIS.index(pl.axis)
        cpml_sides[2 * idx + (0 if pl.side == "-" else 1)] = False
    total_cells = nx * ny * nz
    cell_budget_exceeded = total_cells > num.cell_budget

    cfl = float(cfl_safety) if cfl_safety is not None else provisional_geom_cfl(num)
    dt = cfl * h / (c * SQRT3)
    gap_m = gap_cells * h
    cpml_alpha = num.cpml_alpha_factor * math.pi * cfg.band.f_min
    memory_bytes = _memory_bytes(nx, ny, nz, cpml_cells)

    _pulse_der = Derived(
        f_max=f_max,
        lambda_min=lambda_min,
        h=h,
        dt=0.0,
        realized_W=0.0,
        realized_H=0.0,
        realized_D=0.0,
        radiator_ic=0,
        radiator_jc=0,
        box_nx=0,
        box_ny=0,
        box_nz=0,
        gap_cells=0,
        gap_m=0.0,
        cpml_cells=0,
        cpml_alpha=0.0,
        nx=0,
        ny=0,
        nz=0,
        total_cells=0,
        memory_bytes=0.0,
        pulse_sigma=0.0,
        pulse_t0=0.0,
        run_time_s=0.0,
        n_steps=0,
        ppw_realized=0.0,
        cpml_sides=CPML_SIDES_ALL,
        geometry="parametric",
    )
    sigma, t0 = pulse_params(cfg, _pulse_der)

    dom_extents = [
        (nbx + (0 if rigid_sides.get("x-", False) else gap_cells) + (0 if rigid_sides.get("x+", False) else gap_cells)) * h,
        (nby + (0 if rigid_sides.get("y-", False) else gap_cells) + (0 if rigid_sides.get("y+", False) else gap_cells)) * h,
        (nbz + (0 if rigid_sides.get("z-", False) else gap_cells) + (0 if rigid_sides.get("z+", False) else gap_cells)) * h,
    ]
    l_dom = max(dom_extents)
    run_time_s = t0 + 6.0 * sigma + num.n_transits * l_dom / c
    n_steps = max(1, math.ceil(run_time_s / dt))
    ppw_realized = lambda_min / h

    decimation_m, alias_level_db = resolve_decimation(
        cfg, f_max=f_max, sigma=sigma, t0=t0, dt=dt
    )

    x0 = lo_d - pad_lo.astype(_F64) * h
    for a in range(3):
        extent = float(hi_d[a] - lo_d[a]) + float(pad_lo[a] + pad_hi[a]) * h
        n_cells = (nbx, nby, nbz)[a] + int(pad_lo[a]) + int(pad_hi[a])
        if abs(extent / h - n_cells) > 1e-9:
            raise AssertionError(f"grid extent not integer cells on axis {_AXIS[a]}")

    slabs = _cpml_slabs_for_planes(nx, ny, nz, cpml_cells, rigid_sides)
    gg = GeomGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        h=h,
        cpml=cpml_cells,
        x0=x0,
        cpml_slabs=slabs,
        cfg=cfg,
        rigid_sides=rigid_sides,
        planes=planes,
        h_nominal=h_nominal,
    )

    test_center: tuple[float, float, float] | None = None
    test_index: tuple[int, int, int] | None = None
    if ts_pos is not None:
        ii = int(math.floor((float(ts_pos[0]) - float(x0[0])) / h))
        jj = int(math.floor((float(ts_pos[1]) - float(x0[1])) / h))
        kk = int(math.floor((float(ts_pos[2]) - float(x0[2])) / h))
        cc = x0 + (np.array([ii, jj, kk], dtype=_F64) + 0.5) * h
        test_center = (float(cc[0]), float(cc[1]), float(cc[2]))
        test_index = (ii, jj, kk)

    derived = Derived(
        f_max=f_max,
        lambda_min=lambda_min,
        h=h,
        dt=dt,
        realized_W=rw,
        realized_H=rh,
        realized_D=rd,
        radiator_ic=0,
        radiator_jc=0,
        box_nx=nbx,
        box_ny=nby,
        box_nz=nbz,
        gap_cells=gap_cells,
        gap_m=gap_m,
        cpml_cells=cpml_cells,
        cpml_alpha=cpml_alpha,
        nx=nx,
        ny=ny,
        nz=nz,
        total_cells=total_cells,
        memory_bytes=memory_bytes,
        pulse_sigma=sigma,
        pulse_t0=t0,
        run_time_s=run_time_s,
        n_steps=n_steps,
        ppw_realized=ppw_realized,
        cpml_sides=tuple(cpml_sides),
        geometry="parametric",
        cell_budget_exceeded=cell_budget_exceeded,
        decimation_m=decimation_m,
        alias_level_db=alias_level_db,
        test_source_cell_center=test_center,
        test_source_cell_index=test_index,
    )
    return derived, gg


def validate_geom(
    cfg: Config,
    derived: Derived,
    geometry: Geometry,
    *,
    ram_limit_bytes: float | None = None,
) -> tuple[list[str], list[str]]:
    """Pre-flight checks for parametric geometry (mirrors :func:`boxrad.config.validate` where applicable)."""
    from boxrad.config import DEFAULT_RAM_LIMIT_BYTES
    from boxrad.geom.geometry import clearance_warnings

    if ram_limit_bytes is None:
        ram_limit_bytes = DEFAULT_RAM_LIMIT_BYTES

    errors: list[str] = []
    warnings: list[str] = []
    num = cfg.numerics

    spec = cfg.geometry_spec
    if derived.memory_bytes > ram_limit_bytes:
        errors.append(
            f"estimated memory {derived.memory_bytes / 1e9:.2f} GB exceeds "
            f"limit {ram_limit_bytes / 1e9:.2f} GB"
        )

    if derived.cell_budget_exceeded:
        errors.append(
            f"domain {derived.nx}×{derived.ny}×{derived.nz} = {derived.total_cells:,} cells "
            f"at h={derived.h * 1e3:.4g} mm exceeds numerics.cell_budget="
            f"{num.cell_budget:,}; raise h (lower numerics.cells_across_radiator or "
            f"numerics.ppw, or set numerics.h_override), lower band.f_max, or raise "
            f"numerics.cell_budget if RAM allows"
        )

    _ppw_dispersion_ref = 20.0
    if derived.ppw_realized < _ppw_dispersion_ref - 1e-9:
        warnings.append(
            f"realized {derived.ppw_realized:.1f} points per wavelength < "
            f"default {_ppw_dispersion_ref:g}"
        )

    warnings.extend(clearance_warnings(geometry, derived.h))
    return errors, warnings