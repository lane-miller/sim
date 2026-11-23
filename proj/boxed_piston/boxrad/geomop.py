"""Assemble :class:`~boxrad.cutkernel.CutOperator` for parametric runs (step 6b1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from boxrad.config import Config, Derived
from boxrad.cutcell import Fractions, compute_fractions
from boxrad.cutkernel import CutOperator, FastRadiator, PatchRadiator, build_operator
from boxrad.grid import CpmlSlabs
from boxrad.geom.geometry import Geometry
from boxrad.geomderive import GeomGrid, cut_grid
from boxrad.smallcell import apply_floor, courant_lambda, stability_ok, thin_wall_cells

_F32 = np.float32
_F64 = np.float64


@dataclass
class GeomOpCache:
    """Fractions and floored β̃ reused across multi-radiator runs (step 8a)."""

    fractions: Fractions
    beta_tilde: np.ndarray
    floor_stats: dict[str, Any]
    thin_stats: dict[str, Any]


def _axis_aligned_normal(n_hat: np.ndarray) -> tuple[int, float] | None:
    for a in range(3):
        if abs(float(n_hat[a])) > 1.0 - 1e-12:
            return a, float(np.sign(n_hat[a]))
    return None


def _outward_face_index(axis: int, layer: int, s: float) -> int:
    """Radiator face index: the fluid cell's minus face for n̂=+e_a (index = layer), plus face for n̂=−e_a."""
    _ = axis
    return layer if s > 0 else layer + 1


def _face_flat_index(axis: int, i: int, j: int, k: int, nx: int, ny: int, nz: int, s: float) -> int:
    if axis == 0:
        fi = _outward_face_index(0, i if s > 0 else i, s)
        if s > 0:
            fi = i + 1
        else:
            fi = i
        return int(np.ravel_multi_index((fi, j, k), (nx + 1, ny, nz)))
    if axis == 1:
        fi = j + 1 if s > 0 else j
        return int(np.ravel_multi_index((i, fi, k), (nx, ny + 1, nz)))
    fi = k + 1 if s > 0 else k
    return int(np.ravel_multi_index((i, j, fi), (nx, ny, nz + 1)))


def _fast_path_ok(
    fr,
    fractions_axes,
    nx: int,
    ny: int,
    nz: int,
    axis: int,
    s: float,
    cells: np.ndarray,
    areas: np.ndarray,
) -> tuple[bool, int | None]:
    """Single-layer axis-aligned radiator with α = 0 on the solid-backed face."""
    alpha = fractions_axes[axis]
    layer_idx: int | None = None
    for flat, a in zip(cells, areas):
        if a <= 0.0:
            continue
        i = int(flat) // (ny * nz)
        rem = int(flat) % (ny * nz)
        j = rem // nz
        k = rem % nz
        if axis == 0:
            key = i
        elif axis == 1:
            key = j
        else:
            key = k
        if layer_idx is None:
            layer_idx = key
        elif key != layer_idx:
            return False, None
        wall_face = key if s > 0 else key + 1
        if axis == 0:
            if alpha[wall_face, j, k] > 1e-12:
                return False, None
        elif axis == 1:
            if alpha[i, wall_face, k] > 1e-12:
                return False, None
        elif alpha[i, j, wall_face] > 1e-12:
            return False, None
    if layer_idx is None:
        return False, None
    return True, layer_idx


def _build_radiator(
    j: int,
    fr: Fractions,
    h: float,
    nx: int,
    ny: int,
    nz: int,
    geometry: Geometry,
) -> tuple[str, list[FastRadiator], list[PatchRadiator], float]:
    rd_geom = geometry.radiators[j]
    n_hat = rd_geom.pose.R[:, 2]
    n_hat = n_hat / np.linalg.norm(n_hat)
    cells = fr.radiator_cell[j]
    areas = fr.radiator_a[j]
    S_grid = float(np.sum(areas))
    h2 = h * h
    aligned = _axis_aligned_normal(n_hat)
    fast: list[FastRadiator] = []
    patch: list[PatchRadiator] = []
    path = "patch"
    if aligned is not None:
        axis, s = aligned
        ok_fast, layer_idx = _fast_path_ok(fr, fr.alpha, nx, ny, nz, axis, s, cells, areas)
        if ok_fast and layer_idx is not None:
            idx_list: list[int] = []
            sw_list: list[float] = []
            face_layer = _outward_face_index(axis, layer_idx, s)
            for flat, a in zip(cells, areas):
                if a <= 0.0:
                    continue
                i = int(flat) // (ny * nz)
                rem = int(flat) % (ny * nz)
                jk = rem // nz
                k = rem % nz
                if axis == 0:
                    fidx = int(np.ravel_multi_index((face_layer, jk, k), (nx + 1, ny, nz)))
                elif axis == 1:
                    fidx = int(np.ravel_multi_index((i, face_layer, k), (nx, ny + 1, nz)))
                else:
                    fidx = int(np.ravel_multi_index((i, jk, face_layer), (nx, ny, nz + 1)))
                w = float(a) / h2
                idx_list.append(fidx)
                sw_list.append(s * w)
            fast.append(
                FastRadiator(
                    axis=axis,
                    idx=np.asarray(idx_list, dtype=np.int64),
                    sw=np.asarray(sw_list, dtype=_F32),
                )
            )
            path = "fast"
            return path, fast, patch, S_grid
    coef = (areas / h2).astype(_F32)
    patch.append(PatchRadiator(cells=cells.astype(np.int64), coef=coef))
    return path, fast, patch, S_grid


def build_geom_op_cache(
    cfg: Config,
    derived: Derived,
    gg: GeomGrid,
    geometry: Geometry,
) -> GeomOpCache:
    """One-shot fractions + floor for reuse across ``driven`` runs."""
    cgrid = cut_grid(gg)
    fractions = compute_fractions(geometry, cgrid)
    max_thin = getattr(cfg.numerics, "max_thin_cells", None) or 0
    _, thin_stats = thin_wall_cells(fractions, geometry, cgrid, max_thin=max_thin)
    lam = courant_lambda(cfg, derived)
    beta_tilde, floor_stats = apply_floor(fractions, lam, h=derived.h)
    ok, _worst = stability_ok(beta_tilde, fractions.alpha, lam)
    assert ok, "small-cell stability floor violated"
    return GeomOpCache(
        fractions=fractions,
        beta_tilde=beta_tilde,
        floor_stats=floor_stats,
        thin_stats=thin_stats,
    )


def _assert_geometry_clear_of_cpml_sides(
    op: CutOperator,
    slabs: CpmlSlabs,
    cpml_sides: tuple[bool, ...],
) -> None:
    """Like cutkernel.assert_geometry_clear_of_cpml, but only on absorbing (CPML) sides."""
    c = slabs.cpml_cells
    nx = op.inv_beta.shape[0]
    ny = op.inv_beta.shape[1]
    nz = op.inv_beta.shape[2]
    tol = _F32(1.0 - 1e-5)

    def _check_p(lo: int, hi: int, axis: int) -> None:
        sl = [slice(None)] * 3
        sl[axis] = slice(lo, hi)
        block = op.inv_beta[tuple(sl)]
        if np.any(block < tol):
            raise ValueError("CPML slab: inv_beta != 1")

    if cpml_sides[0]:
        _check_p(0, c, 0)
    if cpml_sides[1]:
        _check_p(nx - c, nx, 0)
    if cpml_sides[2]:
        _check_p(0, c, 1)
    if cpml_sides[3]:
        _check_p(ny - c, ny, 1)
    if cpml_sides[4]:
        _check_p(0, c, 2)
    if cpml_sides[5]:
        _check_p(nz - c, nz, 2)

    def _check_face(afl: np.ndarray, lo: int, hi: int, axis: int) -> None:
        sl = [slice(None)] * 3
        sl[axis] = slice(lo, hi)
        block = afl[tuple(sl)]
        if np.any(block < tol):
            raise ValueError("CPML slab: face afl != 1")

    if cpml_sides[0]:
        _check_face(op.afl_x, 0, c + 1, 0)
    if cpml_sides[1]:
        _check_face(op.afl_x, nx - c, nx + 1, 0)
    if cpml_sides[2]:
        _check_face(op.afl_y, 0, c + 1, 1)
    if cpml_sides[3]:
        _check_face(op.afl_y, ny - c, ny + 1, 1)
    if cpml_sides[4]:
        _check_face(op.afl_z, 0, c + 1, 2)
    if cpml_sides[5]:
        _check_face(op.afl_z, nz - c, nz + 1, 2)


def build_run_operator(
    cfg: Config,
    derived: Derived,
    gg: GeomGrid,
    geometry: Geometry,
    *,
    driven: int = 0,
    cache: GeomOpCache | None = None,
) -> tuple[CutOperator, dict[str, Any]]:
    """Fractions → floor → cut operator; radiator ``driven`` driven, others rigid."""
    n_rad = len(geometry.radiators)
    if not (0 <= driven < n_rad):
        raise ValueError(f"driven index {driven} out of range for {n_rad} radiators")
    if cache is None:
        cache = build_geom_op_cache(cfg, derived, gg, geometry)
    fractions = cache.fractions
    beta_tilde = cache.beta_tilde
    thin_stats = cache.thin_stats
    floor_stats = cache.floor_stats

    path, fast, patch, s_grid = _build_radiator(
        driven, fractions, derived.h, gg.nx, gg.ny, gg.nz, geometry
    )
    op = build_operator(fractions, beta_tilde.astype(_F32), fast, patch, derived.h)
    _assert_geometry_clear_of_cpml_sides(op, gg.cpml_slabs, derived.cpml_sides)

    n_closed = (
        op.closed_faces_x.size + op.closed_faces_y.size + op.closed_faces_z.size
    )
    info: dict[str, Any] = {
        "S_grid": s_grid,
        "path": path,
        "fractions_stats": fractions.stats,
        "floor_stats": floor_stats,
        "thin_wall_stats": thin_stats,
        "n_special_cells": int(op.special_cells.size),
        "n_closed_faces": int(n_closed),
        "fractions": fractions,
        "beta_tilde": beta_tilde,
        "driven": driven,
        "op_cache": cache,
    }
    return op, info