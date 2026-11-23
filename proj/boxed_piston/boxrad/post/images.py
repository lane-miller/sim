"""Rigid-plane image expansion for Kirchhoff–Helmholtz (patch copy method)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from boxrad.post.kirchhoff import SurfacePatches

_F64 = np.float64
_AXIS = {"x": 0, "y": 1, "z": 2}


def _parsed_offsets(planes: list[dict[str, Any]]) -> list[tuple[int, float]]:
    return [(_AXIS[str(pl["axis"])], float(pl["offset"])) for pl in planes]


def plane_reflections(
    planes: list[dict[str, Any]],
) -> list[tuple[tuple[int, float], ...]]:
    """All ``2**n_p`` reflection subsets (identity first).

    Each item is the list of ``(axis_index, offset)`` planes reflected in that term.
    """
    parsed = _parsed_offsets(planes)
    n_p = len(parsed)
    out: list[tuple[tuple[int, float], ...]] = []
    for mask in range(1 << n_p):
        term = tuple(parsed[i] for i in range(n_p) if (mask >> i) & 1)
        out.append(term)
    return out


def _reflect_centers_normals(
    centers: np.ndarray,
    normals: np.ndarray,
    term: tuple[tuple[int, float], ...],
) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(centers, dtype=_F64)
    n = np.asarray(normals, dtype=_F64)
    if not term:
        return c, n
    c = c.copy()
    n = n.copy()
    for axis, offset in term:
        c[:, axis] = 2.0 * offset - c[:, axis]
        n[:, axis] *= -1.0
    return c, n


def expand_patches(
    patches: SurfacePatches,
    planes: list[dict[str, Any]],
    *arrays: np.ndarray,
) -> tuple[SurfacePatches, ...]:
    """Tile patch geometry and trailing patch axes for every image subset."""
    reflections = plane_reflections(planes)
    n_r = len(reflections)
    n0 = int(patches.centers.shape[0])
    centers_list: list[np.ndarray] = []
    normals_list: list[np.ndarray] = []
    for term in reflections:
        cc, nn = _reflect_centers_normals(patches.centers, patches.normals, term)
        centers_list.append(cc)
        normals_list.append(nn)
    exp_patches = SurfacePatches(
        centers=np.vstack(centers_list),
        normals=np.vstack(normals_list),
        areas=np.tile(np.asarray(patches.areas, dtype=_F64), n_r),
        h=float(patches.h),
    )
    expanded: list[np.ndarray] = []
    for arr in arrays:
        a = np.asarray(arr)
        if a.shape[-1] != n0:
            raise ValueError(
                f"expand_patches: last axis length {a.shape[-1]} != patch count {n0}"
            )
        if a.ndim == 1:
            expanded.append(np.concatenate([a] * n_r, axis=0))
        else:
            expanded.append(np.concatenate([a] * n_r, axis=-1))
    return (exp_patches, *expanded)


def in_fluid(
    points: np.ndarray,
    planes: list[dict[str, Any]],
    tol: float = 1e-12,
) -> np.ndarray:
    """True where ``points`` lie in the open acoustic half-space (fluid side of each plane)."""
    pts = np.atleast_2d(np.asarray(points, dtype=_F64))
    ok = np.ones(pts.shape[0], dtype=bool)
    for pl in planes:
        ax = _AXIS[str(pl["axis"])]
        off = float(pl["offset"])
        if pl["side"] == "-":
            ok &= pts[:, ax] >= off - tol
        elif pl["side"] == "+":
            ok &= pts[:, ax] <= off + tol
        else:
            raise ValueError(f"plane side must be '-' or '+', got {pl['side']!r}")
    return ok


def direction_open(
    dirs: np.ndarray,
    planes: list[dict[str, Any]],
    tol: float = 1e-12,
) -> np.ndarray:
    """True for unit directions radiating into the open region defined by the planes."""
    d = np.atleast_2d(np.asarray(dirs, dtype=_F64))
    ok = np.ones(d.shape[0], dtype=bool)
    for pl in planes:
        ax = _AXIS[str(pl["axis"])]
        if pl["side"] == "-":
            ok &= d[:, ax] >= -tol
        elif pl["side"] == "+":
            ok &= d[:, ax] <= tol
        else:
            raise ValueError(f"plane side must be '-' or '+', got {pl['side']!r}")
    return ok


def open_solid_angle(planes: list[dict[str, Any]]) -> float:
    """Solid angle of the open region ``4π / 2**n_p`` (radian²)."""
    n_p = len(planes)
    return 4.0 * math.pi / float(1 << n_p)


def image_sources(
    source: np.ndarray,
    planes: list[dict[str, Any]],
) -> list[np.ndarray]:
    """Monopole image positions for every reflection subset (identity first)."""
    s0 = np.asarray(source, dtype=_F64).reshape(3)
    out: list[np.ndarray] = []
    for term in plane_reflections(planes):
        s = s0.copy()
        for axis, offset in term:
            s[axis] = 2.0 * offset - s[axis]
        out.append(s)
    return out


def first_plane_violation(
    point: np.ndarray,
    planes: list[dict[str, Any]],
    tol: float = 1e-12,
) -> str | None:
    """Name the first plane violated by ``point`` (fluid side), or ``None`` if inside."""
    pt = np.asarray(point, dtype=_F64).reshape(3)
    if bool(in_fluid(pt[np.newaxis, :], planes, tol=tol)[0]):
        return None
    for pl in planes:
        ax = _AXIS[str(pl["axis"])]
        off = float(pl["offset"])
        if pl["side"] == "-" and pt[ax] < off - tol:
            return f"{pl['axis']}{pl['side']} at offset {off}"
        if pl["side"] == "+" and pt[ax] > off + tol:
            return f"{pl['axis']}{pl['side']} at offset {off}"
    return "rigid plane"


def rigid_sides_from_planes(planes: list[dict[str, Any]]) -> dict[str, bool]:
    """``{'x-': True, ...}`` keys matching :func:`huygens_patches_planes`."""
    return {f"{pl['axis']}{pl['side']}": True for pl in planes}