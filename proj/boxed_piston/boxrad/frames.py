"""Rigid-body frames: orthonormal bases and pose transforms (world ↔ local)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_ORTHO_EPS_N = 1e-12
_ORTHO_EPS_PARALLEL = 1e-9


def orthonormal_basis(normal: np.ndarray | list[float], x_axis: np.ndarray | list[float]) -> np.ndarray:
    """Return 3×3 rotation with columns (x̂, ŷ, ẑ) in world coordinates."""
    n = np.asarray(normal, dtype=np.float64)
    a = np.asarray(x_axis, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("x_axis must be finite")
    n_norm = float(np.linalg.norm(n))
    if n_norm < _ORTHO_EPS_N:
        raise ValueError("normal must be non-zero")
    z_hat = n / n_norm
    a_norm = float(np.linalg.norm(a))
    a_proj = a - np.dot(a, z_hat) * z_hat
    proj_norm = float(np.linalg.norm(a_proj))
    if proj_norm < _ORTHO_EPS_PARALLEL * a_norm:
        raise ValueError("x_axis must not be parallel to normal")
    x_hat = a_proj / proj_norm
    y_hat = np.cross(z_hat, x_hat)
    return np.column_stack((x_hat, y_hat, z_hat))


def default_x_axis(normal: np.ndarray | list[float]) -> np.ndarray:
    """World reference direction for completing the basis (see design v2 §1.1)."""
    n = np.asarray(normal, dtype=np.float64)
    n_norm = float(np.linalg.norm(n))
    if n_norm < _ORTHO_EPS_N:
        raise ValueError("normal must be non-zero")
    n_hat = n / n_norm
    ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(n_hat, ex))) > 0.9:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return ex


def _as_vec3(values: np.ndarray | list[float], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _transform(R: np.ndarray, t: np.ndarray, pts: np.ndarray, *, inverse: bool) -> np.ndarray:
    single = pts.ndim == 1
    p = np.atleast_2d(pts).astype(np.float64, copy=False)
    if inverse:
        out = (p - t) @ R
    else:
        out = p @ R.T + t
    if single:
        return out[0]
    return out


@dataclass(frozen=True)
class Pose:
    """Local frame: ẑ along ``normal``, x̂ from ``x_axis`` (Gram–Schmidt)."""

    position: np.ndarray
    normal: np.ndarray
    x_axis: np.ndarray

    @classmethod
    def make(
        cls,
        position: np.ndarray | list[float],
        normal: np.ndarray | list[float] = (0.0, 0.0, 1.0),
        x_axis: np.ndarray | list[float] | None = None,
    ) -> Pose:
        t = _as_vec3(position, name="position")
        n = _as_vec3(normal, name="normal")
        a = default_x_axis(n) if x_axis is None else _as_vec3(x_axis, name="x_axis")
        return cls(position=t, normal=n, x_axis=a)

    @classmethod
    def identity(cls) -> Pose:
        return cls.make((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pose):
            return NotImplemented
        return bool(
            np.allclose(self.position, other.position)
            and np.allclose(self.normal, other.normal)
            and np.allclose(self.x_axis, other.x_axis)
        )

    @property
    def R(self) -> np.ndarray:
        return orthonormal_basis(self.normal, self.x_axis)

    @property
    def T(self) -> np.ndarray:
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = self.R
        mat[:3, 3] = self.position
        return mat

    def to_local(self, pts: np.ndarray | list[float]) -> np.ndarray:
        return _transform(self.R, self.position, np.asarray(pts, dtype=np.float64), inverse=True)

    def to_world(self, pts: np.ndarray | list[float]) -> np.ndarray:
        return _transform(self.R, self.position, np.asarray(pts, dtype=np.float64), inverse=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.tolist(),
            "normal": self.normal.tolist(),
            "x_axis": self.x_axis.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pose:
        return cls.make(
            data["position"],
            data.get("normal", (0.0, 0.0, 1.0)),
            data.get("x_axis"),
        )