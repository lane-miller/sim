"""Directivity-index quadrature over far-field patterns (sphere or half-space)."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

_F64 = np.float64


def sphere_quadrature(
    n_theta: int,
    n_phi: int,
    *,
    half_space: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Unit directions ``(M, 3)`` and solid-angle weights ``(M,)``.

    Gauss–Legendre in ``u = cos θ`` (``u ∈ [−1, 1]``, or ``[0, 1]`` when
    ``half_space``) times ``n_phi`` uniform azimuths at ``(i + ½)·2π/n_phi``.
    Weights sum to ``4π`` (``2π`` for the half space, ``z ≥ 0``).
    """
    from scipy.special import roots_legendre

    u_nodes, w_u = roots_legendre(int(n_theta))
    total = 4.0 * math.pi
    if half_space:
        u_nodes = 0.5 * (u_nodes + 1.0)
        w_u = 0.5 * w_u
        total = 2.0 * math.pi
    phi = (np.arange(int(n_phi), dtype=_F64) + 0.5) * (2.0 * math.pi / int(n_phi))
    w_phi = 2.0 * math.pi / int(n_phi)
    u_grid, phi_grid = np.meshgrid(u_nodes, phi, indexing="ij")
    w_u_grid, _ = np.meshgrid(w_u, phi, indexing="ij")
    sin_theta = np.sqrt(np.maximum(1.0 - u_grid * u_grid, 0.0))
    dirs = np.stack(
        [sin_theta * np.cos(phi_grid), sin_theta * np.sin(phi_grid), u_grid],
        axis=-1,
    ).reshape(-1, 3)
    weights = (w_u_grid * w_phi).reshape(-1)
    weights *= total / float(np.sum(weights))
    return dirs.astype(_F64), weights.astype(_F64)


def di_from_pattern(
    pattern_fn: Callable[[np.ndarray], np.ndarray],
    half_space: bool,
    n_theta: int = 24,
    n_phi: int = 48,
) -> float:
    """On-axis directivity index ``10 log₁₀(4π |P(+z)|² / ∫ |P|² dΩ)`` in dB.

    Args:
        pattern_fn: Maps unit directions ``(M, 3)`` to pattern values ``(M,)``
            (complex or real; any quantity proportional to far-field pressure,
            e.g. pressure on a large sphere). Called once for the quadrature
            nodes and once for the axis ``(0, 0, 1)``.
        half_space: Integrate only over ``z ≥ 0`` (baffle); otherwise the full
            sphere (free-standing box).
        n_theta: Gauss–Legendre nodes in ``cos θ``.
        n_phi: Uniform azimuth nodes.

    Returns:
        DI in dB (0 dB for a uniform sphere; 10·log₁₀ 2 ≈ 3.01 dB for a pattern
        uniform over the half space with ``half_space=True``).
    """
    dirs, w_omega = sphere_quadrature(n_theta, n_phi, half_space=half_space)
    vals = np.asarray(pattern_fn(dirs)).ravel()
    axis_val = np.asarray(pattern_fn(np.array([[0.0, 0.0, 1.0]], dtype=_F64))).ravel()[0]
    integral = float(np.sum(np.abs(vals) ** 2 * w_omega))
    ratio = 4.0 * math.pi * abs(axis_val) ** 2 / max(integral, 1e-300)
    return 10.0 * math.log10(max(ratio, 1e-30))
