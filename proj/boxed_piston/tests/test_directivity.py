"""Directivity-index quadrature on synthetic far-field patterns."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from boxrad.post.directivity import di_from_pattern, sphere_quadrature


def _uniform(dirs: np.ndarray) -> np.ndarray:
    return np.ones(dirs.shape[0], dtype=np.complex128)


def _hemisphere(dirs: np.ndarray) -> np.ndarray:
    return np.where(dirs[:, 2] >= 0.0, 1.0 + 0.0j, 0.0 + 0.0j)


def test_uniform_sphere_is_zero_db():
    di = di_from_pattern(_uniform, False, 24, 48)
    assert abs(di) < 0.01


def test_uniform_hemisphere_half_space_is_3db():
    di = di_from_pattern(_uniform, True, 24, 48)
    assert abs(di - 10.0 * math.log10(2.0)) < 0.01


def test_hemisphere_pattern_full_sphere_matches_half_space():
    """A pattern that vanishes for z < 0 has the same DI either way."""
    di_full = di_from_pattern(_hemisphere, False, 24, 48)
    di_half = di_from_pattern(_uniform, True, 24, 48)
    assert abs(di_full - di_half) < 0.01


def test_dipole_cos_theta_is_4p77_db():
    """|cos θ| pattern: DI = 10·log10(3) on the full sphere."""
    di = di_from_pattern(lambda d: d[:, 2].astype(np.complex128), False, 24, 48)
    assert abs(di - 10.0 * math.log10(3.0)) < 0.01


@pytest.mark.parametrize("half_space", [False, True])
def test_quadrature_weights_and_directions(half_space: bool):
    dirs, w = sphere_quadrature(12, 20, half_space=half_space)
    assert dirs.shape == (240, 3) and w.shape == (240,)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, rtol=0, atol=1e-12)
    total = 2.0 * math.pi if half_space else 4.0 * math.pi
    assert float(np.sum(w)) == pytest.approx(total, rel=1e-12)
    if half_space:
        assert float(np.min(dirs[:, 2])) > 0.0
