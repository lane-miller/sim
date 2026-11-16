"""Tests for boxrad.post.reference (baffled Z_rad)."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from boxrad.post.reference import (
    _moment_closed,
    _moment_series,
    z_baffled_circle,
    z_baffled_rect,
)


def test_circle_low_ka_limit():
    ka = np.array([0.05, 0.08, 0.12], dtype=np.float64)
    z = z_baffled_circle(ka)
    np.testing.assert_allclose(np.real(z), ka**2 / 2.0, rtol=0.02)
    assert np.all(np.imag(z) > 0.0)


def test_rect_low_ka():
    W, H = 0.12, 0.08
    S = W * H
    wh = max(W, H)
    k = wh * np.array([0.02, 0.03, 0.05], dtype=np.float64)
    assert np.all(k * wh <= 0.05 + 1e-15)
    z = z_baffled_rect(k, W, H)
    expect = k**2 * S / (2.0 * math.pi)
    np.testing.assert_allclose(np.real(z), expect, rtol=1e-3)
    assert np.all(np.imag(z) > 0.0)


def test_rect_gauss_nodes_independent():
    W, H = 0.1, 0.05
    wh = max(W, H)
    km = np.linspace(0.1, 20.0, 40)
    k = km / wh
    z64 = z_baffled_rect(k, W, H, n_phi=64)
    z256 = z_baffled_rect(k, W, H, n_phi=256)
    err = float(np.max(np.abs(z64 - z256)))
    assert err <= 1e-8


def test_moment_series_matches_closed_at_half_wavelength():
    """Series and closed ``M_n`` agree at ``|λR| = 0.5``."""
    r_len = 0.08
    lam = np.array([0.5j / r_len], dtype=np.complex128)
    assert abs(lam * r_len).item() == pytest.approx(0.5)
    for n in range(3):
        closed = _moment_closed(n, lam, r_len)
        series = _moment_series(n, lam, r_len)
        np.testing.assert_allclose(closed, series, rtol=1e-12, atol=1e-12)


def _dblquad_rect_z(k: float, W: float, H: float) -> complex:
    from scipy import integrate

    def kernel(w: float, u: float) -> complex:
        r = math.hypot(u, w)
        fac = (W - u) * (H - w)
        if r < 1e-14:
            return fac * (-1j * k)
        return fac * np.exp(-1j * k * r) / r

    def re_part(w: float, u: float) -> float:
        return float(np.real(kernel(w, u)))

    def im_part(w: float, u: float) -> float:
        return float(np.imag(kernel(w, u)))

    re, _ = integrate.dblquad(re_part, 0.0, W, lambda _u: 0.0, lambda _u: H, epsabs=1e-10, epsrel=1e-8)
    im, _ = integrate.dblquad(im_part, 0.0, W, lambda _u: 0.0, lambda _u: H, epsabs=1e-10, epsrel=1e-8)
    s = W * H
    return (2.0j * k / (math.pi * s)) * (re + 1j * im)


@pytest.mark.parametrize("km", [1.0, 5.0])
def test_rect_dblquad_cartesian(km: float):
    W, H = 0.1, 0.06
    wh = max(W, H)
    k = km / wh
    z_fast = z_baffled_rect(k, W, H, n_phi=128)
    z_dq = _dblquad_rect_z(k, W, H)
    np.testing.assert_allclose(z_fast, z_dq, rtol=1e-5)


def test_rect_width_height_symmetric():
    W, H = 0.12, 0.07
    wh = max(W, H)
    k = np.array([1.0, 4.0, 10.0], dtype=np.float64) / wh
    z_wh = z_baffled_rect(k, W, H)
    z_hw = z_baffled_rect(k, H, W)
    np.testing.assert_allclose(z_wh, z_hw, atol=1e-6)