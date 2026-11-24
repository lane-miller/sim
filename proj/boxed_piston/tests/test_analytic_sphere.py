"""Analytic rigid-sphere monopole reference (step 10b2)."""

from __future__ import annotations

import numpy as np
import pytest

from boxrad.validation.analytic import monopole_free, sphere_point_source

_RHO = 1.204
_C = 343.0
_A = 0.05
_SOURCE = np.array([0.0, 0.0, 0.1], dtype=np.float64)
_CENTER = np.array([0.0, 0.0, 0.0], dtype=np.float64)


def test_incident_vs_monopole_free():
    pt = np.array([[0.15, 0.0, 0.0], [0.0, 0.0, -0.15]], dtype=np.float64)
    for f in (300.0, 3000.0):
        inc = sphere_point_source(pt, f, _CENTER, _A, _SOURCE, _RHO, _C, part="inc")
        _, free = monopole_free(pt, f, _SOURCE, _RHO, _C)
        np.testing.assert_allclose(inc, free[0], rtol=1e-12, atol=0)


def test_rigid_bc_gradient():
    f = 1000.0
    dr = 1e-5
    thetas = np.linspace(0.1, np.pi - 0.1, 7)
    worst = 0.0
    for th in thetas:
        u = np.array([np.sin(th), 0.0, np.cos(th)], dtype=np.float64)
        r_in = (_A - dr) * u
        r_out = (_A + dr) * u
        p_in = sphere_point_source(r_in, f, _CENTER, _A, _SOURCE, _RHO, _C, part="total")[0]
        p_out = sphere_point_source(r_out, f, _CENTER, _A, _SOURCE, _RHO, _C, part="total")[0]
        dp_dr = (p_out - p_in) / (2 * dr)
        inc_in = sphere_point_source(r_in, f, _CENTER, _A, _SOURCE, _RHO, _C, part="inc")[0]
        inc_out = sphere_point_source(r_out, f, _CENTER, _A, _SOURCE, _RHO, _C, part="inc")[0]
        dinc = (inc_out - inc_in) / (2 * dr)
        rel = abs(dp_dr) / max(abs(dinc), 1e-30)
        worst = max(worst, rel)
    assert worst <= 1e-6


def test_reciprocity_shifted_center():
    f = 800.0
    center = np.array([0.01, -0.02, 0.005], dtype=np.float64)
    probe = np.array([-0.06, 0.02, -0.08], dtype=np.float64)
    src = np.array([0.03, 0.04, 0.17], dtype=np.float64)
    p_ab = sphere_point_source(probe[np.newaxis, :], f, center, _A, src, _RHO, _C)[0]
    p_ba = sphere_point_source(src[np.newaxis, :], f, center, _A, probe, _RHO, _C)[0]
    assert abs(p_ab - p_ba) <= 1e-12 * max(abs(p_ab), 1.0)


def test_tiny_sphere_scat_small():
    f = 1500.0
    pt = np.array([[0.2, 0.0, 0.0]], dtype=np.float64)
    inc = sphere_point_source(pt, f, _CENTER, 1e-4, _SOURCE, _RHO, _C, part="inc")[0]
    sc = sphere_point_source(pt, f, _CENTER, 1e-4, _SOURCE, _RHO, _C, part="scat")[0]
    assert abs(sc / inc) <= 1e-7


def test_raises_near_equal_radii():
    r0 = float(np.linalg.norm(_SOURCE - _CENTER))
    u = (_SOURCE - _CENTER) / r0
    pt = (_CENTER + 0.98 * r0 * u)[np.newaxis, :]
    with pytest.raises(ValueError, match="probe radius"):
        sphere_point_source(pt, 500.0, _CENTER, _A, _SOURCE, _RHO, _C)