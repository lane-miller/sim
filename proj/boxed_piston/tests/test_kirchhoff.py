"""C4-kh: Kirchhoff–Helmholtz over a box vs analytic monopole / dipole fields."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fdtd.post.kirchhoff import (  # noqa: E402
    box_surface_patches,
    dpdn_from_vn,
    far_field_at_radius,
    face_p_from_cells,
    huygens_box_patches,
    huygens_open_patches,
    kh_far_field,
    kh_pressure,
    kh_pressure_loop,
)

_C128 = np.complex128
_F64 = np.float64

# Air defaults (brief §1)
RHO = 1.204
C = 343.0

# 100 mm cube, h = 2 mm quadrature
L_BOX = 0.1
H_PATCH = 0.002
ORIGIN = (0.0, 0.0, 0.0)

# Off-center monopole (m)
SOURCE = np.array([0.02, 0.01, 0.015], dtype=_F64)
Q = 1.0e-6  # m³/s volume flow

FREQS = np.array([1_000.0, 4_000.0, 10_000.0], dtype=_F64)
OMEGA = 2.0 * np.pi * FREQS
K = OMEGA / C


def _monopole_p(
    x: np.ndarray,
    k: float,
    coeff: complex,
    source: np.ndarray | None = None,
) -> np.ndarray:
    """Analytic pressure from monopole ``coeff = jωρQ/(4π)``."""
    src = SOURCE if source is None else source
    r = np.linalg.norm(x - src, axis=-1)
    r = np.maximum(r, 1e-12)
    return coeff * np.exp(-1j * k * r) / r


def _monopole_dpdn(
    points: np.ndarray,
    normals: np.ndarray,
    k: float,
    coeff: complex,
    source: np.ndarray | None = None,
) -> np.ndarray:
    """Outward normal derivative ∂p/∂n on surface with box-outward ``normals``."""
    src = SOURCE if source is None else source
    eps = points - src
    r = np.linalg.norm(eps, axis=-1, keepdims=True)
    r = np.maximum(r, 1e-12)
    rhat = eps / r
    p = coeff * np.exp(-1j * k * r[..., 0]) / r[..., 0]
    grad = -p[..., np.newaxis] * rhat * (1.0 / r + 1j * k)
    return np.sum(grad * normals, axis=-1)


def _coeff(f_hz: np.ndarray) -> np.ndarray:
    omega = 2.0 * np.pi * f_hz
    return (1j * omega * RHO * Q / (4.0 * np.pi)).astype(_C128)


def _sample_surface(
    patches,
    freqs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact ``p`` and ``∂p/∂n`` at patch centers, shape (F, N)."""
    F = freqs.size
    N = patches.centers.shape[0]
    p_out = np.zeros((F, N), dtype=_C128)
    dp_out = np.zeros((F, N), dtype=_C128)
    for f_i, f in enumerate(freqs):
        kf = 2.0 * np.pi * f / C
        c = _coeff(np.array([f]))[0]
        p_out[f_i] = _monopole_p(patches.centers, kf, c)
        dp_out[f_i] = _monopole_dpdn(patches.centers, patches.normals, kf, c)
    return p_out, dp_out


def _rel_err(est: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.abs(est - ref) / np.maximum(np.abs(ref), 1e-30)


def _rel_err_scalar(est: np.ndarray, ref: np.ndarray) -> float:
    return float(_rel_err(np.asarray(est).ravel()[0], np.asarray(ref).ravel()[0]))


def _fibonacci_sphere(n: int) -> np.ndarray:
    ga = math.pi * (3.0 - math.sqrt(5.0))
    i = np.arange(n, dtype=_F64)
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    theta = ga * i
    return np.stack((r * np.cos(theta), r * np.sin(theta), z), axis=-1)


def test_kh_pressure_matches_reference_loop() -> None:
    """Chunked path must match explicit loops (catches [0] on Green's batches)."""
    patches = box_surface_patches(L_BOX, L_BOX, L_BOX, H_PATCH, ORIGIN)
    p_s, dpdn = _sample_surface(patches, FREQS[:1])
    k = K[:1]
    center = L_BOX / 2.0
    L = L_BOX
    # M = 5 with m_chunk = 3 → remainder chunk of 2 (not divisible)
    points = np.array(
        [
            [center + 1.5 * L, center, center],
            [center + 3.0 * L, center, center],
            [center + 10.0 * L, center, center],
            [center + 2.0 * L, center + 0.02, center],
            [center + 5.0 * L, center, center + 0.03],
        ],
        dtype=_F64,
    )
    ref = kh_pressure_loop(points, patches, p_s, dpdn, k)
    fast = kh_pressure(
        points,
        patches,
        p_s,
        dpdn,
        k,
        m_chunk=3,
        n_chunk=137,
        near_subdivide=False,
    )
    np.testing.assert_allclose(fast, ref, rtol=1e-10, atol=0)


def test_extinction_exterior_and_interior() -> None:
    patches = box_surface_patches(L_BOX, L_BOX, L_BOX, H_PATCH, ORIGIN)
    p_s, dpdn = _sample_surface(patches, FREQS)
    k = K

    center = ORIGIN[0] + L_BOX / 2.0
    scales = np.array([1.5, 3.0, 10.0], dtype=_F64) * L_BOX
    exterior = np.stack(
        [
            center + scales,
            np.full_like(scales, center),
            np.full_like(scales, center),
        ],
        axis=-1,
    )
    interior = np.array(
        [
            [center - 0.015, center - 0.01, center - 0.012],
            [center - 0.005, center + 0.008, center + 0.005],
        ],
        dtype=_F64,
    )

    p_kh_ext = kh_pressure(exterior, patches, p_s, dpdn, k)
    for f_i, f in enumerate(FREQS):
        c = _coeff(np.array([f]))[0]
        p_ex = _monopole_p(exterior, k[f_i], c)
        assert np.all(_rel_err(p_kh_ext[f_i], p_ex) <= 0.01), f"exterior {f} Hz"

    p_kh_in = kh_pressure(interior, patches, p_s, dpdn, k)
    scale = np.max(np.abs(p_kh_ext))
    assert np.all(np.abs(p_kh_in) <= 0.01 * scale)


def test_far_field_monopole_and_dipole() -> None:
    patches = box_surface_patches(L_BOX, L_BOX, L_BOX, H_PATCH, ORIGIN)
    p_s, dpdn = _sample_surface(patches, FREQS)

    dirs = _fibonacci_sphere(720)
    amp_kh = kh_far_field(dirs, patches, p_s, dpdn, K)

    for f_i, f in enumerate(FREQS):
        kf = K[f_i]
        c = _coeff(np.array([f]))[0]
        # Off-center compact source: directivity factor e^{jk d·x_s}
        phase_fac = np.exp(1j * kf * (dirs @ SOURCE))
        amp_ref = c * phase_fac

        mag_err = np.abs(np.abs(amp_kh[f_i]) - np.abs(amp_ref)) / np.abs(amp_ref)
        assert np.all(mag_err <= 0.01), f"monopole mag {f} Hz"

        ph_err = np.angle(amp_kh[f_i] * np.conj(amp_ref), deg=True)
        assert np.all(np.abs(ph_err) <= 1.0), f"monopole phase {f} Hz"

    # Dipole: +Q and −Q along z inside the box
    sep = 0.012
    z0 = ORIGIN[2] + L_BOX / 2.0
    pos_plus = np.array([SOURCE[0], SOURCE[1], z0 + sep / 2.0])
    pos_minus = np.array([SOURCE[0], SOURCE[1], z0 - sep / 2.0])

    F, N = len(FREQS), patches.centers.shape[0]
    p_d = np.zeros((F, N), dtype=_C128)
    dp_d = np.zeros_like(p_d)
    for f_i, ff in enumerate(FREQS):
        kf = 2.0 * np.pi * ff / C
        cc = _coeff(np.array([ff]))[0]
        for src, sign in ((pos_plus, 1.0), (pos_minus, -1.0)):
            p_d[f_i] += sign * _monopole_p(patches.centers, kf, cc, source=src)
            dp_d[f_i] += sign * _monopole_dpdn(
                patches.centers, patches.normals, kf, cc, source=src
            )

    amp_d = kh_far_field(dirs, patches, p_d, dp_d, K)
    for f_i, f in enumerate(FREQS):
        kf = K[f_i]
        cc = _coeff(np.array([f]))[0]
        amp_ref = cc * (
            np.exp(1j * kf * (dirs @ pos_plus)) - np.exp(1j * kf * (dirs @ pos_minus))
        )
        mag_err = np.abs(np.abs(amp_d[f_i]) - np.abs(amp_ref)) / np.maximum(np.abs(amp_ref), 1e-30)
        assert np.all(mag_err <= 0.01), f"dipole mag {f} Hz"
        ph_err = np.angle(amp_d[f_i] * np.conj(amp_ref), deg=True)
        mask = np.abs(amp_ref) > 0.05 * np.max(np.abs(amp_ref))
        assert np.all(np.abs(ph_err[mask]) <= 1.0), f"dipole phase {f} Hz"


def _extinction_max_err(h: float) -> float:
    patches = box_surface_patches(L_BOX, L_BOX, L_BOX, h, ORIGIN)
    p_s, dpdn = _sample_surface(patches, FREQS)
    center = ORIGIN[0] + L_BOX / 2.0
    r = 3.0 * L_BOX
    pt = np.array([[center + r, center, center]], dtype=_F64)
    p_kh = kh_pressure(pt, patches, p_s, dpdn, K)
    errs = []
    for f_i, f in enumerate(FREQS):
        c = _coeff(np.array([f]))[0]
        p_ex = _monopole_p(pt, K[f_i], c)
        errs.append(_rel_err_scalar(p_kh[f_i], p_ex))
    return max(errs)


def test_convergence_order() -> None:
    h_vals = [H_PATCH, H_PATCH / 2.0, H_PATCH / 4.0]
    err = [_extinction_max_err(h) for h in h_vals]
    orders = []
    for i in range(len(h_vals) - 1):
        ratio = h_vals[i] / h_vals[i + 1]
        orders.append(math.log(err[i] / err[i + 1]) / math.log(ratio))
    mean_order = float(np.mean(orders))
    print(f"C4-kh convergence (3×L_BOX probe): h={h_vals}, rel_err={err}, order≈{mean_order:.2f}")
    assert mean_order > 1.5, "expected at least ~2nd-order face quadrature"
    assert err[-1] < err[0]


def test_near_surface_subdivision() -> None:
    patches = box_surface_patches(L_BOX, L_BOX, L_BOX, H_PATCH, ORIGIN)
    p_s, dpdn = _sample_surface(patches, FREQS)
    # 0.5 h outside the +x face (x = L_BOX)
    pt = np.array([[L_BOX + 0.5 * H_PATCH, L_BOX / 2.0, L_BOX / 2.0]], dtype=_F64)

    p_on = kh_pressure(pt, patches, p_s, dpdn, K, near_subdivide=True)
    p_off = kh_pressure(pt, patches, p_s, dpdn, K, near_subdivide=False)

    f4 = 1
    c = _coeff(np.array([FREQS[f4]]))[0]
    p_ex = _monopole_p(pt, K[f4], c)
    err_on = _rel_err_scalar(p_on[f4], p_ex)
    err_off = _rel_err_scalar(p_off[f4], p_ex)
    print(f"near-surface 4 kHz: rel_err subdiv on={err_on:.4f}, off={err_off:.4f}")
    assert err_on <= 0.03
    assert err_off > err_on


def _half_cell_errors_at_3x() -> list[float]:
    """Rel errors at 3×L (box faces) for comparison print in Huygens test."""
    h = H_PATCH
    patches_on = box_surface_patches(L_BOX, L_BOX, L_BOX, h, ORIGIN)
    centers = patches_on.centers
    normals = patches_on.normals
    sample_pts = centers + 0.5 * h * normals
    p_s = np.zeros((FREQS.size, centers.shape[0]), dtype=_C128)
    dpdn = np.zeros_like(p_s)
    for f_i, f in enumerate(FREQS):
        kf = K[f_i]
        c = _coeff(np.array([f]))[0]
        p_s[f_i] = _monopole_p(sample_pts, kf, c)
        dpdn[f_i] = _monopole_dpdn(sample_pts, normals, kf, c)
    center = L_BOX / 2.0
    pt = np.array([[center + 3.0 * L_BOX, center, center]], dtype=_F64)
    p_kh = kh_pressure(pt, patches_on, p_s, dpdn, K, near_subdivide=True)
    return [
        _rel_err_scalar(p_kh[f_i], _monopole_p(pt, K[f_i], _coeff(np.array([f]))[0]))
        for f_i, f in enumerate(FREQS)
    ]


def _fdtd_face_p(
    centers: np.ndarray,
    normals: np.ndarray,
    kf: float,
    c: complex,
    h: float,
    order: int = 4,
) -> np.ndarray:
    """Face pressure from analytic cell-center samples along outward normal."""
    p_m1 = _monopole_p(centers - 0.5 * h * normals, kf, c)
    p_p1 = _monopole_p(centers + 0.5 * h * normals, kf, c)
    if order == 2:
        z = np.zeros_like(p_m1)
        return face_p_from_cells(z, p_m1, p_p1, z, order=2)
    p_m3 = _monopole_p(centers - 1.5 * h * normals, kf, c)
    p_p3 = _monopole_p(centers + 1.5 * h * normals, kf, c)
    return face_p_from_cells(p_m3, p_m1, p_p1, p_p3, order=4)


def _huygens_kh_errors(
    patches,
    order: int,
    exterior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build FDTD-style Huygens data and return (p_kh exterior, far-field max mag err per F)."""
    centers = patches.centers
    normals = patches.normals
    F, N = FREQS.size, centers.shape[0]
    p_fdtd = np.zeros((F, N), dtype=_C128)
    dpdn = np.zeros((F, N), dtype=_C128)
    for f_i, f in enumerate(FREQS):
        kf = K[f_i]
        c = _coeff(np.array([f]))[0]
        p_fdtd[f_i] = _fdtd_face_p(centers, normals, kf, c, patches.h, order=order)
        dpdn_face = _monopole_dpdn(centers, normals, kf, c)
        vn = dpdn_face / (-1j * OMEGA[f_i] * RHO)
        dpdn[f_i] = dpdn_from_vn(vn[np.newaxis, :], OMEGA[f_i : f_i + 1], RHO)[0]

    p_kh = kh_pressure(exterior, patches, p_fdtd, dpdn, K, near_subdivide=True)
    dirs = _fibonacci_sphere(360)
    amp_kh = kh_far_field(dirs, patches, p_fdtd, dpdn, K)
    ff_err = np.zeros(F, dtype=_F64)
    for f_i, f in enumerate(FREQS):
        c = _coeff(np.array([f]))[0]
        amp_ref = c * np.exp(1j * K[f_i] * (dirs @ SOURCE))
        mag_err = np.abs(np.abs(amp_kh[f_i]) - np.abs(amp_ref)) / np.abs(amp_ref)
        ff_err[f_i] = float(np.max(mag_err))
    return p_kh, ff_err, p_fdtd


def test_far_field_matches_kh_pressure_on_axis_large_r() -> None:
    """Fraunhofer ``kh_far_field`` agrees with full ``kh_pressure`` at large r."""
    h = H_PATCH
    patches = huygens_box_patches(L_BOX, L_BOX, L_BOX, h, ORIGIN, offset_cells=2)
    k = np.array([2.0 * math.pi * 400.0 / C], dtype=_F64)
    p = np.ones((1, patches.centers.shape[0]), dtype=np.complex128) * (1.0 + 0.0j)
    dpdn = np.zeros_like(p)
    r = 80.0 * L_BOX
    pt = np.array([[0.0, 0.0, r]], dtype=_F64)
    p_full = kh_pressure(pt, patches, p, dpdn, k, near_subdivide=False)[0, 0]
    ff = kh_far_field(
        np.array([[0.0, 0.0, 1.0]], dtype=_F64), patches, p, dpdn, k
    )[0, 0]
    p_ff = far_field_at_radius(np.array([[ff]], dtype=_C128), k, R_ref=r)[0, 0]
    rel = abs(p_ff - p_full) / max(abs(p_full), 1e-30)
    assert rel < 0.05


def test_image_normals_flip_z() -> None:
    from fdtd.post.kirchhoff import _reflect_normals_z

    n = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=_F64)
    n_img = _reflect_normals_z(n, 0.0)
    assert abs(n_img[0, 2]) < 1e-15
    assert abs(n_img[1, 2] + 1.0) < 1e-15


def test_huygens_open_patches_baffle_extent() -> None:
    """Open Huygens box: no −z face, side centers start at z = h/2 above the baffle."""
    h = H_PATCH
    offset = 3
    patches = huygens_open_patches(0.04, 0.04, 0.0, h, (0.0, 0.0, 0.0), offset_cells=offset)
    assert np.all(patches.centers[:, 2] > 0.0)
    normals = patches.normals
    keys: set[tuple[int, int]] = set()
    for row in normals:
        axis = int(np.argmax(np.abs(row)))
        keys.add((axis, int(np.sign(row[axis]))))
    assert len(keys) == 5
    assert (2, -1) not in keys
    assert abs(patches.centers[:, 2].min() - 0.5 * h) < 1e-15


def test_huygens_surface_emulation() -> None:
    """KH on offset Huygens surface with cubic face p (order 4) and analytic v_n."""
    offset = 2
    patches = huygens_box_patches(L_BOX, L_BOX, L_BOX, H_PATCH, ORIGIN, offset_cells=offset)
    center = L_BOX / 2.0
    exterior = np.array(
        [
            [center + 3.0 * L_BOX, center, center],
            [center + 10.0 * L_BOX, center, center],
        ],
        dtype=_F64,
    )

    p_kh_o4, ff_o4, _ = _huygens_kh_errors(patches, order=4, exterior=exterior)
    p_kh_o2, ff_o2, _ = _huygens_kh_errors(patches, order=2, exterior=exterior)
    half_err = _half_cell_errors_at_3x()

    rtol_ext = {1_000.0: 0.005, 4_000.0: 0.005, 10_000.0: 0.002}
    rtol_ff = {1_000.0: 0.005, 4_000.0: 0.005, 10_000.0: 0.005}
    # Far-field at 10 kHz is limited by the 2 mm patch KH quadrature floor (~0.1–0.2%), not face-p interpolation.
    for f_i, f in enumerate(FREQS):
        c = _coeff(np.array([f]))[0]
        p_ex = _monopole_p(exterior, K[f_i], c)
        err_o4 = _rel_err(p_kh_o4[f_i], p_ex)
        assert np.all(err_o4 <= rtol_ext[f]), f"Huygens order=4 exterior {f} Hz"
        assert ff_o4[f_i] <= rtol_ff[f], f"Huygens order=4 far-field {f} Hz"

        print(
            f"Huygens {f:.0f} Hz order=4: "
            f"3×={_rel_err_scalar(p_kh_o4[f_i, 0], p_ex[0]):.4f}, "
            f"10×={_rel_err_scalar(p_kh_o4[f_i, 1], p_ex[1]):.4f}, "
            f"far-field max={ff_o4[f_i]:.4f} | "
            f"order=2: 3×={_rel_err_scalar(p_kh_o2[f_i, 0], p_ex[0]):.4f}, "
            f"10×={_rel_err_scalar(p_kh_o2[f_i, 1], p_ex[1]):.4f}, "
            f"far-field max={ff_o2[f_i]:.4f} | "
            f"half-cell 3×={half_err[f_i]:.4f}"
        )


def test_half_cell_face_offset() -> None:
    """FDTD-like face values: p and ∂p/∂n at h/2 outside each face; integration on-face."""
    h = H_PATCH
    patches_on = box_surface_patches(L_BOX, L_BOX, L_BOX, h, ORIGIN)
    centers = patches_on.centers
    normals = patches_on.normals
    sample_pts = centers + 0.5 * h * normals

    p_s = np.zeros((FREQS.size, centers.shape[0]), dtype=_C128)
    dpdn = np.zeros_like(p_s)
    for f_i, f in enumerate(FREQS):
        kf = K[f_i]
        c = _coeff(np.array([f]))[0]
        p_s[f_i] = _monopole_p(sample_pts, kf, c)
        dpdn[f_i] = _monopole_dpdn(sample_pts, normals, kf, c)

    center = L_BOX / 2.0
    pt = np.array([[center + 3.0 * L_BOX, center, center]], dtype=_F64)
    p_kh = kh_pressure(pt, patches_on, p_s, dpdn, K, near_subdivide=True)
    for f_i, f in enumerate(FREQS):
        c = _coeff(np.array([f]))[0]
        p_ex = _monopole_p(pt, K[f_i], c)
        err = _rel_err_scalar(p_kh[f_i], p_ex)
        print(f"half-cell face offset {f:.0f} Hz: rel_err={err:.4f}")
        assert np.isfinite(err)


def test_dtypes_complex128() -> None:
    patches = box_surface_patches(L_BOX, L_BOX, L_BOX, H_PATCH, ORIGIN)
    p_s, dpdn = _sample_surface(patches, FREQS)
    assert p_s.dtype == _C128 and dpdn.dtype == _C128
    pt = np.array([[0.2, 0.05, 0.05]], dtype=_F64)
    out = kh_pressure(pt, patches, p_s, dpdn, K)
    assert out.dtype == _C128
    amp = kh_far_field(_fibonacci_sphere(8), patches, p_s, dpdn, K)
    assert amp.dtype == _C128
    p1m = far_field_at_radius(amp, K)
    assert p1m.dtype == _C128


def test_kh_monopole_rigid_plane_image() -> None:
    """Rigid-plane image Green's function matches direct + reflected monopole."""
    from fdtd.post.kirchhoff import _green, _reflect_points

    kf = K[0]
    src = np.array([0.0, 0.0, 0.05], dtype=_F64)
    obs = np.array([0.12, -0.04, 0.11], dtype=_F64)
    plane_z = 0.0
    y = src.reshape(1, 3)
    rdx = obs.reshape(1, 1, 3) - y.reshape(1, 1, 3)
    R = np.linalg.norm(rdx, axis=-1)
    G0 = _green(kf, R)[0, 0]
    y_img = _reflect_points(y, plane_z)
    rdx_i = obs.reshape(1, 1, 3) - y_img.reshape(1, 1, 3)
    G1 = _green(kf, np.linalg.norm(rdx_i, axis=-1))[0, 0]
    coeff = _coeff(np.array([FREQS[0]]))[0]
    src_img = src.copy()
    src_img[2] = -src[2]
    p_ref = _monopole_p(obs.reshape(1, 3), kf, coeff, source=src)[0] + _monopole_p(
        obs.reshape(1, 3), kf, coeff, source=src_img
    )[0]
    p_ker = coeff * (4.0 * math.pi) * (G0 + G1)
    err = abs(p_ker - p_ref) / max(abs(p_ref), 1e-30)
    assert err < 1e-3, f"image Green rel err {err:.4e}"


def test_dpdn_from_vn_export() -> None:
    from fdtd.post.kirchhoff import dpdn_from_vn

    vn = np.ones((3, 4), dtype=_C128)
    omega = OMEGA
    out = dpdn_from_vn(vn, omega, RHO)
    expected = -1j * omega[:, np.newaxis] * RHO * vn
    np.testing.assert_allclose(out, expected, rtol=0, atol=0)