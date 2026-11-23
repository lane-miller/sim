"""Step 7b: image KH (patch expansion) and Result with environment planes."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from boxrad.config import Config
from boxrad.geomrec import GeomRecorder, huygens_patches_planes
from boxrad.geomrun import prepare_geom_run, step_geom
from boxrad.post.directivity import di_from_pattern
from boxrad.post.images import expand_patches, image_sources, rigid_sides_from_planes
from boxrad.post.kirchhoff import SurfacePatches, dpdn_from_vn, kh_far_field, kh_pressure
from boxrad.post.transfer import h_transfer_err
from boxrad.result import Result

SHOWCASE = Path(__file__).resolve().parents[1] / "configs" / "showcase"
_C128 = np.complex128
_F64 = np.float64
_C = 343.0
_RHO = 1.204
_H = 0.005
_HI = np.array([0.20, 0.23, 0.25], dtype=_F64)
_SOURCE = np.array([0.05, 0.08, 0.10], dtype=_F64)


def _load(name: str) -> Config:
    return Config.from_json((SHOWCASE / name).read_text(encoding="utf-8"))


def _lo_for_planes(planes: list[dict]) -> np.ndarray:
    lo = np.full(3, -0.15, dtype=_F64)
    for pl in planes:
        if pl["side"] == "-":
            ax = {"x": 0, "y": 1, "z": 2}[pl["axis"]]
            lo[ax] = float(pl["offset"])
    return lo


def _open_patches(planes: list[dict]) -> SurfacePatches:
    lo = _lo_for_planes(planes)
    rigid = rigid_sides_from_planes(planes)
    domain = np.zeros(3, dtype=_F64)
    return huygens_patches_planes(lo, _HI.copy(), _H, 0, rigid, domain, domain + 1.0)


def _monopole_p(points: np.ndarray, k: float, sources: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(points.shape[0], dtype=_C128)
    for s in sources:
        r = np.linalg.norm(points - s, axis=-1)
        r = np.maximum(r, 1e-12)
        out += np.exp(-1j * k * r) / r
    return out


def _monopole_dpdn(
    centers: np.ndarray,
    normals: np.ndarray,
    k: float,
    sources: list[np.ndarray],
) -> np.ndarray:
    out = np.zeros(centers.shape[0], dtype=_C128)
    for s in sources:
        eps = centers - s
        r = np.linalg.norm(eps, axis=-1, keepdims=True)
        r = np.maximum(r, 1e-12)
        rhat = eps / r
        p = np.exp(-1j * k * r[..., 0]) / r[..., 0]
        grad = -p[..., np.newaxis] * rhat * (1.0 / r + 1j * k)
        out += np.sum(grad * normals, axis=-1)
    return out


def _far_ref_fixed(dirs: np.ndarray, k: float, planes: list[dict]) -> np.ndarray:
    sources = image_sources(_SOURCE, planes)
    phase = np.zeros(dirs.shape[0], dtype=_C128)
    for s in sources:
        phase += np.exp(1j * k * (dirs @ s))
    return phase


def _c19a_planes_cases() -> list[list[dict]]:
    return [
        [{"axis": "z", "side": "-", "offset": 0.0}],
        [{"axis": "z", "side": "-", "offset": 0.0}, {"axis": "y", "side": "-", "offset": 0.0}],
        [
            {"axis": "z", "side": "-", "offset": 0.0},
            {"axis": "y", "side": "-", "offset": 0.0},
            {"axis": "x", "side": "-", "offset": 0.0},
        ],
    ]


@pytest.mark.parametrize("planes", _c19a_planes_cases(), ids=["1p", "2p", "3p"])
def test_c19a_synthetic_kh(planes: list[dict]) -> None:
    freqs = [300.0, 1500.0, 3000.0]
    patches0 = _open_patches(planes)
    thetas = np.deg2rad(np.arange(0, 91, 10, dtype=_F64))
    phi = 0.7
    st = np.sin(thetas)
    dirs = np.column_stack([st * np.cos(phi), st * np.sin(phi), np.cos(thetas)])
    near_pts = np.array(
        [
            [0.5, 0.6, 0.4],
            [1.0, 0.3, 0.05],
            [0.3, 1.2, 0.8],
        ],
        dtype=_F64,
    )
    for f_hz in freqs:
        k = 2.0 * math.pi * f_hz / _C
        sources = image_sources(_SOURCE, planes)
        p_face_scalar = _monopole_p(patches0.centers, k, sources)
        dpdn_scalar = _monopole_dpdn(patches0.centers, patches0.normals, k, sources)
        vn_face_scalar = dpdn_scalar / (-1j * 2.0 * math.pi * f_hz * _RHO)
        exp_patches, p_face, vn_face = expand_patches(
            patches0, planes, p_face_scalar, vn_face_scalar
        )
        omega = np.array([2.0 * math.pi * f_hz], dtype=_F64)
        k_sel = np.array([k], dtype=_F64)
        dpdn = dpdn_from_vn(vn_face[np.newaxis, :], omega, _RHO)
        kh_ff = kh_far_field(dirs, exp_patches, p_face[np.newaxis, :], dpdn, k_sel)[0]
        ref_ff = _far_ref_fixed(dirs, k, planes)
        for h_kh, h_ref in zip(kh_ff, ref_ff, strict=True):
            db, ph = h_transfer_err(h_ref, h_kh)
            assert db <= 0.1, f"far {f_hz} Hz db={db}"
            assert abs(ph) <= 1.0, f"far {f_hz} Hz ph={ph}"
        p_row = p_face[np.newaxis, :]
        kh_nf = kh_pressure(near_pts, exp_patches, p_row, dpdn, k_sel, near_subdivide=False)[0]
        ref_nf = _monopole_p(near_pts, k, image_sources(_SOURCE, planes))
        for h_kh, h_ref in zip(kh_nf, ref_nf, strict=True):
            db, ph = h_transfer_err(h_ref, h_kh)
            assert db <= 0.1, f"near {f_hz} Hz db={db}"
            assert abs(ph) <= 1.0, f"near {f_hz} Hz ph={ph}"


def _di_monopole(planes: list[dict], f_hz: float = 100.0) -> float:
    source = np.array([0.01, 0.01, 0.01], dtype=_F64)
    patches = _open_patches(planes)
    k = 2.0 * math.pi * f_hz / _C
    sources = image_sources(source, planes)
    p = _monopole_p(patches.centers, k, sources)
    dpdn = _monopole_dpdn(patches.centers, patches.normals, k, sources)
    vn = dpdn / (-1j * 2.0 * math.pi * f_hz * _RHO)
    exp_patches, p_face, vn_face = expand_patches(patches, planes, p, vn)
    omega = np.array([2.0 * math.pi * f_hz], dtype=_F64)
    k_sel = np.array([k], dtype=_F64)
    dpdn_row = dpdn_from_vn(vn_face[np.newaxis, :], omega, _RHO)
    p_row = p_face[np.newaxis, :]

    def pattern(dirs: np.ndarray) -> np.ndarray:
        return kh_far_field(dirs, exp_patches, p_row, dpdn_row, k_sel)[0]

    return di_from_pattern(pattern, False, 24, 48) + 10.0 * math.log10(float(1 << len(planes)))


@pytest.mark.parametrize(
    ("planes", "n_p"),
    [
        ([{"axis": "z", "side": "-", "offset": 0.0}], 1),
        (
            [{"axis": "z", "side": "-", "offset": 0.0}, {"axis": "y", "side": "-", "offset": 0.0}],
            2,
        ),
        (
            [
                {"axis": "z", "side": "-", "offset": 0.0},
                {"axis": "y", "side": "-", "offset": 0.0},
                {"axis": "x", "side": "-", "offset": 0.0},
            ],
            3,
        ),
    ],
)
def test_di_image_convention(planes: list[dict], n_p: int) -> None:
    di = _di_monopole(planes)
    assert di == pytest.approx(3.01 * n_p, abs=0.05)


def _short_run(cfg: Config, path: Path, n_steps: int = 150) -> None:
    run = prepare_geom_run(copy.deepcopy(cfg))
    rec = GeomRecorder(
        cfg,
        run.derived,
        run.gg,
        run.geometry,
        run.info,
        path,
        buffer_samples=32,
    )
    for n in range(n_steps):
        step_geom(run, n)
        rec.sample(run.fields, n)
    rec.close()


def _center_sets_allclose(a: np.ndarray, b: np.ndarray, atol: float = 1e-12) -> bool:
    if a.shape != b.shape:
        return False
    used = np.zeros(a.shape[0], dtype=bool)
    for row in b:
        d = np.linalg.norm(a - row, axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] > atol:
            return False
        used[j] = True
    return bool(np.all(used))


def test_result_symmetry_bprime(tmp_path: Path) -> None:
    full = _load("c19_sym_full.json")
    half = _load("c19_half_x.json")
    quarter = _load("c19_quarter_xy.json")
    pf = tmp_path / "full.h5"
    ph = tmp_path / "half.h5"
    pq = tmp_path / "quarter.h5"
    _short_run(full, pf)
    _short_run(half, ph)
    _short_run(quarter, pq)
    freqs = [150.0, 800.0, 2200.0]
    thetas = np.deg2rad([15.0, 35.0, 55.0, 70.0])
    phis = np.deg2rad([20.0, 50.0])
    th_g, ph_g = np.meshgrid(thetas, phis, indexing="ij")
    field_pts = np.array([[0.15, 0.08, 0.12], [0.22, 0.11, 0.18]], dtype=_F64)
    with Result(pf) as rf, Result(ph) as rh, Result(pq) as rq:
        planes_h = rh.planes
        patches_h, _, p_h, vn_h = rh._huygens_at_f(freqs[0])
        exp_h, _, _ = expand_patches(patches_h, planes_h, p_h, vn_h)
        centers_f = np.asarray(rf._file()["huygens/centers"][()], dtype=_F64)
        assert _center_sets_allclose(exp_h.centers, centers_f)
        for f_hz in freqs:
            d_f = rf.directivity(th_g.ravel(), ph_g.ravel(), f_hz, frame="world")
            d_h = rh.directivity(th_g.ravel(), ph_g.ravel(), f_hz, frame="world")
            d_q = rq.directivity(th_g.ravel(), ph_g.ravel(), f_hz, frame="world")
            ref = np.maximum(np.abs(d_f), 1e-30)
            assert float(np.max(np.abs(d_f - d_h) / ref)) <= 1e-5
            assert float(np.max(np.abs(d_f - d_q) / ref)) <= 1e-5
            h_f = rf.field(field_pts, f_hz, frame="world")
            h_h = rh.field(field_pts, f_hz, frame="world")
            h_q = rq.field(field_pts, f_hz, frame="world")
            ref_h = np.maximum(np.abs(h_f), 1e-30)
            assert float(np.max(np.abs(h_f - h_h) / ref_h)) <= 1e-5
            assert float(np.max(np.abs(h_f - h_q) / ref_h)) <= 1e-5


def test_field_and_directivity_masks(tmp_path: Path) -> None:
    half = _load("c19_half_x.json")
    out = tmp_path / "half.h5"
    _short_run(half, out, n_steps=80)
    with Result(out) as res:
        with pytest.raises(ValueError, match="behind rigid plane"):
            res.field([-0.05, 0.1, 0.1], 200.0, frame="world")
        with pytest.raises(ValueError, match="open acoustic region"):
            res.directivity(np.pi / 2, np.deg2rad(150.0), 200.0, frame="world")
        closed = np.array([np.pi / 2, np.deg2rad(150.0)], dtype=_F64)
        open_dir = np.array([np.deg2rad(40.0), np.deg2rad(30.0)], dtype=_F64)
        d_m = res.directivity(
            np.concatenate([closed[:1], open_dir[:1]]),
            np.concatenate([closed[1:], open_dir[1:]]),
            200.0,
            frame="world",
            allow_masked=True,
        )
        assert np.isnan(d_m[0, 0])
        assert np.isfinite(d_m[0, 1])


def test_result_frame_default(tmp_path: Path) -> None:
    half = _load("c19_half_x.json")
    out = tmp_path / "half.h5"
    _short_run(half, out, n_steps=40)
    with Result(out, frame="world") as res:
        assert res.frame_default == "world"
        assert res._resolve_frame(None) == "world"