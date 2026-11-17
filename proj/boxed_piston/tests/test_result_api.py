"""Step-2 API: Result, CLIs, and post-run HDF5 fields."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from boxrad.config import (
    Band,
    Box,
    Config,
    Numerics,
    Output,
    Radiator,
    derive,
)
from boxrad.grid import Grid
from boxrad.post.kirchhoff import dpdn_from_vn, huygens_box_patches, kh_pressure_loop
from boxrad.post.transfer import (
    h_transfer_err,
    kh_huygens_pressure_at_points,
    load_huygens_spectra,
    min_distance_to_huygens_surface,
    probe_fdtd_pressure_spectrum,
)
from boxrad.recorder import resolve_huygens_offset
from boxrad.info import main as info_main
from boxrad.post.finalize import conventions_text
from boxrad.post.zrad import zrad_at_frequencies
from boxrad.result import Result
from boxrad.runner import run
from boxrad.viz import main as plot_main


def _tiny_probe_z(cfg: Config) -> float:
    derived = derive(cfg)
    grid = Grid(cfg, derived)
    assert grid.i_box is not None and grid.j_box is not None and grid.k_box is not None
    origin = (
        float(grid.x_vx[grid.i_box]),
        float(grid.y_vy[grid.j_box]),
        float(grid.z_vz[grid.k_box]),
    )
    off = resolve_huygens_offset(cfg)
    hy = huygens_box_patches(
        derived.realized_W,
        derived.realized_H,
        derived.realized_D,
        derived.h,
        origin,
        offset_cells=off,
    )
    z_min_out = float(np.max(hy.centers[:, 2]))
    return z_min_out + 2.5 * float(derived.h)


def _tiny_config() -> Config:
    base = Config(
        box=Box(W=0.06, H=0.06, D=0.05),
        radiator=Radiator("circle", dims=[0.03], x_c=0.0, y_c=0.0),
        band=Band(f_min=20.0, f_max=4_000.0),
        numerics=Numerics(
            gap_cells=8,
            cpml_cells=8,
            ppw=8.0,
            cells_across_radiator=6,
            cell_budget=10**9,
            cfl_safety=0.99,
            n_transits=0.08,
        ),
        output=Output(path="", probes=[]),
    )
    z_probe = _tiny_probe_z(base)
    return Config(
        box=base.box,
        radiator=base.radiator,
        band=base.band,
        numerics=base.numerics,
        output=Output(path="", probes=[[0.0, 0.0, z_probe]]),
    )


@pytest.fixture(scope="module")
def tiny_h5_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Config]:
    """One full derived-``n_steps`` run for physics checks (not step-truncated)."""
    cfg = _tiny_config()
    work = tmp_path_factory.mktemp("result_api_tiny")
    path = work / "out.h5"
    run(cfg, out_path=path, progress=False)
    with h5py.File(path, "r") as f:
        tail_db = float(f["meta"].attrs["tail_db"])
    assert tail_db < -40.0, (
        f"_tiny_config() FDTD tail_db={tail_db:.1f} dB; physics tests need tail_db < −40 dB"
    )
    return path, cfg


@pytest.fixture
def patch_few_steps(monkeypatch):
    n_steps = 24
    real_derive = derive

    def wrapped(cfg: Config):
        d = real_derive(cfg)
        d.n_steps = n_steps
        d.run_time_s = n_steps * d.dt
        return d

    monkeypatch.setattr("boxrad.runner.derive", wrapped)
    return n_steps


def test_result_z_matches_zrad_from_h5(tiny_h5_run):
    path, cfg = tiny_h5_run
    res = Result(path)
    f_stored, z_stored = res.z(None)
    assert f_stored.ndim == 1 and z_stored.ndim == 1
    f_chk = f_stored[::5]
    _, z_ref = zrad_at_frequencies(path, f_chk)
    np.testing.assert_allclose(z_stored[::5], z_ref, rtol=0, atol=1e-10)
    rho, c = cfg.air.rho, cfg.air.c
    S = res.S_grid
    f1 = f_chk[10]
    za = res.Z_acoustic(f1)
    zm = res.Z_mechanical(f1)
    _, z1 = res.z(f1)
    assert za.shape == (1,) and zm.shape == (1,) and z1.shape == (1,)
    np.testing.assert_allclose(za[0], rho * c * z1[0] / S, rtol=0, atol=1e-12)
    np.testing.assert_allclose(zm[0], rho * c * S * z1[0], rtol=0, atol=1e-12)
    assert float(np.max(res.power(f_chk[5:15]))) > 0.0


def test_field_probe_and_inside_error(tiny_h5_run):
    path, cfg = tiny_h5_run
    res = Result(path)
    f_min = cfg.band.f_min
    f_max = cfg.band.f_max
    f_hz = 600.0
    assert f_hz >= 2.0 * f_min and f_hz <= 0.8 * f_max
    probe_center = np.asarray(cfg.output.probes[0], dtype=np.float64)
    derived = res.derived
    patches, V, p_face, vn_face = load_huygens_spectra(path, derived, f_hz)
    d_hy = min_distance_to_huygens_surface(probe_center, patches)
    assert d_hy >= 2.0 * float(derived.h), (
        f"probe must sit ≥ 2h outside Huygens (dist={d_hy:.4g} m, 2h={2*derived.h:.4g} m)"
    )
    with h5py.File(path, "r") as f:
        t_pr = np.asarray(f["probes/times_p"][()], dtype=np.float64)
        p_ts = np.asarray(f["probes/p"][()], dtype=np.float64)
    p_q = probe_fdtd_pressure_spectrum(p_ts, t_pr, f_hz, 0)
    h_probe = p_q / V
    p_kh = kh_huygens_pressure_at_points(
        probe_center.reshape(1, 3),
        patches,
        p_face,
        vn_face,
        f_hz,
        cfg.air.rho,
        cfg.air.c,
        geometry=cfg.geometry,
    )[0]
    h_kh = p_kh / V
    db, ph = h_transfer_err(h_probe, h_kh)
    h_fd = res.field(probe_center[np.newaxis, :], f_hz)
    assert h_fd.shape == (1, 1)
    h_val = complex(np.ravel(h_fd)[0])
    db_fd, ph_fd = h_transfer_err(h_probe, h_val)
    assert db_fd <= 0.2, (
        f"|Δ|={db_fd:.3f} dB  |h_probe|={abs(h_probe):.4g}  "
        f"|h_val|={abs(h_val):.4g}  |V|={abs(V):.4g}  "
        f"dist_Huygens={d_hy:.4g} m"
    )
    assert abs(ph_fd) <= 2.0, (
        f"∠={ph_fd:.2f}°  |h_probe|={abs(h_probe):.4g}  |h_val|={abs(h_val):.4g}  "
        f"|V|={abs(V):.4g}  dist_Huygens={d_hy:.4g} m"
    )
    assert db <= 0.2 and abs(ph) <= 2.0
    with h5py.File(path, "r") as f:
        center = np.asarray(f["huygens/centers"][0], dtype=np.float64)
    with pytest.raises(ValueError, match="Huygens|probe"):
        res.field(center[np.newaxis, :], f_hz)


def test_directivity_on_axis_normalized(tiny_h5_run):
    path, cfg = tiny_h5_run
    res = Result(path)
    f_hz = 800.0
    assert f_hz >= 2.0 * cfg.band.f_min and f_hz <= 0.8 * cfg.band.f_max
    pat = res.directivity(0.0, 0.0, f_hz, normalize=True)
    assert pat.shape == (1, 1)
    assert np.iscomplexobj(pat)
    assert abs(pat[0, 0] - (1.0 + 0.0j)) < 1e-12
    pat_raw = res.directivity(np.array([0.0, 0.7]), 0.0, f_hz)
    pat_norm = res.directivity(np.array([0.0, 0.7]), 0.0, f_hz, normalize=True)
    np.testing.assert_allclose(pat_norm[0], pat_raw[0] / pat_raw[0, 0], rtol=1e-12, atol=0)


def test_directivity_matches_field_at_50m(tiny_h5_run):
    path, cfg = tiny_h5_run
    res = Result(path)
    f_hz = 1000.0
    r = 50.0
    theta = np.array([0.0, 0.5, 1.0, 1.6, 2.4])
    phi = 0.3
    st = np.sin(theta)
    pts = r * np.column_stack(
        [st * np.cos(phi), st * np.sin(phi), np.cos(theta)]
    )
    D = res.directivity(theta, phi, f_hz)[0]
    H = res.field(pts, f_hz)[0]
    k = 2.0 * np.pi * f_hz / cfg.air.c
    approx = r * np.exp(1j * k * r) * H
    db = 20.0 * np.log10(np.abs(approx) / np.abs(D))
    ph = np.rad2deg(np.angle(approx / D))
    assert np.max(np.abs(db)) < 0.2, db
    assert np.max(np.abs(ph)) < 2.0, ph


def test_field_volume_velocity_is_velocity_over_S_grid(tiny_h5_run):
    path, cfg = tiny_h5_run
    res = Result(path)
    pts = np.array([[0.0, 0.0, 1.0], [0.3, -0.2, 0.4], [-0.4, 0.1, -0.3]])
    f = np.array([300.0, 1500.0])
    h_v = res.field(pts, f)
    h_q = res.field(pts, f, per="volume_velocity")
    np.testing.assert_allclose(h_q * res.S_grid, h_v, rtol=1e-12, atol=0)


def test_field_in_face_plane_beyond_surface_is_accepted(tiny_h5_run):
    path, cfg = tiny_h5_run
    res = Result(path)
    f_hz = 900.0
    patches, V, p_face, vn_face = load_huygens_spectra(path, res.derived, f_hz)
    h = float(res.derived.h)
    top = patches.normals[:, 2] > 0.5
    z_top = float(np.max(patches.centers[top, 2]))
    x_out = float(np.max(patches.centers[:, 0])) + 4.0 * h
    pt = np.array([[x_out, 0.0, z_top]])
    assert min_distance_to_huygens_surface(pt[0], patches) > 3.0 * h
    h_fd = res.field(pt, f_hz)
    p_kh = kh_huygens_pressure_at_points(
        pt, patches, p_face, vn_face, f_hz, cfg.air.rho, cfg.air.c, geometry="box"
    )
    np.testing.assert_allclose(h_fd[0], p_kh / V, rtol=1e-12, atol=0)


def test_kh_box_fast_path_matches_reference_loop(tiny_h5_run):
    path, cfg = tiny_h5_run
    res = Result(path)
    f_hz = 700.0
    rho, c = cfg.air.rho, cfg.air.c
    patches, _, p_face, vn_face = load_huygens_spectra(path, res.derived, f_hz)
    pts = np.array([[0.0, 0.0, 0.3], [0.2, -0.1, 0.05], [-0.15, 0.1, -0.2]])
    fast = kh_huygens_pressure_at_points(
        pts, patches, p_face, vn_face, f_hz, rho, c, geometry="box"
    )
    omega = np.array([2.0 * np.pi * f_hz])
    k = omega / c
    dpdn = dpdn_from_vn(np.asarray(vn_face)[np.newaxis, :], omega, rho)
    ref = kh_pressure_loop(pts, patches, np.asarray(p_face)[np.newaxis, :], dpdn, k)[0]
    np.testing.assert_allclose(fast, ref, rtol=1e-10, atol=0)


def test_band_check_raises_out_of_band(tiny_h5_run):
    path, cfg = tiny_h5_run
    res = Result(path)
    f_min, f_max = res.band
    pt = np.array([0.0, 0.0, 1.0])
    for f_bad in (0.5 * f_min, 1.5 * f_max):
        calls = (
            lambda: res.z(f_bad),
            lambda: res.Z_acoustic(f_bad),
            lambda: res.Z_mechanical(f_bad),
            lambda: res.power(f_bad),
            lambda: res.field(pt, f_bad),
            lambda: res.directivity(0.0, 0.0, f_bad),
            lambda: res.directivity_index(f_bad),
            lambda: res.surface_p(pt, f_bad),
        )
        for call in calls:
            with pytest.raises(ValueError, match="allow_out_of_band"):
                call()
    with pytest.raises(ValueError, match="outside"):
        res.Z_acoustic(np.array([0.5 * f_max, 1.5 * f_max]))


def test_band_check_allow_out_of_band_warns(tiny_h5_run):
    path, cfg = tiny_h5_run
    res = Result(path)
    f_min, f_max = res.band
    f_lo = 0.5 * f_min
    f_hi = 1.05 * f_max
    pt = np.array([[0.0, 0.0, 1.0]])
    with pytest.warns(UserWarning, match="outside"):
        _, z = res.z(f_lo, allow_out_of_band=True)
    assert z.shape == (1,)
    with pytest.warns(UserWarning, match="outside"):
        res.power(f_lo, allow_out_of_band=True)
    with pytest.warns(UserWarning, match="outside"):
        h_fd = res.field(pt, f_hi, allow_out_of_band=True)
    assert h_fd.shape == (1, 1)
    with pytest.warns(UserWarning, match="outside"):
        res.directivity(0.0, 0.0, f_hi, allow_out_of_band=True)


def test_band_check_in_band_and_stored_table_silent(tiny_h5_run):
    import warnings as _warnings

    path, cfg = tiny_h5_run
    res = Result(path)
    f_min, f_max = res.band
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        f_tab, _ = res.z(None)
        res.z(np.array([f_min, f_max]))
        res.power(f_tab[::20])
    assert f_tab.size > 10


def test_result_output_shapes(tiny_h5_run):
    path, cfg = tiny_h5_run
    res = Result(path)
    f_min, f_max = cfg.band.f_min, cfg.band.f_max
    f_two = np.array([max(2.0 * f_min, 120.0), 0.5 * f_max], dtype=np.float64)
    theta = np.linspace(0.1, 0.9, 5)
    pat = res.directivity(theta, 0.0, f_two)
    assert pat.shape == (2, 5)
    probe = np.asarray(cfg.output.probes[0], dtype=np.float64)
    h = res.field(probe, f_two)
    assert h.shape == (2, 1)


def test_meta_self_checks(patch_few_steps):
    cfg = _tiny_config()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.h5"
        run(cfg, out_path=path, progress=False)
        with h5py.File(path, "r") as f:
            meta = f["meta"]
            assert "tail_db" in meta.attrs
            assert "boxrad_version" in meta.attrs
            assert "conventions" in meta.attrs
            conv = str(meta.attrs["conventions"])
            assert conv == conventions_text(cfg)
            assert "origin = radiator center on the front face" in conv
            assert "directivity phase referenced to the origin" in conv
            assert "Yee" not in conv
            acc = f["meta/accuracy"]
            assert "ppw_realized" in acc.attrs
            assert "cells_across_radiator" in acc.attrs
            zrad = f["results/zrad"]
            assert zrad.shape[0] > 10
            assert "f" in zrad.attrs


def test_clis(patch_few_steps, capsys):
    cfg = _tiny_config()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        path = tmp_p / "out.h5"
        run(cfg, out_path=path, progress=False)
        assert info_main([str(path)]) == 0
        out = capsys.readouterr().out
        assert "tail_db" in out or "S_grid" in out
        assert "zrad_source: fdtd" in out
        assert "boxrad_version:" in out
        assert "/results/zrad:" in out and "points" in out
        assert "e^{+jωt}" in out
        png = tmp_p / "cli_geom.png"
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "smoke.json"
        baffle_path = Path(__file__).resolve().parents[1] / "configs" / "c06_baffle.json"
        for src in (cfg_path, baffle_path):
            assert plot_main(["-o", str(png), str(src)]) == 0
            assert png.is_file()
            png.unlink()


def test_run_returns_result_path_compatible(patch_few_steps):
    from boxrad import run as run_fn, Result

    cfg = _tiny_config()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.h5"
        out = run_fn(cfg, out_path=path, progress=False)
        assert out == path
        assert isinstance(Result(path).summary(), str)