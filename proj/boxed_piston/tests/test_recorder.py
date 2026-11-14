"""Unit tests for fdtd.recorder (chunked HDF5 output)."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

h5py = pytest.importorskip("h5py")

from fdtd.config import (  # noqa: E402
    Air,
    Band,
    Box,
    Config,
    Derived,
    Numerics,
    Output,
    Radiator,
    SQRT3,
    derive,
)
from fdtd.grid import Grid, allocate_fields  # noqa: E402
from fdtd.post.kirchhoff import face_p_from_cells  # noqa: E402
from fdtd.recorder import (  # noqa: E402
    Recorder,
    decimation_factor,
    validate_huygens_clearance,
)
from fdtd.source import BoxBoundary, velocity_pulse  # noqa: E402
from fdtd.run import run  # noqa: E402

_F64 = np.float64


def _sync_dt(cfg: Config, d: Derived, cfl_safety: float | None = None) -> float:
    cs = cfg.numerics.cfl_safety if cfl_safety is None else cfl_safety
    dt = cs * d.h / (cfg.air.c * SQRT3)
    d.dt = dt
    return dt


def _small_problem(
    *,
    gap_cells: int = 8,
    n_transits: float = 0.15,
    f_max: float = 4_000.0,
    n_steps_override: int | None = None,
) -> tuple[Config, Derived, Grid, BoxBoundary]:
    cfg = Config(
        box=Box(W=0.06, H=0.06, D=0.05),
        radiator=Radiator("circle", dims=[0.03], x_c=0.0, y_c=0.0),
        band=Band(f_min=20.0, f_max=f_max),
        numerics=Numerics(
            gap_cells=gap_cells,
            cpml_cells=8,
            ppw=8.0,
            cells_across_radiator=6,
            cell_budget=10**9,
            cfl_safety=0.99,
            n_transits=n_transits,
        ),
        output=Output(path="/tmp/recorder_test.h5"),
    )
    d = derive(cfg)
    _sync_dt(cfg, d)
    if n_steps_override is not None:
        d.n_steps = n_steps_override
        d.run_time_s = n_steps_override * d.dt
    g = Grid(cfg, d, radiator_supersample=16)
    boundary = BoxBoundary.from_grid(g, d)
    return cfg, d, g, boundary


def _run_recorder(
    cfg: Config,
    d: Derived,
    g: Grid,
    boundary: BoxBoundary,
    n_steps: int,
    path: Path,
    **rec_kwargs,
) -> None:
    fields = allocate_fields(g)
    fields.p[:] = 1.0
    rec = Recorder(cfg, d, g, boundary, path, **rec_kwargs)
    m = rec.decimation
    for n in range(n_steps):
        fields.p += 0.01 * n
        fields.vx[:] = 0.1 * (n % 3)
        fields.vy[:] = -0.05 * (n % 5)
        fields.vz[:] = 0.02 * (n % 7)
        rec.sample(fields, n)
    rec.close()


def test_hdf5_structure_shapes_and_finite():
    cfg, d, g, boundary = _small_problem(n_steps_override=24)
    n_steps = d.n_steps
    m = decimation_factor(d.f_max, d.dt)
    n_dec = (n_steps + m - 1) // m

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.h5"
        _run_recorder(cfg, d, g, boundary, n_steps, path)

        with h5py.File(path, "r") as f:
            assert f["radiator/v_pulse_integer"].shape == (n_steps,)
            assert f["radiator/v_pulse_half"].shape == (n_steps,)
            assert f["radiator/p_avg"].shape == (n_steps,)
            assert f["radiator/times_p"].shape == (n_steps,)
            assert f["radiator/times_v_half"].shape == (n_steps,)
            nh = f["huygens/p_face"].shape[1]
            nb = f["box_surface/p"].shape[1]
            assert f["huygens/p_face"].shape == (n_dec, nh)
            assert f["huygens/vn"].shape == (n_dec, nh)
            assert f["huygens/times_p"].shape == (n_dec,)
            assert f["huygens/times_vn"].shape == (n_dec,)
            assert f["box_surface/p"].shape == (n_dec, nb)
            assert f["box_surface/times"].shape == (n_dec,)
            assert f["meta"].attrs["n_steps"] == n_steps
            assert f["meta"].attrs["decimation_factor"] == m

            for name in fvisit_datasets(f):
                arr = f[name][()]
                if arr.size:
                    assert np.all(np.isfinite(arr.astype(np.float64)))


def fvisit_datasets(h5file) -> list[str]:
    out: list[str] = []

    def visitor(n, obj):
        if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "fcui":
            out.append(n)

    h5file.visititems(visitor)
    return out


def test_meta_round_trip():
    cfg, d, g, boundary = _small_problem(n_steps_override=6)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "meta.h5"
        _run_recorder(cfg, d, g, boundary, d.n_steps, path)
        with h5py.File(path, "r") as f:
            cfg2 = Config.from_json(f["meta/config_json"][()].decode())
            d2 = json.loads(f["meta/derived_json"][()].decode())
            assert cfg2.box is not None
            assert abs(cfg2.box.W - cfg.box.W) < 1e-12
            assert abs(d2["h"] - d.h) < 1e-12
            assert abs(d2["dt"] - d.dt) < 1e-12


def test_radiator_face_extrapolation_sign():
    """p = a·z + b on the grid; face value at z = 0 with the + sign correction."""
    cfg, d, g, boundary = _small_problem(n_steps_override=1)
    fields = allocate_fields(g)
    assert g.z_p is not None
    assert g.k_face_z is not None
    a, b = 2.0e3, -50.0
    fields.p[:] = (a * g.z_p[np.newaxis, np.newaxis, :] + b).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        rec = Recorder(cfg, d, g, boundary, Path(tmp) / "r.h5", offset_cells=3)
    p_flat = fields.p.reshape(-1)
    h = d.h
    rho = cfg.air.rho
    w = rec._rad_w
    assert w.size > 0
    pc = p_flat[rec._rad_p_idx].astype(_F64)
    # v′ = −a / (ρ·w) cancels the half-cell offset for a linear p(z).
    vprime = -a / (rho * w)
    corr = 0.5 * h * rho * w * vprime
    p_face = pc + corr
    assert np.all(p_face == pytest.approx(b, rel=1e-6, abs=0))
    p_avg = float(np.sum(w * p_face) / rec._rad_w_sum)
    assert p_avg == pytest.approx(b, rel=0, abs=1e-5)


def test_huygens_p_matches_face_p_from_cells_cubic():
    cfg, d, g, boundary = _small_problem(n_steps_override=1)
    with tempfile.TemporaryDirectory() as tmp:
        rec = Recorder(cfg, d, g, boundary, Path(tmp) / "h.h5", offset_cells=3)
    fields = allocate_fields(g)
    idx4 = rec._huygens_idx4
    h = g.h
    u = np.array([-1.5, -0.5, 0.5, 1.5], dtype=_F64) * h
    for q in range(idx4.shape[0]):
        vals = 80.0 * u**3 - 12.0 * u**2 + 0.35 * u + 2.5
        for s in range(4):
            fields.p.reshape(-1)[idx4[q, s]] = np.float32(vals[s])
    p_flat = fields.p.reshape(-1)
    est = rec._huygens_p(p_flat)
    for q in range(idx4.shape[0]):
        samples = [float(p_flat[idx4[q, s]]) for s in range(4)]
        ref = face_p_from_cells(*samples, order=4)
        if hasattr(ref, "item"):
            ref = ref.item()
        assert est[q] == pytest.approx(ref, rel=0, abs=1e-4)


def test_huygens_clearance_validation():
    cfg, d, g, boundary = _small_problem(gap_cells=8)
    with pytest.raises(ValueError, match="offset .* < 2"):
        validate_huygens_clearance(d, 1)
    d_bad = derive(
        Config(
            box=cfg.box,
            radiator=cfg.radiator,
            band=cfg.band,
            numerics=Numerics(
                gap_cells=4,
                cpml_cells=8,
                ppw=8.0,
                cells_across_radiator=6,
                cell_budget=10**9,
            ),
        )
    )
    with pytest.raises(ValueError, match="leaves"):
        validate_huygens_clearance(d_bad, 3)
    with pytest.raises(ValueError, match="leaves"):
        with tempfile.TemporaryDirectory() as tmp:
            Recorder(cfg, d_bad, g, boundary, Path(tmp) / "bad.h5", offset_cells=3)


def test_buffer_sizes_bounded():
    cfg, d, g, boundary = _small_problem(n_steps_override=4)
    buf = 32
    with tempfile.TemporaryDirectory() as tmp:
        rec = Recorder(
            cfg,
            d,
            g,
            boundary,
            Path(tmp) / "buf.h5",
            offset_cells=3,
            buffer_samples=buf,
        )
    assert rec.max_buffer_bytes <= buf * (
        8  # v_int
        + 8  # v_half
        + 8  # p_avg
        + 8  # rad times_p
        + 8  # rad times_v_half
        + 8  # h tp
        + 8  # h tvn
        + 8  # b t
        + rec._n_huy * 4 * 2  # huygens p + vn float32
        + rec._n_box * 4  # box p
    ) + 4096
    rec.close()


def test_decimation_factor_floor():
    assert decimation_factor(10_000.0, 1e-6) >= 1
    m = decimation_factor(4_000.0, 2e-5)
    expect = max(1, int(math.floor(1.0 / (2.5 * 4_000.0 * 2e-5))))
    assert m == expect


def test_recorded_times_match_leapfrog_after_step(monkeypatch):
    """After step(n), p is at (n+1)Δt and recorded v_half at (n+½)Δt."""
    n_steps = 12
    cfg, d, _g, _b = _small_problem(n_steps_override=n_steps)
    real_derive = derive

    def wrapped(c: Config):
        out = real_derive(c)
        out.n_steps = n_steps
        out.run_time_s = n_steps * out.dt
        return out

    monkeypatch.setattr("fdtd.run.derive", wrapped)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "times.h5"
        run(cfg, out_path=path, progress=False)
        with h5py.File(path, "r") as f:
            dt = float(f["meta"].attrs["dt"])
            t_p = f["radiator/times_p"][()]
            t_v = f["radiator/times_v_half"][()]
            v_half = f["radiator/v_pulse_half"][()]
            v_int = f["radiator/v_pulse_integer"][()]
            assert t_p.shape == (n_steps,)
            for n in range(n_steps):
                assert t_p[n] == pytest.approx((n + 1) * dt, rel=0, abs=1e-15)
                assert t_v[n] == pytest.approx((n + 0.5) * dt, rel=0, abs=1e-15)
                ref_half = velocity_pulse(t_v[n], d.pulse_sigma, d.pulse_t0)
                ref_p = velocity_pulse(t_p[n], d.pulse_sigma, d.pulse_t0)
                assert v_half[n] == pytest.approx(ref_half, rel=0, abs=1e-12)
                assert v_int[n] == pytest.approx(ref_p, rel=0, abs=1e-12)