"""Multi-radiator runs (step 8a)."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from boxrad.config import Config
from boxrad.cutkernel import CutOperator, FastRadiator, PatchRadiator
from boxrad.geom.geometry import build_geometry
from boxrad.geomderive import derive_geom
from boxrad.geomop import GeomOpCache, build_geom_op_cache, build_run_operator
from boxrad.geomrec import GeomRecorder
from boxrad.geomrun import prepare_geom_run, step_geom
from boxrad.runner import run

SHOWCASE = Path(__file__).resolve().parents[1] / "configs" / "showcase"
_F64 = np.float64


def _load(name: str) -> Config:
    return Config.from_json((SHOWCASE / name).read_text(encoding="utf-8"))


def _s2_free_coarse(*, n_transits: float = 2.0) -> Config:
    raw = json.loads(_load("s2_bookshelf.json").to_json())
    raw.pop("environment", None)
    raw["numerics"]["h_override"] = 0.01
    raw["numerics"]["n_transits"] = n_transits
    return Config.from_dict(raw)


def _s2_free_woofer_only() -> Config:
    raw = json.loads(_s2_free_coarse().to_json())
    raw["radiators"] = [raw["radiators"][0]]
    return Config.from_dict(raw)


def _prep(cfg: Config, driven: int = 0, cache: GeomOpCache | None = None):
    geom = build_geometry(cfg)
    assert geom is not None
    derived, gg = derive_geom(cfg, geom)
    op, info = build_run_operator(cfg, derived, gg, geom, driven=driven, cache=cache)
    return derived, gg, geom, op, info


def _array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(a, b)


def _operators_equal(op_a: CutOperator, op_b: CutOperator) -> bool:
    for name in (
        "afl_x",
        "afl_y",
        "afl_z",
        "open_x",
        "open_y",
        "open_z",
        "inv_beta",
        "closed_faces_x",
        "closed_faces_y",
        "closed_faces_z",
        "special_cells",
    ):
        if not _array_equal(getattr(op_a, name), getattr(op_b, name)):
            return False

    def _fast_eq(fa: tuple[FastRadiator, ...], fb: tuple[FastRadiator, ...]) -> bool:
        if len(fa) != len(fb):
            return False
        for a, b in zip(fa, fb):
            if a.axis != b.axis:
                return False
            if not _array_equal(a.idx, b.idx) or not _array_equal(a.sw, b.sw):
                return False
        return True

    def _patch_eq(pa: tuple[PatchRadiator, ...], pb: tuple[PatchRadiator, ...]) -> bool:
        if len(pa) != len(pb):
            return False
        for a, b in zip(pa, pb):
            if not _array_equal(a.cells, b.cells) or not _array_equal(a.coef, b.coef):
                return False
        return True

    return _fast_eq(op_a.radiators_fast, op_b.radiators_fast) and _patch_eq(
        op_a.radiators_patch, op_b.radiators_patch
    )


def test_operator_parity_driven0_cache() -> None:
    cfg = _s2_free_coarse()
    geom = build_geometry(cfg)
    assert geom is not None
    derived, gg = derive_geom(cfg, geom)
    op0, _ = build_run_operator(cfg, derived, gg, geom, driven=0, cache=None)
    cache = build_geom_op_cache(cfg, derived, gg, geom)
    op1, _ = build_run_operator(cfg, derived, gg, geom, driven=0, cache=cache)
    assert _operators_equal(op0, op1)


def test_tweeter_operator_driven1() -> None:
    cfg = _s2_free_coarse()
    derived, gg, geom, op, info = _prep(cfg, driven=1)
    assert info["path"] == "fast"
    fr = info["fractions"]
    rd = geom.radiators[1]
    n_hat = rd.pose.R[:, 2].astype(_F64)
    n_hat = n_hat / max(float(np.linalg.norm(n_hat)), 1e-30)
    h = derived.h
    h2 = h * h
    nx, ny, nz = gg.nx, gg.ny, gg.nz
    fast = op.radiators_fast[0]
    a = fast.axis
    s = 1.0 if n_hat[a] > 0.0 else -1.0
    face_indices: set[int] = set()
    for flat, area in zip(fr.radiator_cell[1], fr.radiator_a[1]):
        if float(area) / h2 <= 1e-5:
            continue
        flat_i = int(flat)
        i = flat_i // (ny * nz)
        rem = flat_i % (ny * nz)
        j = rem // nz
        k = rem % nz
        layer = (i, j, k)[a]
        face_layer = int(layer) if s > 0 else int(layer) + 1
        if a == 0:
            fidx = int(np.ravel_multi_index((face_layer, j, k), (nx + 1, ny, nz)))
        elif a == 1:
            fidx = int(np.ravel_multi_index((i, face_layer, k), (nx, ny + 1, nz)))
        else:
            fidx = int(np.ravel_multi_index((i, j, face_layer), (nx, ny, nz + 1)))
        face_indices.add(fidx)
    sig_idx = set(int(i) for i, w in zip(fast.idx, fast.sw) if abs(float(w)) > 1e-5)
    assert sig_idx == face_indices
    s1 = float(np.sum(fr.radiator_a[1]))
    wh2 = float(np.sum(np.abs(fast.sw.astype(_F64))) * h2)
    assert wh2 == pytest.approx(s1, rel=1e-6)  # sw is float32


def _run_steps(cfg: Config, driven: int, n_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    run = prepare_geom_run(copy.deepcopy(cfg), driven=driven)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
        path = Path(tf.name)
    rec = GeomRecorder(
        cfg,
        run.derived,
        run.gg,
        run.geometry,
        run.info,
        path,
        buffer_samples=32,
        driven=driven,
    )
    for n in range(n_max):
        step_geom(run, n)
        rec.sample(run.fields, n)
    rec.close()
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not installed")
    with h5py.File(path, "r") as f:
        p_avg = np.asarray(f["radiator/p_avg"][()], dtype=_F64)
        p_face = np.asarray(f["huygens/p_face"][()], dtype=np.float32)
        vn = np.asarray(f["huygens/vn"][()], dtype=np.float32)
    path.unlink(missing_ok=True)
    return p_avg, p_face, vn


def test_c20i_bit_identity_woofer_only() -> None:
    both = _s2_free_coarse()
    woofer_only = _s2_free_woofer_only()
    n_max = 150
    p0, h0, v0 = _run_steps(both, 0, n_max)
    p1, h1, v1 = _run_steps(woofer_only, 0, n_max)
    assert np.array_equal(p0, p1)
    assert np.array_equal(h0, h1)
    assert np.array_equal(v0, v1)


def test_p_avg_all_and_zrad_row() -> None:
    cfg = _s2_free_coarse()
    n_max = 150
    for driven in (0, 1):
        run = prepare_geom_run(copy.deepcopy(cfg), driven=driven)
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
            path = Path(tf.name)
        rec = GeomRecorder(
            cfg,
            run.derived,
            run.gg,
            run.geometry,
            run.info,
            path,
            buffer_samples=32,
            driven=driven,
        )
        for n in range(n_max):
            step_geom(run, n)
            rec.sample(run.fields, n)
        rec.close()
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not installed")
        with h5py.File(path, "r") as f:
            p_avg = np.asarray(f["radiator/p_avg"][()], dtype=_F64)
            p_all = np.asarray(f["radiator/p_avg_all"][()], dtype=_F64)
            assert np.array_equal(p_all[:, driven], p_avg)
            zrad = np.asarray(f["results/zrad"][()], dtype=np.complex128)
            zrow = np.asarray(f["results/zrad_row"][()], dtype=np.complex128)
            den = max(float(np.max(np.abs(zrad))), 1e-30)
            rel = float(np.max(np.abs(zrow[:, driven] - zrad))) / den
            assert rel <= 1e-12
        path.unlink(missing_ok=True)


def test_runner_output_files() -> None:
    cfg = _s2_free_coarse(n_transits=0.35)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.h5"
        run(copy.deepcopy(cfg), out_path=out, progress=False)
        import h5py

        sib = out.parent / "out.r1_tweeter.h5"
        assert out.is_file() and sib.is_file()
        with h5py.File(out, "r") as f0, h5py.File(sib, "r") as f1:
            run_files = json.loads(f0["meta"].attrs["run_files_json"])
            assert run_files == ["out.h5", "out.r1_tweeter.h5"]
            assert int(f0["meta"].attrs["driven_index"]) == 0
            assert int(f1["meta"].attrs["driven_index"]) == 1
            pos_tw = np.asarray(
                cfg.geometry_spec.radiators[1].pose.position, dtype=_F64
            )
            pos_sib = np.asarray(f1["meta"].attrs["radiator0_position"], dtype=_F64)
            assert np.allclose(pos_sib, pos_tw, atol=0.0, rtol=0.0)


def test_drive_subset_and_unknown() -> None:
    cfg = _s2_free_coarse()
    raw = json.loads(cfg.to_json())
    raw["numerics"]["drive_radiators"] = ["tweeter"]
    c_subset = Config.from_dict(raw)
    assert c_subset.numerics.drive_radiators == ["tweeter"]
    roundtrip = Config.from_json(c_subset.to_json())
    assert roundtrip.numerics.drive_radiators == ["tweeter"]
    from boxrad.geomrun import resolve_drive_indices

    assert resolve_drive_indices(c_subset) == [0, 1]
    bad = copy.deepcopy(c_subset)
    bad.numerics.drive_radiators = ["nope"]
    with pytest.raises(ValueError, match="unknown"):
        resolve_drive_indices(bad)