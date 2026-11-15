"""Tests for validation._common helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import validation._common as vc


@pytest.mark.skipif(sys.platform != "darwin", reason="keep_awake uses caffeinate on macOS")
def test_keep_awake_starts_and_stops_on_darwin():
    import shutil

    if shutil.which("caffeinate") is None:
        pytest.skip("caffeinate not installed")
    with vc.keep_awake("test"):
        pass


def test_keep_awake_noop_off_darwin():
    if sys.platform == "darwin":
        pytest.skip("covered by darwin test")
    with vc.keep_awake("test"):
        pass


def test_run_workdir_temp_removed():
    seen: Path | None = None
    with vc.run_workdir("pytest_temp", keep=False) as wd:
        seen = wd
        (wd / "probe.h5").write_bytes(b"\x00" * 12)
        assert wd.is_dir()
    assert seen is not None
    assert not seen.exists()


def test_run_workdir_temp_removed_on_exception():
    seen: Path | None = None
    with pytest.raises(RuntimeError):
        with vc.run_workdir("pytest_exc", keep=False) as wd:
            seen = wd
            (wd / "x.h5").write_bytes(b"1")
            raise RuntimeError("abort")
    assert seen is not None
    assert not seen.exists()


def test_run_workdir_keep_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_PROJECT_ROOT", tmp_path)
    (tmp_path / "outputs").mkdir()
    with vc.run_workdir("pytest_keep", keep=True) as wd:
        (wd / "saved.h5").write_bytes(b"data")
        kept = wd
    assert kept.is_dir()
    assert (kept / "saved.h5").is_file()