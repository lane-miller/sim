# Boxed piston FDTD

Config-driven 3-D acoustic FDTD for radiation from a radiator on a rigid box (or free-field for Stage A checks). See `docs/brief_v2_fdtd.md`.

## Layout

- `fdtd/` — solver modules (`config.py` first)
- `tests/` — pytest unit tests

Run tests from this directory with `PYTHONPATH=.` (or `pip install -e .` once packaging exists).
