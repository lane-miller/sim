# Boxed piston FDTD (boxrad)

Config-driven 3-D acoustic FDTD for radiation from a radiator on a rigid box (or free-field for Stage A checks). See `docs/brief.md`.

## Layout

- `boxrad/` — installable solver package (`pip install -e .`)
- `tests/` — pytest unit tests (require installed `boxrad`)
- `validation/` — C1–C10 check scripts (not in the wheel; optional extra `bem` for bempp-cl)

Run a simulation: `boxrad-run configs/default.json` (or `python -m boxrad.run`).