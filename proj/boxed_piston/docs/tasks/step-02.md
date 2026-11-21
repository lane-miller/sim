# boxrad v0.3 — Step 2: frames and config schema

Spec: design v2 §1.1–1.2, §1.5, §6.1, §9 step 2, §9.1. Branch `boxrad-v0.3`. All paths are relative to `sim/proj/boxed_piston` unless noted.

## Rules for this task
- **Read existing code first:** the config loader, `derive()`, `run()`, and the existing tests and configs. Follow the current style, naming and error types.
- **Do not change:** solver, kernel, recorder, KH/post, `Result`, or any existing config field's meaning or default. **Legacy box configs must behave exactly as before.**
- **New third-party imports:** none. numpy and the stdlib only.
- **Allowed commands:** exactly one, run from `sim/` with the `simenv` Python 3.11:
  `python -m pytest proj/boxed_piston/tests/test_frames.py proj/boxed_piston/tests/test_geomconfig.py -q -x --tb=short`
  - First check that `python --version` is 3.11.x. If it isn't, stop and report.
  - At most **3** run/fix cycles. If the tests are still red after 3, stop and report.
  - Do not run the full suite, the showcase, any other script, pip, or git.
- **End with a report of ≤ 10 lines:**
  - files created or changed;
  - the final pytest summary line;
  - the number of cycles used;
  - assumptions made;
  - anything left open.

## 1. `boxrad/frames.py` (new)
- `orthonormal_basis(normal, x_axis) -> R` (3×3 float64, **columns** = local x̂, ŷ, ẑ in world coordinates). This must match acs-lpm `_orthonormal_basis` exactly:
  - ẑ = n / |n|
  - x̂ = (a − (a·ẑ) ẑ) / |a − (a·ẑ) ẑ|, where a = x_axis
  - ŷ = ẑ × x̂
- Raise `ValueError` if |n| < 1e-12, if a is non-finite, or if |a − (a·ẑ)ẑ| < 1e-9 |a| (x_axis parallel to normal).
- `default_x_axis(normal)`: return world (1,0,0). If |n̂·(1,0,0)| > 0.9, return world (0,1,0) instead.
- `@dataclass(frozen=True) class Pose`:
  - Fields: `position` (3,), `normal` (3,), `x_axis` (3,), all float64.
  - `Pose.make(position, normal=(0,0,1), x_axis=None)`, where x_axis=None → `default_x_axis(normal)`. Validate that all values are finite.
  - Properties:
    - `R`;
    - `T`: 4×4 homogeneous local → world, [[R, t], [0, 1]].
  - Methods:
    - `to_local(pts)`: Rᵀ(p − t);
    - `to_world(pts)`: R p + t.
    - Both accept (3,) or (N, 3) and return the same shape.
  - `to_dict()` / `from_dict()`, with JSON-friendly lists.
  - `Pose.identity()`.

## 2. `boxrad/geomconfig.py` (new): parse and validate only, nothing is built yet
Frozen dataclasses, plus `parse_geometry(cfg_dict) -> GeometrySpec`.

### Dataclasses
- `PlaneSpec(axis: "x"|"y"|"z", side: "-"|"+", offset: float)`.
- `EnvironmentSpec(planes: tuple[PlaneSpec, ...])`. Error if two planes share an axis.
- `PrimitiveSpec(kind: "box"|"cylinder"|"sphere", params: dict, pose: Pose)`.
  - Config entry format: `{"<kind>": {params}, "position": [...], "normal": [...]?, "x_axis": [...]?}`.
  - `box {size:[sx,sy,sz]}`, `cylinder {d, length}` (axis along the local z), `sphere {d}`.
  - All dimensions must be finite and > 0.
- `StepPartSpec(file: str, enclosure_part: str = "enclosure", placement: Pose = identity)`.
  - Config entry format: `{"step": {"file", "enclosure_part"?, "placement"?: {position, normal, x_axis}}}`.
- `ShapeSpec(kind: "rect"|"circle"|"racetrack"|"ellipse", params: dict, hole: ShapeSpec | None)`.
  - `rect {w, h}`;
  - `circle {d}`;
  - `racetrack {w, h}`: overall length w along local x, width h, semicircular ends, w ≥ h;
  - `ellipse {a, b}`: **semi-axes** along local x and y.
  - `hole` uses the same format, and nested holes are an error. Whether the hole fits inside the shape is checked in step 3, not here.
  - All dimensions finite and > 0.
- `RadiatorSpec(name, shape: ShapeSpec | None, part: str | None, axis: tuple | None, pose: Pose | None, motion: "axial")`.
  - Parametric/box entry: exactly one shape key, plus `position`, `normal`, `x_axis?`.
  - STEP entry: `part`, and optional `axis`.
  - `motion` defaults to `"axial"`, and any other value is a `ValueError`. (`uniform_normal` is a hidden numerics test option, added later.)
  - Radiator names must be unique.
- `GeometrySpec(tag, parts, radiators, environment)`.

### Parsing rules
- **`geometry` tag:** absent → `"box"`. Allowed: `"box"`, `"parametric"`, `"step"`. Anything else, including a dict, is a `ValueError` that names the allowed values.
- **`"box"`:**
  - `parts = ()`.
  - radiators = [legacy radiator as radiator 0, with `name = "radiator"`, the legacy shape mapped to `ShapeSpec`, and `pose = Pose.identity()`], followed by any entries from an optional `radiators` list.
  - Read the existing config to map the legacy radiator shape types (circle/rect/racetrack, …). If a legacy type can't be mapped, keep `shape = None` and note it in the report. **Do not change the legacy radiator section.**
- **`"parametric"`:**
  - `parts` is a non-empty list of primitive entries only. A `step` entry here is an error ("mixing STEP and parametric parts is not supported in v0.3").
  - `radiators` is non-empty, and every entry needs a shape and a pose.
- **`"step"`:**
  - `parts` is exactly one `step` entry.
  - `radiators` is non-empty, and every entry needs `part`.
- **`environment`:** optional for all tags. Absent → no planes.

## 3. Wire-in (minimal)
- The config object gets one new attribute, `geometry_spec` (a `GeometrySpec`), built by `parse_geometry` during the existing load/validate path. **The existing `config.geometry` string stays exactly as is.**
- **Guard in `run()`/`derive()`:** raise `NotImplementedError` with a clear message if any of these hold:
  - tag ≠ `"box"` (message: "parametric/STEP geometry lands in v0.3 step 3+");
  - box mode with extra radiators ("multi-radiator lands in v0.3 step 8");
  - any environment planes ("environment planes land in v0.3 step 7").
- Legacy configs must reach the existing code paths unchanged.

## 4. Fixtures: `configs/showcase/` (new JSON files, reused by later steps)
Units are meters. The world frame is z up, with the floor at z = 0 where present.

- **`s1_cylinder.json`** (`"parametric"`):
  - Cylinder d = 0.20, length = 0.30, position (0, 0, −0.15), normal +z, so its top face is at z = 0.
  - Radiator `driver`: circle d = 0.13, position (0, 0, 0), normal +z.
  - No environment.
- **`s2_bookshelf.json`** (`"parametric"`):
  - Pedestal box, size [0.18, 0.18, 0.15], at position (0, 0, 0.075).
  - Speaker box, size [0.20, 0.25, 0.30] (x width, y depth, z height), at position (0, 0, 0.30).
  - The front face is y = +0.125 with normal +y.
  - Radiators:
    - `woofer`: circle d = 0.13, position (0, 0.125, 0.25), normal +y;
    - `tweeter`: circle d = 0.025, position (0, 0.125, 0.40), normal +y.
  - Floor plane: axis z, side "-", offset 0.
- Copy all the other required config sections (air, band, numerics, …) from `configs/c09_box.json`.

## 5. Tests
### `tests/test_frames.py`
- 1000 random poses (seeded):
  - RᵀR = I to ≤ 1e-12, and det R = +1;
  - ẑ ∥ normal;
  - x̂·a > 0, and x̂ ⊥ ẑ;
  - `to_world(to_local(p))` round trip ≤ 1e-12 (absolute, for |p| ≤ 10).
- Known cases:
  - normal +y with x_axis +x → R = [[1,0,0],[0,0,−1],[0,1,0]] read as columns x̂ = (1,0,0), ŷ = (0,0,−1), ẑ = (0,1,0). Assert the columns.
  - Identity.
- Errors: zero normal, parallel x_axis, NaN.
- `default_x_axis`: the threshold cases at n = (0.95, 0.31, 0) normalized, and at (0, 0, 1).
- `T` agrees with `to_world`. `to_dict`/`from_dict` round trip.

### `tests/test_geomconfig.py`
- **Every existing JSON in `configs/` (not `configs/showcase/`):**
  - tag is `"box"`, there is 1 radiator, and its pose is the identity;
  - `config.geometry` is still `"box"`;
  - a dry run into `tmp_path` still works.
- **`s1_cylinder.json` and `s2_bookshelf.json`:** they parse to the expected parts, radiators, poses and plane. `run(..., dry_run=True)` raises `NotImplementedError`.
- **Errors, one test each:**
  - two planes on the z axis;
  - axis "w";
  - racetrack with w < h;
  - negative cylinder d;
  - two shape keys in one radiator;
  - nested hole;
  - duplicate radiator names;
  - a STEP entry inside a parametric config;
  - a STEP config with two parts;
  - an unknown geometry tag;
  - `geometry` given as a dict;
  - motion `"uniform_normal"`.
- **Box config plus `environment` → `NotImplementedError` on dry run.**

## 6. Showcase: `showcase/step02_frames.py` (write it; do not run it)
- Load the S1 and S2 fixtures with the normal config loader. Print a table per radiator:
  - name, position, normal, x_axis;
  - R (3×3), det R, and ‖RᵀR − I‖_max;
  - the round-trip error of the radiator center;
  - local (0, 0, 1) mapped to world, expected position + n̂, and the error;
  - **PASS** if every error is ≤ 1e-12, otherwise **CHECK**.
- Also print, for S2, the woofer-to-tweeter center distance (expected 0.15 m), and that both radiators lie on the plane y = 0.125 (expected 0).
- **Plot** with a matplotlib 3-D axis, equal aspect:
  - primitives as wireframes (box edges; a cylinder as two circles plus 4 generators);
  - radiators as circle outlines in their planes, using their poses;
  - a pose triad at each radiator (x red, y green, z blue, length 0.05 m) and a world triad at the origin;
  - the floor as a translucent 0.6 × 0.6 m square at z = 0.
  - Two subplots: S1 and S2.
- Save to `showcase/out/step02_frames.png`, overwritten in place. `--show` opens the window; the default is no window.
- The script prints its start, the PASS/CHECK summary and the output path. No `caffeinate` is needed; it runs in under 5 s.
- Add `proj/boxed_piston/showcase/out/` to `sim/.gitignore` on its own line, keeping the file's trailing newline.
