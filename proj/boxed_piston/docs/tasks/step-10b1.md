# Step 10b1: validation point source (`numerics.test_source`), free-field gate

Step 10b is split in two:
- **10b1 (this file):** the hidden test source and its free-field gate.
- **10b2:** C13a (rigid sphere, series reference) and C16 (convergence order).

## Goal
- Add a hidden, validation-only monopole volume-velocity source in the medium for parametric configs (spec v2 §6.1 `numerics.test_source`, gate C13a).
- It is not public API. Do not document it in the user guide.

## Hard rules
- Additive only. With no `test_source` key, every existing path is bit-identical (box, baffle, parametric, planes, multi-radiator). The full suite must stay green.
- No new dependencies.
- No changes to the radiator pickup, the CPML, the floor, or the KH/image code.
- One showcase, one JSON, one PNG, all overwritten in place.

## 1. Config
`numerics.test_source = {"position": [x, y, z]}`, in the world frame, meters.

Parsing and validation (`geomconfig.py` / `config.py`):
- **Geometry tag.** Allowed only with `geometry == "parametric"`. For box or baffle, raise `ValueError`.
- **Parts and radiators.** When `test_source` is set, `parts` may be empty and `radiators` may be empty. When it is not set, the existing rules apply unchanged.
- **Radiators are not driven.** Any radiators that are present are held rigid (v = 0). There is exactly one run, and `drive_radiators` is ignored.
- **`to_json`** round-trips the key.

## 2. Grid and derive (`geomderive.py`)
- **Grid anchor.** Planes keep the top priority. Where the existing rules would anchor to radiator 0 and there are no radiators, anchor instead so that `position` is a **cell center** (the same offset rule, with `position` in place of the radiator-0 center).
- **No radiators.** Skip the radiator clearance terms.
- **Extents.** The domain bbox is the solid bbox ∪ {position}, or {position} alone if there are no parts. Add the existing Huygens + CPML margins on absorbing sides.
- **Snapping.** Record the snapped source cell-center coordinates in `derived` as `test_source_cell_center` (3 floats), plus its integer index.

## 3. Fractions with an empty solid
If the solid has zero triangles, return all-fluid fractions (α = 1, β = 1, no patches) without calling the clipping engine.

## 4. Source term (stepping kernel used by `prepare_geom_run`)
The source cell must be full fluid: β = 1 and all six α = 1. Otherwise raise `ValueError` naming the cell and its β. Check this before stepping.

In the pressure update, after the divergence term, add to the source cell only:
$$p_c^{n+1} \mathrel{+}= \frac{\rho c^2\,\Delta t}{h^3}\,Q^{n+\frac12},\qquad Q(t) = (1\ \mathrm{m^2})\cdot v_{\text{drive}}(t).$$

- v_drive is the existing radiator drive waveform. Sample it at **exactly the same time sample** that the radiator flux uses in the same pressure update.
- The recorder's existing division by the drive spectrum then yields transfers per unit volume velocity (Pa per m³/s).
- Implement it as a separate small step (e.g. one scalar add per step on the flat index), not inside the numba bulk kernel. Leave the bulk kernel untouched.

## 5. Recorder, files, Result
- **Recorded:** probes, the Huygens surface, and KH as usual.
- **With no radiators:** skip `radiator/*`, `results/zrad`, `zrad_row` and `p_avg_all`. With rigid radiators present, record them as usual; they are passive pickups.
- **Meta attrs:**
  - `test_source_json` (position, cell center, index);
  - `drive_kind = "test_source"`;
  - `transfer_units = "Pa/(m^3/s)"`.
- **Result:**
  - `frame_default = "world"` when there are no radiators.
  - `.field` / `.directivity` / `.directivity_index` work unchanged.
  - `.z`, `.Z_acoustic`, `.Z_mechanical` raise `ValueError("test-source run: no radiator impedance")` when there are no radiators.
  - Add a read-only property `test_source` (dict or None).

## 6. Analytic reference (`boxrad/validation/analytic.py`, new)
`monopole_free(points, f, source, rho, c) -> (F, P) complex`, using e^{+jωt}:
$$\frac{p}{Q} = \frac{j\omega\rho\,e^{-jkR}}{4\pi R}.$$

## 7. Tests (`tests/test_test_source.py`)
Fast tests:
1. **Config rejections** (`ValueError`):
   - box + test_source;
   - baffle + test_source;
   - parametric, empty parts and radiators, without test_source;
   - a source inside a solid or in a cut cell (`prepare_geom_run`).
2. **Null equals absent:** on `c19_sym_full`, `numerics.test_source: null` gives `prepare_geom_run` operator arrays (α, β̃, radiator weights, grid origin and shape) identical to the key being absent. Regression against existing behavior is covered by the existing C12/C19/C20 bit-identity tests.
3. **Empty geometry:** fractions are all ones; the anchor puts `position` exactly at a cell center (≤ 1e-12).
4. **Injection unit:** on a tiny all-fluid grid (e.g. 8³ interior), one pressure update from zero state with Q = 1e-6 gives p = ρc²Δt·1e-6/h³ in the source cell (relative ≤ 1e-6, float32) and 0 elsewhere.
5. **`monopole_free`** matches a hand value at one point (≤ 1e-12).

Slow test (`@pytest.mark.slow`):

6. **Coarse free-field run** (f_max 1 kHz, source at the origin, probe at (0.1, 0, 0)): within 0.3 dB / 2° of `monopole_free` at 300, 500 and 800 Hz.

## 8. Showcase (`showcase/step10b1_point_source.py`)
- **Config `configs/c13_free_ps.json`:**
  - Copy the numerics, air and band of `configs/c19_sym_full.json` (f_max 3 kHz, production h = 5 mm).
  - `geometry = "parametric"`, `parts = []`, `radiators = []`, `test_source.position = [0, 0, 0]`.
  - Probes: (0.05, 0, 0), (0.1, 0, 0), (0, 0.05, 0), (0.06, 0.06, 0.06). These must be cell centers: verify against `derived.test_source_cell_center` and the grid; adjust the coordinates by whole cells if needed.
- **Run it** (caffeinate, progress prints).
- **Compare** at 300, 500, 800, 1000, 1500, 2000, 2500 and 3000 Hz:
  - probes vs `monopole_free`;
  - KH field at r = 2 m in the directions (1,0,0), (1,1,0)/√2, (1,1,1)/√3 and (0,0,−1) vs `monopole_free`;
  - DI vs 0 dB.
- **Print** per f: dB and degree errors per probe and direction, R/λ, and the DI.
- **Gates:**
  - probes ≤ 0.15 dB / 1°;
  - far field ≤ 0.15 dB / 1.5°;
  - |DI| ≤ 0.05 dB.
- **Outputs:** `outputs/step10b1/c13_free.json` and one PNG with 3 panels (probe dB, probe phase, far-field dB vs f).

**Reference numbers** (Claude's numpy Yee prototype, same h, λ = 0.99/√3, rigid box with time gating; worst over 0.3–3 kHz). Expect production to land in the same range. If it is off by more than 2× in dB, report it rather than tuning anything.

| Probe | Distance | dB | deg |
|---|---|---|---|
| axis | 0.1 m | −0.004 | −0.48 |
| face | 0.05 m | +0.022 | +0.17 |
| diagonal | 0.104 m | +0.098 | −0.13 |
| axis (report only) | 0.2 m | −0.04 | −1.1 |

The 0.2 m axis case shows grid dispersion of about 0.6° per wavelength at 23 ppw.

## Report back
- Files changed, with line counts.
- Any deviation, and why.
- Where the source term sits in the update.
