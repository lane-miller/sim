# Boxed piston radiation via FDTD

Location: `sim/proj/boxed_piston/`, in the conda environment `simenv` (Numba and bempp-cl).

## 1. Goal
A config-driven solver that computes, for **one physical system per run**:
1. **Z_rad(f)**, the radiation impedance (required);
2. **far-field directivity**, complex (required);
3. **surface pressure on the box**, stored so that near-field and surface probes can be computed later.

**System:** a rigid radiator flush on one face of a rigid rectangular box, with uniform normal velocity on the radiator and v_n = 0 everywhere else. Air is linear and lossless. The radiator is circular by default; the mask also supports rectangles and racetracks. **Radiator geometry (config `radiator`):** `circle` — `dims[0]` is the **diameter**; `rect` — `dims` is **(width, height)**; `racetrack` — `dims` is **(width, height)** with semicircular ends of diameter **min(width, height)** along the longer axis. **`(x_c, y_c)`** is the offset of the radiator center from the **front-face center**. The **front face** is **+z**, size **W×H**; the box interior extends in **−z** behind it (depth **D**).

**Purpose:** coupling to acs-lpm, as the radiation load and directivity for lumped-parameter models.

**Conventions:**
- Time convention e^{+jωt}. NumPy's forward FFT, P(ω) = Σ p(t)·e^{−jωt}, is already consistent with it, so no conjugation is needed for FDTD data. Conjugate only data imported from e^{−iωt} codes (e.g. bempp in C9).
- z = ∫_radiator p dS / (ρc·S·v), the dimensionless impedance (acoustic Z_A normalized by ρc/S).
- The origin is at the radiator center, with +z the outward face normal.
- SI units throughout. Air defaults: c = 343 m/s, ρ = 1.204 kg/m³. Both are configurable.

**Range (defaults, all configurable):** f from f_min (default 20 Hz) to f_max (default min(20 kHz, the frequency at which ka = 10)). Above ka ≈ 10 a real cone has broken up, so a rigid piston is no longer physical. Intended box dimensions are 10–600 mm, with aspect ratio ≤ 4. Values outside these ranges are allowed, with a pre-flight warning (§3).

## 2. Method
**Derived, not fixed.** Every number below (h, Δt, points per wavelength, cells across the radiator, gap, CPML thickness and α, pulse width, run length, decimation, cell budget) is an illustrative default. The solver computes each one per system from the config (f_min, f_max, box and radiator geometry, tolerances) using the rules in this section. Defaults live in one place, can be overridden in the config, and are written to the output metadata. Check results (§5) update the defaults and the pre-flight thresholds.

- **Scheme:** linear acoustic FDTD on a staggered (Yee) grid, with p at cell centers and v on cell faces. 2nd order in space and time.
- **Box:** aligned with the grid. It is rigid through v_n = 0 on its faces, so edges and corners are exact. Realized dimensions are rounded to h and reported.
- **Grid spacing:** default **ppw = 20**, so h = min(λ_min/20, D/10), where D is the radiator diameter (the smallest dimension for non-circular masks). C8-convergence justifies ppw 20 (dispersion order ≈ 2; ≈1% z error at f_max; directivity within 0.1 dB of the extrapolated result at kL = 10). This resolves both the waves and the radiator.
- **Time step:** Δt = h/(c√3), the CFL limit. Example: at 20 kHz, h ≈ 0.858 mm and Δt ≈ 1.44 µs. f_max is set by h, not by Δt.
- **Radiator:** a prescribed v_n(t) on face cells, weighted by the area fraction each cell has inside the radiator outline, so the volume velocity is exact (staggered flux ``h² v_z``; weight on the drive, ``Q = h² Σ v_z``, ``S_grid = h² Σ w``). The same weights are used to average pressure (``⟨p⟩ = Σ w p / Σ w``). The radiator face also drives the closed box interior (rear radiation into a sealed rigid cavity); it can't affect the exterior because v_n is prescribed (tested).
- **Cell budget:** if the domain at the chosen h exceeds `numerics.cell_budget` (default 6×10⁷ cells), validation fails; this happens with tiny radiators on comparatively large boxes, such as mic ports. There is no coarser fallback and no analytic Z_rad substitute: the error message says to raise h (lower `cells_across_radiator` or `ppw`, or set `h_override`), lower f_max, or raise the budget if RAM allows. z always comes from the FDTD (`zrad_source = "fdtd"`).
- **Config validation:** reject radiators closer than 2–3 cells to any box edge. Tangent radiators are not physical and can't be resolved. In baffle mode the radiator mask is the radiator bounding box (rounded to cells) shifted by −(x_c, y_c), so an offset clips the outline by max(|x_c| + w/2 − W_bbox/2, |y_c| + h_r/2 − H_bbox/2): a warning up to one cell, an error above.
- **Excitation:** one band-limited Gaussian velocity pulse per run, with no derivative so there is energy down to DC. Its width is chosen so the spectrum falls to about −40 dB near 1.2·f_max, which keeps out content the grid can't represent. Transfer functions are output spectrum divided by input spectrum.
- **Run length:** until the response decays below a threshold that accounts for low frequency, typically a few box-transit times plus the pulse length. Example: a 300 mm box needs about 5–10 ms, or 3,000–6,000 steps. Low frequencies come out of the same run, because a box in free space doesn't ring.
- **Domain:** the box, plus an air gap default 10 cells (C3 passes at 5; margin until box tests C9), plus the CPML. The far field comes from Kirchhoff–Helmholtz over the box faces, so the grid doesn't need to reach the far field.
- **Absorber:** CPML 16 cells, R=1e-6, m=3, κ_max=1, α=π·f_min (C3-cpml: −100 dB normal / −100 dB oblique; low-f Z gap-converged ≤0.5% from 5 cells). Large α (≥100·π·f_min) degrades low-f Re Z; keep α small.
- **Precision:** fields in float32; spectral and DFT accumulation in float64. Constants are cast explicitly so nothing is silently promoted.
- **Time staggering:** p is sampled at integer steps and v at half steps. Correct the v spectra by e^{jωΔt/2}, with the sign set by the convention, before forming z or Kirchhoff–Helmholtz terms.
- **Face pressure:**
  - Rigid faces: p_face = p_c, which is O(h²) because ∂p/∂n = 0.
  - Radiator cells: p_face = p_c + (h/2)ρ∂v_n/∂t for outward normal n.
  - Edge-adjacent cells are singular (p ~ r^(−1/3)); their convergence is checked separately.
  - Far field / KH uses a **Huygens surface** (C4-kh), not the box faces; box-face p is for surface probes.

## 3. Software design
- **Input:** a JSON config with **`geometry`**: `box` (default) or `baffle`. Box mode requires box dimensions; baffle mode uses a half-space rigid plane at z-min (no box, no z-min CPML) with lateral domain sized to the radiator bounding box. Also: radiator shape, size and position, f_min, f_max, tolerances, optional overrides, far-field angle grid, output path. Examples: `configs/default.json` (box, the C9 geometry with h from the default rules), `configs/smoke.json`, `configs/c06_baffle.json`. Unknown top-level keys (e.g. `"_doc"`) are ignored; unknown keys inside a section are errors.
- **Pre-flight check:** before any computation, `runner.py` validates the config and prints warnings; any error stops the run (and the dry run) with `error: …`. `--dry-run` then prints a summary of the derived parameters and the estimates (cells, RAM, disk, run time).
  - **Errors** (the run stops): a radiator tangent to an edge or within 2–3 cells of one; geometry that doesn't fit the box; the domain exceeds the cell budget; a baffle-mode offset clips the radiator by more than one cell; the RAM estimate exceeds available memory.
  - **Warnings** (the run continues):
    - ka > 10 at f_max (beyond the rigid-piston regime);
    - a baffle-mode offset (or bounding-box rounding) clips the radiator by up to one cell;
    - points per wavelength below 20 at f_max (dispersion);
    - box dimensions or aspect ratio outside the intended range;
    - the CPML thinner than its default, or the gap smaller than its default;
    - long estimated run time;
    - large output file.
- **Core:** the update kernels use Numba `@njit(parallel=True)` and run multithreaded on the M1. CPML updates go in separate slab kernels, with ψ arrays allocated only inside the slabs. Arrays are laid out as a struct of arrays.
- **Output (store raw, reduce later):** one HDF5 file per run, written incrementally in chunks so it never sits fully in RAM. It contains:
  - the config and all derived parameters;
  - the input pulse v(t);
  - the area-weighted, radiator-averaged p(t) at the full step rate;
  - the surface p(t) on all box faces at full spatial resolution, decimated in time only (to about 2.5·f_max, which is lossless within the band); roughly 0.1–2 GB per run;
  - **p** and **v_n** on a Huygens surface offset by **N** cells (default 3, inside the air gap) at the decimated time rate; Huygens face **p** by 4-point cubic interpolation of cell-center **p** along the normal (C4-kh: error ≈ 3(kh)⁴/128); Huygens offset ≥ 2 cells from box and CPML (default 3);
  - run metadata (h, Δt, number of steps, decimation, run time, peak RAM, `zrad_source`, always `fdtd`).

  The volume field is not stored. Optional mid-plane snapshots can be saved for debugging. Compression (spatial downsampling, decomposition) comes after the solver is verified.
- **Post-processing:** a separate module computes:
  - z(f), from the radiator-averaged p(t) and v(t);
  - the far field and near-field probes **outside** the Huygens surface via Kirchhoff–Helmholtz on the stored Huygens **p** and **v_n**; box-surface p is used for on-surface probes only.

  The far-field output (`Result.directivity`) is the complex pattern D(θ, φ, f) = lim r·e^{+jkr}·p/v [Pa·m/(m/s)], phase referenced to the origin (radiator center), at any requested directions. The directivity index integrates |D|² over the full sphere in box mode and over the half space z ≥ 0 in baffle mode. Every `Result` method that takes f raises outside [f_min, f_max] unless `allow_out_of_band=True` (then it warns).
- **Storage:** run outputs go to a configurable path, such as the external SSD, and are excluded from git.
- **Package `boxrad/`** (install via `pyproject.toml`; `pip install -e .` in `proj/boxed_piston`). Console: `boxrad-run CONFIG [-o OUT] [--dry-run] [--no-show] [--plot] [--quiet]`; geometry: `boxrad-plot CONFIG_OR_H5 [-o PNG] [--no-show] [--huygens]`; `boxrad-info H5` prints the summary, `boxrad_version`, `zrad_source`, the `/results/zrad` range and point count, the conventions, accuracy metadata, and datasets.
- **Module layout** (one Cursor prompt per module under `boxrad/`, each with a small unit test):

  | Module | Contents |
  |---|---|
  | `config.py` | Schema, defaults, derived-parameter rules, validation |
  | `grid.py` | Domain, box mask, radiator area weights |
  | `kernels.py` | Numba p and v updates |
  | `cpml.py` | CPML profiles and slab updates |
  | `source.py` | Pulse design and radiator injection |
  | `recorder.py` | Chunked HDF5 output |
  | `runner.py` | Entry point: config → run → HDF5 (`python -m boxrad.runner` or `boxrad-run`) |
  | `post/` | z(f), Kirchhoff–Helmholtz, probes |
  | `validation/` | Check scripts (`c01_perf.py` … `c11_bandedge.py`; not part of the wheel) |

## 4. Local compute envelope
Measured in C1-perf at ~375 M cell-updates/s, ~3.4 box transits, without surface recording. Thread scaling is ~3.2× on 8 threads (memory-bandwidth bound).

| Box | f_max at 20 points/λ | Cells | Est. RAM | Projected run time (C1-perf, Numba, M1) |
|---|---|---|---|---|
| 200 mm | 20 kHz | 1.9×10⁷ | 0.37 GB | ~70 s |
| 300 mm | 17 kHz | 3.7×10⁷ | 0.68 GB | ~3 min |
| 300 mm | 20 kHz | 5.8×10⁷ | 1.04 GB | ~5 min |
| ~500 mm | 20 kHz | ~2.6×10⁸ | ~5 GB (RAM limit) | ~25 min (extrapolated) |

## 5. Checks (in run order, gated)
**Principle:** build only the code each check needs, and run the checks most likely to kill the approach first. If a check fails its gate, stop and replan before writing more code.

**Stage A: core solver only** (lean `config.py`, `grid.py`, `kernels.py`, `cpml.py`, plus a point source; no box, radiator, recorder or post-processing).

| Check | What | Reference | Gate (fail → action) |
|---|---|---|---|
| C1-perf | Timing and memory vs grid size on the M1; Numba compile time, thread scaling, dtype | — | Local envelope too small → bring the GPU backend forward |
| C2-dispersion | Free-field point source, on axis and diagonal; phase error vs points per wavelength; long-run stability | Analytic | Phase error unacceptable at affordable points per wavelength → change the scheme before going further |
| C3-cpml | Absorber on a small case vs a very large reference domain. Vary σ_max, α, thickness and gap; include grazing incidence. | Large domain | Can't reach about −80 dB at f_min → rethink low-frequency handling |

**Stage A results:** C1-perf **PASS** (2026-11-13, ~390 M cell-updates/s peak). C2-dispersion **PASS** (2026-11-13, stability ratio ≪10⁻³; ≈1°/λ axis lag at ppw 20). C3-cpml **PASS** (2026-11-14, −100 dB normal and oblique reflection).

**Stage B: post-processing, standalone** (no FDTD needed).

| Check | What | Reference | Gate |
|---|---|---|---|
| C4-kh | Kirchhoff–Helmholtz over a closed box surface, applied to an analytic point-source field with face-extrapolated p | Analytic | Quadrature, normal or sign errors → fix before Stage C |

**Stage B results:** C4-kh **PASS** (2026-11-14): exact-surface extinction ≤0.1% at 1.5–10× box size; ~2nd-order convergence in h; half-cell offset on **box** faces (p at h/2 outside) gives 3× probe rel err ≈2% / 4% / 14% at 1 / 4 / 10 kHz; **Huygens** (2 cells out, face **v_n** exact): 2-point face **p** (order 2) ≈0.3% / 0.7% / **1.5%** at 1 / 4 / 10 kHz (3× probe, ∝(kh)²); 4-point cubic face **p** (order 4) ≈0.1% / 0.2% / **≤0.2%** at 1 / 4 / 10 kHz on 3× & 10× probes and far field.

**Stage C: box and radiator** (`grid.py` box mask and area weights, `source.py`, `recorder.py`, `runner.py`).

| Check | What | Reference | Gate |
|---|---|---|---|
| C5-lowfreq | 0.2 m cube, φ0.1 m centered radiator, f=20–1500 Hz; fit Re z = A k²(1+B k²), Im z = k δ(1+C k²) on kL≤0.2 (report kL≤1); gap×2 truncation run | Monopole: Re z → (ka)²/4; δ/a ≈ 0.61–0.85 | **G1** \|A/(S/4π)−1\|≤3%; **G2** Im z>0 on gate band and 0.55≤δ/a≤0.90; **G3** max\|z_gap2−z\|/\|z\|≤2% on gate band. Fail → debug face pressure, half-step / sampling, CPML |
| C6-baffle | **Baffle** geometry (`configs/c06_baffle.json`): φ0.1 m centered circle, f=100–6000 Hz (~35 cells across, ka_max≈5.5); half-space rigid plane at z-min (no z-min CPML), five-face Huygens; KH with `image_plane_z=0` and mirrored image normals; spectra via `post.spectra.spectrum` (Δt-weighted sum at stored times); optional on-axis probes (`output.probes_baffle_z_cells`) | Baffled piston: `z_ref` via `post.reference.z_baffled_circle` | **G1** low ka: fit Re z = A k²(1+B k²) for ka≤0.3; \|A/(S/2π)−1\|≤2%. **G2** max \|z−z_ref\|≤0.03 on band. **G3** transfer \(H=P/V\): `kh_pressure` on axis at r=100·a vs \(H_{\mathrm{exact}}(r)=\rho c[e^{-jkr}-e^{-jk\sqrt{r^2+a^2}}]\) at ka=1,3,5 (±0.3 dB, 5°); \|H(θ)/H(0)\| vs \(2J_1(ka\sinθ)/(ka\sinθ)\) with ref level ≥−20 dB: max \|ΔdB\|≤0.5 (included in G3 summary). Report `kh_far_field` vs `kh_pressure`. **G3-probe** (report): \(H_{\mathrm{KH}}, H_{\mathrm{exact}}\) vs \(H_{\mathrm{FDTD}}\) at probe heights. Tail ≤−80 dB |
| C7-subcell | **Baffle** mode (`configs/c07_*.json`): sub-cell radiator offset and mask resolution. **G1** cell-aligned φ0.1 m circle vs same grid with center offset `(0.25h, 0.25h)` and `(0.5h, 0.37h)`; `max |z_shift−z_0|≤0.01` on the band (default f 100–6000 Hz). **G2** rectangle 0.10×0.05 m vs `post.reference.z_baffled_rect`; `max |z−z_ref|≤0.03` on band; report largest `ka` (`a=√(S/π)`) with error ≤0.01. **G3** circle d=0.1 m, `cells_across∈{6,10,16}`, `f_max` so `ka_max≈1.5`; `max |z−z_ref|≤0.02` at 10 cells (analytic circle ref); report all three and observed convergence order. **Report** racetrack 0.10×0.05 m at `{10,16,24}` cells: `max |z_10−z_24|`, `|z_16−z_24|`. PNGs: z vs ref (G1/G2), error vs `ka` (G3). RAM check before each run. | `boxrad/post/reference.py` | Gates G1–G3 as stated; area weighting / sub-cell error too large → fix `grid` weights |
| C8-convergence | **G1** baffle dispersion (`configs/c08_g1_baffle.json`): φ0.1 m circle, `cells_across=6` so **λ/ppw** sets *h*; `f_max` 6000 Hz; ppw ∈ {10, 15, 20, 30} (verify distinct *h* per run); common *ka* band vs `z_baffled_circle`; report max \|z−z_ref\| per ppw and observed order. **Gate:** ppw=20 max \|z−z_ref\|≤0.015 (≈1% z error at f_max, falling as (f/f_max)²). **G2** box self-convergence (`configs/c08_g2_box.json`): 0.2³ m cube, φ0.1 m centered, f 50–3000 Hz; `numerics.h_override = L/N` with *N* ∈ {24, 36, 54} (refinement ratio 1.5, nominal *L*=*W*); assert realized *W*, *H*, *D* and radiator center match nominal; print realized ppw at *f_max*. Richardson *p* = ln(‖z_c−z_m‖/‖z_m−z_f‖)/ln 1.5 on the common design band (same for reported directivity *p* on gated angles); fallback *p*=2 if *p*∉[1,3]. **Gates:** max \|z_36−z_∞\|≤0.01 on band; closed-surface KH directivity at *r*=10·L in **xz** and **yz** at kL=2, 5, 10 — max \|ΔdB\| gate level vs extrapolated ≤0.5 dB (pattern ≥−20 dB re on-axis). **G3** (*N*=36 run): probes midway Huygens–CPML at (0,0,+g), (W/2+g,0,−D/2), (0,0,−D−g); KH (`kh_pressure_loop`, full **G**) vs FDTD at G2 kL values; *H*=P/V via `post.spectra.spectrum`. **Gate:** ≤0.2 dB, ≤2°. Report recommended default ppw (smallest in sweep meeting G1+G2); do not change config default. PNGs: \|z−z_ref\| vs *ka* per ppw; directivity overlays; probe table in stdout. RAM check and wall/RAM plan before runs. | Self-consistency | G1–G3 as stated; sets ppw recommendation (brief §8) |

**Stage D: independent confirmation.**

| Check | What | Reference | Gate |
|---|---|---|---|
| C9-bempp | **Box** with edge-offset rectangular radiator (`configs/c09_box.json`: W×H×D = 0.20×0.15×0.25 m, 0.08×0.06 m rad, *h_override* = 5 mm, f = 50–3000 Hz). FDTD with Huygens + **box_surface** + mid-gap **probes** vs **bempp-cl** exterior Neumann Burton–Miller (`boxrad/post/bem_ref.py`: Neumann ∂p/∂n = −jωρ*v*\_n, structured box mesh, default epw=12 with radiator-span edge spacing ≤ min(rad_w, rad_h)/8 unless non-default `rad_div` overrides, P1/DP0, e^{−iωt}→e^{+jωt} conjugation). All BEM **H** in Pa/(m/s) via **D[p]−S[∂p/∂n]** (`transfer_at`); on-surface compare uses P1 **interpolation**. Per-frequency BEM cache: `validation/results/bem_cache/c09_<hash>.npz` (stable hash: `cache_version`, geometry; `rad_div` only if ≠8; stdout **cache hit** / **cache miss** per *f*); `--reuse-bem` default on. Optional `--h-refine`: FDTD at 2.5 mm and 5/3 mm; per-*f* Im-excess order *p* from successive Δ (ratios 2 and 1.5, not through zero), `z_FDTD,∞` Richardson (f ≤ 1.2 kHz). BEM `z_BEM,∞` at ≈253 and ≈639 Hz from standard vs G3-low mesh (report *p*=1 and *p*=2). **Absolute-scale** at kL=2: area-weighted ⟨*p*⟩\_rad/(ρc) vs *z*\_BEM (assert ≤0.1%), on-axis \|*H*\_BEM\| within 2× monopole. **G0** sphere *z* + **field_at** (r = 2a, 10a). **G1-report** (report only): default-resolution \|z_FDTD−z_BEM\| table and max \|Δz\| (reference tol 0.02); BEM Richardson at top 3 freqs. **G1x** (with `--h-refine`): \|z_FDTD,∞−z_BEM,∞(p=1)\| ≤ 0.005 at ≈253 and ≈639 Hz (print *p*=2). **G2** *r* = 10L, FDTD KH vs BEM *H*. **G3** \|z_epw12−z_epw17\| ≤ 0.005. **G3-low** \|Δz\| ≤ 0.005 at ≈253 and ≈639 Hz vs standard mesh (*h\_max* = 10 mm, rad/16). **KH-norm** probe ≤ 0.2 dB, 2°. Fail G0 → abort. | bempp-cl Burton–Miller | G0 + **G1x** (if h-refine) + G2–G3 + G3-low + KH-norm; G1-report not gated |
| C10-power | **Box** (`configs/c09_box.json`, *h_override* = 5 mm). One FDTD with Huygens recording. Frequencies: 12 log-spaced in [100 Hz, *f_max*] plus kL = 2, 5, 10. Per *f*, transfer functions at \|V\| = 1 m/s (`post.spectra.spectrum` / `load_huygens_spectra`): *W_rad* = ½ρc*S*\|V\|² Re *z* with *S* = *S_grid*; *W_H* = ½ Re Σ *p*\_face *v_n\** d*A* on the closed Huygens surface (*p* and *v_n* at their stored time levels); *W_ff* = Σ \|*H*\|²/(2ρc) *r*² *w_i* on a sphere at *r* = 50·*L* via full KH (*kh_pressure*, full **G**), Gauss–Legendre in cos θ × uniform φ (*n_θ* = 24, *n_φ* = 48; at top *f* also *n_θ* = 36, *n_φ* = 72 for quadrature convergence report). **G1** max \|*W_H*/*W_rad* − 1\| ≤ 0.02; **G2** max \|*W_ff*/*W_rad* − 1\| ≤ 0.02 on the frequency set. Report table (*f*, kL, *W_rad*, ratios), directivity index DI = 10 log₁₀(4π*r*²\|*H_axis*\|²/(2ρc·*W_ff*)) per *f*, PNG ratios vs kL; print every gate even if one fails; RAM/disk pre-checks. | Self-consistency | **G1–W_H PASS** max \|*W_H*/*W_rad* − 1\| = 0.52 % (tol 2 %; *O*((kh)²) discrete-product error, ≈0.03 % at 71 ppw per C11); **G2–W_ff PASS** 0.43 % |
| C11-bandedge | **Box** (`configs/demo/bandedge_box.json`): 0.20×0.15×0.12 m, ⌀0.06 m centered, band 200–800 Hz. **V0** default numerics (alias-aware Huygens decimation from `derive`). At 600 / 700 / 750 / 800 Hz: *P_face* = ½ Re *Z_acoustic* \|U\|² with \|U\| = *S_grid* at \|v\| = 1 m/s; *P_huyg* = ½ Re Σ *p v_n\** d*A* from decimated Huygens data; *P_far* = far-sphere KH integral at *r* = 30 m (Gauss–Legendre 32×64). **G1** max \|*P_far*/*P_face* − 1\| ≤ 0.002 on V0; **G2** max \|*P_huyg*/*P_face* − 1\| ≤ 0.002 on V0. **V1–V8** (one change each: coarser decimation, wider band, pulse width/level, *n_transits*, *cpml_cells*) are report-only sweeps for alias diagnosis. JSON under `validation/results/`. | Self-consistency | **PASS** V0: \|*P_far*/*P_face* − 1\| and \|*P_huyg*/*P_face* − 1\| ≤ 0.2 % at 0.75–1.0 *f_max*; result 0.0094 % / 0.0335 % (*m* = 39, alias −107 dB) |

The pre-flight UI, README and example configs come after Stage D.

## 6. Timeline (about 1 week)
| Day | Work |
|---|---|
| 1 | Stage A code; C1-perf, C2-dispersion |
| 2 | C3-cpml; Stage B code; C4-kh |
| 3 | Stage C code; C5–C8 |
| 4 | Stage D: C9-bempp, C10-power; pre-flight check |
| 5 | Buffer, README, example configs |

Days 1–3 are the critical path. If they slip, reduce the scope of C9; don't skip verification.

## 7. Future work
- **Baffle subtraction for under-resolved radiators.** Run the same radiator mask on the same grid in a half-space with an infinite rigid plane, then take Z = Z_baffled,exact + (Z_FDTD,box − Z_FDTD,baffle), so the grid's self-term error cancels. This would allow radiators that are currently rejected by the cell-budget check. Non-circular masks need the exact baffled Z from a series solution or a numerical Rayleigh self-impedance integral. The approach is unpublished for this use and needs its own validation test.
- **Superposition library.** One run per face patch gives a full column of mutual pressures, so N runs give exact Z and far field for any radiator on that box. That's about 256 runs per box shape at 16×16 patches per quadrant, so it needs a GPU.
- **Data reduction.** Per-frequency spatial downsampling and decomposition of the stored surface pressure.
- **Parameter sweeps** across normalized box and radiator shapes, with fitted functions and lookup for acs-lpm. One run per normalized shape covers all physical sizes via kL.
- **CuPy or GPU backend.** Higher-order or low-dispersion schemes if C2-dispersion and C8-convergence show the high-kL phase error is too large.
- Fillets, recessed or domed radiators, multiple radiators, and non-uniform velocity.

## 8. Open questions
- Dispersion (C2-dispersion, PASS): solver matches exact Yee dispersion (mean lag within 5%). At ppw 20 the worst-direction (axis) phase lag is ≈1°/λ, giving ≈35–42° over a 600 mm path at 20 kHz; ≤10° needs ppw ≈ 35–38 (~10–13× cost).
- Revisit ppw if band-edge dispersion error compounds in downstream (acs-lpm) fits.
- FDTD radiator-edge reactance error ≈1.5% at 12 cells across the short side (C9); consider raising cells_across_radiator if acs-lpm fits are sensitive.
- acs-lpm interface needs (frequency grid, angle grid): to be specified with acs-lpm. The directivity phase reference is the origin (radiator center).