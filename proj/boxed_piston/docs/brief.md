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
- **Cell-budget fallback:** this applies only when h = D/10 would exceed the cell budget, which happens with tiny radiators on comparatively large boxes, such as mic ports. Then h = λ_min/20 is kept and the radiator spans fewer than 10 cells.
  - The far field and surface p still come from FDTD; the radiator acts as a near-point source with exact volume velocity.
  - Z_rad comes from the analytic baffled-piston closed form, (ρc/S)[1 − J₁(2ka)/ka + j·H₁(2ka)/ka]. The output is flagged `zrad_source = "analytic_baffled"` and a console warning states that the box's effect on Re z is neglected (up to 2× when the box is also small). Im z is dominated by the radiator itself and stays reliable.
- **Config validation:** reject radiators closer than 2–3 cells to any box edge. Tangent radiators are not physical and can't be resolved.
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
- **Input:** a JSON config with **`geometry`**: `box` (default) or `baffle`. Box mode requires box dimensions; baffle mode uses a half-space rigid plane at z-min (no box, no z-min CPML) with lateral domain sized to the radiator bounding box. Also: radiator shape, size and position, f_min, f_max, tolerances, optional overrides, far-field angle grid, output path. Examples: `configs/default.json`, `configs/smoke.json`, `configs/c06_baffle.json`.
- **Pre-flight check:** before any computation, `run.py` prints a summary of the derived parameters and the estimates (cells, RAM, disk, run time), then lists warnings and errors. It waits for confirmation unless `--yes` is passed.
  - **Errors** (the run stops): a radiator tangent to an edge or within 2–3 cells of one; geometry that doesn't fit the box; the RAM estimate exceeds available memory.
  - **Warnings** (the run continues):
    - ka > 10 at f_max (beyond the rigid-piston regime);
    - the cell-budget fallback is active (the radiator spans fewer than 10 cells, so Z_rad is analytic);
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
  - run metadata (h, Δt, number of steps, decimation, run time, peak RAM, `zrad_source`).

  The volume field is not stored. Optional mid-plane snapshots can be saved for debugging. Compression (spatial downsampling, decomposition) comes after the solver is verified.
- **Post-processing:** a separate module computes:
  - z(f), from the radiator-averaged p(t) and v(t);
  - the far field and near-field probes **outside** the Huygens surface via Kirchhoff–Helmholtz on the stored Huygens **p** and **v_n**; box-surface p is used for on-surface probes only.

  The default far-field output is p at 1 m per 1 m/s radiator velocity, on a full sphere at 5° spacing.
- **Storage:** run outputs go to a configurable path, such as the external SSD, and are excluded from git.
- **Module layout** (one Cursor prompt per module, each with a small unit test):

  | Module | Contents |
  |---|---|
  | `config.py` | Schema, defaults, derived-parameter rules, validation |
  | `grid.py` | Domain, box mask, radiator area weights |
  | `kernels.py` | Numba p and v updates |
  | `cpml.py` | CPML profiles and slab updates |
  | `source.py` | Pulse design and radiator injection |
  | `recorder.py` | Chunked HDF5 output |
  | `run.py` | Entry point: config → run → HDF5 (`python -m fdtd.run CONFIG [-o OUT] [--dry-run] [--quiet]`) |
  | `post/` | z(f), Kirchhoff–Helmholtz, probes |
  | `validation/` | Check scripts (`c01_perf.py` … `c10_power.py`) |

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

**Stage C: box and radiator** (`grid.py` box mask and area weights, `source.py`, `recorder.py`, `run.py`).

| Check | What | Reference | Gate |
|---|---|---|---|
| C5-lowfreq | 0.2 m cube, φ0.1 m centered radiator, f=20–1500 Hz; fit Re z = A k²(1+B k²), Im z = k δ(1+C k²) on kL≤0.2 (report kL≤1); gap×2 truncation run | Monopole: Re z → (ka)²/4; δ/a ≈ 0.61–0.85 | **G1** \|A/(S/4π)−1\|≤3%; **G2** Im z>0 on gate band and 0.55≤δ/a≤0.90; **G3** max\|z_gap2−z\|/\|z\|≤2% on gate band. Fail → debug face pressure, half-step / sampling, CPML |
| C6-baffle | **Baffle** geometry (`configs/c06_baffle.json`): φ0.1 m centered circle, f=100–6000 Hz (~35 cells across, ka_max≈5.5); half-space rigid plane at z-min (no z-min CPML), five-face Huygens; KH with `image_plane_z=0` and mirrored image normals; spectra via `post.spectra.spectrum` (Δt-weighted sum at stored times); optional on-axis probes (`output.probes_baffle_z_cells`) | Baffled piston: `z_ref` via `post.reference.z_baffled_circle` | **G1** low ka: fit Re z = A k²(1+B k²) for ka≤0.3; \|A/(S/2π)−1\|≤2%. **G2** max \|z−z_ref\|≤0.03 on band. **G3** transfer \(H=P/V\): `kh_pressure` on axis at r=100·a vs \(H_{\mathrm{exact}}(r)=\rho c[e^{-jkr}-e^{-jk\sqrt{r^2+a^2}}]\) at ka=1,3,5 (±0.3 dB, 5°); \|H(θ)/H(0)\| vs \(2J_1(ka\sinθ)/(ka\sinθ)\) with ref level ≥−20 dB: max \|ΔdB\|≤0.5 (included in G3 summary). Report `kh_far_field` vs `kh_pressure`. **G3-probe** (report): \(H_{\mathrm{KH}}, H_{\mathrm{exact}}\) vs \(H_{\mathrm{FDTD}}\) at probe heights. Tail ≤−80 dB |
| C7-subcell | **Baffle** mode (`configs/c07_*.json`): sub-cell radiator offset and mask resolution. **G1** cell-aligned φ0.1 m circle vs same grid with center offset `(0.25h, 0.25h)` and `(0.5h, 0.37h)`; `max |z_shift−z_0|≤0.01` on the band (default f 100–6000 Hz). **G2** rectangle 0.10×0.05 m vs `post.reference.z_baffled_rect`; `max |z−z_ref|≤0.03` on band; report largest `ka` (`a=√(S/π)`) with error ≤0.01. **G3** circle d=0.1 m, `cells_across∈{6,10,16}`, `f_max` so `ka_max≈1.5`; `max |z−z_ref|≤0.02` at 10 cells (analytic circle ref); report all three and observed convergence order. **Report** racetrack 0.10×0.05 m at `{10,16,24}` cells: `max |z_10−z_24|`, `|z_16−z_24|`. PNGs: z vs ref (G1/G2), error vs `ka` (G3). RAM check before each run. | `fdtd/post/reference.py` | Gates G1–G3 as stated; area weighting / sub-cell error too large → fix `grid` weights |
| C8-convergence | **G1** baffle dispersion (`configs/c08_g1_baffle.json`): φ0.1 m circle, `cells_across=6` so **λ/ppw** sets *h*; `f_max` 6000 Hz; ppw ∈ {10, 15, 20, 30} (verify distinct *h* per run); common *ka* band vs `z_baffled_circle`; report max \|z−z_ref\| per ppw and observed order. **Gate:** ppw=20 max \|z−z_ref\|≤0.015 (≈1% z error at f_max, falling as (f/f_max)²). **G2** box self-convergence (`configs/c08_g2_box.json`): 0.2³ m cube, φ0.1 m centered, f 50–3000 Hz; `numerics.h_override = L/N` with *N* ∈ {24, 36, 54} (refinement ratio 1.5, nominal *L*=*W*); assert realized *W*, *H*, *D* and radiator center match nominal; print realized ppw at *f_max*. Richardson *p* = ln(‖z_c−z_m‖/‖z_m−z_f‖)/ln 1.5 on the common design band (same for reported directivity *p* on gated angles); fallback *p*=2 if *p*∉[1,3]. **Gates:** max \|z_36−z_∞\|≤0.01 on band; closed-surface KH directivity at *r*=10·L in **xz** and **yz** at kL=2, 5, 10 — max \|ΔdB\| gate level vs extrapolated ≤0.5 dB (pattern ≥−20 dB re on-axis). **G3** (*N*=36 run): probes midway Huygens–CPML at (0,0,+g), (W/2+g,0,−D/2), (0,0,−D−g); KH (`kh_pressure_loop`, full **G**) vs FDTD at G2 kL values; *H*=P/V via `post.spectra.spectrum`. **Gate:** ≤0.2 dB, ≤2°. Report recommended default ppw (smallest in sweep meeting G1+G2); do not change config default. PNGs: \|z−z_ref\| vs *ka* per ppw; directivity overlays; probe table in stdout. RAM check and wall/RAM plan before runs. | Self-consistency | G1–G3 as stated; sets ppw recommendation (brief §8) |

**Stage D: independent confirmation.**

| Check | What | Reference | Gate |
|---|---|---|---|
| C9-bempp | **Box** with edge-offset rectangular radiator (`configs/c09_box.json`: W×H×D = 0.20×0.15×0.25 m, 0.08×0.06 m rad, *h_override* = 5 mm, f = 50–3000 Hz). FDTD with Huygens + **box_surface** + mid-gap **probes** vs **bempp-cl** exterior Neumann Burton–Miller (`fdtd/post/bem_ref.py`: Neumann ∂p/∂n = −jωρ*v*\_n, structured box mesh, default epw=12 with radiator-span edge spacing ≤ min(rad_w, rad_h)/8 unless non-default `rad_div` overrides, P1/DP0, e^{−iωt}→e^{+jωt} conjugation). All BEM **H** in Pa/(m/s) via **D[p]−S[∂p/∂n]** (`transfer_at`); on-surface compare uses P1 **interpolation**. Per-frequency BEM cache: `validation/results/bem_cache/c09_<hash>.npz` (stable hash: `cache_version`, geometry; `rad_div` only if ≠8; stdout **cache hit** / **cache miss** per *f*); `--reuse-bem` default on. Optional `--h-refine`: FDTD at 2.5 mm and 5/3 mm; per-*f* Im-excess order *p* from successive Δ (ratios 2 and 1.5, not through zero), `z_FDTD,∞` Richardson (f ≤ 1.2 kHz). BEM `z_BEM,∞` at ≈253 and ≈639 Hz from standard vs G3-low mesh (report *p*=1 and *p*=2). **Absolute-scale** at kL=2: area-weighted ⟨*p*⟩\_rad/(ρc) vs *z*\_BEM (assert ≤0.1%), on-axis \|*H*\_BEM\| within 2× monopole. **G0** sphere *z* + **field_at** (r = 2a, 10a). **G1-report** (report only): default-resolution \|z_FDTD−z_BEM\| table and max \|Δz\| (reference tol 0.02); BEM Richardson at top 3 freqs. **G1x** (with `--h-refine`): \|z_FDTD,∞−z_BEM,∞(p=1)\| ≤ 0.005 at ≈253 and ≈639 Hz (print *p*=2). **G2** *r* = 10L, FDTD KH vs BEM *H*. **G3** \|z_epw12−z_epw17\| ≤ 0.005. **G3-low** \|Δz\| ≤ 0.005 at ≈253 and ≈639 Hz vs standard mesh (*h\_max* = 10 mm, rad/16). **KH-norm** probe ≤ 0.2 dB, 2°. Fail G0 → abort. | bempp-cl Burton–Miller | G0 + **G1x** (if h-refine) + G2–G3 + G3-low + KH-norm; G1-report not gated |
| C10-power | Power balance: Re Z vs far-field radiated power | Self-consistency | Global sanity |

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
- **Baffle subtraction for under-resolved radiators.** Run the same radiator mask on the same grid in a half-space with an infinite rigid plane, then take Z = Z_baffled,exact + (Z_FDTD,box − Z_FDTD,baffle), so the grid's self-term error cancels. This would replace the analytic fallback. Non-circular masks need the exact baffled Z from a series solution or a numerical Rayleigh self-impedance integral. The approach is unpublished for this use and needs its own validation test.
- **Superposition library.** One run per face patch gives a full column of mutual pressures, so N runs give exact Z and far field for any radiator on that box. That's about 256 runs per box shape at 16×16 patches per quadrant, so it needs a GPU.
- **Data reduction.** Per-frequency spatial downsampling and decomposition of the stored surface pressure.
- **Parameter sweeps** across normalized box and radiator shapes, with fitted functions and lookup for acs-lpm. One run per normalized shape covers all physical sizes via kL.
- **CuPy or GPU backend.** Higher-order or low-dispersion schemes if C2-dispersion and C8-convergence show the high-kL phase error is too large.
- Fillets, recessed or domed radiators, multiple radiators, and non-uniform velocity.

## 8. Open questions
- Dispersion (C2-dispersion, PASS): solver matches exact Yee dispersion (mean lag within 5%). At ppw 20 the worst-direction (axis) phase lag is ≈1°/λ, giving ≈35–42° over a 600 mm path at 20 kHz; ≤10° needs ppw ≈ 35–38 (~10–13× cost).
- Revisit ppw if band-edge dispersion error compounds in downstream (acs-lpm) fits.
- FDTD radiator-edge reactance error ≈1.5% at 12 cells across the short side (C9); consider raising cells_across_radiator if acs-lpm fits are sensitive.
- acs-lpm interface needs (frequency grid, angle grid, phase reference): to be specified once single runs are verified.