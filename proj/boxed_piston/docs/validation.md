# boxrad validation (C1–C10)

All ten checks pass. They ran in order, and each gate had to pass before the
next stage was built (see the [brief](brief.md) §5 for full definitions). C1
through C3 check the bare solver, C4 checks the Kirchhoff–Helmholtz (KH)
post-processing alone, C5 through C8 check the box and radiator against
analytic and self-convergence references, and C9 and C10 are independent
confirmations: boundary-element method (BEM) and energy balance.

Notation: $z$ is the dimensionless radiation impedance, $H = p/v$ is the
pressure transfer, and $L$ is the largest box side. $\Delta z$ errors are
absolute and dimensionless.

## Summary

| Check | What it tests | Reference | Gate | Result |
|---|---|---|---|---|
| **C1-perf** | Speed and memory vs grid size on the M1; Numba compile, thread scaling, dtype | — | Local envelope large enough (else a GPU backend first) | **PASS**: ~390 M cell-updates/s peak (375 M used for estimates); ~3.2× on 8 threads |
| **C2-dispersion** | Free-field point source on axis and diagonal: phase error vs ppw; long-run stability | Analytic (exact Yee dispersion) | Acceptable phase error at affordable ppw | **PASS**: matches Yee dispersion (mean lag within 5 %); ≈1°/λ axis lag at ppw 20; stability ratio ≪ 10⁻³ |
| **C3-cpml** | CPML reflection vs a very large domain; σ_max, α, thickness, gap; grazing incidence | Large-domain run | About −80 dB at f_min | **PASS**: −100 dB normal and oblique; low-frequency Z gap-converged to ≤ 0.5 % from 5 cells |
| **C4-kh** | KH over a closed box applied to analytic monopole and dipole fields; Huygens face-pressure rule | Analytic | No quadrature, normal, or sign errors | **PASS**: extinction ≤ 0.1 %; ~2nd order in h; Huygens 4-point cubic face p ≈ 0.1 / 0.2 / ≤ 0.2 % at 1 / 4 / 10 kHz |
| **C5-lowfreq** | 0.2 m cube, ⌀0.1 m centered circle, 20–1500 Hz; low-kL fit $\mathrm{Re}\,z = Ak^2(1+Bk^2)$, $\mathrm{Im}\,z = k\delta(1+Ck^2)$; gap ×2 run | Monopole: $\mathrm{Re}\,z \to (ka)^2/4$; $\delta/a \approx 0.61$–$0.85$ | G1 $\lvert A/(S/4\pi)-1\rvert \le 3\%$; G2 Im z > 0 and $0.55 \le \delta/a \le 0.90$; G3 gap ×2 change ≤ 2 % | **PASS**: G1 9×10⁻⁴; G2 δ/a = 0.787; G3 6.3×10⁻⁶ |
| **C6-baffle** | Baffle mode, ⌀0.1 m circle, 100–6000 Hz (ka ≤ 5.5); five-face Huygens + image KH; on-axis probes | Analytic baffled piston (z, on-axis H, $2J_1(x)/x$ pattern) | G1 low-ka $\lvert A/(S/2\pi)-1\rvert \le 2\%$; G2 $\max\lvert z-z_\mathrm{ref}\rvert \le 0.03$; G3 on-axis H ±0.3 dB / 5° at ka = 1, 3, 5, pattern ≤ 0.5 dB; tail ≤ −80 dB | **PASS**: G1 1.6×10⁻⁴; G2 0.0104; on-axis far field and probes ≤ 0.01 dB / 0.3°; directivity ≤ 0.043 dB; tail −151 dB |
| **C7-subcell** | Baffle mode: sub-cell radiator offsets; rectangle mask; circle resolution at 6 / 10 / 16 cells | Analytic baffled circle and rectangle (`post/reference.py`) | G1 offset change ≤ 0.01; G2 rect ≤ 0.03; G3 circle at 10 cells ≤ 0.02 ($ka_\mathrm{max} \approx 1.5$) | **PASS**: G1 0.0026; G2 0.0242; G3 0.0126 / **0.0065** / 0.0029 at 6 / **10** / 16 cells |
| **C8-convergence** | G1 baffle dispersion vs ppw ∈ {10, 15, 20, 30}; G2 box self-convergence at $h = L/N$, N ∈ {24, 36, 54}; G3 KH vs FDTD probes in the box gap | Analytic (G1); Richardson extrapolation (G2); FDTD probes (G3) | G1 ppw 20 ≤ 0.015; G2 $\max\lvert z_{36}-z_\infty\rvert \le 0.01$ and directivity ≤ 0.5 dB at kL = 2, 5, 10; G3 ≤ 0.2 dB / 2° | **PASS**: G1 0.0102 at f_max, order 1.96; G2 0.0080, Richardson p = 2.26, directivity ≤ 0.08 dB; G3 box KH probes ≤ 0.07 dB |
| **C9-bempp** | Box 0.20×0.15×0.25 m, 0.08×0.06 m rect off center, h = 5 mm, 50–3000 Hz: z, far field, probes, surface p | bempp-cl exterior Neumann Burton–Miller BEM (+ Richardson on both sides) | G0 sphere ≤ 0.01; G1x Richardson $\lvert z_{\mathrm{FDTD},\infty}-z_{\mathrm{BEM},\infty}\rvert \le 0.005$; G2 on-axis ≤ 0.3 dB / 5°, directivity ≤ 0.5 dB; G3 BEM mesh ≤ 0.005; G3-low ≤ 0.005; KH-norm ≤ 0.2 dB / 2° | **PASS**: G1x 0.0038; G1-report (default h) max 0.0251, from edge reactance, report only; on-axis ≤ 0.09 dB / 0.6°; directivity ≤ 0.26 dB; G3 0.003; G3-low 0.0027; KH-norm 0.01 dB / 0.02°; surface ≤ 0.43 dB (report) |
| **C10-power** | C9 box: radiator power vs Huygens flux vs far-field sphere integral; 12 log frequencies in [100 Hz, f_max] plus kL = 2, 5, 10 | Self-consistency ($W_\mathrm{rad} = \tfrac12\rho c S\,\mathrm{Re}\,z$) | G1 $\max\lvert W_H/W_\mathrm{rad}-1\rvert \le 2\%$; G2 $\max\lvert W_\mathrm{ff}/W_\mathrm{rad}-1\rvert \le 2\%$ | **PASS**: $W_\mathrm{ff}/W_\mathrm{rad}$ within 0.39 %; $W_H/W_\mathrm{rad}$ within 0.27 % through kL 10 and 1.04 % at f_max |

## What the results mean for users

- **Radiation impedance:** about 1 % $|\Delta z|$ at $f_\mathrm{max}$ for
  ppw 20, falling as $(f/f_\mathrm{max})^2$ (C8). At 10–12 cells across the
  radiator's short side there is about 1.5–2.6 % excess edge reactance in
  Im z, which shrinks with resolution (C7, C9). The same summary is stored in
  every result file under `/meta/accuracy/notes`.
- **Far field and directivity:** within 0.1 dB of the converged result in the
  box (C8) and within 0.26 dB of BEM (C9).
- **Energy:** radiator, Huygens, and far-field power agree within 0.4 % up to
  kL 10 (C10).
- **Low frequency:** the monopole limit and end correction are reproduced
  (C5, C6). The CPML does not bias low-frequency Re z at the default α (C3,
  C5 G3).

## Rerunning the checks

Install boxrad first (from `sim/`: `python -m pip install -e proj/boxed_piston`).
Run the scripts from `proj/boxed_piston/`; they locate their configs relative
to the script, so other working directories also work. Each script prints
progress and gate lines and writes `validation/results/<check>_<UTC stamp>.json`
(plus PNGs unless `--no-plot`). It returns exit code 0 only if every gate
passes. On macOS, long scripts keep the machine awake with `caffeinate`.

Flags common to C5–C10:

| Flag | Effect |
|---|---|
| `--out DIR` | Results directory (default `validation/results/`) |
| `--no-plot` | Skip PNGs |
| `--ram-gb X` | RAM limit for the pre-flight check (default 5 GiB ≈ 5.37) |
| `--keep-h5` | Keep FDTD HDF5 under `outputs/<check>_<stamp>/` (default: a temporary directory, removed afterward) |

| Check | Command | Notes |
|---|---|---|
| C1 | `python validation/c01_perf.py [--quick]` | `--sizes`, `--threads`, `--steps` |
| C2 | `python validation/c02_dispersion.py [--quick]` | `--f-ref`, `--ppw`, `--stability-steps` |
| C3 | `python validation/c03_cpml.py [--quick]` | `--only NAME` runs one variant |
| C4 | `python -m pytest tests/test_kirchhoff.py` | C4 lives in the unit tests |
| C5 | `python validation/c05_lowfreq.py` | Config `configs/c05_lowfreq.json` |
| C6 | `python validation/c06_baffle.py` | Config `configs/c06_baffle.json`; `c06_diag.py` is an optional diagnostic |
| C7 | `python validation/c07_subcell.py` | Configs `configs/c07_*.json` |
| C8 | `python validation/c08_convergence.py` | Configs `configs/c08_g1_baffle.json`, `configs/c08_g2_box.json` |
| C9 | `python validation/c09_bempp.py --h-refine` | **Requires bempp-cl.** `--h-refine` adds FDTD at 2.5 mm and 5/3 mm and is needed for gate G1x. BEM solutions are cached per frequency in `validation/results/bem_cache/`; `--no-reuse-bem` forces a re-solve |
| C10 | `python validation/c10_power.py` | Config `configs/c09_box.json` (forces `h_override = 5 mm`) |

**bempp-cl** is needed only for C9 and `tests/test_bem_ref.py`. Install it
with the optional extra, from `sim/`:
`python -m pip install -e "proj/boxed_piston[bem]"`. The reference
environment is the conda env `simenv`, which has Numba and bempp-cl. BEM
data from bempp (time convention $e^{-i\omega t}$) is conjugated to
$e^{+j\omega t}$ in `boxrad/post/bem_ref.py`.

To list kept validation outputs under `outputs/`, run
`python validation/clean_outputs.py`; add `--yes` to delete them.

Unit tests: `python -m pytest tests -m "not slow"` (fast set) or
`python -m pytest tests` (everything).
