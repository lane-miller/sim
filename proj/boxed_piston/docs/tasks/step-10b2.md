# Step 10b2: C13a (point source near a rigid sphere) and C16 (convergence order)

## Goal
Validate the cut-cell wall against an exact solution: a monopole near a rigid sphere.
- **C13a:** the gate at ppw ≈ 20.
- **C16:** observed order over four resolutions, plus a sub-cell shift sweep.

The 10b1 free-field showcase showed that the source cell and grid dispersion contribute up to ~0.1 dB / 0.5° on their own. So C13a and C16 compare the **scattering ratio**
$$R(\mathbf x) = \frac{p_{\text{sphere}}(\mathbf x)}{p_{\text{free}}(\mathbf x)},$$
taken between an FDTD run with the sphere and an FDTD run without it, with the same source cell and the same probe cells. This is compared against the analytic ratio. Source and dispersion errors cancel, leaving the wall error.

## Hard rules
- No changes to the kernel, CPML, floor, KH or recorder.
- This step adds:
  - one analytic function;
  - one config template;
  - one showcase;
  - tests;
  - the one-line plot fix from 10b1.
- Every run is a `test_source` run (step 10b1 machinery).
- One JSON and one PNG per gate, overwritten in place.

## 1. Analytic reference: `boxrad/validation/analytic.py`
Add `sphere_point_source(points, f, center, a, source, rho, c, *, nmax=None, part="total")`. It returns p/Q (Pa per m³/s) for a monopole near a rigid sphere, with e^{+jωt}. Here r and r₀ are the probe and source radii about the sphere center, and γ is the angle between them:
$$\frac{p_{\text{inc}}}{Q} = j\omega\rho\,\Big(\!-\frac{jk}{4\pi}\Big)\sum_n (2n+1)\,j_n(kr_<)\,h_n^{(2)}(kr_>)\,P_n(\cos\gamma),$$
$$\frac{p_{\text{scat}}}{Q} = j\omega\rho\,\Big(\frac{jk}{4\pi}\Big)\sum_n (2n+1)\,\frac{j_n'(ka)}{h_n^{(2)\prime}(ka)}\,h_n^{(2)}(kr_0)\,h_n^{(2)}(kr)\,P_n(\cos\gamma).$$

Use this verified implementation verbatim (vectorized over points; scipy only):
```python
from scipy.special import spherical_jn, spherical_yn, eval_legendre

def _h2(n, x, d=False):
    return spherical_jn(n, x, derivative=d) - 1j * spherical_yn(n, x, derivative=d)

def sphere_point_source(points, f, center, a, source, rho, c, *, nmax=None, part="total"):
    pts = np.atleast_2d(np.asarray(points, float)) - np.asarray(center, float)
    s = np.asarray(source, float) - np.asarray(center, float)
    r = np.linalg.norm(pts, axis=1); r0 = float(np.linalg.norm(s))
    cosg = np.clip((pts @ s) / (r * r0), -1.0, 1.0)
    w = 2 * np.pi * f; k = w / c; ka = k * a
    if nmax is None:
        q = float(np.max(np.minimum(r, r0) / np.maximum(r, r0)))
        if q > 0.9:
            raise ValueError("probe radius within 10% of source radius: series converges too slowly")
        nmax = min(400, int(k * max(r0, a)) + 20 + int(np.log(1e-12) / np.log(q)))
    rl = np.minimum(r, r0); rg = np.maximum(r, r0)
    pin = np.zeros(len(r), complex); psc = np.zeros(len(r), complex)
    for n in range(nmax + 1):
        Pn = eval_legendre(n, cosg)
        pin += (2 * n + 1) * spherical_jn(n, k * rl) * _h2(n, k * rg) * Pn
        coef = spherical_jn(n, ka, derivative=True) / _h2(n, ka, True)
        if not np.isfinite(coef):
            coef = 0.0
        psc += (2 * n + 1) * coef * _h2(n, k * r0) * _h2(n, k * r) * Pn
    pre = 1j * w * rho
    pin *= -1j * k / (4 * np.pi) * pre
    psc *= 1j * k / (4 * np.pi) * pre
    return {"inc": pin, "scat": psc, "total": pin + psc}[part]
```
Wrap or suppress numpy warnings locally (`np.errstate`). Do not change the math.

**Prototype verification** (Claude, a = 0.05, source (0, 0, 0.1), 300–4000 Hz):
- incident series vs `monopole_free`: ≤ 6e-15;
- rigid BC |∂p/∂r| / |∂p_inc/∂r| at r = a: ≤ 2.4e-7 (finite-difference limited);
- reciprocity (source ↔ probe swapped, shifted center): ≤ 4e-16;
- a = 1e-4: |p_scat/p_inc| ≤ 6e-9;
- truncation (nmax + 20 vs nmax): ≤ 4e-16.

## 2. Geometry and probes (common to all runs)
- **Sphere:** parametric part `{"sphere": {"d": 0.1}}` at `position = center`, normally (0, 0, 0). No radiators.
- **Source:** `numerics.test_source.position = [0, 0, 0.1]`, 50 mm from the surface.
- **Resolutions:** h = 0.05/n for n ∈ {6, 9, 12, 18}, via `numerics.h_override`. That is h ≈ 8.33, 5.56, 4.17, 2.78 mm, or ppw at 3 kHz ≈ 13.7, 20.6, 27.4, 41.2 (spec C16: 14/20/28/40).
- **Cell centers.** With no radiators, the grid is anchored with the source at a cell center (10b1). All probe coordinates below are multiples of 1/60 m, so they are cell centers at every h above. Assert this in the showcase (|offset| ≤ 1e-9 m).

| Probe | Position (m) | Role |
|---|---|---|
| P0 | (0, 0, −0.15) | shadow |
| P1 | (0.15, 0, 0) | side |
| P2 | (0, 0, 0.15) | front, beyond the source |
| P3 | (0, 1/15, 0) | near wall, side (16.7 mm off the surface) |
| P4 | (0, 0, −1/15) | near wall, shadow |

- **Far field:** KH at r = 2 m from the origin (world frame), in the directions +z, −z, +x, and (1, 0, −1)/√2.
- **Free-field companion run:** same h, same source, `parts = []`, same probes.
- **gap_cells per run.** Compute it so the CPML-free interior covers every probe with at least 3 cells of margin:
  $$\text{gap\_cells} = \Big\lceil \frac{\max(\text{probe distance beyond the solid/source bbox, per side})}{h} \Big\rceil + 3,$$
  computed separately for the sphere run and the free run. The 10b1 CPML probe guard must not fire.

## 3. C13a gate (n = 9, ppw 20.6)
Frequencies: 300, 500, 800, 1000, 1500, 2000, 2500, 3000 Hz.
- **Probes P0–P4:** compare the measured ratio R_fd = p_sphere/p_free against R_an = `sphere_point_source(...)` / `monopole_free(...)`. Error in dB: 20 log₁₀|R_fd/R_an|; error in degrees: ∠(R_fd/R_an).
- **Far field:** the same, with the KH fields at 2 m.
- **Gate:** probes and far field ≤ 0.1 dB / 1° at every f (spec C13a).
- **Report:**
  - the absolute error of the sphere run alone vs `sphere_point_source` (not gated);
  - the analytic |R_an| in dB at each probe, so shadow-zone sensitivity is visible.

## 4. C16 convergence (n = 6, 9, 12, 18)
- **Error metric:**
  $$e(h) = \max_{\text{P0–P4},\ f \in \{500, 1000, 2000, 3000\}} \left|\frac{R_{fd}}{R_{an}} - 1\right| \quad \text{(complex).}$$
- **Observed order p:** the least-squares slope of log e vs log h over the four points. Also print the pairwise orders.
- **Gate:** p ≥ 1.5 (spec).
- **Shift sweep** (n = 9 only): translate the **sphere only** by δh·(1, 1, 1) for δ ∈ {0, 0.25, 0.37, 0.5}. The source and probes stay fixed; the analytic reference uses the shifted center. Report the spread (max − min over δ) of |R_fd/R_an − 1| per probe at 1 kHz and 3 kHz. **Gate:** spread ≤ 0.3% (spec, with probe ratios in place of z).
- **Modes:** if `numerics.boundary` already supports `staircase` on the parametric path, also run n = 9 in staircase mode and gate e_cutcell(n = 9) ≤ ½·e_staircase(n = 9). **If it is not implemented, skip it and print "staircase mode not available, comparison skipped". Do not implement it.**

## 5. Showcase: `showcase/step10b2_c13a_c16.py`
- Builds configs in code from `configs/c13_sphere_ps.json` (new template: the sphere, source and probes above). Free and shifted variants are derived in code; no extra fixture files.
- Runs: 4 sphere + 4 free + 3 shifted, plus 1 staircase if available. Use caffeinate and per-run progress lines with an ETA.
- Run outputs go under `outputs/step10b2/`, one HDF5 per run, overwritten.
- **Printed tables:**
  - C13a per f × probe (dB, deg, |R_an| dB);
  - far field per f × direction;
  - C16 e(h), p, pairwise orders;
  - shift spread;
  - the gate lines.
- **Outputs:** `outputs/step10b2/c13a_c16.json` and one PNG with 3 panels: C13a probe dB error vs f; e(h) vs h on log-log with a slope-2 reference line; shift spread.
- Print the dry-run cell counts and the step count per run before starting. At n = 18 the free run is about 9–10 M cells.

## 6. Tests
**`tests/test_analytic_sphere.py`** (fast):
1. incident part vs `monopole_free` ≤ 1e-12 at 300 and 3000 Hz;
2. rigid BC: central difference of `total` across r = a at 7 polar angles, relative to the incident gradient, ≤ 1e-6;
3. reciprocity with a shifted center ≤ 1e-12;
4. a = 1e-4: |scat/inc| ≤ 1e-7;
5. a probe at r = 0.98 r₀ raises `ValueError`.

**`tests/test_test_source.py`** (slow, add one): a coarse sphere ratio check at n = 6, f = 500 Hz, P1 only, |R_fd/R_an − 1| ≤ 0.05. This is a smoke test, not the gate.

## 7. 10b1 plot fix
`showcase/step10b1_point_source.py`: the probe panels omit probe 3 (diagonal). Plot all probes.

## Report back
- Files changed, with line counts.
- Whether staircase mode exists.
- Dry-run cell counts.
- Any deviation.
