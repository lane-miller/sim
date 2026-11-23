# boxrad v0.3 — step 10a: C18(i) h-refinement; C17 stability soak

Scope: two validation gates that use the existing kernel. New production code is limited to one energy helper. No kernel, recorder, derive or `Result` changes.

Do not modify: any existing `boxrad/` module, `tests/ref_*.py`. New files only (plus `docs/tasks/step-10a.md`).

## 1. `boxrad/validation/energy.py` (new; also `boxrad/validation/__init__.py`)

```python
discrete_energy(fields, op, rho, c, h, v_prev) -> float
```

The conserved quantity of the cut-cell leapfrog (see `tests/ref_refstep.py`, A5), with $\tilde\beta$ from the operator, $\alpha$ the face fractions, and $v^{n\mp1/2}$ the velocities before/after the step:

$$
E^n = \frac{h^3}{2}\left[\sum_c \frac{\tilde\beta_c\,(p_c^n)^2}{\rho c^2} \;+\; \sum_f \rho\,\alpha_f\,v_f^{\,n-1/2}\,v_f^{\,n+1/2}\right].
$$

Read `cutkernel.py` and `ref_refstep.py` to get the exact scaling the kernel uses, and match it (the test below checks that this E is conserved by `ref_refstep` to 1e-12 in float64). Sum in float64.

## 2. Gate C17 — `showcase/step10a_c17_soak.py`

**(a) Closed soak.** A parametric config with six rigid planes, so the domain is closed with no CPML: a 0.20 m cube room (planes x−, x+, y−, y+, z−, z+ at ±0.10), h = 5 mm, containing
- a sphere d 0.06 at (−0.04, −0.03, −0.02),
- a box 0.05 × 0.04 × 0.03 rotated by $R_z(37°)R_x(20°)$ at (0.04, 0.03, 0.02),
- a box boss 0.02 × 0.02 × 0.004 at (0.035, −0.045, −0.0612), shifted so some cells have $\beta < 10^{-3}$ (print min raw β and the number of floored cells; adjust the z position by ≤ 1 mm if no such cell appears and say so).

**Known risk:** with all six sides rigid there is no absorbing side, so `cutcell`'s sweep start (`Grid.absorbing_side`) and the flood-fill seeds may not support it. If `compute_fractions` rejects or mishandles this case, **stop and report**; do not modify `cutcell.py`.

There is no radiator drive; the radiator is a tiny rect on the sphere-facing side of the box, held rigid (needed only for the parser). Initial condition: random $p \sim U(-1, 1)$ in fluid cells (seed 0), v = 0. Step with `step_cut` and the production operator for $2\times10^5$ steps with no source. Record E every 100 steps.

Gate: $\max_n |E^n - E^0|/E^0 \le 10^{-4}$ (float32 round-off random walk; the float64 reference conserves to 1e-15), **and** no secular growth: the mean of E over the last 10 % is within $10^{-4}$ of the mean over the first 10 %.

**(b) Floor disabled (documents the need).** The same with the raw β instead of $\tilde\beta$ (build the operator directly with `fractions.beta`). Stop at the first non-finite or $|p| > 10^6$. Expected: it blows up; print the step. Not a gate (PASS = blew up).

**(c) Open-domain late time.** S4 and S6 at production settings with `numerics.n_transits` × 10. Report $\max|p_{\text{avg}}|$ over the last 10 % of the record relative to its peak (dB), and whether $|p_{\text{avg}}|$ over the last half is non-increasing in 10 equal windows (max per window). Gate: last-10 % level ≤ the file's `tail_db`, and no window more than 3 dB above the previous one.

Output: one `outputs/step10a/c17.json` (numeric) and one `c17.png`: (a) $(E^n-E^0)/E^0$ vs step; (b) floor-disabled max|p| vs step (log); (c) $|p_{\text{avg}}|$ vs t (dB re peak) for S4 and S6.

## 3. Gate C18(i) — `showcase/step10a_c18.py`

s3p (`s3p_c09_parametric.json`) and S4 (`s4_tilted.json`) at `numerics.h_override` ∈ {5.0, 3.333…, 2.5} mm (h = 0.005/1.5 exactly as `0.005/1.5`). Six runs; keep_awake; progress; print the dry-run cell count and wall estimate first. Store the HDF5 in `outputs/step10a/` (`s3p_h{1,2,3}.h5`, `s4_h{1,2,3}.h5`); reuse if present.

At each h compute:
- $e_z = \max_f |z_{S4}-z_{s3p}|/\max_f|z_{s3p}|$, plus the same separately for Re and Im, over the stored `zrad` grid;
- on-axis far field: $\max_f |20\log_{10}|D_{S4}(0)/D_{s3p}(0)||$ in each file's **local** frame, at 12 log-spaced f in $[f_{\min}, f_{\max}]$;
- directivity: normalized, θ 0–90° step 5°, φ ∈ {0, 90°}, at 500, 1000, 2000, 3000 Hz, where s3p is above −20 dB re its max: max |Δ dB|.

Observed order between successive h: $p = \ln(e_1/e_2)/\ln(h_1/h_2)$ for each metric.

Gate (spec): at the finest h, z ≤ 1 % (Re and Im separately) and far field ≤ 0.2 dB. Also report the orders and a Richardson estimate of the $h\to0$ limit of each metric (from the two finest levels, using the observed order, clamped to [1, 2]); a limit clearly above 0 means a systematic bias, not discretization.

Output: one `outputs/step10a/c18.json` (all numbers) and one `c18.png`: log-log metric vs h with slope-1 and slope-2 guide lines; Re z and Im z of S4 and s3p at the three h.

## 4. Tests — `tests/test_energy.py` (fast)

1. `discrete_energy` on the `ref_refstep` closed domain (float64 reference stepper, 2000 steps): $|E^n-E^0|/E^0 \le 10^{-12}$.
2. Production kernel, a closed 6-plane 24³ domain with a sphere, 2000 steps from random p: $|E^n-E^0|/E^0 \le 10^{-5}$.

## 5. Docs

`docs/tasks/step-10a.md` = this file.
