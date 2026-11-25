# Step 12b — unpin `cfl_safety` from the showcase fixtures

**Goal.** S1–S6 follow the auto cfl rule (A12): S1, S4, S6 have cut cells → 0.7 (`"auto_cutcell"`); S2, S3p, S5 have none → 0.99 (`"default"`, bit-identical dt).

## Guardrails
- Only read/write code; don't run code, tests, or git commands.
- Touch only the files named below plus any file found under R2. Do not edit `boxrad/` source unless R2 finds a direct float read of `numerics.cfl_safety` there; report that instead of fixing it silently.
- Do not loosen any tolerance. Do not add new hardcoded magic numbers; derive expected dt/step counts from the resolved cfl.
- Do not touch: `configs/c13_sphere_ps.json`, `configs/c13_free_ps.json` (A12 sweep), box/baffle/demo/c05–c09/c19 configs, hardcoded λ in `test_cutkernel`, `test_energy`, `test_kernels`, `test_cpml`, `ref_refstep`, `showcase/step05_smallcell.py`, `showcase/step06a_kernel.py`, the own-config pin in `showcase/step10a_c17_soak.py:75`, `showcase/step10b2_c13a_c16.py`.

## Changes

**R1 — fixtures.** In `configs/showcase/{s1_cylinder,s2_bookshelf,s3p_c09_parametric,s4_tilted,s5_platform,s6_downfire}.json`, delete the line `"cfl_safety": 0.99,` (line 15; mid-object, so no comma fix-ups). Nothing else in these files changes.

**R2 — direct reads.** Find every read of `numerics.cfl_safety` (or `["numerics"]["cfl_safety"]`) used as a float in `tests/`, `showcase/`, and `boxrad/`. With the key absent it is `None`. Replace with the resolved value (`derived.cfl_safety_resolved`, or the existing helper the tests already use, `cfl_for_legacy_derive` for box paths). Check at least: `tests/test_smallcell.py::_showcase_geom`, `tests/test_geomrun.py`, `tests/test_geomrec.py`, `tests/test_result_frames.py`, `showcase/step06b1_geomrun.py`, `step06b2a_gates.py`, `step06b2b_frames.py`, `step10a_c17_soak.py`, `step10a_c18.py`, `step12_report_plot.py`. List every hit in the reply, changed or not.

**R3 — `test_smallcell.py` keeps the A5 reference.** Its S1/S4/S6 floor and stability tests stay at `LAM_SHOW = 0.99/√3` explicitly (not from the config). `test_delta_eq_s1` stays at 0.047 ± 0.003. Add `test_delta_eq_s1_auto`: same S1 geometry at λ = `CFL_CUTCELL/√3`; assert `delta_eq_over_h < 0.01` and status `"OK"`.

**R4 — new gate test** (in `tests/test_cfl_auto.py`), parametrized over the six fixtures:
- the raw JSON has no `numerics.cfl_safety` key; `Config.from_json(...).numerics.cfl_safety is None`;
- dry-run estimate (same call as the existing line-116 test): S1, S4, S6 → `cfl_safety_resolved == CFL_CUTCELL`, source `"auto_cutcell"`; S2, S3p, S5 → `CFL_DEFAULT`, `"default"`;
- for S2/S3p/S5, dt equals dt of the same config with an explicit 0.99 (exact equality).

**R5 — stored numbers.** Any test assertion on dt, n_steps, wall-time estimate, floored-cell counts, or δ_eq/h taken from S1/S4/S6 at cfl 0.99 is updated by deriving from the resolved cfl, or pinned to 0.99 explicitly in that test's local config copy if the test is about the 0.99 behavior. Say which you chose for each.

**R6 — docs.** `docs/user_guide.md`:
- line 162 table row: default column `auto`; description: 0.99 on box/baffle and on parametric/STEP grids without cut cells, 0.7 when the grid has cut cells; user value always wins.
- line 287: replace the fixed $\Delta t = 0.99\,h/(c\sqrt3)$ with $\Delta t = s\,h/(c\sqrt3)$, $s$ = resolved `cfl_safety`, and one sentence on the auto rule.

## Tests (Lane runs, from `sim/`)
```
python -m pytest proj/boxed_piston/tests/test_cfl_auto.py proj/boxed_piston/tests/test_smallcell.py proj/boxed_piston/tests/test_report_plot.py proj/boxed_piston/tests/test_geomrun.py proj/boxed_piston/tests/test_geomrec.py proj/boxed_piston/tests/test_result_frames.py proj/boxed_piston/tests/test_geomconfig.py proj/boxed_piston/tests/test_geometry.py -q
python -m pytest proj/boxed_piston/tests -q
```

## Gate reruns (after green; report vs previous, no tuning)
| Script | Previous | Expect |
|---|---|---|
| `showcase/step06b2a_gates.py` power consistency S1/S4/S6 | ≤ 0.71% at f_max | similar or lower |
| `showcase/step10a_c17_soak.py` open-domain S4/S6 | decay to −210 dB | same |
| `showcase/step12_report_plot.py` δ_eq/h | S1 0.047, S4 0.045 CHECK; S6 0.018 OK | S1/S4 ≈ 0.0025–0.01 OK; S6 lower, OK |

Step 9 surface gates use the sphere fixture (pinned), so they are unaffected. Log results as A14.
