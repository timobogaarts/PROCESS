# Objective function audit record

Schema adapted from `_audit/schema.md`'s "Constraint record" (the closest existing
template — an objective branch and a constraint body are the same shape: a pure read/
scaling of already-produced `data` fields, selected at assembly time by an external
PROCESS ID). One section per `FiguresOfMerit` value.

---
kind: model-unit
status: reviewed
confidence: high
---

## source

`process/core/solver/objectives.py`, single function `objective_function(i_figure_merit,
data)`, lines 11-105. No stellarator/`istell` special-casing anywhere (checked both by
docstring text and by grepping the function body for `.stellarator.`/`istell` reads —
neither present). Every branch is `objective_metric = <plain read/scale of 1-2 data
fields>`; sign (`objective_sign = np.sign(i_figure_merit)`) is applied once, outside the
branch, to whichever metric was picked.

## why no single `objective_function` node/port

See `objectives.py`'s own module docstring for the full argument (already agreed this
session, `_audit/next_steps.md` §6): the branch selection is a per-run, assembly-time
choice of *which already-produced output is wanted*, not a computation. Reproducing the
`if`/`elif` chain as one traced function would be the same "switch hiding inside one
function" shape this codebase splits by default everywhere else. Ported instead as
sixteen standalone `objective_metric_<id>` functions (named after PROCESS's own
`i_figure_merit` ID, matching `constraints.py`'s `constraint_<id>` convention) plus an
`OBJECTIVE_METRICS` lookup dict from `FiguresOfMerit` to the function that ports it.

## per-branch data footprint / hole-in-MDA

"Hole-in-MDA" here means: does a ported node in this codebase already produce (own, via
a declared `Output`) the field(s) this branch reads? A "no" is not necessarily a bug —
several of these fields (`rmajor` notably) are read everywhere in this codebase as an
`Input` but never appear as anyone's `Output`, consistent with being a genuine free
design/iteration variable in PROCESS (an unknown a solver picks, not a derived
quantity) rather than a missing port. Checked by grepping
`Output(lambda s: s.<area>.<field>)` across `functional_process/models/**` (excluding
tests/`_wip`).

| id | name | field(s) read | producer found? | note |
|---|---|---|---|---|
| 1 | `MAJOR_RADIUS` | `.physics.rmajor` | **no `Output` found anywhere** | read as `Input` throughout (`Build`, `geometry.py`, etc.) but never owned — consistent with `rmajor` being a genuine free design variable (PROCESS iteration variable candidate), not a hole. Not re-verified against `iteration_variables.py`'s ID table this pass — flagged as an open question below, not asserted as fact. |
| 3 | `NEUTRON_WALL_LOAD` | `.physics.pflux_fw_neutron_mw` | yes | `plasma_physics.py` |
| 4 | `P_TF_PLUS_P_PF` | `.tfcoil.tfcmw`, `.pf_power.srcktpm` | `tfcmw`: yes (`tf_coil_power.py`). `srcktpm`: **no `Output` found** | `srcktpm` (PF coil circuit power) — `pf_power.py` itself is out of this session's stellarator scope entirely (no file of that name under `functional_process/models/`); a real hole, not a design-variable case like `rmajor`. |
| 5 | `FUSION_GAIN_Q` | `.current_drive.big_q_plasma` | yes | `stellarator/heating.py` |
| 6 | `COST_OF_ELECTRICITY` | `.costs.coe` | **no `Output` found** | `costs.py`'s node classes are ported as pure functions but almost entirely unregistered per this session's own consolidation audit (see `_audit/next_steps.md`'s alternates-audit item) — real hole, tracked there already, not new. |
| 7 | `CAPITAL_COST` | `.costs.cdirt`, `.costs.concost`, `.costs.ireactor` (switch) | **no `Output` found for either** | same `costs.py` gap as id 6. |
| 8 | `ASPECT_RATIO` | `.physics.aspect` | yes | `geometry.py` |
| 9 | `DIVERTOR_HEAT_LOAD` | `.divertor.pflux_div_heat_load_mw` | yes | `stellarator/divertor.py` |
| 10 | `TOROIDAL_FIELD` | `.physics.b_plasma_toroidal_on_axis` | **no `Output` found** | read everywhere (`structure.py`, `density_limits.py`, ...) but not owned — same "possible free/input variable" caveat as `rmajor`, not independently confirmed this pass. |
| 11 | `TOTAL_INJECTED_POWER` | `.current_drive.p_hcd_injected_total_mw` | yes | `stellarator/heating.py` |
| 14 | `PULSE_LENGTH` | `.times.t_plant_pulse_burn` | **no `Output` found** | `times.py`/pulse-timing model is out of this session's stellarator scope entirely — real hole. |
| 15 | `PLANT_AVAILABILITY_FACTOR` | `.costs.i_plant_availability` (switch, input not output), `.costs.f_t_plant_available` | yes | `availability.py` — three separate producer classes (per-availability-model arms), consistent with `availability.py`'s already-known alternate-arm registration gaps (see the alternates audit). |
| 16 | `MIN_R0_MAX_TAU_BURN` | `.physics.rmajor`, `.times.t_plant_pulse_burn` | see ids 1, 14 | |
| 17 | `NET_ELECTRICAL_OUTPUT` | `.heat_transport.p_plant_electric_net_mw` | **no `Output` found** | `electric_production.py` reads it but a producer wasn't found this pass — flagged, not fully traced (this file's own output set wasn't exhaustively checked; may be a false negative from the grep pattern used, see open questions). |
| 18 | `NULL_FIGURE_OF_MERIT` | none | n/a | `f(x) = 1`, no data access at all. |
| 19 | `MAX_Q_MAX_T_PLANT_PULSE_BURN` | `.current_drive.big_q_plasma`, `.times.t_plant_pulse_burn` | see ids 5, 14 | |

**Net**: this port's own pure functions are all tier-1 (plain arithmetic, fully
verified against PROCESS below) regardless of hole-in-MDA status — a hole here means
"this objective can't yet be assembled into a real `Optimise` problem against the
current graph," not "this port is wrong." None of the holes found are new; they're the
same costs.py/`pf_power.py`/`times.py`/`electric_production.py` coverage gaps
already tracked elsewhere in this session's audit trail.

## two real PROCESS docstring inaccuracies found (not code bugs)

1. `objective_function`'s own inline docstring lists **both** id 16 and id 19 as "Major
   radius/burn time." The `FiguresOfMerit` enum (`process/data_structure/numerics.py:
   105-113`) has the real, non-duplicate descriptions, and the actual `if`/`elif` bodies
   compute genuinely different formulas (16 reads `rmajor`, 19 reads `big_q_plasma` —
   confirmed by reading `objectives.py:92-103` directly, not inferred from either
   docstring). Ported as the two distinct formulas the code computes; the stale
   docstring text is not reproduced.
2. `objective_function`'s top docstring says the id-15 precondition is "not used with
   `i_plant_availability=1`" — imprecise; the real condition
   (`objectives.py:83-90`) is "must not be `AvailabilityModel.USER_INPUT` (0)," i.e.
   any of `WARD_TAYLOR`(1)/`MORRIS`(2)/`ST`(3) is fine, not only `1`. Ported against the
   actual code, not the docstring's looser phrasing.

## switches touched

- `.costs.ireactor` (id 7): static, selects `cdirt` vs `concost`.
- `.costs.i_plant_availability` (id 15): static, precondition — see `objectives.py`'s
  own docstring and `objective_metric_15`'s `Raises` section. Same taxonomy as
  `constraints.py`'s `constraint_24` (`i_beta_component`'s `ValueError` on an invalid
  static switch value) — a real `raise`, not a domain-guard NaN, since nothing at this
  branch point is traced.

## cottax node

None — deliberately. See `objectives.py`'s module docstring: branch selection is an
assembly-time query (`OBJECTIVE_METRICS[figure_of_merit]`), not a node. Each
`objective_metric_<id>` is a plain pure function a later `Optimise` assembly would bind
as that problem's `objective` `In`, reading whichever already-produced `VarPath`(s) the
table above lists.

## open questions

1. `.physics.rmajor` and `.physics.b_plasma_toroidal_on_axis` (ids 1/16, 10) have no
   `Output` producer found by this pass's grep sweep — is either a genuine PROCESS
   iteration variable (free design unknown, no producer needed), or an actual porting
   gap? Not resolved here; check against `process/core/solver/iteration_variables.py`'s
   `ITERATION_VARIABLES` table before assuming either way.
2. `.heat_transport.p_plant_electric_net_mw` (id 17) — no producer found, but this
   pass's search was a single grep pattern (`Output(lambda s: s.<path>)` on one line);
   `electric_production.py` reads the field, so a multi-line `Output(...)`
   definition there is plausible and would be a false negative here. Worth a direct
   check before treating this as a confirmed hole.
3. The `costs.py`/`pf_power.py`/`times.py` coverage gaps (ids 4, 6, 7, 14) are not new
   findings — they duplicate what this session's alternates-audit already has queued.
   Listed here only so this record's own hole-in-MDA table is self-contained, not as a
   second, independent tracking entry.
