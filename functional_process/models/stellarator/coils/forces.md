---
kind: model-unit
status: reviewed
confidence: high
---

**Ported.** `forces.py` / `test_forces.py`, all 7 functions tier-1, tests passing
(fuzz only — no matching PROCESS unit test found for any of these 7 in
`tests/unit/models/stellarator/test_stellarator.py`, grepped by function name). Two of
the seven (`calculate_max_force_density`, `calculate_maximum_stress`) get `cottax`
nodes; the other five do not — see below.

## source
`process/models/stellarator/coils/forces.py` (133 lines, full file in scope). 7
module-level functions, all called only from `coils/calculate.py` (registry unit #9)'s
`st_coil`, lines ~121-136. No calls into or out of `coils.py`/`mass.py`/`quench.py`.

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.a_tf_wp_no_insulation` | read | explicit-arg | `calculate_max_force_density`, `calculate_max_lateral_force_density`, `calculate_max_radial_force_density` — passed as an explicit arg at all 3 call sites already (not read via `data` in the source) |
| `.stellarator_config.stella_config_max_force_density` | read | explicit-arg | `calculate_max_force_density` |
| `.stellarator.f_st_i_total` | read | explicit-arg | all 7 functions |
| `.stellarator.f_st_n_coils` | read | explicit-arg | all 7 functions |
| `.tfcoil.b_tf_inboard_peak_symmetric` | read | explicit-arg | all 7 functions |
| `.stellarator_config.stella_config_wp_bmax` | read | explicit-arg | all 7 functions |
| `.stellarator_config.stella_config_wp_area` | read | explicit-arg | `calculate_max_force_density`, `calculate_max_lateral_force_density`, `calculate_max_radial_force_density` |
| `.tfcoil.max_force_density` | write, then **read by a different function** (`calculate_maximum_stress`) | implicit-io-via-callee-adjacent (see note) | `calculate_max_force_density` writes this; `calculate_maximum_stress`, a *separate* function in the same file, reads it back off `data` rather than receiving it as an argument. Not quite `implicit-io-via-callee` (that label is for a deep-copied proxy object) and not `local-intermediate` (the write and read are in different functions, not the same straight-line body) — this is the shape `naming_convention.md`'s open questions flagged as needing a category: two *sibling* pure functions chained only through `data`. Ported as an explicit argument (see proposed signatures) rather than inventing a new classification label unilaterally. |
| `.tfcoil.dr_tf_wp_with_insulation` | read | explicit-arg | `calculate_maximum_stress` |
| `.tfcoil.sig_tf_wp` | write | explicit-arg | `calculate_maximum_stress`'s output |
| `.stellarator_config.stella_config_max_force_density_mnm` | read | explicit-arg | `calculate_max_force_density_mnm` — **return value feeds only `write()` (unit #13, confirmed reporting-only), never written to `data`** |
| `.stellarator_config.stella_config_max_lateral_force_density` | read | explicit-arg | `calculate_max_lateral_force_density` — same "report-only consumer" note |
| `.stellarator_config.stella_config_max_radial_force_density` | read | explicit-arg | `calculate_max_radial_force_density` — same |
| `.stellarator_config.stella_config_centering_force_{max,min,avg}_mn` | read | explicit-arg | the three `calculate_centering_force_*_mn` functions — same "report-only consumer" note |
| `.stellarator_config.stella_config_coillength` | read | explicit-arg | the three `calculate_centering_force_*_mn` functions |
| `.tfcoil.n_tf_coils` | read | explicit-arg | the three `calculate_centering_force_*_mn` functions |
| `.tfcoil.len_tf_coil` | read | explicit-arg | the three `calculate_centering_force_*_mn` functions |

## proposed signature(s)

All seven, tier-1, as ported (see `forces.py` — omitted here to avoid duplicating 150
lines; signatures follow the data-footprint table above 1:1, PROCESS's own names kept).

## cottax node

Only `MaxForceDensity` (`calculate_max_force_density`) and `MaximumStress`
(`calculate_maximum_stress`) are wrapped as `ExplicitFunction`s, registered in
`functional_process/total_process.py`. `MaximumStress` reads `max_force_density` as a
real graph edge from `MaxForceDensity`'s output — closing exactly the "two sibling
functions chained through `data`" gap noted above, mechanically, once both are declared
nodes in the same graph.

The other five (`calculate_max_force_density_mnm`, `calculate_max_lateral_force_density`,
`calculate_max_radial_force_density`, `calculate_centering_force_max_mn`,
`calculate_centering_force_min_mn`, `calculate_centering_force_avg_mn`) are **not**
wrapped: none of their return values are written to `data` anywhere — `coils/calculate.py`
consumes every one of them only as a local passed into `write()` (unit #13, confirmed
reporting-only). Same convention already applied to `stellarator_D_structure.md`'s
`calculate_intercoil_mass_scaling_reference`: a function whose only consumer is the
report gets no node, since there is nothing downstream in the computational graph for it
to feed.

## tier signal
**Tier 1**, all seven. No internal solve, no calls into other models, no switches, no
data-dependent control flow.

## switches touched
None.

## calls into other models
None.

## JAX-difficulty flags
None. Plain scalar arithmetic throughout.

## open questions
1. Whether the "sibling functions chained only through `data`" pattern
   (`calculate_max_force_density` → `calculate_maximum_stress`) recurs often enough
   elsewhere to deserve its own classification label alongside the existing six — this
   is the second time it's been noted (see `naming_convention.md`'s open questions);
   recommend deciding once a third example turns up.
2. Whether the five report-only functions should eventually get nodes anyway once a
   real reporting layer is designed (a node whose only reader is "the printed output"
   is still a node in a fuller sense of "the graph", just not one anything else in the
   *computation* depends on) — left as a policy question, not resolved here, consistent
   with this audit's standing "reporting is out of scope" stance.
