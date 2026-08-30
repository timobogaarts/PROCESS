---
kind: model-unit
status: draft
confidence: medium
---

**Ported (both reachable arms, and since 2026-08-30 the temperature margin too).**
`models/pfcoil/superconductor.py` / `tests/functional_process/models/pfcoil/
test_superconductor.py`: `calculate_cs_critical_current_density`,
`calculate_cs_critical_current_density_iter_nb3sn`,
`calculate_cs_critical_current_density_wst_nb3sn`,
`calculate_cs_strand_critical_current_density` — tier-1, five contracts (three until
2026-08-30). Two cottax occupants of one slot (`.tokamak.cs_coil.critical_current`):
`CSCriticalCurrentDensitiesIterNb3Sn` and `CSCriticalCurrentDensitiesWstNb3Sn`. Since
2026-08-30, also `calculate_cs_superconductor_current_density`,
`calculate_cs_temperature_margin_iter_nb3sn` and
`calculate_cs_temperature_margin_wst_nb3sn`, with two occupants of a **second** slot,
`.tokamak.cs_coil.temperature_margin` (`CSTemperatureMarginIterNb3Sn`,
`CSTemperatureMarginWstNb3Sn`) — see § "2026-08-30 — the temperature margin" below.

## source

`process/models/pfcoil.py:3577-3684` — two `superconpf` calls (end of flat top,
beginning of pulse) and the four `.pf_coil.*` fields a constraint reads;
`superconpf` itself is `:4641-4924`.

## what it ports

`superconpf`'s `ITER_NB3SN` and `WST_NB3SN` arms plus `ohcalc`'s cable-space-to-
cross-section scaling. Two occupants, the second a one-`staticmethod` subclass of the
first — identical reads, identical outputs.

**`indat.CS_SUPERCONDUCTOR` is total and has no `UNPORTED` entries**, the second such
registry after `SHIELD_HALF_HEIGHT`. `superconpf` dispatches on eight values but
`_pf_coil_system_arm` refuses six of them before this slot is built; only `1` and `5`
survive its `(i_pf_superconductor, i_cs_superconductor)` pair, and both are written. The
first draft of this wave wrote only the ITER arm and listed the other seven as
`UNPORTED` — which **turned `low_aspect_ratio_DEMO.IN.DAT` from a machine that assembles
into a `NotImplementedError`**, caught by
`test_machine.test_a_switch_that_decides_two_slots_decides_both`. A new slot may not
narrow the set of files the port accepts; the WST arm is written because of that, not
because a constraint asked for it.

**The switch is asked twice on purpose.** `_pf_coil_system_arm` reads it as half of the
pair that selects the *masses* occupant, and that pair's arm `1` covers both surviving
values — because "which `.tfcoil.dcond` element a mass reads" is a different question
from "which critical surface a current density comes from". Reusing the first answer
would have given a WST Nb3Sn CS the ITER Nb3Sn critical surface.

`j_pf_wp` is **not** declared: `ohcalc` computes it at both call sites and neither ported
arm reads it (only `BI2212` does), so declaring it would invent an edge from
`.pf_coil.c_pf_cs_coils_peak_ma`. `.pf_coil.j_pf_wp_critical` is left unowned; the CS
slot is `j_cs_critical_pulse_start` under a second name and the six PF slots have no
producer. **+5 MDA-harness agreements**, all five exact.

## 2026-08-30 — the temperature margin, constraint 60

**Ported. The section below it is the plan this section carried out**, kept verbatim
because the one place it was wrong is worth seeing next to what happened.

`.pf_coil.temp_cs_superconductor_margin` was a boundary `0.0` against PROCESS's
`3.4208032` K on `tests/regression/input_files/large_tokamak_nof.IN.DAT` — found by
`boundary.unproduced_but_computed`, which measures PROCESS's write set for one pipeline
pass and reports which of the port's boundary `input`s are in it. Constraint 60
(`temp_cs_superconductor_margin >= .tfcoil.temp_cs_superconductor_margin_min`) was
comparing the frozen zero against a real bound.

**What is ported**: `min(tmarg1, tmarg2)` (`pfcoil.py:3679`), where each `tmarg` is
`superconpf`'s fourth return at one of the two peak fields — the root of
`j_crit_sc(T) - j_sc = 0` less `temp_pf_peak_field` (`:4906-4922`). `j_sc` is
`superconpf`'s own `j_pf_wp / (1-fhe) / (1-fcu)` (`:4874-4878`) with `ohcalc`'s spelling
of `j_pf_wp` folded in, which is `calculate_cs_superconductor_current_density`.

**Where the plan below was wrong, and it is the interesting part.** Steps (1), (2) and
(4) predicted a declared `ImplicitFunction`/`RootFind` pair with its own driver, and a
tier-2 unit. What landed is an `ExplicitFunction` whose body is
`models/tfcoil/superconducting.py::solve_current_sharing_temperature`, a fixed-trip
`jax.lax.fori_loop` replicating `scipy.optimize.newton`'s secant branch step for step.
Step (3) — *use the shared driver, take both margins together* — is what decided it: the
TF coil's margin was ported first and took that shape, for the reason its own docstring
gives. The endpoint **is** the quantity constraint 60 compares against PROCESS's, so
matching scipy's stopping rule is what makes the comparison a value test rather than a
tolerance negotiation, and that is a tier-1 claim, not a tier-2 one. So the unit is tier
1 after all, and it agrees with PROCESS to 2 ulp on the reference point.

**New reads.** One: `.pf_coil.c_pf_cs_coils_peak_ma`, which the note two paragraphs above
says the critical-current arms must not declare because they never use it. That stands —
the edge is declared by the margin node, which does use it, and that is why the margin is
a slot of its own rather than a fifth output of `critical_current`. Same split as the TF
coil's `cicc_superconductor_properties` / `tf_superconductor_temperature_margin`.

**`indat.CS_TEMPERATURE_MARGIN`** is the second registry keyed on
`.pf_coil.i_cs_superconductor`, total over the two reachable values for the same reason
`CS_SUPERCONDUCTOR` is, and both arms are written — the WST one because
`low_aspect_ratio_DEMO.IN.DAT` would otherwise stop assembling, exactly as it did the
first time this file was written with one arm.

## open questions — constraint 60, `.pf_coil.temp_cs_superconductor_margin`

*(Superseded 2026-08-30 by the section above; kept as written.)*

**Not ported, and this is the §11.5 row this half of the wave leaves open.**
`superconpf` finishes with `scipy.optimize.newton`'s secant iteration on
`j_crit_sc(T) - j_sc = 0` (`pfcoil.py:4894-4921`; `fprime=None`, `x1 = 2*T_op`,
`tol = rtol = 1e-6`, `maxiter = 50`, `disp=False`). In cottax's terms that is an
`ImplicitFunction`/`RootFind` pair, of exactly the shape
`models/stellarator/coils/coils.py::Intersect` already has, with a concrete
`AbstractDriver` beside it.

**Next steps**, in order: (1) write the residual as an `ImplicitFunction` over `T` with
the arm's critical-surface function, reusing the `_critical_surface` `staticmethod` hook
the two occupants already carry; (2) declare a `RootFind` problem and a secant driver
matching PROCESS's tolerances, so the *algorithm* is the thing being compared, not just
the root; (3) own `.pf_coil.temp_cs_superconductor_margin = T_zero - T_op`; (4) tier 2,
not tier 1 — there is an internal iteration on both sides.

**It is deliberately left for one commit rather than two**: `.tfcoil.temp_tf_superconductor_
margin` (constraint 36) is the *same solve on the same function* from
`superconducting.py`, and the two halves were being ported by concurrent agents. One
shared driver written once is worth more than two that have to be merged. Whoever takes
it should take both.

## record provenance

Written 2026-08-27 inside `pfcoil/fields.md` § "the CS chain", because the wave that
wrote this module was asked to leave `unit_registry.md` alone while two sibling agents
had it open. Split out to its own record with its own registry row on 2026-08-29; the
material is unchanged apart from the heading levels.
