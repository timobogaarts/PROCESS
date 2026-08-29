---
kind: model-unit
status: draft
confidence: medium
---

**Ported (both reachable arms; the temperature-margin root find is not).**
`models/pfcoil/superconductor.py` / `tests/functional_process/models/pfcoil/
test_superconductor.py`: `calculate_cs_critical_current_density`,
`calculate_cs_critical_current_density_iter_nb3sn`,
`calculate_cs_critical_current_density_wst_nb3sn`,
`calculate_cs_strand_critical_current_density` — tier-1, three contracts. Two cottax
occupants of one slot (`.tokamak.cs_coil.critical_current`):
`CSCriticalCurrentDensitiesIterNb3Sn` and `CSCriticalCurrentDensitiesWstNb3Sn`.

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

## open questions — constraint 60, `.pf_coil.temp_cs_superconductor_margin`

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
