---
kind: model-unit
status: draft
confidence: medium
---

**Ported (both reachable arms; since 2026-08-30 the temperature margin and the PF
coils' strand critical current density too).**
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
`CSTemperatureMarginWstNb3Sn`) — see § "2026-08-30 — the temperature margin" below. And
`calculate_pf_strand_critical_current_density` + `PFStrandCriticalCurrentDensity`, one
occupant of `.tokamak.pf_coil.strand_critical_current` — the module's only unit that is
`pfcoil()`'s rather than `ohcalc`'s; see § "2026-08-30 — the PF coils' strand critical
current density" below.

## source

`process/models/pfcoil.py:3577-3684` — two `superconpf` calls (end of flat top,
beginning of pulse) and the four `.pf_coil.*` fields a constraint reads;
`superconpf` itself is `:4641-4924`. Since 2026-08-30 also `:871-904`, `pfcoil()`'s own
`superconpf` call for the PF coils.

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

## 2026-08-30 — the PF coils' strand critical current density, Account 222.2

**Ported**, and it is the first thing in this module that is not `ohcalc`'s.
`calculate_pf_strand_critical_current_density` +
`PFStrandCriticalCurrentDensity` (`.tokamak.pf_coil.strand_critical_current`), one
tier-1 contract against PROCESS's own `superconpf` at `isumat = 3`.

**source**: `process/models/pfcoil.py:871-904` — the `bmax = max(|b_pf_coil_peak[i]|,
|bpf2[i]|)` at `:871-874`, the `superconpf` call at `:877-892`, and the strand branch at
`:898-904`. The arm reached is `OLD_LUBELL_NBTI` (`:4773-4784`), whose critical surface
is `superconductors.jcrit_nbti` at `bc20m = 15.0`, `tc0m = 9.3`, `c0 = 1e10` — three
literals inside the arm, named as module constants here.

**data footprint**

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_coil.b_pf_coil_peak[5]` | read | explicit-arg | inner-edge field of the **last** PF coil |
| `.pf_coil.bpf2[5]` | read | explicit-arg | outer-edge field of the same coil |
| `.tfcoil.tftmp` | read | explicit-arg | `superconpf`'s `temp_pf_peak_field` |
| `.pf_coil.fcupfsu` | read | explicit-arg | strand copper fraction, the branch's only factor |
| `.pf_coil.j_crit_str_pf` | write | — | scalar, last-write-wins over the loop |

**Why index 5 and not an array.** The assignment at `:900`/`:902` carries no index: it
is inside `pfcoil()`'s group-then-coil loop and overwrites a **scalar** on every one of
the six passes, so the value PROCESS leaves in `.pf_coil.j_crit_str_pf` is the last
coil's. Owning the six-entry array PROCESS never stores would be inventing a variable;
computing the five overwritten values would be work no edge leaves. This is `masses.md`'s
"dropped rather than computed and discarded" rule applied to the one part of that block
whose output does now have a reader.

**Four reads, not `superconpf`'s fifteen.** The NbTi arm's `j_crit_sc` is the third
return and comes straight off the fit; `fhe` and `fcu` scale it into `j_crit_cable`,
which is the *first* return and unread here. So `.pf_coil.f_a_pf_coil_void` is a read of
`superconpf` and not a read of this unit, and `fcupfsu` enters only through
`j_crit_sc * (1 - fcu)` at `:903`. The harness adapter still runs the whole real
function, temperature-margin root find included, so a regression in arm selection shows
as a value mismatch rather than a division by zero.

**Not a family.** `.pf_coil.i_pf_superconductor` is `3` on both ported arms of
`_pf_coil_system_arm` (`_pf_coil_system_deviations`' `-6` refuses everything else), so
there is one critical surface and the slot takes an instance default — unlike its two CS
neighbours, whose switch has two reachable values. Written as the `else` half of the
`:898` strand branch for the same reason
`calculate_cs_strand_critical_current_density` is: that branch tests
`i_cs_superconductor in {2, 6, 8}` — the **CS** switch deciding the **PF** value, which
is PROCESS as written and reproduced as written — and both values that reach this package
are outside the set.

**Why now.** `.pf_coil.j_crit_str_pf` is a field PROCESS computes (`1.1017899e9` A/m^2
on `large_tokamak_nof`) that nothing owned, and Account 222.2's `PER_KAM` arm reads it.
`_audit/cost_boundary_inputs.md` §13.2 refused to register `PfMagnetCost` partly for
that: as one node it would have taken `.costs.c2222` off the missing-producer pin and
put this field on. Landed together with the account's split into
`PfMagnetCostPerKg`/`PerKam`, so the pin went 2 → 1 with nothing arriving.

**Value**: exact against PROCESS on the reference point
(`1101789868.9092896` both sides, bit-for-bit).

**Still unowned**: `.pf_coil.j_pf_wp_critical`, `superconpf`'s *first* return for the six
PF slots. The reason has changed from "unported" to "no reader" — nothing in the graph
consumes it and PROCESS only prints it (`outpf`, `pfcoil.py:2570-2603`) — so it is
invisible to `unproduced_but_computed` and owning it would mean six critical-surface
evaluations no edge leaves.

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
