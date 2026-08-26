---
kind: model-unit
status: draft
confidence: high
---

**Ported.** `confinement_time.py` now declares `calculate_confinement_time` and
`calculate_double_and_triple_product` (registry unit #10's exact scope) plus the 48
individual scaling-law statics they call transitively, all within this same source file,
and `calculate_iter_physics_basis_elongation` (out-of-file, see "calls into other
models"). **Six cottax slots, not two nodes**: the composite `ConfinementTime` this
line originally described was split into `power_loss`/`scaling`/`tail` when its three
static switches became occupants, and `inputs`, `elongation` and
`double_and_triple_product` sit beside them. See "## cottax node".

## source

`process/models/physics/confinement_time.py` (4185 lines). Registry unit #10, in-scope
methods `calculate_confinement_time` (L58-1035) and `calculate_double_and_triple_product`
(L1037-1065). `find_other_h_factors`, `output_confinement_time_info`,
`output_confinement_comparison` are explicitly out of scope per the registry row (not
called by `calculate_confinement_time` itself — `find_other_h_factors` *calls*
`calculate_confinement_time` in a `scipy.optimize.root_scalar` loop, the reverse
direction, and the two `output_*` methods are reporting shells).

`calculate_confinement_time`'s body calls 48 same-file `@staticmethod`/`@classmethod`
scaling-law functions (one per `ConfinementTimeModel` enum value, `physics_variables.py:
90-353`) — all reachable only from this one dispatcher, so all 48 are in scope
transitively, same "the registry names the entry point, not everything it reaches"
situation `fusion_reactions.md`/`plasma_physics.md` already established. Verified
exhaustively (not sampled): every one of the 48 already takes plain arguments with no
`self.data` access of its own — the pure/impure split for this unit is entirely at the
`calculate_confinement_time` level, not distributed across the 48.

## A latent PROCESS bug, ported faithfully: `KAYE_GOLDSTON`

`calculate_confinement_time`'s `KAYE_GOLDSTON` branch (`i_confinement_time == 5`,
source L268-278) calls `kaye_goldston_confinement_time` with positional arguments
`(cur_plasma_ma, rmajor, rminor, kappa, nd_plasma_electron_line_20,
b_plasma_toroidal_on_axis, m_fuel_amu, p_plasma_loss_mw)` — but that function's own
declared parameter order (source L1598-1606) is `(kappa95, cur_plasma_ma, n20, rmajor,
afuel, b_plasma_toroidal_on_axis, rminor, p_plasma_loss_mw)`. Binding positionally, six
of the eight arguments land on the wrong physical quantity: the value meant as `kappa95`
receives the plasma current, the value meant as `cur_plasma_ma` receives `rmajor`, the
value meant as `n20` receives `rminor`, the value meant as `rmajor` receives the
elongation `kappa`, the value meant as `afuel` receives the line density, and the value
meant as `rminor` receives the fuel mass number. Only `b_plasma_toroidal_on_axis`
(position 6) and `p_plasma_loss_mw` (position 8) happen to land correctly.

Found by an exhaustive AST-level cross-check of every one of the 48 call sites against
its callee's own declared parameter order (script-assisted, not spot-checked) — every
*other* mismatch the same check flagged turned out to be a benign rename at a matching
position (`afuel`/`m_fuel_amu`, `dene20`/`n20`, `q`/`q95`, `aion`/`m_ions_total_amu`, all
verified against each function's own docstring to denote the same physical quantity).
`KAYE_GOLDSTON` is the only genuine positional scramble.

**Reproduced exactly, not fixed** — same policy `radiation_power.md` already set for its
own latent bug. The port's `kaye_goldston_confinement_time` (standalone) is correct
against its own parameter names and independently legacy-tested; the composite
`calculate_confinement_time`'s `KAYE_GOLDSTON` branch calls it with the identical
(scrambled) positional order PROCESS does, with an inline comment marking it as a known
upstream bug.

## A dead branch: `USER_INPUT`

**Confirmed empirically against the live PROCESS reference, not just read off source.**
`i_confinement_time == 0` (`ConfinementTimeModel.USER_INPUT`, meant to read
`.physics.tauee_in` directly as the confinement time) **always raises**
`ProcessValueError("Illegal value for i_confinement_time")` in real PROCESS. Cause:
source's dispatch opens with two independent `if` statements rather than one `if` /
`elif` chain (L220-223 `if model == USER_INPUT: ...`, L228-230 `if model ==
NEO_ALCATOR: ...`, and only from L236 onward does the chain become `elif`). So the
`USER_INPUT` arm's assignment to `t_electron_confinement` is always immediately
discarded — execution falls through the second `if` (false, since `model` is 0) and then
every `elif` in the chain (all false), landing on the final `else: raise`. Verified by
instantiating a real `PlasmaConfinementTime` bound to a `DataStructure` and calling
`calculate_confinement_time(..., i_confinement_time=0, ...)` directly: it raises, for any
input.

**Reproduced faithfully**: the port's `calculate_confinement_time` raises `ValueError` for
`i_confinement_time == 0` too, with an inline comment. `tauee_in` is still threaded
through the port's signature (an unavoidably dead argument for this one switch value,
matching source), since the promoted-implicit-read decision was made before this branch's
unreachability was discovered, and keeping it costs nothing while documenting exactly
what PROCESS itself would also compute-and-discard if the raise weren't there first.

## A dead branch: `PAZ_SOLDAN_NT`

`ConfinementTimeModel.NCST` and `ConfinementTimeModel.PAZ_SOLDAN_NT`
(`physics_variables.py:344-353`) are both declared with enum value **51**. Python's
`IntEnum` makes the second declaration an *alias* of the first — `ConfinementTimeModel(51)
is ConfinementTimeModel.NCST` — so `model == ConfinementTimeModel.PAZ_SOLDAN_NT` is
identical to `model == ConfinementTimeModel.NCST` and the `NCST` `elif` (tried first in
source order) always wins. `calculate_confinement_time`'s `PAZ_SOLDAN_NT` arm
(source L941-947) is therefore dead code — no value of `i_confinement_time` can ever
reach it. Also affects `N_CONFINEMENT_SCALINGS = len(ConfinementTimeModel)`
(`physics_variables.py:407`), which counts 51 canonical members, not 52 — a second,
independent symptom of the same aliasing, used by `output_confinement_comparison`'s
scan range (out of this unit's scope).

**Reproduced faithfully**: the port's dispatch has no `PAZ_SOLDAN_NT` arm at all — `NCST`
is the only branch bound to `i_confinement_time == 51`, matching what PROCESS actually
executes. `paz_soldan_nt_confinement_time` is still ported and independently
legacy-tested as a standalone formula (a legitimate, self-consistent scaling law in its
own right — the bug is only in how the enum numbering makes it unreachable via the
switch), just never wired into the composite dispatcher.

## A minor naming inconsistency (not a value bug): `iter_pb98py_confinement_time`

This function's own parameter is named `kappa` (docstring: "Plasma separatrix
elongation"), but `calculate_confinement_time`'s one call site always feeds it
`.physics.kappa_ipb` — the IPB-corrected elongation every sibling IPB98-family scaling
(`iter_ipb98y1..y4`, `murari`, `petty08`, `menard_nstx*`, `itpa20*`) receives under a
parameter *named* `kappa_ipb`. The value supplied is consistent with the whole family, so
this reads as a naming/docstring slip in the one function whose parameter was never
renamed to match, not a value-correctness bug. Not reproduced-as-a-bug (there is nothing
to reproduce): the port's `iter_pb98py_confinement_time` keeps the source's `kappa`
parameter name unchanged, and the composite dispatcher feeds it `kappa_ipb`, exactly as
source does.

## The NON_IGNITED power-loss arm, and what refusing it had already found

*(Added by the tokamak dispatch pass. The `## cottax node` section below has since been
rewritten to describe the head/law/tail split rather than the deleted composite
`ConfinementTime`; the authoritative node list is
`models/physics/confinement_time.py` itself.)*

`PlasmaPowerLoss` — the head of `calculate_confinement_time`, extracted so that
`i_plasma_ignited` and `i_rad_loss` can be occupants rather than static kwargs — had
exactly one occupant, `PlasmaPowerLossIgnitedCoreRadiation`, written for the arm both
stellarator reference runs use (`stellarator_helias.IN.DAT:126` sets
`i_plasma_ignited = 1`). `_audit/tokamak_boundary.md` § "What blocked the real file"
records what happened when a conventional tokamak was first assembled:
`large_tokamak_eval.IN.DAT` **did not assemble**, and was refused at
`.physics.confinement_time.power_loss`, because the file never sets `i_plasma_ignited`
and therefore takes PROCESS's own default `0` (`physics_variables.py:881`, `NON_IGNITED`).
It assembles now — this arm is written and registered, and `_slot_occupant`'s
`PLASMA_POWER_LOSS` registry has two entries where it had one.

That refusal is worth keeping in the record because of what it *was*. It is not a
missing model: `i_plasma_ignited` is not one of the seventeen new decisions
`tokamak_scope.md` counted, precisely because it is not a new switch — it is one this
port already read, pinned to the single value two stellarator runs happen to share. The
occupant split is what turned "a value we never varied" into "a slot with one arm
filled", and the boundary measurement is what made the empty arm fail loudly instead of
being silently answered by the wrong formula.

**`PlasmaPowerLossNonIgnitedCoreRadiation` is that arm**, and it is as small as the
prediction said: the same formula with one extra term, reading exactly one extra
variable.

| | reads |
|---|---|
| `PlasmaPowerLossIgnitedCoreRadiation` | `.physics.f_p_alpha_plasma_deposited`, `.p_alpha_total_mw`, `.p_non_alpha_charged_mw`, `.p_plasma_ohmic_mw`, `.pden_plasma_core_rad_mw`, `.vol_plasma` |
| `PlasmaPowerLossNonIgnitedCoreRadiation` | the same six, **plus `.current_drive.p_hcd_injected_total_mw`** |

The extra term is `process/models/physics/confinement_time.py:143-144` — `if
PlasmaIgnitionModel(i_plasma_ignited) == NON_IGNITED: p_plasma_loss_mw +=
p_hcd_injected_total_mw`, guarded by that switch and nothing else. Both occupants own
`.physics.p_plasma_loss_mw` and neither reads `.physics.pden_plasma_rad_mw`, which is the
`FULL_RADIATION` arm's read.

`i_rad_loss` is **not** a second unwritten arm here: `large_tokamak_eval.IN.DAT` does not
set it either, so it takes its own default `1` (`physics_variables.py:954`, `CORE_ONLY`)
— the same value both stellarator runs use, and the value
`PlasmaPowerLossIgnitedCoreRadiation` and `ConfinementTailCoreRadiation` were already
written for. So the tokamak needed one new class, not two, and the `.physics.
confinement_time.tail` slot needed none.

**The one read is no longer a boundary entry.** `tokamak_boundary.md` priced this arm as
"adds a second reader on `.current_drive.p_hcd_injected_total_mw` and no new variable at
all, since nothing in this port produces it either way". As of the same pass, something
does: `models/physics/current_drive.py::HcdInjectedPowerTotal`
(`_audit/units/models/physics/current_drive.md`). The two halves of that section's
prediction landed together.

**Still UNPORTED**, and each is a real PROCESS branch reading genuinely different
variables:

| `i_plasma_ignited` | `i_rad_loss` | status |
|---|---|---|
| `1` IGNITED | `1` CORE_ONLY | ported (`PlasmaPowerLossIgnitedCoreRadiation`) |
| `0` NON_IGNITED | `1` CORE_ONLY | ported (`PlasmaPowerLossNonIgnitedCoreRadiation`) |
| `1` IGNITED | `0` FULL_RADIATION | UNPORTED — subtracts `.physics.pden_plasma_rad_mw` instead of `.pden_plasma_core_rad_mw`; a read neither written arm makes |
| `0` NON_IGNITED | `0` FULL_RADIATION | UNPORTED — same read, plus the injected-heating term |
| `1` IGNITED | `2` NO_RADIATION | UNPORTED — subtracts nothing, so it reads neither radiation density *and* not `.physics.vol_plasma` for that term |
| `0` NON_IGNITED | `2` NO_RADIATION | UNPORTED — same, plus the injected-heating term |

The three `ConfinementTail` arms are counted separately (`ConfinementTailCoreRadiation`
is written; `FULL_RADIATION` and `NO_RADIATION` are not), since `i_rad_loss` decides that
node a second time and there its arms read genuinely different variables — see
`confinement_from_scaling`'s docstring.

Validated by `tests/functional_process/models/physics/test_confinement_time.py`:
`test_power_loss_occupants_match_process` calls each occupant with the reads it declares
and diffs the result against `PlasmaConfinementTime.calculate_confinement_time`'s own
`p_plasma_loss_mw` at the same point, for both arms — the one boundary PROCESS exposes
for a function it does not itself have.
`test_ignition_switch_decides_exactly_one_read_of_the_power_loss_head` pins that the arm
added one read and nothing else.

## data footprint

`calculate_confinement_time`'s own explicit signature (25 arguments, source-order
unchanged) plus 9 further arguments promoting implicit `self.data.physics.*`/
`self.data.current_drive.*` reads/writes the source performs mid-body — none of these 9
are in the source method's own parameter list, all are reached by grepping every
`self.data.` occurrence within the method body (L58-1035), not sampled.

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.m_fuel_amu` | read | explicit-arg | |
| `.physics.p_alpha_total_mw` | read | explicit-arg | |
| `.physics.aspect` | read | explicit-arg | |
| `.physics.b_plasma_toroidal_on_axis` | read | explicit-arg | |
| `.physics.nd_plasma_electrons_vol_avg` | read | explicit-arg | |
| `.physics.nd_plasma_electron_line` | read | explicit-arg | |
| `.physics.eps` | read | explicit-arg | |
| `.physics.hfact` | read | explicit-arg | |
| `.physics.i_confinement_time` | read | explicit-arg (switch) | not a `VarPath` on the pure port's own terms per `naming_convention.md` — a plain Python `int` used for ordinary (non-traced) branching; `static_argnames` in the harness |
| `.physics.i_plasma_ignited` | read | explicit-arg (switch) | same treatment |
| `.physics.kappa` | read | explicit-arg | |
| `.physics.kappa95` | read | explicit-arg | |
| `.physics.p_non_alpha_charged_mw` | read | explicit-arg | |
| `.current_drive.p_hcd_injected_total_mw` | read | explicit-arg | cross-area |
| `.physics.plasma_current` | read | explicit-arg | |
| `.physics.pden_plasma_core_rad_mw` | read | explicit-arg | |
| `.physics.rmajor` | read | explicit-arg | |
| `.physics.rminor` | read | explicit-arg | |
| `.physics.temp_plasma_electron_density_weighted_kev` | read | explicit-arg | |
| `.physics.q95` (tokamak call site, `physics.py:1080`) / `.stellarator.iotabar` (stellarator call site, `stellarator.py:2292`) | read | explicit-arg | same parameter, two different producers depending on device mode (`istell`) — a topology question, out of scope here (registry's own scope-rule paragraph); the pure port takes it as one plain `q95` argument, caller's job to bind the right producer |
| `.physics.qstar` | read | explicit-arg | |
| `.physics.vol_plasma` | read | explicit-arg | |
| `.physics.n_charge_plasma_effective_vol_avg` | read | explicit-arg | bound to the port's `zeff` parameter — a call-site rename, not an ambiguity (docstring: "zeff: Plasma effective charge") |
| `.physics.eden_plasma_electrons_thermal_vol_avg` | read | explicit-arg | |
| `.physics.eden_plasma_ions_thermal_vol_avg` | read | explicit-arg | |
| `.physics.f_p_alpha_plasma_deposited` | read | explicit-arg (promoted) | read directly off `self.data`, not in source's own signature |
| `.physics.p_plasma_ohmic_mw` | read | explicit-arg (promoted) | ″ |
| `.physics.i_rad_loss` | read (×4 in source) | explicit-arg (promoted, switch) | read once via `ConfinementRadiationLossModel(int(...))` for the loss-power adjustment, then re-read three more times (unwrapped) for the `hstar` branch — same unchanged value throughout one call, `local-intermediate` in spirit; promoted to one explicit `int` parameter, read once in the port |
| `.physics.pden_plasma_rad_mw` | read (×2 in source) | explicit-arg (promoted) | read once for the `FULL_RADIATION` loss-power term, again for the `FULL_RADIATION` `hstar` term — same value both times, `local-intermediate` |
| `.physics.tauee_in` | read | explicit-arg (promoted) | `USER_INPUT` branch only — dead in practice, see "A dead branch: `USER_INPUT`" |
| `.physics.pden_plasma_sync_mw` | read | explicit-arg (promoted) | `hstar`'s `CORE_ONLY` term only |
| `.physics.p_plasma_inner_rad_mw` | read | explicit-arg (promoted) | `hstar`'s `CORE_ONLY` term only |
| `.physics.triang` | read | explicit-arg (promoted) | `ITPA20`/`ITPA20_IL` branches only |
| `.physics.m_ions_total_amu` | read | explicit-arg (promoted) | `ITPA20`/`ITPA20_IL` branches only, as `aion` |
| `.physics.e_plasma_beta` | read | explicit-arg (promoted) | feeds `t_energy_confinement_beta` only |
| `.physics.kappa_ipb` | write, then read (×11 branches) | local-intermediate | computed unconditionally near the top of the method via a call into `plasma_geometry.py` (see "calls into other models"), then read by 11 of the 48 scaling-law branches later in the same call — always the value just computed, no possibility of divergence |
| `.physics.t_energy_confinement_beta` | write | — | a **second**, independent write the source method performs beyond its own returned `ConfinementTimeData`: `self.data.physics.t_energy_confinement_beta = (self.data.physics.e_plasma_beta / 1e6) / p_plasma_loss_mw`, computed at the very end of the method and never part of the dataclass the source returns. The pure port appends it as a 9th return value. |
| `.physics.pden_electron_transport_loss_mw`, `.pden_ion_transport_loss_mw`, `.t_electron_energy_confinement`, `.t_ion_energy_confinement`, `.t_energy_confinement` (from `ConfinementTimeData.t_plasma_energy_confinement`), `.p_plasma_loss_mw`, `.hstar` | write (by caller, from the returned dataclass) | — | `PlasmaConfinementTime.calculate_confinement_time` itself does not write these — both call sites (`stellarator.py:2320-2336`, `physics.py`, unit #9) unpack the returned `ConfinementTimeData` field-by-field onto `self.data.physics.*` themselves. Recorded here since they are the function's real outputs; the writing is the caller's, not this unit's |

`calculate_double_and_triple_product` (source L1037-1065): already a clean,
self-contained `@staticmethod`, no `self.data` access at all.

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.nd_plasma_electrons_vol_avg` | read | explicit-arg | |
| `.physics.t_energy_confinement` | read | explicit-arg | the same field `calculate_confinement_time`'s `t_plasma_energy_confinement` output is assigned to by the caller, one call later |
| `.physics.temp_plasma_electron_vol_avg_kev` | read | explicit-arg | bound to the port's `temp_plasma_electrons_vol_avg_kev` parameter — source's own parameter is spelled with the trailing `s`, matching neither VarPath exactly; kept as source spells it |
| `.physics.ntau`, `.physics.nTtau` | write (by caller) | — | tuple-unpacked directly at the call site (`stellarator.py:2338-2344`), same shape as above |

## proposed signature(s)

```python
def calculate_iter_physics_basis_elongation(vol_plasma, rmajor, rminor) -> float: ...
    # ported from plasma_geometry.py, see "calls into other models"

def <name>_confinement_time(<source's own parameters, unchanged>) -> float: ...
    # 48 functions, one per ConfinementTimeModel value; see the .py file for the full
    # list, each individually docstringed with its ConfinementTimeModel member and value

def calculate_confinement_time(
    m_fuel_amu, p_alpha_total_mw, aspect, b_plasma_toroidal_on_axis,
    nd_plasma_electrons_vol_avg, nd_plasma_electron_line, eps, hfact,
    i_confinement_time, i_plasma_ignited, kappa, kappa95, p_non_alpha_charged_mw,
    p_hcd_injected_total_mw, plasma_current, pden_plasma_core_rad_mw, rmajor, rminor,
    temp_plasma_electron_density_weighted_kev, q95, qstar, vol_plasma, zeff,
    eden_plasma_electrons_thermal_vol_avg, eden_plasma_ions_thermal_vol_avg,
    f_p_alpha_plasma_deposited, p_plasma_ohmic_mw, i_rad_loss, pden_plasma_rad_mw,
    pden_plasma_sync_mw, p_plasma_inner_rad_mw, triang, m_ions_total_amu, e_plasma_beta,
    tauee_in,
) -> tuple[float, ...]:  # 9 values, see "cottax node"
    ...

def calculate_double_and_triple_product(
    nd_plasma_electrons_vol_avg, temp_plasma_electrons_vol_avg_kev, t_energy_confinement
) -> tuple[float, float]:  # (ntau, nTtau)
    ...
```

## cottax node

**Head, law, tail — five slots of `.physics.confinement_time`, and no composite.** This
section described a single `ConfinementTime` node carrying `i_confinement_time`,
`i_rad_loss` and `i_plasma_ignited` as static fields and branching on all three
internally. **That node no longer exists.** It declared the union of every arm's reads —
**32, where a law needs 6 to 8** — and two of the 32 were dead at the reference machine's
own switch values, one of them inventing a `.current_drive → .physics` subsystem edge
that no run makes. The authoritative node list is
`functional_process/models/physics/confinement_time.py`; what follows is what replaced
the composite and why.

| slot | occupant(s) | switch | owns |
|---|---|---|---|
| `inputs` | `ConfinementScalingInputs` | — | the unit conversions every law takes (`nd_plasma_electron_line_19`, `cur_plasma_ma`) |
| `elongation` | `IterPhysicsBasisElongation` | — | `.physics.kappa_ipb` |
| `power_loss` | `PlasmaPowerLossIgnitedCoreRadiation`, `PlasmaPowerLossNonIgnitedCoreRadiation` | `i_plasma_ignited` × `i_rad_loss` | `.physics.p_plasma_loss_mw` |
| `scaling` | `Iss04ConfinementTime`, `IterIpb98y2ConfinementTime` | `i_confinement_time` | `.physics.t_energy_confinement` and the law's own outputs |
| `tail` | `ConfinementTailCoreRadiation` | `i_rad_loss` | everything downstream of the law |
| `double_and_triple_product` | `DoubleAndTripleProduct` | — | `.physics.ntau`, `.physics.nTtau` |

**The 48 scaling laws still get no individual node, and that is now a rule rather than a
deferral.** `switch_kwarg_survey.md` band (d): one occupant per value **this port
supports**, not one per value PROCESS has. Two exist — ISS04 (38, the Helias run) and
IPB98(y,2) (34, the conventional tokamak) — and the other ~46 are neither written nor
refused-in-advance; they are values with no entry, which `_slot_occupant` reports as
such. So the question this section left open ("48 alternatives or one dispatcher?") is
answered in neither of the ways it framed: **the unit of rebinding is the declaring
class**, so a law whose reads differ cannot be a kwarg, and a law nobody runs need not
exist.

**`IterPhysicsBasisElongation` is registered now**, and was deliberately not while the
composite owned `kappa_ipb` itself — registering both would have been a duplicate-owner
conflict on one `VarPath`. Several laws read it and only one law ever runs, which is
where it belonged.

**`StellaratorConfinementTime` is gone.** It existed to rebind exactly one read: PROCESS
calls ISS04's twentieth argument `q95` and hands it the rotational transform instead.
With one class per law that is not a rebinding at all —
`iss04_stellarator_confinement_time`'s own parameter *is* `iotabar`, so
`Iss04ConfinementTime` reads `.stellarator.iotabar` because that is what its law takes.
The read follows from the law, not from the device, and the registry keyed on `istell`
had nothing left to decide.

**`i_rad_loss` is answered once and *used* twice** — it decides `power_loss` (which
radiation term is subtracted from the loss power) and `tail` (which term `hstar` reads).
`model_tree_design.md` §8 step 4d's "a switch is answered once" holds; two slots reading
one resolved value is not a second transcription.

## tier signal

**Tier 1** throughout — no internal iteration anywhere in either in-scope method or any
of the 48 scaling laws (`find_other_h_factors`'s `scipy.optimize.root_scalar` loop is the
unit that would be tier 2, and it is out of scope — see "source"). Every one of the 48
scaling laws is closed-form algebra; `calculate_confinement_time` itself is a large
`if`/`elif` dispatch with two `jnp.maximum`/`jnp.minimum` clamps and one continuous
interpolation (`menard_nstx_petty08_hybrid_confinement_time`), none of which are
iterative.

Verified against the live PROCESS reference at every legacy sample, not merely
translated: all 48 individual scaling laws against `tests/unit/models/physics/
test_confinement_time.py`'s own all-ones parametrisation (a free, pre-validated oracle —
lifted verbatim), and the composite `calculate_confinement_time`/
`calculate_double_and_triple_product` against a real `PlasmaConfinementTime` bound to a
`DataStructure`, at representative operating points spanning ohmic/L-mode/H-mode/
stellarator scalings and all three `i_rad_loss` arms.

## switches touched

- **`i_confinement_time`** (`.physics.i_confinement_time`, `ConfinementTimeModel`,
  values 0-51 minus the `PAZ_SOLDAN_NT` alias — see `core/solver/switches.md`, already
  listed there as "draft — pilot"). **Reads-set genuinely differs per value** (each of
  the 48 scaling laws takes a different argument subset), so per `naming_convention.md`'s
  own rule this does not qualify for "kept as a static branch, provably identical
  reads-set" — the textbook case would be a topology-changing switch, one node/subgraph
  per value, resolved at graph-assembly time. **Not resolved that way here**: this port
  keeps PROCESS's own granularity (one dispatcher function, one node), with the switch as
  a plain non-traced `int` used for ordinary Python branching — the same move already
  used by every other unit in this registry that folds a large dispatch into one pure
  function rather than splitting it (see `CLAUDE.md`'s "variant dispatch... has no clean
  node-graph shape" difficulty). Whether 51 separate `Alternative` nodes is ever worth
  building (v.s. one node with a huge static-kwarg space) is an open design question, not
  decided in this pass — flagging for whoever does the consolidation/registration work,
  not resolving it unilaterally.
- **`i_rad_loss`** (`.physics.i_rad_loss`, `ConfinementRadiationLossModel`, values 0-2).
  Same shape as `i_confinement_time` — reads-set differs per value (`FULL_RADIATION`
  reads `pden_plasma_rad_mw`, `CORE_ONLY` reads `pden_plasma_core_rad_mw`/
  `pden_plasma_sync_mw`/`p_plasma_inner_rad_mw`, `NO_RADIATION` reads neither) — kept as
  a second static, non-traced `int` on the same node for the same reason.
  Not previously in `core/solver/switches.md`; worth a row there.
- **`i_plasma_ignited`** (`.physics.i_plasma_ignited`, `PlasmaIgnitionModel`, values 0-1).
  Reads-set differs by exactly one term (`p_hcd_injected_total_mw`, added only when
  `NON_IGNITED`) — same treatment, third static `int`. **Superseded**: both values now
  have an occupant of `PlasmaPowerLoss` and neither is a static kwarg — see "The
  NON_IGNITED power-loss arm" above. The same switch is split the same way in
  `current_drive.md` (`HcdElectricTotalNonIgnited`/`HcdElectricTotalIgnited`), and the
  two units agree.

## calls into other models

- **`PlasmaGeom.calculate_iter_physics_basis_elongation`**
  (`process/models/physics/plasma_geometry.py:1003-1032`) — called unconditionally from
  `calculate_confinement_time`'s body (`self.data.physics.kappa_ipb = PlasmaGeom.
  calculate_iter_physics_basis_elongation(...)`). `plasma_geometry.py` is **not in
  `unit_registry.md` at all** — a new, small scope-sweep finding, same shape as
  `radiation_power.md`'s `impurity_radiation.py` miss. The called method is one line, pure,
  self-contained (`vol_plasma`/`rmajor`/`rminor` -> `kappa_ipb`, no further dependencies),
  so it is ported here rather than deferred, same call `radiation_power.md`/
  `fusion_reactions.md` already made for their own out-of-registry one-liners. Not
  claiming the rest of `plasma_geometry.py` — only this one method was reached.
- **`physics.py`'s `calculate_total_plasma_heating_power`** is *not* called by this unit
  — it produces `p_plasma_heating_total_mw`, which is `exhaust.py`'s input, not this
  unit's. Noted here only because both units were audited in the same pass; see
  `exhaust.md`.
- No other model calls anywhere in the in-scope methods or their 48 transitive
  scaling-law callees.
- **Dependency on a unit being audited in parallel**: none. Every implicit read this unit
  needed was a plain `.physics`/`.current_drive` field, not a call into `physics.py`
  (unit #9), `superconductors.py` (unit #22), or `impurity_radiation.py` (unit #23) — all
  three currently being audited by other agents in this same wave. No blocking dependency
  either direction.

## JAX-difficulty flags

- **Python `min`/`max` over differentiable arguments**, 3 sites: `p_plasma_loss_mw =
  max(p_plasma_loss_mw, 1.0e-3)` (main dispatcher), `denfac = min(1.0, denfac)`
  (`t10_confinement_time`), `min(iter_89p_confinement_time(...),
  iter_89_0_confinement_time(...))` (`MINIMUM_OF_ITER_89P_AND_ITER_89_0` branch). All
  three are `workaround-known`: JAX cannot trace a Python `min`/`max` producing a
  boolean-tracer `if` internally, so all three become `jnp.maximum`/`jnp.minimum` in the
  port — ordinary, not a blocker.
- **`menard_nstx_petty08_hybrid_confinement_time`'s three-way `if`/`elif`/`else` on
  `1/aspect`** — `workaround-known` (`needs-lax-cond-or-where`). Replaced with a clipped
  linear blend (`jnp.clip` + weighted sum), verified algebraically equivalent at both
  boundaries (`1/aspect = 0.4` reduces to pure Petty08, `1/aspect = 0.6` to pure NSTX) —
  not an approximation, the exact same piecewise-linear function, continuous everywhere
  and differentiable except at the two kinks (measure zero).
- **`i_confinement_time`/`i_rad_loss`/`i_plasma_ignited` switches drive ordinary Python
  `if`/`elif` branching, not `jnp.where`** — legitimate since all three are declared
  `static_argnames` in the harness (never traced, never differentiated), consistent with
  `naming_convention.md`'s "switches are not ports". Not a difficulty, recorded for
  completeness: a caller that ever wanted to differentiate *with respect to* one of these
  switches (not physically meaningful — they are integers selecting a formula) would need
  a different design entirely.
- No CoolProp calls, no `scipy.optimize`/`fsolve`, no `copy.deepcopy` anywhere in either
  in-scope method or its 48 transitive callees.

## open questions

- **`i_confinement_time`/`i_rad_loss` node-shape decision** (see "switches touched") —
  one composite node with two large static-kwarg spaces (this pass's choice, matching
  PROCESS's own granularity) versus 51x3 `Alternative` nodes. Not resolved here;
  flagging for the consolidation pass or a future dedicated design pass.
- **[RESOLVED] `q95`'s two producers** (`.physics.q95` tokamak / `.stellarator.iotabar`
  stellarator) — a device-mode topology question, deferred here as the caller's
  decision. **The caller then got it wrong**: `total_process.py` registered the base
  `ConfinementTime` unconditionally, binding `.physics.q95`, so the ISS04 branch — whose
  own parameter is *named* `iotabar` and raises it to `0.41` — was fed a safety factor
  where PROCESS feeds a rotational transform.

  Found by `mda_harness.py`'s block-by-block comparison, not by this unit's own tests,
  which structurally cannot see it: every `Tier1Contract` case here passes `q95`
  positionally to the pure function, and the pure function is correct. Only the
  *binding* was wrong, and the binding lives in the node declaration. Confirmed
  arithmetically rather than inferred: on `stellarator_helias.IN.DAT`'s converged run
  `q95 = 1.03`, `iotabar = 1.0`, and `1.03**0.41 = 1.0121928428817748` against the
  harness's reported `rel_diff` of `1.219e-02` on `.physics.t_energy_confinement`
  (`1.205e-02` on `.physics.f_t_alpha_energy_confinement`, which scales as `1/tau`).

  Fixed by `StellaratorConfinementTime`, a subclass rebinding exactly that one read via
  `_rebound_signature`, registered as the `value=6` arm of `total_process.py`'s
  `.stellarator.istell` switch. A subclass because reads are class-level `__call__`
  parameter defaults, so no per-instance static field can vary them; a *derived*
  signature rather than a restated one so the arm cannot silently drift from the base.
  `test_confinement_time.py::test_stellarator_arm_rebinds_only_the_q95_read` pins that
  exactly one read differs.

  **The general lesson, worth carrying**: a positional parameter whose *name* comes from
  one device's vocabulary is a standing trap in this port. PROCESS calls the slot `q95`
  in the signature and passes `iotabar` into it; nothing in the pure function's own
  tests can catch a caller binding the name rather than the role. Any other
  `calculate_*` with device-dependent call sites deserves the same check.
- **`USER_INPUT`/`PAZ_SOLDAN_NT` findings are new** and not yet cross-checked against
  whether any tracked regression input file actually sets `i_confinement_time = 0` or
  `51` (if one does, it would currently be hitting/masking these same bugs in real
  PROCESS runs — worth a `tests/regression` grep by whoever picks this up next, not done
  here since it is outside this unit's own scope).

## Derivative-safe power laws (`safe_pow` / `safe_sqrt`)

198 fractional power laws and 26 square roots in this file have been rewritten from `x ** p` / `jnp.sqrt(x)` to
`models/safe_math.py`'s `safe_pow(x, p)` / `safe_sqrt(x)`.

**Why.** For `0 < p < 1` the function is continuous at `x == 0` and its derivative is
not: `d/dx x**p = p * x**(p-1) -> +inf`. JAX's JVP then returns `inf` along the
direction that perturbs `x` and `nan` (`inf * 0`) along every other, so the *value* is
right everywhere and the *Jacobian row* is poisoned. That is the defect class
`_audit/next_steps.md` §9 records; the most recent instance produced 46 non-finite
Jacobian cells and stalled a cold optimiser start at zero SQP steps, reported by the
solver as "the problem seems to be non-convex".

**Value identity, checked not asserted.** `safe_pow`/`safe_sqrt` dispatch on `x == 0`
and evaluate the identical expression otherwise, so every `x != 0` result is bit-for-bit
what it was, and the `x == 0` result is `0.0 ** p` / `sqrt(0.0)` -- again exactly what
the bare expression returns. Verified two ways: a hex-exact diff of every Tier-1
contract's output over every declared sample plus eight fresh fuzz draws (3655 points,
zero differing bits), and `run_mda_harness.py` unchanged at 492 agreements / 34
disagreements. PROCESS itself does not raise at `x == 0` here -- it is plain Python
`float.__pow__` / `numpy.sqrt`, both of which return `0.0` -- and the reference was
re-evaluated at each boundary point to confirm it returns the port's number.

**What changed is only the derivative at exactly `x == 0`**, which becomes `0` instead
of `inf`/`nan` -- the same convention JAX already uses at `jnp.maximum`'s kink.

`Tier1Contract.test_gradient_finite_at_zero` (`--fp-gradients`) now checks the whole
class automatically: it zeroes each differentiable argument in turn and requires a
finite Jacobian wherever the value is finite.
