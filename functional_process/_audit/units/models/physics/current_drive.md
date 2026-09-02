---
kind: model-unit
status: draft
confidence: high
---

**Partially ported — two arms of four switches.** `models/physics/current_drive.py`
declares the minimal closure of `CurrentDrive.current_drive` that produces
`_audit/tokamak_boundary.md` § `.tokamak.current_drive`'s three boundary reads, for the
combinations the tracked tokamak input files actually hold:
`tests/regression/input_files/large_tokamak_eval.IN.DAT`'s `i_hcd_primary = 10`, and —
since the 2026-08-27 pass, § "2026-08-27" below — the two spherical tokamak files'
`i_hcd_primary = 13` (O-mode); both with `i_hcd_secondary = 0`,
`i_hcd_calculations = 1`, `i_plasma_ignited = 0`. Seven pure functions, two composites,
nine cottax nodes across four occupant families. Every other heating-and-current-drive
scheme is UNPORTED, with a per-value reason below — and for two of them the reason is
that **PROCESS itself cannot execute them** (see "A live PROCESS bug in two sibling
arms").

## source

`process/models/physics/current_drive.py` (2996 lines). The in-scope entry point is
`CurrentDrive.current_drive` (L1651-2309), a 660-line method that writes ~30 fields onto
`self.data` and returns nothing. The file's other four `Model` classes —
`NeutralBeam` (L133), `ElectronCyclotron` (L787), `IonCyclotron` (L1166),
`ElectronBernstein` (L1221), `LowerHybrid` (L1310) — are the efficiency models
`current_drive` dispatches into, and **none of them is reached on this arm**: `hcd_models`
(L1697-1771) is a dict of eleven *lambdas*, exactly one of which is called (L1795-1798),
and `i_hcd_primary = 10`'s is `eta_cd_norm_ecrh / (dene20 * rmajor)` (L1744-1747), three
reads and no sub-model call. That is checked, not assumed: the harness constructs
`CurrentDrive(None, None, None, None, None, None)` and the reference runs.

`CurrentDrive.output` (L2390-2800+) is a reporting shell, out of scope.
`calculate_normalised_current_drive_efficiency` (L2312) and
`calculate_dimensionless_current_drive_efficiency` (L2341) are already-pure statics but
produce fields outside this unit's closure — see "What this unit does *not* port and why".

## A live PROCESS bug in two sibling arms: `calculate_profile_y` returns `None`

`_audit/next_steps.md` §2 flagged that `profiles.py`'s `calculate_profile_y` returns
`None` on both concrete profile classes while six call sites in `current_drive.py` use
the return value arithmetically. **This is the audit that flag was waiting for. It is a
real, live bug, and it is worse than the flag said.**

The two implementations — `NeProfile.calculate_profile_y`
(`process/models/physics/profiles.py:161-211`) and `TeProfile.calculate_profile_y`
(`:383-...`) — set `self.profile_y` and fall off the end. Both return `None` for every
input; verified by calling each directly with well-formed array arguments, not read off
the source. The six call sites are:

| site | in | uses the result as |
|---|---|---|
| `current_drive.py:815` | `ElectronCyclotron.culecd` | `tlocal`, then `eccdef(tlocal, ...)` |
| `current_drive.py:826` | `ElectronCyclotron.culecd` | `1.0e-20 * <result>` |
| `current_drive.py:1353` | `LowerHybrid.cullhy` | `1.0e-19 * <result>` |
| `current_drive.py:1361` | `LowerHybrid.cullhy` | `tlocal`, then `np.sqrt`/division |
| `current_drive.py:1498` | `LowerHybrid.lheval` | `1.0e-19 * <result>` |
| `current_drive.py:1509` | `LowerHybrid.lheval` | `tlocal`, then arithmetic |

`1.0e-20 * None` raises `TypeError: unsupported operand type(s) for *: 'float' and
'NoneType'`, confirmed directly.

**A second, independent defect on the same call reaches first.** All six sites pass a
**scalar** `rho` — `rrr = 1.0/3.0` at `:813`, `rratio` at `:1348`/`:1496` — into a
function whose own signature annotates `rho: np.array` and whose body does
`self.profile_y[rho_index] = ...` (`profiles.py:203`). With a scalar `rho`,
`self.profile_y` is a numpy scalar and the assignment raises before the `None` ever
escapes. Reproduced live against the `process_port` reference:

```
  File "process/models/physics/current_drive.py", line 815, in culecd
    tlocal = self.plasma_profile.teprofile.calculate_profile_y(
TypeError: 'float' object is not subscriptable

  File "process/models/physics/current_drive.py", line 1443, in lhrad
    g0 = self.lheval(drfind, rat0)
TypeError: 'float' object is not subscriptable
```

So `culecd()` and `cullhy()` **cannot be called at all** in the current tree. Either
defect alone is sufficient; both are present.

**Which `i_hcd_primary`/`i_hcd_secondary` values this kills:** `6`
(`CULHAM_LOWER_HYBRID` → `cullhy`) and `7` (`CULHAM_ELECTRON_CYCLOTRON` → `culecd`).
Those two are unreachable in PROCESS as it stands, not merely unported.

**Not live on this run, and not live on any tracked run.** The six sites sit inside
lambdas that only fire when the switch selects them. Every input file in the tree
selects `10` or `13`:
`large_tokamak_eval.IN.DAT:124`, `large_tokamak_nof.IN.DAT:432`,
`low_aspect_ratio_DEMO.IN.DAT:676` and `tests/integration/data/large_tokamak_IN.DAT:425`
set `10`; `spherical_tokamak_eval.IN.DAT:133` and `st_regression.IN.DAT:2522` set `13`.
No tracked input selects `6` or `7`, and no test in `tests/` calls `culecd`, `cullhy`,
`lhrad` or `lheval` (the one grep hit,
`tests/functional_process/models/stellarator/test_heating.py:6`, is a docstring saying
`culnbi` is out of scope). That is why the tree is green with two unusable branches in
it.

**Nothing was guessed and nothing was worked around.** The port does not attempt models
6 or 7; they are UNPORTED with this as the reason. Reported upward rather than fixed —
fixing PROCESS is not this pass's remit, and the fix is not obvious (it needs a
scalar-`rho` evaluator, which is a different function from the array profile filler,
and choosing which one the call sites meant is a physics decision).

## data footprint

`CurrentDrive.current_drive`, restricted to the ported arm. Every row is a
`self.data.<area>.<field>` access inside L1651-2309, located by reading the method, not
by grep.

### reads

| VarPath | read at | classification | note |
|---|---|---|---|
| `.current_drive.i_hcd_primary` | L1676, L1795 | switch | selects the efficiency lambda *and*, through `CurrentDriveModel.method`, the wall-plug block at L2068/2099/2131/2162/2191 |
| `.current_drive.i_hcd_secondary` | L1677, L1681, L1784 | switch | same, for the second system; `0` means there is none |
| `.current_drive.i_hcd_calculations` | L1688 | switch | `0` skips the entire body |
| `.physics.i_plasma_ignited` | L2296 | switch | decides `.heat_transport.p_hcd_electric_total_mw` outright |
| `.current_drive.eta_cd_norm_ecrh` | L1745 | explicit-arg | model 10 only |
| `.physics.nd_plasma_electrons_vol_avg` | L1690 (as `dene20`) | explicit-arg | model 10 divides by `dene20 * rmajor` |
| `.physics.rmajor` | L1746 | explicit-arg | ″ |
| `.current_drive.p_hcd_secondary_injected_mw` | L1823, L1867, L2268, L2275 | explicit-arg | **no writer anywhere in `process/`** — a genuine boundary input, default `0.0` (`current_drive_variables.py:256`) |
| `.physics.plasma_current` | L1830, L1840, L1854 | explicit-arg | |
| `.physics.f_c_plasma_auxiliary` | L1837 | explicit-arg | produced by `physics.py:585` — another unit's node, an ordinary declared input here |
| `.current_drive.p_hcd_primary_extra_heat_mw` | L2134, L2140, L2149, L2267 | explicit-arg | `large_tokamak_eval.IN.DAT:125` sets `75.0`; iteration variable 11 |
| `.current_drive.eta_ecrh_injector_wall_plug` | L2141, L2144, L2155 | explicit-arg | `IN.DAT:122` sets `0.5` |
| `.current_drive.eta_cd_hcd_secondary` | L1815, L1822 | local-intermediate | on this arm nothing writes it (see "the three zeros"); it holds its `current_drive_variables.py:98` default of `0.0` |
| `.current_drive.p_hcd_secondary_extra_heat_mw` | L2269 | local-intermediate | written `0.0` by this same method at L1682 |
| `.heat_transport.p_hcd_secondary_electric_mw` | L2291 | local-intermediate | never written on this arm; `heat_transport_variables.py:127` default `0.0` |
| `.current_drive.eta_cd_hcd_primary` | L1808, L1841, L1846 | local-intermediate | written at L1796 earlier in the same call |
| `.current_drive.f_c_plasma_hcd_secondary` | L1838 | local-intermediate | written at L1828 |
| `.current_drive.p_hcd_primary_injected_mw` | L1847, L1864, L2133, L2139, L2148, L2266, L2274 | local-intermediate | written at L1834 |
| `.heat_transport.p_hcd_primary_electric_mw` | L2290 | local-intermediate | written at L2138; **default is `None`**, not `0.0` (`heat_transport_variables.py:130`), so a run that skips the primary block crashes at L2289 — one more reason `i_hcd_calculations = 0` is UNPORTED rather than trivially "everything stays default" |
| `.current_drive.p_hcd_ecrh_injected_total_mw` | L2147 (`+=`), L2154 | local-intermediate | zeroed at L1663; see "the accumulators" |
| `.physics.temp_plasma_electron_vol_avg_kev` | L1862, L1873 | explicit-arg | **outside this unit's closure** — feeds only `eta_cd_dimensionless_hcd_*` |
| `.physics.p_fusion_total_mw`, `.physics.p_plasma_ohmic_mw` | L2303, L2307 | explicit-arg | feed only `big_q_plasma`; **inside the closure since 2026-09-02** — see § "2026-09-02" |
| `.current_drive.p_beam_orbit_loss_mw` | L2306 | explicit-arg | zeroed by this same method at L1669 on every ported arm, and a **boundary input** of the graph (`reference_boundary_tokamak.txt:197`), not an output of any node — see § "2026-09-02" |

### writes (ported)

| VarPath | written at | owned by |
|---|---|---|
| `.current_drive.eta_cd_hcd_primary` | L1796 | `HcdPrimaryEfficiencyUserInputEcrh` |
| `.current_drive.eta_cd_hcd_secondary` | (never, on this arm) | `HcdSecondaryHeatingNone` |
| `.current_drive.p_hcd_secondary_extra_heat_mw` | L1682 | `HcdSecondaryHeatingNone` |
| `.heat_transport.p_hcd_secondary_electric_mw` | (never, on this arm) | `HcdSecondaryHeatingNone` |
| `.current_drive.c_hcd_secondary_driven` | L1821 | `HcdSecondaryDrivenCurrent` |
| `.current_drive.f_c_plasma_hcd_secondary` | L1828 | `HcdSecondaryDrivenCurrent` |
| `.current_drive.p_hcd_primary_injected_mw` | L1834 | `HcdPrimaryInjectedPower` |
| `.current_drive.p_hcd_ecrh_injected_total_mw` | L1663 (`= 0`), L2147 (`+=`) | `HcdPrimaryPowersElectronCyclotronNoSecondary` |
| `.current_drive.p_hcd_ecrh_electric_mw` | L2153 | ″ |
| `.current_drive.eta_hcd_primary_injector_wall_plug` | L2143 | ″ |
| `.heat_transport.p_hcd_primary_electric_mw` | L2138 | ″ |
| `.current_drive.p_hcd_injected_total_mw` | L2265 | `HcdInjectedPowerTotal` |
| `.heat_transport.p_hcd_electric_total_mw` | L2289, L2299 | `HcdElectricTotalNonIgnited` / `HcdElectricTotalIgnited` |
| `.current_drive.big_q_plasma` | L2302 | `FusionGain` (2026-09-02) |

Three of these — `.current_drive.p_hcd_ecrh_injected_total_mw`,
`.current_drive.p_hcd_injected_total_mw`, `.heat_transport.p_hcd_electric_total_mw` — are
exactly `_audit/tokamak_boundary.md` § `.tokamak.current_drive`'s three rows, which is
what this unit was dispatched to close. `tests/functional_process/models/physics/
test_current_drive.py::test_the_three_boundary_reads_are_produced` asserts it.

### the three zeros

`HcdSecondaryHeatingNone` owns three fields and reads none. PROCESS assigns exactly one
of them (`p_hcd_secondary_extra_heat_mw = 0.0`, L1682, guarded by `if i_hcd_secondary ==
0`) and leaves the other two at their `DataStructure` defaults: `eta_cd_hcd_secondary`
because `0` is not a key of `hcd_models` and the `elif ... != 0` at L1788 is false, and
`p_hcd_secondary_electric_mw` because every block that assigns it (L1893, L1919, L1945,
L1971, L2040) is guarded on a `secondary_cdm.method` that `NO_CURRENT_DRIVE` does not
have. Both defaults are `0.0` and no other model in `process/` writes either.

**Declaring the zeros is a judgement call and is recorded as one.** The alternative is to
leave the two unassigned fields as boundary inputs, which would put two *computed*
quantities on the boundary standing for "a switch skipped this code" —
`tokamak_boundary.md` § "The twelve that are simply inputs" is explicit that the boundary
is for variables PROCESS computes nowhere. A test
(`test_the_no_secondary_occupant_declares_the_three_zeros`) pins both defaults so that a
change upstream fails here rather than silently shifting a run.

### the accumulators

`p_hcd_ecrh_injected_total_mw` is zeroed at L1663 and then `+=`-ed by the *secondary*
ECRH block at L1955 and the *primary* ECRH block at L2147. Two writers of one field, in
one method, selected by two different switches. Ported literally it is a self-loop, which
cottax rejects.

It is not one here. The prior value is the secondary system's contribution, and on the
`i_hcd_secondary = 0` arm that contribution is the literal `0.0`, passed as a plain
argument (`p_hcd_ecrh_injected_secondary_mw`) rather than read. That is why
`HcdPrimaryPowersElectronCyclotronNoSecondary` is keyed on the **pair** of switches, the
same shape `confinement_time.py::PlasmaPowerLossIgnitedCoreRadiation` uses for
`i_plasma_ignited`/`i_rad_loss`. **A per-technology "secondary contribution" field would
let the two sides be independent nodes keyed on one switch each; PROCESS has no such
field**, so the pair is the honest key until someone adds one. Flagged rather than
invented — `naming_convention.md`'s "do not invent new names; port the existing one".

## proposed signature(s)

```python
def user_input_electron_cyclotron_efficiency(
    *, eta_cd_norm_ecrh, nd_plasma_electrons_vol_avg, rmajor) -> float: ...

def hcd_secondary_driven_current(
    *, eta_cd_hcd_secondary, p_hcd_secondary_injected_mw, plasma_current
) -> tuple[float, float]: ...              # (c_hcd_secondary_driven, f_c_plasma_hcd_secondary)

def hcd_primary_injected_power_mw(
    *, f_c_plasma_auxiliary, f_c_plasma_hcd_secondary, plasma_current,
    eta_cd_hcd_primary) -> float: ...

def electron_cyclotron_primary_powers(
    *, p_hcd_ecrh_injected_secondary_mw, p_hcd_primary_injected_mw,
    p_hcd_primary_extra_heat_mw, eta_ecrh_injector_wall_plug
) -> tuple[float, float, float, float]: ...

def hcd_injected_power_total_mw(
    *, p_hcd_primary_injected_mw, p_hcd_primary_extra_heat_mw,
    p_hcd_secondary_injected_mw, p_hcd_secondary_extra_heat_mw) -> float: ...

def hcd_electric_total_mw(
    *, p_hcd_primary_electric_mw, p_hcd_secondary_electric_mw,
    i_plasma_ignited) -> float: ...

def calculate_current_drive_ecrh_primary_no_secondary(...) -> tuple[float, ...]:  # 10
    ...
```

## cottax nodes

Four occupant families and three switch-independent nodes:

| family | slot switch | occupant | answers |
|---|---|---|---|
| `HcdPrimaryEfficiency` | `i_hcd_primary` | `HcdPrimaryEfficiencyUserInputEcrh` | `10` |
| ″ | `i_hcd_primary` × `i_ecrh_wave_mode` | `HcdPrimaryEfficiencyFreethyEcrhOMode` | `13` × `0` (see § "2026-08-27") |
| `HcdSecondaryHeating` | `i_hcd_secondary` | `HcdSecondaryHeatingNone` | `0` |
| `HcdPrimaryPowers` | `i_hcd_primary` × `i_hcd_secondary` | `HcdPrimaryPowersElectronCyclotronNoSecondary` | primary method ECRH (`3, 7, 10, 13`) × secondary `0` |
| `HcdElectricTotal` | `i_plasma_ignited` | `HcdElectricTotalNonIgnited`, `HcdElectricTotalIgnited` | `0`, `1` |

plus `HcdSecondaryDrivenCurrent`, `HcdPrimaryInjectedPower`, `HcdInjectedPowerTotal` and
(since 2026-09-02) `FusionGain`, which are switch-independent once the arms above have
produced their outputs — a real result of the split, not a shortcut: PROCESS computes
these four lines outside every `if` in the method (L1821-1855, L2265-2270, L2301-2308).

Full bodies in `models/physics/current_drive.py`. Which occupant belongs in which
`Tokamak` slot is the consolidation pass's call, not this record's.

## What this unit does *not* port and why

Deliberately outside the closure, all on the *same* arm and all cheap to add later —
listed so the omission is a decision rather than a gap:

- `.current_drive.eta_cd_norm_hcd_primary` / `_secondary` (L1807, L1814) and
  `.current_drive.eta_cd_dimensionless_hcd_primary` / `_secondary` (L1859, L1870) —
  diagnostic efficiencies, no reader in the current graph. The secondary dimensionless
  one is additionally behind `if p_hcd_secondary_injected_mw > 0.0` (L1867), a
  **data-dependent branch on a continuous quantity** — `needs-lax-cond-or-where` if it is
  ever ported, and the only such branch in this method.
- `.current_drive.c_hcd_primary_driven`, `.f_c_plasma_hcd_primary` (L1845, L1852).
- `.current_drive.p_hcd_injected_current_total_mw`, `.p_hcd_injected_electrons_mw`,
  `.p_hcd_injected_ions_mw` (L2273, L2279, L2284).
- ~~`.current_drive.big_q_plasma` (L2302) — the fusion gain, read by constraints rather
  than by any node currently in the graph.~~ **Ported 2026-09-02** (`fusion_gain`,
  `FusionGain`); the sentence above is left struck through because it is the exact
  reasoning that let `st_regression` read a frozen `0.0` — see § "2026-09-02".
- The other four technologies' zeroed accumulators (`p_hcd_beam_/lowhyb_/icrh_/ebw_
  injected_total_mw`, L1664-1667), `c_beam_total` and `p_beam_orbit_loss_mw` (L1668-1669)
  — zero on this arm, and a node declaring them would be asserting five switch values at
  once rather than one.

## UNPORTED switch values

| switch | value | reason |
|---|---|---|
| `i_hcd_primary` | `1` `FENSTERMACHER_LOWER_HYBRID` | needs `LowerHybrid.lower_hybrid_fenstermacher` and `.feffcd`; not written |
| | `2` `IPDG89_ION_CYCLOTRON` | needs `IonCyclotron.ion_cyclotron_ipdg89`; not written |
| | `3` `FENSTERMACHER_ELECTRON_CYCLOTRON` | needs `ElectronCyclotron.electron_cyclotron_fenstermacher` and `.physics.dlamee`; not written |
| | `4` `EHST_LOWER_HYBRID` | needs `LowerHybrid.lower_hybrid_ehst`; not written |
| | `5` `ITER_NEUTRAL_BEAM` | needs `NeutralBeam.iternb` and the whole beam wall-plug block (L2191-2260); not written |
| | `6` `CULHAM_LOWER_HYBRID` | **PROCESS cannot execute it** — `cullhy` → `lhrad` → `lheval` raises `TypeError` at `current_drive.py:1498`; see "A live PROCESS bug in two sibling arms" |
| | `7` `CULHAM_ELECTRON_CYCLOTRON` | **PROCESS cannot execute it** — `culecd` raises `TypeError` at `current_drive.py:815`; same section |
| | `8` `CULHAM_NEUTRAL_BEAM` | needs `NeutralBeam.culnbi` (and its `sigbeam`/`cfnbi`/`xlmbdabi` chain) plus the beam block; not written |
| | `12` `USER_INPUT_ELECTRON_BERNSTEIN` | needs `ElectronBernstein.electron_bernstein_freethy` and the EBW block (L2162-2187); not written |
| | `13` `FREETHY_ELECTRON_CYCLOTRON` | **ported 2026-08-27** for `i_ecrh_wave_mode = 0` (O-mode), the value both selecting files set; X-mode (`1`) remains UNPORTED — see § "2026-08-27" |
| | `0` `NO_CURRENT_DRIVE` | raises `ProcessValueError` at L1800 — a primary system is mandatory |
| `i_hcd_secondary` | `1`-`8`, `10`, `12`, `13` | each needs its efficiency model *and* its wall-plug block (L1885-2063), *and* changes which technology accumulator the primary block's `+=` starts from — see "the accumulators" |
| `i_hcd_calculations` | `0` | the whole body is skipped, so `.heat_transport.p_hcd_primary_electric_mw` keeps its `None` default and any consumer of it fails. Not "trivially all zeros"; not written |

## tier signal

**Tier 1.** No internal iteration anywhere on the ported arm: every stage is closed-form
rational arithmetic. (`LowerHybrid.lhrad`'s Newton-Raphson at L1411-1470 *is* the tier-2
shape in this file, and it is on the unreachable model-6 path.)

Verified against the live PROCESS reference, not merely translated: all ten outputs of
`calculate_current_drive_ecrh_primary_no_secondary` diffed against a real `CurrentDrive`
bound to a `DataStructure` at four declared operating points plus fuzz, values and
gradients.

## switches touched

- **`i_hcd_primary`** (`.current_drive.i_hcd_primary`, `CurrentDriveModel`, eleven
  values). **Split**, one occupant per value — the reads differ radically (model 10 reads
  three variables; model 13 reads six plus two further switches; models 6/7/8 reach the
  plasma profile machinery entirely), and there is no shared body at all: `hcd_models` is
  eleven independent lambdas. This is the `i_tf_sc_mat` shape from
  `traceability_policy.md`, not the `coelc`/`itart` shape.
  Note the switch decides **two** things at different granularities: the efficiency
  lambda per *value*, and the wall-plug block per `CurrentDriveModel.method`, so one
  block serves four values. Both are declared; they are separate families.
- **`i_hcd_secondary`** (twelve values including `0`). Same treatment.
- **`i_hcd_calculations`** (`0`/`1`). Topology: `1` means these nodes exist, `0` means
  none of them does. Not a port on any node.
- **`i_plasma_ignited`** (`.physics.i_plasma_ignited`, `PlasmaIgnitionModel`). Split into
  two occupants of `HcdElectricTotal`; the `IGNITED` arm reads **nothing**, which is two
  invented edges removed. Same switch `confinement_time.md` records, and this unit and
  that one now agree on the treatment.

## calls into other models

- `PlasmaProfile` / `NeProfile` / `TeProfile` (`process/models/physics/profiles.py`) —
  reached by models 6 and 7 only, and broken; see the bug section. **Not ported, not
  needed on this arm.**
- `NeutralBeam`, `IonCyclotron`, `LowerHybrid`, `ElectronCyclotron`, `ElectronBernstein`
  — same file, reached by the ten unported models only. The harness proves the ported arm
  reaches none of them by passing `None` for all five.
- `.physics.f_c_plasma_auxiliary` comes from `physics.py:585`, which is `physics.py`'s
  unit, being ported in parallel. An ordinary declared input here; no blocking dependency
  either direction.
- **Consumed by**: `confinement_time.py::PlasmaPowerLossNonIgnitedCoreRadiation` reads
  `.current_drive.p_hcd_injected_total_mw`, which `HcdInjectedPowerTotal` now produces.
  That pairing is the whole of `tokamak_boundary.md` § "What blocked the real file", and
  both halves landed in the same pass.

## JAX-difficulty flags

- No CoolProp, no `scipy.optimize`, no `copy.deepcopy`, no array mutation, no fractional
  powers or square roots anywhere on the ported arm — so no `safe_pow`/`safe_sqrt` is
  needed and `test_gradient_finite_at_zero` has no boundary to trip on. (Zeroing any of
  `eta_cd_norm_ecrh`, `nd_plasma_electrons_vol_avg`, `rmajor`, `plasma_current` or
  `eta_ecrh_injector_wall_plug` makes the *value* non-finite through an ordinary
  division, which that test excuses by construction.)
- **`if p_hcd_secondary_injected_mw > 0.0` (L1867)** is a data-dependent branch on a
  continuous quantity — `needs-lax-cond-or-where`. Not hit: it guards
  `eta_cd_dimensionless_hcd_secondary`, which is outside this unit's closure. Recorded so
  whoever extends the closure meets it in the record rather than in a trace error.
- `min(0.999, f_p_beam_orbit_loss)` (L1994, L2195) is a Python `min` over a data value —
  the `workaround-known` shape `confinement_time.md` already records — on the neutral
  beam arms only, which are unported.
- `i_hcd_primary`/`i_hcd_secondary`/`i_hcd_calculations` never appear in a ported body at
  all: they select occupants. `i_plasma_ignited` appears once, in the composite, as a
  `static_argnames` int.

## open questions

- **No per-technology "secondary contribution" field exists**, so
  `HcdPrimaryPowers`'s occupants are keyed on a switch *pair* and the family is
  combinatorial in principle (5 primary methods × 6 secondary methods). Only one cell is
  written and only one is needed, but if a second secondary scheme is ever ported this
  should be revisited — introducing `p_hcd_<tech>_injected_secondary_mw` as a real
  variable would make both sides single-switch nodes. Not decided here: it means minting
  a name PROCESS does not have, which `naming_convention.md` forbids doing quietly.
- **Two unreachable `i_hcd_primary` values (`6`, `7`) are a PROCESS bug this pass found
  and did not fix.** Whether to fix them upstream, and what a scalar-`rho` profile
  evaluator should be, is a physics decision for the PROCESS maintainers.
- `.heat_transport.p_hcd_primary_electric_mw`'s `None` default
  (`heat_transport_variables.py:130`) is the only `None` default this port has met. It is
  benign on the ported arm and a latent crash on `i_hcd_calculations = 0`; noted, not
  acted on.

## 2026-08-27 — `i_hcd_primary = 13` (`FREETHY_ELECTRON_CYCLOTRON`), O-mode

The arm that was blocking **both** spherical tokamak files:
`spherical_tokamak_eval.IN.DAT:133` and `st_regression.IN.DAT:2522` set
`i_hcd_primary = 13`, and before this pass `machine_from_indat` refused each at exactly
that key (verified live before writing anything). With the arm wired, the refusal moves
on — see "frontier probe" below.

### What the files select, checked not assumed

| switch | spherical_tokamak_eval | st_regression | PROCESS default |
|---|---|---|---|
| `i_hcd_primary` | `13` (`:133`) | `13` (`:2522`) | `5` (`current_drive_variables.py:190`) |
| `i_ecrh_wave_mode` | `0` (`:130`) | `0` (`:2665`) | `0` (`:116`) |
| `i_hcd_secondary` | unset → `0` | commented out (`:2565`) → `0` | `0` (`:206`) |
| `i_hcd_calculations` | `1` (`:134`) | `1` (`:2492`) | `1` (`:223`) |
| `n_ecrh_harmonic` | `2` (`:129`) | `2` (`:2661`) | `2.0` (`:113`) |
| `feffcd` | `1.0` (`:132`) | `1.0` (`:2558`) | `1.0` (`:158`) |
| `eta_ecrh_injector_wall_plug` | `0.45` (`:131`) | `0.45` (`:2681`) | — |

**`i_hcd_secondary` is `0` in both files**, so the already-written
`HcdSecondaryHeatingNone` occupant serves them and the secondary system is *not* the
next frontier for these two files (the probe below says what is).

### The port

- `freethy_electron_cyclotron_efficiency` — ports
  `ElectronCyclotron.electron_cyclotron_freethy`
  (`process/models/physics/current_drive.py:992-1088`) **plus** the `* feffcd` factor
  its `hcd_models[13]` lambda applies (`:1759-1770`). Model 10's lambda has no `feffcd`
  factor; model 13's does — the same asymmetry the first pass's reads-tests pinned from
  the other side, now asserted in both directions.
- `HcdPrimaryEfficiencyFreethyEcrhOMode` — the second occupant of
  `HcdPrimaryEfficiency`. Seven reads (`.physics.temp_plasma_electron_vol_avg_kev`,
  `.physics.n_charge_plasma_effective_vol_avg`, `.physics.rmajor`,
  `.physics.nd_plasma_electrons_vol_avg`, `.physics.b_plasma_toroidal_on_axis`,
  `.current_drive.n_ecrh_harmonic`, `.current_drive.feffcd`) where model 10 has three,
  overlapping in two. The staticmethod's `te`/`zeff` abbreviations are spelled as the
  `VarPath` leaves the lambda actually reads (`:1760-1767`); no name was minted.
- `calculate_current_drive_freethy_ecrh_primary_no_secondary` — the composite for this
  arm. `CurrentDriveModel(13).method` is `ELECTRON_CYCLOTRON`, so stages 2-6 are the
  *same functions* as arm 10's (`_hcd_primary_powers_arm` already answered `13` before
  this pass); only stage 1 differs.
- Defect preserved: the X-mode cut-off puts `n_ecrh_harmonic` on `fc**2` **inside** the
  square root (`:1077`, `sqrt(n*fc**2 + 4*fp**2)`, not `sqrt((n*fc)**2 + ...)`).
  Transcribed as written, flagged in the function's docstring, not fixed.
- First JAX-visible nonlinearity in this unit: `jnp.sqrt` (plasma frequency) and
  `jnp.tanh` (the coupling factor, `:1082-1085`). Both smooth on the sampled domain; the
  `"near-cutoff-coupling"` legacy sample sits deliberately on the tanh *slope* (density
  `1.5e20` against `2.2` T) so the gradient check exercises the one place this model's
  derivative has structure model 10's does not.

### The nested switch: representation choice

`i_ecrh_wave_mode` is read at `:1767` by model 13's lambda and nowhere else in any model
body (`:2541-2542` is the reporting shell). Two decisions, separately grounded:

1. **Within the pure function it is a static kwarg, not a split** — the one switch in
   this unit taking `traceability_policy.md`'s static-kwarg exception, because the two
   branches' reads-sets are **provably identical**: both cut-offs are formed from `fc`
   and `fp`, which both modes compute from the same two reads. The evidence is asserted
   in `test_the_wave_mode_switch_is_static_with_identical_reads`, and the differing body
   is one line inside a shared ~20, the exact shape the policy's deviation clause
   (`coelc`/`itart`) exists for. The occupant pins `0`; the reference's `ValueError` on
   an invalid mode is transcribed, defects included.
2. **In `indat.py` it dispatches as a nested registry, not a joint arm.**
   `_hcd_primary_efficiency(i_hcd_primary, i_ecrh_wave_mode)` keeps the outer registry
   keyed on `i_hcd_primary` (eleven refusals untouched, still keyed on the switch a
   user must change) and consults `HCD_PRIMARY_EFFICIENCY_FREETHY = {0: ...}` only for
   value 13. The `i_plasma_ignited_i_rad_loss` / `hcd_primary_powers_arm` joint-arm
   shape was considered and rejected: joint arms are for products where both switches
   shape every cell, and a joint key here would have had to refuse `(1, O-mode)` and
   `(1, X-mode)` as distinct cells when PROCESS never reads the wave mode on model 1's
   arm — two refusals for one branch, the invented-edge defect at the registry level.
   The nesting also mirrors the source, where the wave-mode `if` sits *inside*
   `electron_cyclotron_freethy`, not beside `hcd_models`.

X-mode (`1`) is `UNPORTED[("i_ecrh_wave_mode", 1)]`: live on no tracked input (both
selecting files set `0`, also the default), so no occupant pins it — the
`plasma_geometry_arm` 1 precedent, porting a formula and binding it being different
acts. The branch *is* transcribed inside the shared pure function and value- and
gradient-checked against the reference by the `"x-mode-transcription-check"` sample, so
a future X-mode occupant starts from verified ground: it is these same seven reads over
the other branch, one class and one registry line away.

### Registration diff (`indat.py`), exact

1. Import: `HcdPrimaryEfficiencyFreethyEcrhOMode` added to the
   `functional_process.models.physics.current_drive` import list.
2. `UNPORTED`: the `("i_hcd_primary", 13)` refusal **removed** (the value dispatches
   now), **replaced** by `("i_ecrh_wave_mode", 1)` with the reason above.
3. New registry `HCD_PRIMARY_EFFICIENCY_FREETHY = {0: HcdPrimaryEfficiencyFreethyEcrhOMode}`
   (plain-int keys: PROCESS has no enum for this switch, `process/core/input.py:1096`
   declares `int, choices=[0, 1]`) and new dispatcher `_hcd_primary_efficiency`.
4. `machine_from_indat`: `primary_efficiency=_hcd_primary_efficiency(i_hcd_primary,
   switches.get("i_ecrh_wave_mode", 0))` replaces the direct `_slot_occupant` call.

Consequential (same pass): `tests/functional_process/test_machine.py` gained
`NESTED_UNPORTED_COMPANIONS = {"i_ecrh_wave_mode": {"i_hcd_primary": 13}}` and
`i_ecrh_wave_mode` in `TOKAMAK_ONLY_UNPORTED_FIELDS`, because
`test_a_refused_value_says_why` writes each `UNPORTED` key over a baseline whose
`i_hcd_primary` is `10` — under which PROCESS itself ignores the wave mode, so the case
must also select the value that nests it.

### Harness evidence

`tests/functional_process/models/physics/test_current_drive.py`:
`TestCurrentDriveFreethyEcrhPrimaryNoSecondary` (Tier 1, all ten outputs, reference =
a real `CurrentDrive` bound to a `DataStructure` with `i_hcd_primary = 13` and a real
`ElectronCyclotron(None)` — the constructor needs the instance because the lambda
reaches the `@staticmethod` *through the attribute*, and `plasma_profile=None` is still
the proof no profile machinery is touched). Five legacy samples (the
`spherical_tokamak_eval` operating point with the file's own values where stated; the
ignited reset; non-zero secondary power; near-cutoff; the X-mode transcription check)
plus fuzz over twelve bounded arguments, `i_plasma_ignited`/`i_ecrh_wave_mode` static.
Fuzz bounds keep `n_ecrh_harmonic * fc` above the O-mode cut-off at every corner so no
point lands in the decoupled regime where the efficiency underflows and the downstream
division explodes (PROCESS would produce the same explosion; the tracked files run
coupled). Structural tests: the seven reads (and `eta_cd_norm_ecrh` *not* among them,
nor `.current_drive.i_ecrh_wave_mode` — switches are not ports), the reads-identity
evidence for the static kwarg, the no-self-loop sweep extended to the new occupant, and
a composition test tying the occupant chain to the composite.

Runs (this file only): plain — 35 passed, 24 skipped; `--fp-gradients` — 59 passed;
`--fp-gradients --fp-fuzz 10` — 131 passed. Values agree exactly (0.0 diff at the
smoke points, machine-precision contract in the harness), gradients within PROCESS's
own finite-difference error bars.

### Frontier probe (after wiring, merged base incl. consolidation round 2)

`machine_from_indat` + `graph_for`, both files, this arm no longer refuses; the next
refusal is **identical for both** and is not this unit's:

```
spherical_tokamak_eval.IN.DAT / st_regression.IN.DAT →
NotImplementedError: divertor_geometry_arm == -1 is a real PROCESS branch but is not
ported: `.physics.itart == 1`: `divgeom` returns `1.75 * rminor` at
`process/models/build.py:863` and **never writes `.build.rspo`** -- a different
write-set, not just a different formula, so it is a different occupant. Not written
```

So the spherical tokamaks' next frontier is the TART divertor geometry arm
(`build.py`'s unit), not heating and current drive.

## 2026-09-02 — `.current_drive.big_q_plasma` (L2301-2308): registration, not a port

**The verdict first, with the evidence, because the two answers are different jobs.**
`_audit/next_steps.md` §28.3 item 3 asked whether `st_regression`'s failure was *"a
**registration** problem — the node exists in `models/stellarator/heating.py` and may
simply be absent from the tokamak graph — or a real port."* Measured on
`graph_for(machine_from_indat("st_regression.IN.DAT"))`, before any edit:

| path | owner | on the boundary? |
|---|---|---|
| `.physics.p_fusion_total_mw` | `.physics.set_fusion_powers` | no |
| `.physics.p_plasma_ohmic_mw` | `.tokamak.physics.ohmic_heating` | no |
| `.current_drive.p_hcd_injected_total_mw` | `.tokamak.current_drive.injected_power_total` | no |
| `.current_drive.p_beam_orbit_loss_mw` | none | **yes, an ordinary pinned input** |
| `.current_drive.big_q_plasma` | **none** | no — *nothing in the model graph reads it* |

All four operands of PROCESS's own tokamak formula are already on the tokamak graph;
three are produced by nodes that have been there for weeks and the fourth is a boundary
input the pin has carried since the tokamak boundary was first written
(`reference_boundary_tokamak.txt:197`). Nothing had to be reverse-engineered, no switch
value had to be decided, no new read was introduced. **Registration.**

`models/stellarator/heating.py::calculate_fusion_gain` is the *same physics* and was
ported (unit #5) with `FusionGain` registered in the stellarator namespace. It was not
simply reused here for two reasons, one of principle and one of arithmetic:

- a unit is a PROCESS source file, and `st_heat` is not the function a tokamak runs;
- **the two source lines are not identical.** `st_heat` guards the denominator
  (`< 1e-6` → `1e18`, `heating.py:129-137`); `current_drive.py:2301` divides straight.
  Transcribing the guard onto the tokamak would be inventing an arm PROCESS does not
  have, in exactly the direction that hides a degenerate configuration.

So: `fusion_gain` (the pure core, two lines) and `FusionGain` (the node) in
`models/physics/current_drive.py`, one slot in `TokamakCurrentDrive`. Eight lines of
code for a failure that had stood since §26.

### Why it went missing, which is the transferable part

`.current_drive.big_q_plasma` has **no reader inside the model graph**. Its readers are
the problem layer's — `objective_metric_5` (`i_figure_merit = -5`, `FUSION_GAIN_Q`) and
`constraint_28`. Every instrument that could have caught it looks at the model graph:
`boundary.py`'s pins, `provider.py`'s classification, `unproduced_but_computed`. An
output nothing reads is invisible to all of them, and the record's own § "What this unit
does *not* port and why" wrote the reason down in 2026-08 — *"read by constraints rather
than by any node currently in the graph"* — as a justification rather than as a debt.
`boundary.inert_conditions` (2026-09-01) is the instrument that closes the gap, and
`drivers._refuse_inert_objective` is its numeric twin.

### Measured

`$PY -m functional_process.boundary --inert`: `st_regression` 14 design, 19 driven
condition(s) → **0 inert** (was 1: `.Objective`, `1/1 operand(s) frozen`). The other six
rows unchanged, `helias_5b`'s `.Constraint11` included.

`run_cold_matrix --input st_regression.IN.DAT --native --compare-process`, twice, and
once more with `MDF_TOLERANCE` rebound to the file's own `epsvmc = 1e-9` (§31.29.2's
method) — **all three runs bit-identical**:

| form | SQP | status | objf | PROCESS | d objf | worst dx | max\|eq\| | min ie |
|---|---|---|---|---|---|---|---|---|
| MDF | **10** | converged | `-16.588576507853947` | `-16.58857650779728` in **10** | `3.42e-12` | `1.18e-04` at x140 | `2.66e-15` | `2.83e-12` |
| SAND | **10** | converged | `-16.588576508201427` | ″ | `2.44e-11` | `1.59e-02` at x93 | `8.88e-16` | `3.74e-13` |

Both formulations, from a **native** cold start with no `DataStructure` anywhere in the
solve path, reach PROCESS's own answer in PROCESS's own iteration count. The `1e-8` and
`1e-9` runs agree to the last digit, so this row is converged well inside both stopping
criteria and the tolerance caveat §31.29.2 raises does not bite here.

SAND's `KeyError: VarPath(^cond.numerics.objf)` is gone, and it had the stated root: the
objective node was minted over a path the graph could not place, so the condition map had
no `^cond.numerics.objf` to look up. Verified, not assumed — the same build now assembles
and solves.

### The other six, controlled

The reference table in `reference_cold_matrix.txt` was measured at `2a6902f2` and the
tree has moved since, so a diff against it confounds this change with several others
(`stellarator_helias`'s MDF count is 66 at `HEAD`, against the pinned 108 — none of it
this node's doing). Measured instead against **`HEAD` with the one slot commented out**,
same command, same cache:

| configuration | form | baseline | with `FusionGain` |
|---|---|---|---|
| `large_tokamak_nof` | MDF | 7 it, `1.6`, d objf `1.14e-11`, max\|eq\| `2.39e-06` | **identical** |
| ″ | SAND | 10 it, `1.6`, d objf `4.83e-12`, max\|eq\| `7.19e-06` | **identical** |
| `stellarator_helias` | MDF | 66 it, `1.21775739` | **identical** |
| ″ | SAND | 169 it, `1.21775743` | **identical** |

Every solver column is bit-identical. The only columns that move are `graph` (246→247),
`nodes` and `blks` — the one node and its one schedule slot. The stellarator is
untouched by construction: its graph is 154 nodes before and after, and ownership is
per-graph (`TokamakCurrentDrive.electric_total`'s docstring makes the same argument for
`.heat_transport.p_hcd_electric_total_mw`).

`cold_start --write` moves exactly five rows of `reference_cold_start.txt`, each
`agree +1`, on the five tokamak configurations — `.current_drive.big_q_plasma` agrees
with PROCESS at the cold point on every one of them. No new `off` row, no change to
`errors` or `nocompare`. Both boundary pins re-check clean (tokamak 389, stellarator 295).

### What is *not* covered

`FusionGain` has no harness case of its own. The unit's Tier-1 evidence runs through
`calculate_current_drive_ecrh_primary_no_secondary`, the composite diffed sample-by-sample
against a live `CurrentDrive`, and this line is deliberately **not** in it: the composite
is scoped to the closure that produces the three `.tokamak.current_drive` boundary reads,
and `.physics.p_fusion_total_mw`/`.p_plasma_ohmic_mw` are two more `DataStructure` fields
a caller would have to set up for a two-line division. The value evidence is the cold
point instead — five configurations agreeing with PROCESS to `1e-9` on a path the port
now computes for itself — and it is weaker than a fuzz-and-gradient contract. Adding the
line to the composite (three more arguments, one more tuple slot) is the cheap way to
close it and is left as a follow-up rather than done blind.
