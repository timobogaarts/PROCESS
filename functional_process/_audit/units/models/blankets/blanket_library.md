---
kind: model-unit
status: draft
confidence: high
---

**Ported (4 of the 14 entered functions; the other 10 are out of scope with evidence).**
This is the tokamak wave's companion to `hcpb.md`: `blankets/blanket_library.py` has no
registry row of its own and is reached **only as a base class** --
`CCFE_HCPB(OutboardBlanket, InboardBlanket)` (`process/models/blankets/hcpb.py:25`)
inherits from `BlanketLibrary` (`blanket_library.py:56`), while
`models.blanket_library` itself is constructed at `process/main.py:678` and never called.
`_audit/tokamak_call_surface.md` §A lists it as one of three files a `caller.py`-only
reading misses entirely.

## source

`process/models/blankets/blanket_library.py`, 3828 lines, **822 entered** across **14
entered functions** on the reference run (`tokamak_call_surface.md` §B). The entered set
was re-measured for this port with the same `sys.setprofile` hook over one
`Caller._call_models_once` on `tests/regression/input_files/large_tokamak_eval.IN.DAT`
and reproduced §B's fourteen exactly:

| # | entered function | in this port's scope? |
|---|---|---|
| 1 | `BlanketLibrary.component_volumes` (`:70-167`) | the orchestration; its four callees are ported, it is not a node itself |
| 2 | `BlanketLibrary.calculate_blkt_half_height` (`:169-232`) | **ported** (single-null arm) |
| 3 | `BlanketLibrary.calculate_elliptical_blkt_areas` (`:380-449`) | **ported** |
| 4 | `BlanketLibrary.calculate_elliptical_blkt_volumes` (`:451-530`) | **ported** |
| 5 | `BlanketLibrary.apply_coverage_factors` (`:532-584`) | **ported** (single-null arm) |
| 6 | `OutboardBlanket.blkt_outboard_poloidal_plasma_angle` (`:3639-3670`) | no -- writes `.blanket.deg_blkt_outboard_poloidal_plasma` only |
| 7 | `OutboardBlanket.f_deg_blkt_outboard_poloidal_plasma` (`:3672-3677`) | no -- writes `.blanket.f_deg_blkt_outboard_poloidal_plasma` only |
| 8 | `InboardBlanket.calculate_blkt_inboard_poloidal_plasma_angle` (`:3772-3799`) | no -- `.blanket.deg_blkt_inboard_poloidal_plasma` only |
| 9 | `BlanketLibrary.pipe_hydraulic_diameter` (`:3337-3368`) | no -- `.fwbs.radius_blkt_channel` only |
| 10 | `BlanketLibrary.set_blanket_module_geometry` (`:906-1081`) | no -- `.blanket.*`, `.fwbs.b_bz_liq`/`a_bz_liq`/`n_blkt_*_modules_poloidal` only |
| 11 | `BlanketLibrary.calculate_elliptical_inboard_blkt_segment_poloidal` (`:1661-1730`) | no -- callee of 10 |
| 12 | `BlanketLibrary.calculate_elliptical_outboard_blkt_segment_poloidal` (`:1732-1796`) | no -- callee of 10 |
| 13 | `InboardBlanket.calculate_blanket_inboard_module_geometry` (`:3801-3828`) | no -- `.blanket.len_blkt_inboard_segment_toroidal` only |
| 14 | `OutboardBlanket.calculate_blanket_outboard_module_geometry` (`:3721-3749`) | no -- `.blanket.len_blkt_outboard_segment_toroidal` only |

**The scope rule applied here is `next_steps.md`'s "minimal closure of functions that
produces the slot's listed output variables", and the ten exclusions are a measurement,
not a judgement.** Every write of functions 6-14 was collected and intersected with the
sixteen variables `_audit/tokamak_boundary.md` §`.tokamak.ccfe_hcpb` lists and with the
transitive reads of `CCFE_HCPB.component_masses`/`nuclear_heating_*`/`powerflow_calc`:
the intersection is **empty**. The `.blanket.*` and channel-geometry fields they write
are consumed by `primary_coolant_properties`/`thermo_hydraulic_model`, which live behind
`i_p_coolant_pumping == 2` and are *not entered* on this run (`tokamak_call_surface.md`
§D). They become live the moment that switch moves to 2 -- which is also the moment
CoolProp becomes live -- so this exclusion is scoped to the arm, not permanent.

**`.fwbs.vol_blkt_total` is the one variable this file exists to supply.**
`CCFE_HCPB.component_masses` reads it at `hcpb.py:306`, `:419`, `:425`, `:444` and
nothing else in the 38-file tokamak call surface writes it (grep for
`vol_blkt_total\s*=`: `blanket_library.py:581` and stellarator/IFE sites only). Without
this file, five of the slot's sixteen boundary variables have no producer.

## data footprint

### `calculate_blkt_half_height` (`:169-232`), single-null arm

`@staticmethod`, already pure -- no `self` access at all.

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.build.z_plasma_xpoint_lower` | read | explicit-arg | |
| `.build.dz_xpoint_divertor` | read | explicit-arg | |
| `.divertor.dz_divertor` | read | explicit-arg | |
| `.build.dz_blkt_upper` | read | explicit-arg | |
| `.build.z_plasma_xpoint_upper` | read | explicit-arg | **single-null arm only** (`:224`) |
| `.build.dr_fw_plasma_gap_inboard` | read | explicit-arg | single-null arm only |
| `.build.dr_fw_plasma_gap_outboard` | read | explicit-arg | single-null arm only |
| `.build.dr_fw_inboard` | read | explicit-arg | single-null arm only |
| `.build.dr_fw_outboard` | read | explicit-arg | single-null arm only |
| `.divertor.n_divertors` | read | switch | selects the occupant; **not a port** |
| `.blanket.dz_blkt_half` | write (via caller, `:77`) | own-write (returned) | |

The `n_divertors == 2` arm (`z_top = z_bottom`, `:221`) reads **five fewer** fields. That
is the concrete cost of a `jnp.where`: five invented edges on a double-null machine.

### `calculate_elliptical_blkt_areas` (`:380-449`) / `calculate_elliptical_blkt_volumes` (`:451-530`)

Both `@staticmethod`, already pure. Identical read sets except the volumes routine also
reads `.build.dz_blkt_upper`.

| VarPath | read/write | classification |
|---|---|---|
| `.physics.rmajor`, `.physics.rminor`, `.physics.triang` | read | explicit-arg |
| `.build.r_shld_inboard_inner`, `.build.dr_shld_inboard`, `.build.dr_blkt_inboard` | read | explicit-arg |
| `.build.r_shld_outboard_outer`, `.build.dr_shld_outboard`, `.build.dr_blkt_outboard` | read | explicit-arg |
| `.blanket.dz_blkt_half` | read | explicit-arg (produced by `BlanketHalfHeightSingleNull`) |
| `.build.dz_blkt_upper` | read | explicit-arg (**volumes only**) |
| `.build.a_blkt_inboard_surface_full_coverage`, `_outboard_`, `a_blkt_total_surface_full_coverage` | write (`:132-134`) | own-write (returned) -- areas |
| `.fwbs.vol_blkt_inboard_full_coverage`, `_outboard_`, `vol_blkt_total_full_coverage` | write (`:149-151`) | own-write (returned) -- volumes |

Both call `eshellarea`/`eshellvol` from `process/models/engineering/ivc_functions.py`
(`:99-130`, `:170-246`), ported here as the module-private `_eshellarea`/`_eshellvol`.
See the deviation note below.

### `apply_coverage_factors` (`:532-584`), single-null arm

Not a `@staticmethod` -- genuine `self.data` extraction.

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.divertor.n_divertors` | read (`:538`) | switch | selects the occupant; not a port |
| `.build.a_blkt_total_surface_full_coverage` | read | explicit-arg | |
| `.build.a_blkt_inboard_surface_full_coverage` | read | explicit-arg | read three times (`:547`, `:562`, `:578`) |
| `.fwbs.f_ster_div_single` | read | explicit-arg | `divertor.py:42` |
| `.fwbs.f_a_fw_outboard_hcd` | read | explicit-arg | |
| `.fwbs.vol_blkt_total_full_coverage` | read | explicit-arg | |
| `.fwbs.vol_blkt_inboard_full_coverage` | read | explicit-arg | read three times |
| `.build.a_blkt_outboard_surface` | write (`:540`/`:551`) | own-write (returned) | |
| `.build.a_blkt_total_surface` | write (`:561`) | own-write (returned) | |
| `.fwbs.vol_blkt_outboard` | write (`:566`) | own-write (returned) | |
| `.fwbs.vol_blkt_inboard` | write (`:575`) | own-write (returned) | a copy of the full-coverage value |
| `.build.a_blkt_inboard_surface` | write (`:577`) | own-write (returned) | also a copy |
| `.fwbs.vol_blkt_total` | write (`:581`) | own-write (returned) | **the target** |

**No self-loop.** `a_blkt_outboard_surface` and `a_blkt_total_surface` are read back
inside the body (`:563`, `:583` read the local just written), but the read is of a value
this same block produced, so both are `local-intermediate`, not a read of an owned field.
Verified line by line: the only fields read are the seven above, none of which this
function writes.

## proposed signature(s)

As written in `functional_process/models/blankets/blanket_library.py`. Four public pure
functions plus two module-private shell helpers:

```python
def calculate_blkt_half_height_single_null(...) -> float
def calculate_elliptical_blkt_areas(...) -> tuple[float, float, float]
def calculate_elliptical_blkt_volumes(...) -> tuple[float, float, float]
def apply_coverage_factors_single_null(...) -> tuple[float, ...]   # six, PROCESS's order
def _eshellarea(rshell, rmini, rmino, zminor)
def _eshellvol(rshell, rmini, rmino, zminor, drin, drout, dz)
```

## cottax node

Four `ExplicitFunction`s: `BlanketHalfHeightSingleNull`, `EllipticalBlanketAreas`,
`EllipticalBlanketVolumes`, `BlanketCoverageFactorsSingleNull`. Edges within the unit:
half-height -> areas and volumes (`.blanket.dz_blkt_half`); areas and volumes ->
coverage factors. `component_volumes` itself becomes **no node at all** -- it is
orchestration, and the tree's slot order carries what its call order carried.

Assembled together with `hcpb.py`'s eleven nodes, `Blocking.scc` returns **15 blocks of
size 1** in exactly PROCESS's own call order. There is no SCC in this subsystem once
`hcpb.py`'s two apparent self-loops are dissolved (see `hcpb.md`).

## tier signal

All four: **tier 1**. No internal iteration, no calls into other models, no CoolProp, no
`scipy`, no data-dependent loop or early exit.

## switches touched

| switch | value on the reference run | source | ported | unported arm's reason |
|---|---|---|---|---|
| `.divertor.n_divertors` | 1 | IN.DAT-derived; read out of the assembled `DataStructure` | single-null | `== 2` differs in *reads* in `calculate_blkt_half_height` (five fewer) and in a *literal* in `apply_coverage_factors` (`2.0 * f_ster_div_single`, `:544`). Under `next_steps.md` §14.2 both are separate occupants; neither is written here because the reference run cannot reach them. |
| `.fwbs.i_fw_blkt_vv_shape` | 2 (`ELLIPTICAL_SHAPED`, `build.py:26-30`) | default | elliptical | `D_SHAPED` (1) needs `calculate_dshaped_blkt_areas`/`_volumes` (`:234-378`) and hence ports of `dshellarea`/`dshellvol`. Not written. |
| `.physics.itart` | 0 | default (`physics_variables.py:994`) | conventional | `component_volumes:91-94` takes the D-shaped arm on `itart == 1 or i_fw_blkt_vv_shape == D_SHAPED`, so `itart == 1` is unported here for the same reason as `D_SHAPED`. It is a **joint key** over two switches, which is worth flagging for the consolidation pass -- the tree has joint keys already (`blktmodel`/`ipowerflow`, `switch_kwarg_survey.md` §4.3) but this is a new one. |

**`n_divertors` is classified as a switch here, and that is a judgement the consolidation
pass may want to revisit.** It is a *count*, and `models/vacuum/vacuum.py`'s existing port
reads it as an ordinary `From(divertor)` value (`vacuum.py:1019`) -- correctly, because
there it is used multiplicatively (`ndiv = n_divertors`) and branches nothing. Here it
appears only as `if n_divertors == 2`, i.e. as single-null vs double-null, which is
`_audit/tokamak_call_surface.md` §E's topology decision 3 (`i_single_null`) wearing a
different name. Treating it as a switch is what §14.2's rule implies; treating it as a
value would put five dead edges back into the single-null occupant. Recorded rather than
resolved: nothing else in the tree currently branches on it.

## calls into other models

`eshellarea`/`eshellvol` from `process/models/engineering/ivc_functions.py` -- plain
module functions, no `Model` instance, no state. Nothing else. `component_volumes`'s
other callees are all in-file and out of scope (see the source table).

## JAX-difficulty flags

- **`n_divertors`'s and the shape switch's branches** -- resolved structurally (one
  occupant per value), so no `jnp.where` and no `needs-lax-cond-or-where` flag remains.
- **`_eshellarea`/`_eshellvol` divide by `rmini`/`rmino`** (and by `rmini + drin`,
  `rmino + drout`). These go through zero when the radial build closes up, which is
  outside the model's domain rather than a porting problem; the harness's fuzz bounds are
  set tight around the reference point for exactly this reason, and the reason is
  recorded in `test_blanket_library.py` rather than left implicit. No `safe_pow`/
  `safe_sqrt` site: every exponent here is a positive integer.
- No CoolProp, no `scipy`, no `copy.deepcopy`, no logging side effect, no in-place array
  mutation anywhere in the four ported functions.

## deviations from PROCESS

1. **`_eshellarea`/`_eshellvol` are duplicated into this file** rather than imported from
   a port of `process/models/engineering/ivc_functions.py`. That module is not a `Model`
   and has no slot in the tree; it is imported by `fw.py`, `shield.py` and `vacuum.py`
   as well, all of which are other agents' scope in this wave. Twelve and forty lines
   respectively, transcribed with the arithmetic order preserved exactly (PROCESS's
   `elong = b / a` then `2.0 * np.pi * elong * (...)` is `2.0 * jnp.pi * (b / a) * (...)`
   here, which associates identically). **The consolidation pass should lift both into
   one `functional_process/models/engineering/ivc_functions.py`.**
2. **`component_volumes` is not a node.** PROCESS's method is pure orchestration once its
   four callees are nodes; making it one as well would give one node two owners for every
   field.

## open questions

1. **`n_divertors`: switch or value?** See "switches touched". The port has committed to
   *switch* for this file; `vacuum.py` has committed to *value* for its own use, and the
   two are compatible only because `vacuum.py` does not branch on it. A one-line policy
   from the consolidation pass would settle it for both.
2. **The `itart`/`i_fw_blkt_vv_shape` joint key.** `component_volumes:91` is
   `itart == 1 or i_fw_blkt_vv_shape == D_SHAPED` -- one arm selected by two switches, the
   same shape as `switch_kwarg_survey.md` §4.3's `blktmodel`/`ipowerflow` pair, which that
   survey found the factory feeding *arm indices* where switch *values* were expected.
   Whoever wires this slot should key it the way §4.3 recommends rather than re-deriving.


## 2026-08-27 — the `n_divertors == 2` arms ported (double-null wave)

Both of this unit's `n_divertors` slots are total now. Driven by
`tests/regression/input_files/spherical_tokamak_eval.IN.DAT:292` and
`st_regression.IN.DAT:638`, which set `i_single_null = 0`; `process/core/init.py:606-617`
derives `n_divertors = 2` from that, and `indat.py`'s `('n_divertors', 2)` refusal — which
named this file's two sites first — is gone.

| slot | single-null occupant | double-null occupant |
|---|---|---|
| `.tokamak.ccfe_hcpb.blanket_half_height` | `BlanketHalfHeightSingleNull` | `BlanketHalfHeightDoubleNull` |
| `.tokamak.ccfe_hcpb.blanket_coverage_factors` | `BlanketCoverageFactorsSingleNull` | `BlanketCoverageFactorsDoubleNull` |

Both are children of a new family base (`BlanketHalfHeight`, `BlanketCoverageFactors`),
which is what `models/blankets/namespace.py` now annotates the slots with — the
`ShieldHalfHeight` shape from `shield.md`, reused rather than reinvented.

**Half-height (`blanket_library.py:220-229`).** `z_top = z_bottom` on the double-null
arm, so PROCESS's `0.5 * (z_top + z_bottom)` collapses to `z_bottom` itself, and the
port writes it reduced. The reduction is exact in floating point as well as
algebraically (`x + x` is representable; `0.5 * (x + x)` is `x` to the bit), so this is
not a tolerance question. **Five fewer reads**: `z_plasma_xpoint_upper` and the four
gap/first-wall thicknesses are the whole of the `else` arm's `z_top` and a double-null
machine never looks at them. That reads-set difference is what makes this an occupant
rather than a `jnp.where` — folding the arms would declare five edges the machine does
not have.

**Coverage factors (`blanket_library.py:538-584`) — and a defect, transcribed.** The
`if n_divertors == 2` branch covers **only** the `a_blkt_outboard_surface` assignment
(`:541-547`), which subtracts `2.0 * f_ster_div_single`. The volume assignment
`vol_blkt_outboard` (`:565-572`) sits *below* the branch and subtracts the
single-divertor `f_ster_div_single` on both arms. So on a double-null machine PROCESS
removes two divertors' solid angle from the blanket **surface** and one divertor's from
the blanket **volume**. Nothing in the file justifies the asymmetry and no comment
mentions it; it reads as an arm that was edited where the branch was and not where it was
not. **Reproduced exactly, not repaired** (`traceability_policy.md`: the port's job is to
agree with PROCESS, including where PROCESS is wrong), and executed rather than argued
about by `test_blanket_library.py::TestApplyCoverageFactorsDoubleNull`, whose reference is
PROCESS's own bound `apply_coverage_factors()` at `n_divertors = 2` — had the port
"fixed" the volume line, that contract would fail. `models/fw.py`'s analogous coverage arm
is *symmetric* (both its assignments are inside the `if`), which is worth naming here,
because the two functions look like the same edit and only one of them was made
completely.

**Tests.** `TestBlktHalfHeightDoubleNull` and `TestApplyCoverageFactorsDoubleNull`, both
Tier 1, both at the same operating points as their single-null siblings so a difference
between the pair is the branch and not the domain. The half-height adapter **poisons**
the five parameters the arm does not read with `nan` rather than zeroing them, so "PROCESS
does not look at these" is executed: were the branch not taken, the reference would return
`nan` and the value comparison would fail instead of quietly agreeing on a zero. Green at
`--fp-gradients --fp-fuzz 40`.

No new boundary input: both double-null arms read strictly a subset of what the
single-null arms already read.
