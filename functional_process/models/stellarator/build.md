---
kind: model-unit
status: reviewed
confidence: high
---

**Ported.** `build.py` / `test_build.py`, four tier-1 contracts passing (fuzz only —
no existing PROCESS unit test covers `st_build`, see `test_build.py`'s module docstring).

## source
`process/models/stellarator/build.py` (439 lines, full file in scope). Two module-level
functions: `st_build` (the whole computation, one straight-line body gated by two
switches) and `output` (reporting shell, not audited further — no computation invoked,
unlike `density_limits.py`'s `output()`).

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.fwbs.blktmodel` | read | explicit-arg (switch) | topology-changing, see "switches touched" |
| `.build.blbuith`, `.blbmith`, `.blbpith` | read | explicit-arg | only inside `blktmodel > 0` branch |
| `.build.blbuoth`, `.blbmoth`, `.blbpoth` | read | explicit-arg | same |
| `.build.dr_shld_inboard`, `.dr_shld_outboard` | read | explicit-arg | read twice each: once inside the `blktmodel > 0` branch (feeds `dz_shld_upper`), again later unconditionally (feeds `r_shld_inboard_inner`/`r_shld_outboard_outer` etc.) — same value both times, no risk, but worth noting since it crosses the switch boundary |
| `.build.dr_blkt_inboard`, `.dr_blkt_outboard` | write (`blktmodel > 0`) **or** read as a plain external input (`blktmodel <= 0`) | **conditional-ownership-by-run-config** | identical pattern to `.physics.aspect` in `stellarator_C_geometry.md` — ownership is a run-configuration fact, not a static property of the function. See "cottax node" below for how the port handles this. |
| `.build.dz_shld_upper` | write (`blktmodel > 0` only) | explicit-arg | never read again *within this file*; consumed by `st_fwbs` output (`stellarator.py` ~line 1371, chunks 1E1-1E3) — confirmed by grep, out of this unit's scope to verify further |
| `.fwbs.radius_fw_channel`, `.fwbs.dr_fw_wall` | read | explicit-arg | |
| `.physics.rmajor`, `.physics.rminor` | read | explicit-arg | `rminor` used as a divisor (`a_fw_total` calc) — defaults to `0.0` in a bare `DataStructure()`, a real degenerate-input risk worth flagging for whoever assembles the graph (not a bug in this file, just a precondition: `rminor != 0`) |
| `.build.dr_cs`, `.dr_cs_tf_gap`, `.dr_tf_inboard` | read | explicit-arg | |
| `.build.dr_shld_vv_gap_inboard`, `.dr_vv_inboard` | read | explicit-arg | |
| `.build.dr_fw_plasma_gap_inboard`, `.dr_fw_plasma_gap_outboard` | read | explicit-arg | |
| `.stellarator.r_coil_minor`, `.f_coil_shape`, `.f_st_rmajor` | read | explicit-arg | |
| `.stellarator_config.stella_config_derivative_min_lcfs_coils_dist`, `.stella_config_rminor_ref` | read | explicit-arg | |
| `.build.gapomin`, `.dr_vv_outboard` | read | explicit-arg | |
| `.physics.a_plasma_surface` | read | explicit-arg | |
| `.build.dz_blkt_upper`, `.dr_fw_inboard`, `.dr_fw_outboard`, `.dr_bore`, `.rbld`, `.required_radial_space`, `.available_radial_space`, `.r_shld_inboard_inner`, `.r_shld_outboard_outer`, `.dr_tf_outboard`, `.dr_shld_vv_gap_outboard`, `.r_tf_outboard_mid`, `.z_tf_inside_half`, `.rspo` | write | explicit-arg | unconditional, straight-line, no cross-references outside this file found |
| `.heat_transport.ipowerflow` | read | explicit-arg (switch) | topology-changing, see "switches touched" |
| `.fwbs.fhole` | read | explicit-arg | both `ipowerflow` branches |
| `.fwbs.f_ster_div_single`, `.f_a_fw_outboard_hcd` | read | explicit-arg | `ipowerflow != 0` branch only |
| `.first_wall.a_fw_total` | write | explicit-arg | final value; which of the two `ipowerflow` branches owns it is a graph-assembly choice, see "cottax node" |

No `implicit-io`, `implicit-io-via-callee`, or `redundant-duplicate-write` in this file.
`local-intermediate`: `awall` (unnamed local, folded directly into the port's arithmetic,
no `data` field of its own — not tabulated).

## proposed signature(s)

Split three ways — the `blktmodel` preamble, the unconditional body, and the
`ipowerflow` split — per the switches. See `build.py`'s module docstring and function
docstrings for the exact signatures; not repeated here since the port already exists and
is the authoritative copy (`_audit/schema.md`: "the record stays the thing you read to
know what a port should look like" applies best when there's no port yet — once ported,
the `.py` file is more current than a duplicate written back into this record).

Names, for cross-reference: `calculate_blktmodel_blanket_thickness`,
`calculate_build`, `calculate_a_fw_total_no_powerflow`,
`calculate_a_fw_total_with_powerflow`.

## cottax node

**Actually written**, in `build.py` (`BlktmodelBlanketThickness`, `Build`,
`AFwTotalNoPowerflow`, `AFwTotalWithPowerflow` — all `ExplicitFunction`s), registered in
`functional_process/total_process.py`.

`Build.a_fw_total_unadjusted` mints an invented `VarPath`
(`.first_wall.a_fw_total_unadjusted`) — PROCESS never stores this value, it is a local
inside `st_build`. Whichever of `AFwTotalNoPowerflow`/`AFwTotalWithPowerflow` is wired in
(graph-assembly time, per `ipowerflow`) owns the real `.first_wall.a_fw_total`. This is
the resolution mechanism `naming_convention.md`'s "switches are not ports" describes:
neither variant node is instantiated except by the choice made when the graph is built.

`BlktmodelBlanketThickness` is symmetrically optional: instantiated (feeding `Build`'s
`dr_blkt_inboard`/`dr_blkt_outboard` inputs) only when `blktmodel > 0`; when it isn't,
those two names are external inputs to the graph instead. Not resolved here which is
PROCESS's default/more common case — a question for whoever assembles the full graph.

## tier signal
**Tier 1** for all four functions — no internal solve, no calls into other models, no
data-dependent Python control flow beyond the two switches (both resolved at
graph-assembly time, not as runtime branches inside any ported function). `output` is a
pure reporting shell (like `stellarator_G_output.md`), not audited further.

## switches touched
- `.fwbs.blktmodel` — **split** (topology: changes which fields are inputs vs. outputs
  of this unit, see data footprint). Confirms and extends the existing high-fan-out
  finding from chunks 1E1/1E2/1E3 (`switches.md` still needs a `blktmodel` entry — see
  `_audit/unit_registry.md`'s outstanding consolidation item). This is the *first* unit
  where `blktmodel`'s effect is read directly rather than inferred from reporting-branch
  density.
- `.heat_transport.ipowerflow` — **split**. Reads-sets differ (`fhole` alone vs. `fhole,
  f_ster_div_single, f_a_fw_outboard_hcd`), same shape as the `i_pflux_fw_neutron`/
  `ipowerflow` compound finding in `switches.md`'s existing entries. Also still needs its
  own `switches.md` entry (same outstanding item).

## calls into other models
None. `st_build` reads only `data.*`; no calls to another `Model`'s method anywhere in
this file.

## JAX-difficulty flags
None. Plain arithmetic throughout, no external calls, no dynamic shapes.

## open questions
1. **Which `blktmodel`/`ipowerflow` combination is PROCESS's actual default/common
   case?** Not answered here — both switches are left as genuine graph-assembly
   choices rather than one being silently assumed. Worth resolving once more of the
   `.fwbs`/`.heat_transport` switch family (`blktmodel`, `blkttype`, `ipowerflow`,
   already flagged pending consolidation from chunks 1E1-1E3) is looked at together.
2. **`dz_shld_upper` is only ever produced when `blktmodel > 0`.** What is its value
   supposed to be, structurally, when `blktmodel <= 0`? The source leaves it whatever a
   bare `DataStructure()` (or a previous run) already has — this is the same
   "conditional-ownership" shape as `dr_blkt_inboard`/`dr_blkt_outboard`, but for a field
   this file doesn't also consume as an input when it isn't the owner, so there's no
   symmetric "or it's an external input" story for it the way there is for the blanket
   thicknesses. Flagging rather than guessing.
3. **No existing PROCESS unit test covers `st_build`.** Every sample in `test_build.py`
   is `fuzz`; there is no "legacy" validated operating point pinning this file down the
   way `helias_5b.IN.DAT`-derived tests do for `density_limits.py`/`stellarator_D`/`_F`.
   Value/gradient agreement is still checked against a real call into PROCESS's own
   `st_build` (not re-derived), so this isn't a weaker *correctness* check, just a
   narrower *coverage* one — worth a legacy sample if/when a stellarator input file is
   run through PROCESS's own build stage for another purpose.
