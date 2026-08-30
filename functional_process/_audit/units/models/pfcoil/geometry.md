---
kind: model-unit
status: draft
confidence: medium
---

**Ported (partial by switch, full by function).** `pfcoil/geometry.py` /
`test_geometry.py`: `calculate_cs_geometry`, `calculate_cs_turn_geometry_eu_demo`,
`place_cs_filaments`, `calculate_pf_coil_group_positions`,
`calculate_pf_coil_positions` — tier-1, the geometric half of `PFCoil.pfcoil()` and of
`CSCoil`. Four cottax nodes: `CSCoilGeometry` (`.tokamak.cs_coil.geometry`),
`CSCoilTurnGeometry` (`.tokamak.cs_coil.turn_geometry`), `PFCoilPlacement`
(`.tokamak.pf_coil.placement`), `PFCoilPositions` (`.tokamak.pf_coil.positions`).

## source

`process/models/pfcoil.py`, 5286 lines. In scope here:

| lines | what |
|---|---|
| `3005-3072` | `CSCoil.calculate_cs_geometry` — a `@staticmethod`, pure |
| `3074-3149` | `CSCoil.calculate_cs_turn_geometry_eu_demo` — a `@staticmethod`, pure (added 2026-08-30) |
| `3296-3319` | `ohcalc`'s `a_cs_turn` division and the five writes it feeds (added 2026-08-30) |
| `3151-3226` | `CSCoil.place_cs_filaments` — a `@staticmethod`, pure |
| `127`, `237-238`, `247-354` | `pfcoil()`'s `i_pf_location` dispatch loop and its `top_bottom`/`signn` state |
| `239-242` | `r_pf_outside_tf_midplane` |
| `1178-1263` | `PFCoil.place_pf_above_tf` — instance method, one `self.data` read |
| `1265-1343` | `PFCoil.place_pf_outside_tf` — likewise |
| `663-672` | the group-array → per-coil flattening (`ncl` loop) |
| `176-198`, `3237-3259` | the CS's own writes into the same per-coil arrays |

Out of scope in this file, UNPORTED with reasons:

| lines | what | why not |
|---|---|---|
| `1115-1176` | `place_pf_above_cs` | `i_pf_location = 1`; not reachable on this arm |
| `1345-1401` | `place_pf_generally` | `i_pf_location = 4`; not reachable on this arm |
| `1508-1565` | `tf_pf_collision_detector` | pure reporting (`logger.error`), writes nothing to `data`; and gated on `i_tf_shape == 2`, which is not this arm |
| `3074-3149` | `calculate_cs_turn_geometry_eu_demo` | feeds `.pf_coil.dz_cs_turn`/`dr_cs_turn`/`radius_cs_turn_cable_space` and `.cs_fatigue.*`, none of which any mass in this pass's closure reads |

## data footprint

`calculate_cs_geometry` (`CSCoilGeometry`):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.build.z_tf_inside_half` | read | explicit-arg | `pfcoil.py:170` |
| `.pf_coil.f_z_cs_tf_internal` | read | explicit-arg | `:171` |
| `.build.dr_cs` | read | explicit-arg | `:172` |
| `.build.dr_cs_bore` | read | explicit-arg | `:173` |
| `.pf_coil.z_cs_upper` | write | explicit-arg | `:176-178` |
| `.pf_coil.z_cs_lower` | write | explicit-arg | `:179-181` |
| *(`CSGeometry.r_cs_coil_middle`)* | — | **not a `VarPath`** | `PfCoilVariables` has no field of that name (checked against `dataclasses.fields`, not assumed). It is bit-for-bit `r_cs_middle` (`:3030` vs `:3042`) and PROCESS stores it only into `r_pf_coil_middle[n_cs_pf_coils - 1]` (`:182-184`), which `PFCoilPositions` owns. The pure function returns it, in the source's field order; `CSCoilGeometry.__call__` drops it |
| `.pf_coil.r_cs_middle` | write | explicit-arg | `:185` |
| `.pf_coil.z_cs_middle` | write | explicit-arg | `:186-188` |
| `.pf_coil.r_cs_outer` | write | explicit-arg | `:189-191` |
| `.pf_coil.r_cs_inner` | write | explicit-arg | `:192-194` |
| `.pf_coil.a_cs_poloidal` | write | explicit-arg | `:195` |
| `.pf_coil.a_cs_toroidal` | write | explicit-arg | `:196` |
| `.pf_coil.dz_cs_full` | write | explicit-arg | `:197` |
| `.pf_coil.dr_cs_full` | write | explicit-arg | `:198` |

`calculate_pf_coil_group_positions` (`PFCoilPlacement`):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.superconducting_tfcoil.r_tf_outboard_out` | read | explicit-arg | `:240` |
| `.pf_coil.dr_pf_tf_outboard_out_offset` | read | explicit-arg | `:241` |
| `.physics.rmajor` | read | explicit-arg | `:281` |
| `.physics.triang` | read | explicit-arg | `:282` |
| `.physics.rminor` | read | explicit-arg | `:283`, `:310` |
| `.physics.itart` / `.physics.itartpf` | read | switch | `:284-285` — see § switches touched |
| `.build.z_tf_inside_half` | read | **not declared** | `:286`, passed to `place_pf_above_tf` but read there only inside the `itart == 1` branch (`:1250-1253`). Not a read of this occupant; declaring it would be an invented edge |
| `.build.dz_tf_upper_lower_midplane` | read | explicit-arg | `:287` |
| `.build.z_tf_top` | read | explicit-arg | `:288` |
| `.pf_coil.rpf2` | read | explicit-arg | `:290` |
| `.pf_coil.zref` | read | explicit-arg | `:291`, `:311` — first four entries only |
| `.tfcoil.i_tf_shape` | read | switch | `:312` |
| `.pf_coil.i_r_pf_outside_tf_placement` | read | switch | `:313` |
| `.pf_coil.i_pf_location` | read | switch | `:248`, `:272`, `:302`, `:325` |
| `.pf_coil.rref` | read | **not declared** | `:338`, `place_pf_generally` only (`i_pf_location = 4`) |
| `.pf_coil.r_pf_outside_tf_midplane` | write | explicit-arg | `:239-242` |
| `.pf_coil.r_pf_coil_middle_group_array` | write | explicit-arg | `:265-267` etc. |
| `.pf_coil.z_pf_coil_middle_group_array` | write | explicit-arg | `:268-270` etc. |

`calculate_pf_coil_positions` (`PFCoilPositions`):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_coil.r_pf_coil_middle_group_array` | read | explicit-arg | `:667-669` |
| `.pf_coil.z_pf_coil_middle_group_array` | read | explicit-arg | `:670-672` |
| `.pf_coil.r_cs_middle` | read | explicit-arg | `:182-185` (the CS's slot) |
| `.pf_coil.r_pf_coil_middle` | write | explicit-arg | `:667`, `:182` |
| `.pf_coil.z_pf_coil_middle` | write | explicit-arg | `:670`, `:186-188` |

`place_cs_filaments` is called from `pfcoil()` (`:223-234`) and its three outputs
(`.pf_coil.r/z/c_pf_cs_current_filaments`) are **owned by no node in this package** —
see `fields.md` § "A PROCESS defect ported faithfully" and `currents.md`. The function
itself is ported and contract-tested; only the storage is refused.

## proposed signature(s)

```python
def calculate_cs_geometry(z_tf_inside_half, f_z_cs_tf_internal, dr_cs, dr_cs_bore) -> tuple  # 11
def place_cs_filaments(r_cs_middle, z_cs_inside_half, c_cs_flat_top_end,
                       f_j_cs_start_pulse_end_flat_top) -> tuple  # (r, z, c), each NFXF
def calculate_pf_coil_group_positions(rmajor, rminor, triang, rpf2, z_tf_top,
                                      dz_tf_upper_lower_midplane, zref,
                                      r_pf_outside_tf_midplane) -> tuple  # two (4, 2)
def calculate_pf_coil_positions(r_pf_coil_middle_group_array,
                                z_pf_coil_middle_group_array, r_cs_middle) -> tuple  # two (7,)
```

## cottax node

Three, all `ExplicitFunction`, in `functional_process/models/pfcoil/geometry.py`:
`CSCoilGeometry`, `PFCoilPlacement`, `PFCoilPositions`. Ownership as in the tables above.

## tier signal

**Tier 1** for all four. No iteration, no call into another `Model`, no CoolProp, no
external library.

**Sample provenance.** There is no `tests/unit/models/test_pfcoil.py` case for any of
these four, so the legacy points are taken the way the `process_port` env exists to allow:
`process.main.SingleRun` was run in-process on
`tests/regression/input_files/large_tokamak_eval.IN.DAT` to convergence and the resulting
live `DataStructure` read directly. Re-running `PFCoil.pfcoil()` on that state is
idempotent to `1e-13` relative, which is what makes a single-pass comparison meaningful
at all (the routine is one step of a Gauss-Seidel iteration — see `currents.md`).

**`calculate_pf_coil_positions` has no contract of its own.** It ports an inline
stretch of `pfcoil()` that PROCESS never exposes as a callable, so the only honest
tier-1 reference for it is `pfcoil()` itself; that is `test_masses.py`'s
`TestPFCoilChain`, which covers it. Writing a hand-rolled "reference" here would have
compared the port against a copy of the port.

## switches touched

| switch | reachable values | live on `large_tokamak_eval` | decision | evidence |
|---|---|---|---|---|
| `.pf_coil.i_pf_location[g]` | `1` `ABOVE_CS`, `2` `ABOVE_TF`, `3` `OUTSIDE_TF`, `4` `GENERALLY_PLACED` | `(2, 2, 3, 3)` (set in the file, `:244`) | **split** — one occupant per *group pattern*, since the per-group choice also decides how many coils each group has and therefore every downstream array index | `pfcoil.py:248-354`, four disjoint bodies with disjoint read sets (`ABOVE_CS` reads `r_cs_middle`/`dr_pf_cs_middle_offset`/`dr_tf_inboard`, `GENERALLY_PLACED` reads `rref`, neither of which `ABOVE_TF`/`OUTSIDE_TF` touch) |
| `.physics.itart` / `.physics.itartpf` | `0`/`1` | `0` / `0` (defaults) | **split** | `:1250-1253` — the `itart == 1 and itartpf == 0` arm places the coil from `z_tf_inside_half - zref[g]` and ignores `top_bottom`, `z_tf_top` and `dz_tf_upper_lower_midplane` entirely. A genuinely different read set |
| `.tfcoil.i_tf_shape` | `1` `D_SHAPE`, `2` `PICTURE_FRAME`, … | `1` (default) | **split** | `:1323-1333` — `PICTURE_FRAME` puts the coil at `r_pf_outside_tf_midplane` flat, dropping the `sqrt(r^2 - z^2)` and its `isinf` kludge |
| `.pf_coil.i_r_pf_outside_tf_placement` | `0`, `1` | `0` (default, `pfcoil_variables.py:287`) | **split**, same line as above | `1` collapses to the same flat placement as `PICTURE_FRAME` |
| `.build.iohcl` | `0`, `1` | `1` (default) | **split** | `:202-234` — `iohcl = 0` means no CS filaments at all and `c_cs_flat_top_end = 0` |

**UNPORTED switch values**, for `indat.py`'s `UNPORTED` table: every
`i_pf_location` pattern other than `(2, 2, 3, 3)`; `itart = 1`;
`i_tf_shape = PICTURE_FRAME`; `i_r_pf_outside_tf_placement = 1`; `iohcl = 0`.

**The coil-count topology is not a switch and not a port.** `n_pf_coil_groups = 4`,
`n_pf_coils_in_group = (1, 1, 2, 2)` and `n_cs_current_filaments = 7` are read once to
fix every array index in the package, which `_audit/naming_convention.md` §
"Switches are not ports" puts on the graph assembler. They are module constants in
`functional_process/models/pfcoil/__init__.py`, and `.pf_coil.n_cs_pf_coils`,
`.pf_coil.n_pf_cs_plasma_circuits` and `.pf_coil.nfxf` — which `pfcoil()` writes into
`data` purely as loop bookkeeping (`:140-158`, `:206`) — are correspondingly **owned by
no node**.

`pfcoil()` also *mutates one of its own inputs*: `:155` writes
`n_pf_coils_in_group[n_pf_coil_groups] = 1` to give the CS a group of its own. That is a
write to a run input, and it is not reproduced; the port's constants already account for
the CS.

## calls into other models

None. `place_pf_outside_tf` imports `TFCoilShapeModel` from
`process/models/tfcoil/base.py` but only to compare an integer.

## JAX-difficulty flags

- **`np.isinf` kludge** in `place_pf_outside_tf` (`pfcoil.py:1334-1339`) — ported as
  `jnp.where(jnp.isinf(r_raw), 1e10, r_raw)`, the same treatment `models/structure.py`
  gives the identical kludge in `coldmass`. The `logger.error` beside it is dropped.
- `jnp.sqrt(r_pf_outside_tf_midplane**2 - z**2)` goes NaN, not `inf`, when a coil is
  placed further from the midplane than the outboard TF radius. PROCESS's own kludge
  catches only `inf`, so both sides return NaN there and the harness's finiteness check
  is the thing that would notice; the fuzz bounds keep `rminor * zref` well inside
  `r_pf_outside_tf_midplane` rather than papering over it.
- No CoolProp, no elliptic integrals, no external library.

## open questions

- **Should `PFCoilPlacement` be four occupant classes (one per group) instead of one?**
  It is one class answering one *pattern* of `i_pf_location`, because the pattern also
  fixes the coil-index layout every other node in the package depends on. A per-group
  decomposition would be more faithful to the switch-per-occupant rule but would need
  the flattening to become a runtime concatenation of variable-length pieces. Not
  decided here.

## the spherical tokamaks miss on five of seven dimensions at once (measured 2026-08-30)

**This section said "four of seven" until 2026-08-30 and was wrong by one**, in exactly
the way `consolidation_round_3.md` §5 warns about. The ST closing wave probed
`_pf_coil_system_arm` past its first refusal, because that function returns at the
**first** deviating dimension and a single refusal message therefore sizes nothing — but
it probed by neutralising one dimension at a time, which still lets a short-circuit hide
the next. Evaluating all seven predicates independently (`_pf_coil_system_deviations`,
added 2026-08-30) says **five**. Both `spherical_tokamak_eval.IN.DAT` and
`st_regression.IN.DAT` deviate identically:

| arm | dimension | ST value |
|---|---|---|
| `-1` | `iohcl` | `0` — **no central solenoid at all** |
| `-2` | `n_pf_coil_groups` / `i_pf_location` / `n_pf_coils_in_group` | `4` / `(2, 3, 3, 4)` / `(2, 2, 2, 2)` — **not** the ported `(2, 2, 3, 3)` / `(1, 1, 2, 2)` pattern |
| `-3` | `itart` / `itartpf` | `1` / `1` — spherical tokamak PF placement |
| `-6` | `(i_pf_superconductor, i_cs_superconductor)` | `(9, 1)` — `HAZELTON_ZHAI_REBCO` PF conductor |
| `-7` | `i_tf_shape` / `i_r_pf_outside_tf_placement` | `2` (picture frame) / `1` |

`-4` (`i_pf_current`) and `-5` (`i_pf_conductor`) agree with the reference configuration.

`-2` is not a rounding error on the count: the group pattern fixes every array index in
`pfcoil/__init__.py`'s module constants, so it is a different occupant per node in the
same sense `-1` is. The two files place four *pairs* of coils in locations 2/3/3/4 where
the ported pattern places one, one, two and two in locations 2/2/3/3.

**The refusal now names all five.** `machine_from_indat` calls
`_pf_coil_system_deviations` and raises one `NotImplementedError` carrying every refused
dimension's recorded reason plus the count, instead of leaving `_slot_occupant` to
report the one arm index it can see. `test_machine.py::
test_the_pf_coil_refusal_names_every_deviating_dimension` pins the tuple, so the day one
of the five is ported the tuple shrinks in a test rather than in somebody's memory.

**This is a package, not an arm.** `-1` alone is what `UNPORTED` already says it is —
"a different occupant set for every node in the package", across `geometry.md`,
`currents.md`, `fields.md`, `masses.md` and `inductance.md` — and four more dimensions
sit behind it. The earlier brief's expectation that `pf_coil_system_arm == -3` was the
remaining item is off by four dimensions in one direction and by a whole subsystem in the
other.

Two things worth carrying forward:

- **`-6` is portable, contrary to the shape of the `i_tf_sc_mat = 9` refusals elsewhere.**
  `models/pfcoil.py:4851` has a real `HAZELTON_ZHAI_REBCO` branch calling
  `superconductors.hijc_rebco`. The "PROCESS has no arm at value 9" statement in
  `UNPORTED[("i_tf_sc_mat", 9)]` is about `models/stellarator/coils/coils.py`'s
  `jcrit_from_material`, which handles 1..8; it does not transfer to the tokamak PF path.
- **The refusal is invisible to `machine_survey`'s switch table**, because
  `pf_coil_system_arm` is a derived arm index and appears in no `IN.DAT`. That is why
  `report()` now ends with one real `machine_from_indat` attempt (`assembly_verdict`);
  see `physics.md`'s 2026-08-29 section for the other half of the same blind spot.


## the EU DEMO CS turn geometry (added 2026-08-30)

**Ported because `ncycle` needs it, not because this unit's own boundary asked for it.**
`CSCoil.calculate_cs_turn_geometry_eu_demo` writes six fields, and until 2026-08-30 not
one of them had a reader in this graph. `.tokamak.cs_fatigue` gave two of them one --
`.cs_fatigue.dr_cs_turn_conduit` and `.cs_fatigue.dz_cs_turn_conduit` are `ncycle`'s two
crack-size limits, i.e. the thresholds its integration stops at -- and
`boundary.unproduced_but_computed` reported both as missing producers the moment that
node landed.

**The numbers say why it could not be left.** `pfcoil_variables.py` gives the two
conduit thicknesses input defaults of `0.07` and `0.022`; `ohcalc` overwrites both with
`0.00990` on `low_aspect_ratio_DEMO`. A `CsFatigue` node reading the defaults would
therefore have produced a wrong `n_cycle` that looked entirely plausible, in place of a
wrong `n_cycle` of zero that did not -- the strictly worse failure, and the one the
missing-producer measure exists to prevent.

**One node, not two.** `a_cs_turn = a_cs_poloidal / n_pf_coil_turns[CS]`
(`pfcoil.py:3297-3300`) is folded into the same function: `.pf_coil.a_cs_turn` has no
other reader, so a node of its own would exist to hold one division.

### data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_coil.a_cs_poloidal` | read | explicit-arg | owned by `CSCoilGeometry`, this namespace's own first slot |
| `.pf_coil.n_pf_coil_turns` | read | explicit-arg | element `CS_INDEX` only; owned by `.tokamak.pf_coil.sizes`, and inside the PF package's four-node cycle |
| `.pf_coil.f_dr_dz_cs_turn`, `.radius_cs_turn_corners`, `.f_a_cs_turn_steel` | read | explicit-arg | run inputs; `f_a_cs_turn_steel` is iteration variable 123 on `low_aspect_ratio_DEMO` |
| `.pf_coil.a_cs_turn`, `.dz_cs_turn`, `.dr_cs_turn`, `.radius_cs_turn_cable_space` | **write** | explicit-arg | `pfcoil.py:3297`, `:3309-3313`; no reader in this graph |
| `.cs_fatigue.dr_cs_turn_conduit`, `.dz_cs_turn_conduit` | **write** | explicit-arg | `:3314-3319` — written by `ohcalc`, read only by `CsFatigue.ncycle`. PROCESS's own cross-area placement, reproduced |

### JAX-difficulty flags

- **`(a_cs_turn / f_dr_dz_cs_turn) ** 0.5`** — `needs-safe-pow`, resolved with
  `safe_sqrt`. Reachable at zero rather than theoretical: `a_cs_turn` is proportional to
  the CS cross-section, and `f_a_cs_turn_steel` -- an iteration variable -- sits under
  the cable-space square root beside it.
- **The `< 1 mm` clamp on `dr_cs_turn_conduit`** (`pfcoil.py:3138-3141`) —
  `needs-lax-cond-or-where`, resolved with `jnp.maximum`. It is a kink in the value, not
  only in the derivative, and the fuzz bounds keep clear of it; both tracked operating
  points sit an order of magnitude above it.

### suspected defects in PROCESS

**D2 — the 1 mm floor is applied to the radial conduit thickness and not the vertical
one**, though `pfcoil.py:3136-3141` sets them to the same number one line apart and
PROCESS's own `logger.error` beside the clamp calls it a kludge. Both are then read by
`ncycle` as two independent crack-size limits, so a run that reaches the clamp gets a
radial limit floored at 1 mm and a vertical limit left at whatever negative or
sub-millimetre value produced the clamp -- an asymmetry with no stated physical
justification. Reproduced as written; not reachable on any tracked input (both compute
~9.9 mm).
