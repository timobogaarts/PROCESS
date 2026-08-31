---
kind: model-unit
status: draft
confidence: medium
---

**Ported 2026-08-31.** `functional_process/models/initialisation.py` /
`tests/functional_process/models/test_initialisation.py`, with the resolutions themselves
in `indat.py` beside `resolve_i_tf_bucking`. Eight nodes across seven slots, and the
acceptance test is a number rather than a description: **every one of the thirteen `off`
rows in the seven `reference_provider_*.txt` pins is gone, and no new disagreement
appeared.**

## source

`process/core/init.py` (1,302 lines) and `process/models/stellarator/initialization.py`'s
`st_init` (67 lines). The full audit of what they write is
`functional_process/_audit/init_audit.md`, which this record does not repeat. The three
facts it rests on:

- `init.py` writes **35** `DataStructure` fields and `st_init` **18**. Not one has a
  physics formula on its right-hand side.
- **12 of the 35 have a dataclass default that is not an answer** — 8 sentinels and 4
  presence flags — and two of the sentinels *look* like answers (`eyoung_ins` at `1e8` Pa,
  `eyoung_cond_axial` at `6.6e8` Pa, both replaced by two orders of magnitude).
- **13 writes land on a value neither the input file nor the dataclass default supplies.**
  They split **7 `init.py` / 6 `st_init`**, one behaviour each, and they are exactly the
  pins' `off` rows.

## what is ported

| node | owns | `init.py`/`st_init` | occupied on |
|---|---|---|---|
| `TfCryoplantEfficiency` | `.tfcoil.eff_tf_cryo` | `:933-940` | **7/7** |
| `TfInsulationYoungsModulus` | `.tfcoil.eyoung_ins` | `:961-975` | the 5 tokamaks |
| `TfConductorYoungsModulus` | `.tfcoil.eyoung_cond_axial`, `.eyoung_cond_trans` | `:992-1034` | the 5 tokamaks |
| `PfCoilResistivity` | `.pf_coil.rho_pf_coil` | `:1140` | the 5 tokamaks |
| `BeamElectronDensityFraction` | `.physics.f_nd_beam_electron` | `:1145-1147` | the 5 tokamaks |
| `EnergyStorageBuildingVolume` | `.buildings.esbldgm3` | `:827` | the 4 non-pulsed |
| `DoubleNullUpperBuild` | `.build.dz_shld_upper`, `.dz_vv_upper` | `:610-612` | the 2 double-null tokamaks |
| `StellaratorSolenoidAbsent` | `.build.dr_cs`, `.build.dr_cs_tf_gap` | `st_init:23,26` | the 2 stellarators |
| `StellaratorPulseTimes` | the four `.times.t_plant_pulse_*` phase durations | `st_init:43-46` | the 2 stellarators |

Fourteen fields, thirteen of them `off` rows; the fourteenth is `.tfcoil.eyoung_cond_trans`,
a **latent** row (`init_audit.md` §5c) that agreed only because its branch did not fire.
`.build.dz_vv_upper` is the same shape: the double-null branch writes it, and on both
double-null files it writes the value that was already there.

## proposed signature(s)

Every occupant is an `ExplicitFunction`. Seven of the nine have **no reads at all** and
carry their answer as a static field; `DoubleNullUpperBuild` reads `.build.dz_shld_lower`
and `.build.dz_vv_lower` and is the only one of the nine with a graph edge.

A node with no reads looks degenerate and is not, and the reason is the shape of what is
being ported: these values are *literature constants selected by a switch* (20 GPa ITER
insulation, 32/80/6.8/145 GPa conductor moduli with their DOIs, 0.13 ITER cryoplant
efficiency), and a switch is resolved at assembly in this port and never read as a graph
value. What reaches the graph is one number. What the node buys is that the number has an
**owner** — and an owned path is not a boundary path, which is the whole mechanism
(`next_steps.md` §24.1).

The resolutions live in `indat.py` (`resolve_eff_tf_cryo`, `resolve_eyoung_ins`,
`resolve_eyoung_cond`, `resolve_rho_pf_coil`, `resolve_f_nd_beam_electron`,
`resolve_esbldgm3`), following `resolve_i_tf_bucking`'s precedent: one copy of each rule,
in the module that reads the file, taking the file's raw value and this machine's
switches.

## decisions

**1. Raw → resolved, with `raw` static rather than a graph root.** §24.2 item 2 asks for
`.raw.<area>.<field>` to be a read so a sentinel resolution is an edge instead of a
self-loop. It is an edge here, but the raw side is assembly-time data, for two measured
reasons. First, no model writes any of these fields, so the only way one could move during
a solve is by being an iteration variable — checked, not assumed, by
`indat._refuse_seed_owned_unknowns`, the same guard `_quench_helium_table` carries for the
helium property table. Second, **a `.raw.*` root would be seeded at `0.0` in silence**:
`mdf.seed` grounds a `VarPath` it cannot resolve on the `DataStructure` at `0.0`
(`mdf.py:411-414`) rather than raising, so every sentinel would resolve against zero and
no value test could see it. Static is the spelling that cannot do that. If a `raw` root
ever becomes seedable, these nodes gain a read and lose a field; nothing else moves.

**2. A slot is empty where the write has no next use.** Two kinds of emptiness, and both
are `None` occupants rather than kwargs (`next_steps.md` §14.2):

- **PROCESS does not write.** `EnergyStorageBuildingVolume` is absent on a pulsed plant
  and `DoubleNullUpperBuild` on a single-null machine, because there `init.py` writes
  nothing and the file's own value is the answer. An identity node would remove a genuine
  input from the boundary and put nothing in its place.
- **Nothing in this port reads it.** The two Young's moduli, the PF resistivity and the
  beam fraction are absent on a stellarator: their only readers are the TF stress chain
  and the tokamak physics nodes, and `.tfcoil.eyoung_ins` is not a boundary path of the
  assembled stellarator graph at all. `.build.dz_shld_upper` has a further reason —
  `models/stellarator/build.py:343` already owns it there, so an occupant would be a
  second producer.

**3. `init.py:610`'s third write is deliberately not ported.** The double-null branch also
sets `dz_fw_plasma_gap = dz_xpoint_divertor`, and on both double-null configurations
`.build.dz_fw_plasma_gap` is not a boundary path — nothing in the assembled graph reads
it. Porting it would be a write with no next use. It joins the moment a reader does.

**4. `st_init` is a variant arm, not a parse step.** Its two slots are `None` on a tokamak
because `st_init` returns at its first line on `istell == 0`. `init_audit.md` §4 corrected
§22.6's attribution of the zeroed solenoid and the rewritten pulse times to `init.py`;
this is that correction in the tree.

## PROCESS defects reproduced, not repaired

- **`init.py:933-940` has no copper arm.** A water-cooled copper magnet's `eff_tf_cryo`
  stays at the `-1.0` sentinel, and `.power.thermal_cryo` divides by it.
  `resolve_eff_tf_cryo` returns the sentinel there and `test_the_cryoplant_efficiency_
  sentinel_resolves_by_conductor` pins it. No tracked file is copper.
- **`init.py:961`'s test is `<= 1e8`, an inequality.** A file asking for a genuinely soft
  insulator gets ITER's 20 GPa instead of its own number. Transcribed, not tightened.
- **`init.py:611` destroys a genuine input.** `dz_shld_upper` is overwritten by
  `dz_shld_lower` on a double-null machine; PROCESS logs a warning and does it anyway. It
  is why that row is `off (input/indat)` rather than `off (default/defaults)` — the file
  said `0.3` and PROCESS solved with `0.6`.

## open questions

**OQ1. The material table's home is still §14.5's question.** `eyoung_cond_axial`'s
literature values are `dcond[]`-shaped — engineering data keyed on a switch — and
`init_audit.md` §7.3 says they belong wherever `dcond[]` ends up rather than in the
provider. They are in `indat.EYOUNG_COND_AXIAL_LITERATURE` now, keyed on
`SuperconductorMaterial` rather than on `i_tf_sc_mat`, which is the part of the answer that
does not depend on where the table lives.

**OQ2. Zero of `init.py`'s 51 raises and 16 warnings are ported**, including the dozen that
encode physics-validity ranges. Out of scope here and named in `init_audit.md` §7.6.

**OQ3. The four presence flags and `boundl[3] = teped * 1.001` are structurally not nodes**
(§24.2 items 1 and 3). Two of the four are already answered by
`indat.presence_flags_from_indat`; the bound belongs to the problem statement.

**OQ4. `initialise_imprad` is a fifth initialisation source** owning 4 `derived` rows on
every pin. Untouched.

## verification

- `tests/functional_process/models/test_initialisation.py` — 44 cases. **The resolvers arm
  by arm** against `init.py`'s branches, including arms no tracked file takes; **and the
  assembled machines against a real `init_process`**, field by field, on all seven
  configurations, with exact equality. That second half is the oracle the nodes took away:
  a path a node produces is no longer a boundary path, so `provider.disagreements` stops
  checking it, and the check has to come back somewhere.
- `tests/functional_process/test_machine.py::NODE_COUNTS` — +5 on the three single-null
  pulsed tokamaks, +7 on the two spherical tokamaks, +4 on the two stellarators, each split
  asserted rather than left to the total.
- `tests/functional_process/test_boundary.py` — the stellarator's input half 297 → 289 and
  the tokamak's 384 → 378, both regenerated pins.
- The seven provider pins, regenerated: **13 `off` rows → 0**, with `from_process`
  unchanged on every configuration (32/27/52/35/51/31/41), which is what says the paths
  left the boundary rather than moving to the seed.
