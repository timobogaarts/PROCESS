# The ten cost nodes that read nothing the graph computes — input, or missing producer?

**Question asked**: `GRAPH` (159 nodes) has 19 nodes whose entire read set is unowned —
pure sources. Ten are `.costs.*`. A cost model that reads no computed quantity is reading
its masses, powers and areas from the boundary. For each of the ten, and for each variable
it reads: is the read unowned because PROCESS itself treats the field as an input, or
because the model that computes it is not ported?

**Short answer: all ten are settled; none is a port-coverage gap.** Zero of the **50**
distinct variables the ten nodes read is a field this port fails to produce that PROCESS
produces *on this run's path*. **But the brief's binary is missing its dominant case.**
The 50 split three ways, not two:

| answer | count | example |
|---|---|---|
| PROCESS treats it as an input (IN.DAT-settable or a module constant) | **32** | `.costs.uccase`, `.pf_coil.fcupfsu`, `.tfcoil.dcond` |
| **PROCESS computes it, but only on a path this configuration never executes** | **16** | 9 `.pf_coil.*`, 6 `.pf_power.*`, `.heat_transport.peakmva` |
| PROCESS computes it on *this* path, to an unconditional literal `0.0` | **2** | `.structure.gsmass`, `.structure.fncmass` |
| PROCESS never produces it at all — a real gap | **0** | — |

(25 of the 32 inputs are `.costs.*` unit-cost scalars; the other 7 are 5 `.pf_coil.*`
fractions/switches plus `.tfcoil.dcond` and `.tfcoil.j_crit_str_0`.)

That middle category is `boundary_inputs_audit.md` §4b's **category (d)**, and it is
neither "genuine input" nor "unported producer". It is *correct for this configuration and
wrong for any other*, which is a different kind of settled: it is settled conditionally,
on `istell != 0`, and the condition is recorded per row below so a tokamak re-run is
mechanical.

**The loudest thing found is not a gap but a hollow.** Fourteen of the fifteen outputs of
the three interesting nodes are **exactly zero on both sides**, and their agreement in the
MDA harness is therefore vacuous — `compare` already counts them in
`trivial_agreements`. `.costs.c2214` (reactor structure), `.costs.c2222` (PF magnets) and
`.costs.c2252` (PF coil power conditioning) are **identically 0.0 in PROCESS's own
converged stellarator run**, so three whole cost accounts are switched off and 21 of
`pf_magnet_cost`'s 27 declared reads are traced into arithmetic that is multiplied by an
empty loop. Nothing is wrong; a reader should simply not count those 14 as coverage.

**And the `.costs.coe` question is answered negatively and by measurement**: none of these
boundary inputs contributes to the `rel_diff = 1.733e-02`. Their contribution is exactly
zero on both sides, so their contribution to the *difference* is exactly zero. See §7.

*Status: investigation only. **No code, no registration, no harness change was made.***
Nothing under `~/jaxgraph` was touched. `total_process.py` and `_audit/next_steps.md` were
deliberately not edited (concurrently owned).

---

## 1. The measurement, reproduced

```python
from functional_process.total_process import GRAPH
owned = set(GRAPH.owners)
srcs = [n for n in GRAPH.nodes if not (set(GRAPH[n].reads) & owned)]
```

`len(GRAPH.nodes) == 159`, `len(srcs) == 19`. Verbatim output, with each node's read count:

```
.costs.reactor_structure_cost                 4
.costs.pf_magnet_cost                        27
.costs.pf_coil_power_conditioning_cost       15
.costs.fuelling_system_cost                   2
.costs.instrumentation_and_control_cost       2
.costs.maintenance_equipment_cost             2
.costs.switchyard_cost                        2
.costs.diesel_generators_cost                 2
.costs.auxiliary_facility_power_cost          2
.costs.misc_plant_equipment_cost              2
.stellarator.machine_config                   0
.stellarator.heating                          2
.stellarator.coils.winding_pack_geometry      3
.stellarator.coils.coil_casing                1
.stellarator.beam_current                     2
.stellarator.pulse_durations                  6
.physics.profiles.parameterisation.ecrh_density_limit   2
.physics.profiles.parameterisation.l_mode_profile_reset 0
.physics.profiles.profile_grid                0
```

`GRAPH` is `graph_for()`'s default, `REFERENCE_CONFIGURATION` (`total_process.py`) — the
helias run's switches (`istell = 6`, `isthtr = 1`, `i_plasma_pedestal = 0`,
`i_cost_model = 0`, `ireactor = 1`). Every claim below is for that configuration.

MDA harness state at the time of this audit, unchanged by it
(`$PY -m functional_process.run_mda_harness`):

```
agreements: 499 (of which array-valued: 23, both-sides-exactly-zero: 73)
disagreements: 34   (in driven blocks: 0)
unverifiable: 3     ungrounded inputs: 0     errors: 21
owned variables walked: 557, unaccounted: 0
static switch kwargs checked: 61, mismatched: 0, not data-backed: 3, unresolved: 0
```

## 2. Method

Three independent checks per variable, in this order; a variable is only called an input
when all three agree.

1. **Does any `process/models/**` code assign it?** An `ast` walk of every `.py` under
   `process/` outside `data_structure/`, collecting `Assign`/`AugAssign`/`AnnAssign`
   targets, recursing through `Tuple`/`List`/`Starred`/`Subscript`, and reconstructing the
   **full dotted target** so `.area.field` must match — the same scanner shape
   `boundary_inputs_audit.md` §2 justified (a bare regex both misses tuple-unpack writes
   and counts f-strings as writes). Re-implemented here rather than reused, and it
   reproduces §2's per-field results everywhere the two overlap.
2. **If it is assigned, is that assignment on this run's path?** Arbitrated by
   `process/core/caller.py:272-275`, which for `istell != 0` calls
   `self.models.stellarator.run()` and **returns immediately**, and by the call sequence
   inside it (`process/models/stellarator/stellarator.py:114-186`). Neither `PFCoil`,
   `CSCoil` nor `Power.pfpwr` appears anywhere in that sequence.
3. **What value does the field actually hold?** Both in a bare `DataStructure()` and in
   PROCESS's own converged run (`mda_harness.converged_data`, the cached
   `stellarator_helias` solve). A field whose default equals its converged value is one
   nothing wrote; a field whose converged value differs is one something did.

Check 3 is the one that makes the verdicts falsifiable rather than argued, and it is where
the `.times.*` rows below get their (different) answer.

## 3. The seven small nodes — settled, unremarkable

Seven of the ten read **nothing but `.costs.*` scalars**. Every one is either
IN.DAT-settable (confirmed against `process.core.input.INPUT_VARIABLES`, matching on
*name and area*) or a module constant in `cost_variables.py` that PROCESS's input parser
does not expose at all.

| node | reads | PROCESS status | verdict |
|---|---|---|---|
| `.costs.fuelling_system_cost` | `.costs.ucf1`, `.costs.fkind` | both IN.DAT inputs | **settled** |
| `.costs.instrumentation_and_control_cost` | `.costs.uciac`, `.costs.fkind` | both IN.DAT inputs | **settled** |
| `.costs.maintenance_equipment_cost` | `.costs.ucme`, `.costs.fkind` | both IN.DAT inputs | **settled** |
| `.costs.switchyard_cost` | `.costs.UCSWYD`, `.costs.lsa` | `UCSWYD` constant (`cost_variables.py:781`, not IN.DAT-settable); `lsa` IN.DAT (`:261` of the reference file) | **settled** |
| `.costs.diesel_generators_cost` | `.costs.UCDGEN`, `.costs.lsa` | `UCDGEN` constant (`:632`) | **settled** |
| `.costs.auxiliary_facility_power_cost` | `.costs.UCAF`, `.costs.lsa` | `UCAF` constant (`:569`) | **settled** |
| `.costs.misc_plant_equipment_cost` | `.costs.ucmisc`, `.costs.lsa` | `ucmisc` IN.DAT input | **settled** |

Zero dotted writes to any of these nine fields anywhere in `process/`. This confirms and
does not extend `boundary_inputs_audit.md` §3's aggregate treatment of the "35 `.costs.*`
constants" — it just names the four of them that these nodes actually read, and separates
them from the IN.DAT-settable `uc*` the same paragraph lumped in.

Their outputs are **real, non-trivial agreements** in the harness (`c2271 = 22.3`,
`c228 = 150.0`, `c229 = 300.0`, `c241 = 14.444`, `c244 = 5.338`, `c245 = 1.1775`,
`c25 = 22.125`). A node reading only inputs is not thereby vacuous; that is the contrast
that makes §4-§6 worth writing separately.

## 4. `.costs.reactor_structure_cost` — settled, and the brief's suspicion is wrong

| variable | PROCESS's own status | port status | verdict |
|---|---|---|---|
| `.costs.UCGSS` | module constant, `cost_variables.py:677`; not in `INPUT_VARIABLES` | n/a | input |
| `.costs.fkind` | IN.DAT input | n/a | input |
| `.costs.lsa` | IN.DAT input (`:261`) | n/a | input |
| `.structure.gsmass` | **computed on this path — to an unconditional literal `0.0`** | deliberately unported, documented | **settled** |

The brief reasons "that is suspicious for `reactor_structure_cost` — PROCESS computes a
reactor structure mass". **PROCESS's tokamak does; PROCESS's stellarator does not.**
`Stellarator.st_strc` (`process/models/stellarator/stellarator.py:334-337`) opens with

```python
self.data.structure.fncmass = 0.0e0
# Reactor core gravity support mass
self.data.structure.gsmass = 0.0e0   # ? Not sure about this.
```

with the method docstring's own explanation — "In practice, many of the masses are simply
set to zero to avoid double-counting of structural components that are specified
differently for tokamaks". The tokamak producer (`process/models/structure.py:47-52`,
tuple-unpacked, which is why a naive regex misses it) is unreachable behind
`caller.py:272-275`. The IFE producer (`process/models/ife.py:1665-1666`) likewise.

The port's decision is already on the record and is not merely implicit:
`functional_process/models/stellarator/stellarator_D_structure.py:11` — "`fncmass` and
`gsmass` are not ported: both are unconditional literal `0.0`". This **confirms**
`boundary_inputs_audit.md` §4a, which classified both as category (a) with the same
evidence.

**Measured consequence, which the existing records do not state**: `.costs.c2214` is
therefore **exactly `0.0`** in PROCESS's converged reference run. Account 221.4 — reactor
structure — contributes nothing to a stellarator's direct cost, and the whole node is a
constant zero with a structurally zero gradient with respect to every iteration variable.
That is faithful to PROCESS. Whether PROCESS *should* omit gravity-support cost for a
stellarator is a physics-model question this port has no business answering, but the
source comment `# ? Not sure about this.` suggests PROCESS is not sure either. Recorded,
not acted on.

## 5. `.costs.pf_magnet_cost` — settled, and the brief's reasoning is right

The brief's hypothesis — "this port models a stellarator and may have no PF coil model at
all, in which case every PF input is genuinely external" — is **correct, with one
refinement**: the fields are not *external* in PROCESS's data model, they are *unwritten*.
PROCESS has a full PF coil model; `caller.py:272-275` never reaches it on a stellarator, so
every `.pf_coil.*` field the cost model reads keeps its `pfcoil_variables.py` dataclass
default for the entire run.

| variable | PROCESS's own status | on this path? | port status | verdict |
|---|---|---|---|---|
| `.costs.cconfix`, `.cconshpf`, `.uccase`, `.uccu`, `.ucfnc`, `.ucsc`, `.ucwindpf`, `.fkind`, `.lsa` | IN.DAT inputs | — | n/a | input |
| `.costs.sc_mat_cost_0` | module constant, `cost_variables.py:770` | — | n/a | input |
| `.tfcoil.j_crit_str_0` | module constant list, `tfcoil_variables.py:353` | — | n/a | input |
| `.tfcoil.dcond` | IN.DAT input (list of 9 densities) | — | n/a | input |
| `.pf_coil.f_a_cs_void`, `.fcuohsu`, `.fcupfsu`, `.i_cs_superconductor`, `.i_pf_superconductor` | IN.DAT inputs, **never assigned** anywhere in `process/` | — | n/a | input |
| `.pf_coil.j_pf_coil_wp_peak` | `PFCoil.pfcoil`, `pfcoil.py:764` | **no** | unported | **settled (d)** |
| `.pf_coil.m_pf_coil_structure_total` | `PFCoil.pfcoil`, `pfcoil.py:1054,1061` | **no** | unported | **settled (d)** |
| `.pf_coil.n_pf_coil_turns` | `PFCoil.pfcoil`, `pfcoil.py:607,757,808,1079`; `waveform`, `:3284` | **no** | unported | **settled (d)** |
| `.pf_coil.r_pf_coil_middle` | `PFCoil.pfcoil`, `pfcoil.py:182,667`; `:3243` | **no** | unported | **settled (d)** |
| `.pf_coil.j_crit_str_pf` | `PFCoil.pfcoil`, `pfcoil.py:900,902` | **no** | unported | **settled (d)** |
| `.pf_coil.c_pf_cs_coils_peak_ma` | `PFCoil.waveform`, `pfcoil.py:2891,2904,2918,3264,3274` | **no** | unported | **settled (d)** |
| `.pf_coil.a_cs_cable_space` | `CSCoil.ohcalc`, `pfcoil.py:3548,3557` | **no** | unported | **settled (d)** |
| `.pf_coil.f_a_pf_coil_void` | `CSCoil.ohcalc`, `pfcoil.py:3322` | **no** | unported | **settled (d)** |
| `.pf_coil.j_crit_str_cs` | `CSCoil.ohcalc`, `pfcoil.py:3622,3626` | **no** | unported | **settled (d)** |
| `.structure.fncmass` | `st_strc`, literal `0.0` (§4) | yes | deliberately unported | **settled** |

Nine `.pf_coil.*` fields have producers PROCESS never calls here. Each was checked
against a `DataStructure()` default *and* the converged run:

```
pf_coil.a_cs_cable_space          default all-zero   converged 0.0
pf_coil.c_pf_cs_coils_peak_ma     default all-zero   converged all-zero (22,)
pf_coil.j_crit_str_cs             default all-zero   converged 0.0
pf_coil.j_crit_str_pf             default all-zero   converged 0.0
pf_coil.m_pf_coil_structure_total default all-zero   converged 0.0
pf_coil.n_pf_coil_turns           default all-zero   converged all-zero (22,)
pf_coil.r_pf_coil_middle          default all-zero   converged all-zero (22,)
pf_coil.j_pf_coil_wp_peak         default 3.0e7      converged 3.0e7   (untouched default)
pf_coil.f_a_pf_coil_void          default 0.3        converged 0.3     (untouched default)
```

**Default == converged for every one**, which is the direct evidence that nothing wrote
them. The port's boundary seed and PROCESS's stored value are therefore the same number by
construction, cold or converged, and there is no staleness to be had.

### 5.1 Twenty-one of the twenty-seven reads are dead here, not merely zero

The node's two loops are bounded by the **static** kwarg `.pf_coil.n_cs_pf_coils`, which
is `0` for this run (its `pfcoil_variables.py` default; the reference IN.DAT does not set
it), and `.build.iohcl`, which is `0`. Both loops unroll at trace time to **zero
iterations**, so `costs.py`'s

```
pfwndl = 0 -> c22222 = 0 ;  c22221 = 0 (empty loop, no CS block)
c22223 = scale * 1e-6 * uccase * m_pf_coil_structure_total   -> 0 (mass is 0)
c22224 = scale * 1e-6 * ucfnc  * fncmass                     -> 0 (mass is 0)
```

leaves **only six** of the twenty-seven declared reads with any influence at all
(`uccase`, `ucfnc`, `fkind`, `lsa`, `m_pf_coil_structure_total`, `fncmass`), and four of
those six are multiplied by an exact zero. The other twenty-one appear in the node's `In`
set and in no live expression.

This is **already stated** in the port's own docstring
(`functional_process/models/costs/costs.py:1438-1442`: "On the reference stellarator run
every output of this function is exactly zero … PROCESS never runs its PF coil model for a
stellarator. That is reproduced, not special-cased"), and the loop-bound reasoning is
`costs.md`'s open question 3, recorded there as **withdrawn**. This audit adds only the
count and the point that the node's *declared* read set overstates its dependence by 21
variables under this configuration — a structural observation, not a defect: cottax
declares a node's ports once, and the ports are correct for the general case.

## 6. `.costs.pf_coil_power_conditioning_cost` — settled, same cause, different producer

| variable | PROCESS's own status | on this path? | port status | verdict |
|---|---|---|---|---|
| `.costs.ucpfb`, `.ucpfbk`, `.ucpfbs`, `.ucpfcb`, `.ucpfdr1`, `.ucpfic`, `.ucpfps`, `.fkind` | IN.DAT inputs | — | n/a | input |
| `.pf_power.acptmax` | `Power.pfpwr`, `power.py:576,590` | **no** | unported | **settled (d)** |
| `.pf_power.ensxpfm` | `Power.pfpwr`, `power.py:562` | **no** | unported | **settled (d)** |
| `.pf_power.pfckts` | `Power.pfpwr`, `power.py:572` | **no** | unported | **settled (d)** |
| `.pf_power.spfbusl` | `Power.pfpwr`, `power.py:575` | **no** | unported | **settled (d)** |
| `.pf_power.srcktpm` | `Power.pfpwr`, `power.py:352,411` | **no** | unported | **settled (d)** |
| `.pf_power.vpfskv` | `Power.pfpwr`, `power.py:571` | **no** | unported | **settled (d)** |
| `.heat_transport.peakmva` | `Power.pfpwr`, `power.py:569` | **no** | unported | **settled (d)** |

`Power.pfpwr`'s only callers are `Power.run` (`power.py:54,81`), and
`stellarator.py:114-186` calls `tfpwr`, `component_thermal_powers`,
`calculate_cryo_loads`, `acpow` and `plant_electric_production` — never `run`, never
`pfpwr`. All seven fields are `0.0` in both a bare `DataStructure()` and the converged
run. `.costs.c2252` and its seven sub-accounts are all exactly `0.0`.

`Power.pfpwr` is **not ported anywhere** in `functional_process/` (grepped; the only
occurrences of the name are `boundary_inputs_audit.md`'s own prose). It is not
"written-but-unregistered" — there is nothing to register.

## 7. Written-but-unregistered producers: **none, in either direction**

The brief asks this to be called out loudly if true. It is not true here, and the check
was made rather than assumed:

- **`PFCoil` / `CSCoil` / `Power.pfpwr`** — no port exists. `unit_registry.md` has no unit
  for any of them; the only mention of `.pf_power.*` in it is `objectives.md`'s
  hole-in-MDA note that `pf_power.srcktpm` has no producer.
- **`.structure.gsmass` / `.fncmass`** — a port exists for the surrounding unit
  (`stellarator_D_structure.py`, registered) and deliberately excludes these two fields,
  with the reason in its module docstring. Registering them would add two constant-`0.0`
  nodes and change no number.
- The only ported-but-unregistered node anywhere in the cost unit is
  `TfMagnetCostResistive` (`costs.md`'s method inventory), which is `acc2221`'s
  `i_tf_sup != 1` arm and irrelevant to all ten nodes here.

So the cheap win the brief was hoping for does not exist in this set. The cheap wins in
this vicinity were already taken: `boundary_inputs_audit.md` §7 items 3-7 closed six such
producers (S4 blanket masses, `len_tf_coil`, `tfcryoarea`, `fusden_total`/`_alpha_total`,
`p_plasma_inner_rad_mw`, `Cryo`/`CryoLoads`) and this query returns nothing of that shape
left among cost sources.

## 8. Does any of this explain `.costs.coe`'s `rel_diff = 1.733e-02`? No — measured

**No, and the reason is arithmetic rather than argument: the three affected accounts are
exactly `0.0` on both sides, so their contribution to the difference is exactly `0.0`.**

```
costs.c2214 = 0.0   costs.c2222 = 0.0   costs.c2252 = 0.0    (PROCESS, converged)
```

and the port reproduces all three exactly, from identical boundary values (§5). A term
that is zero in both operands cannot appear in their difference. There is no
stale-versus-converged distinction available either, because §5's default-equals-converged
measurement shows nothing ever writes the inputs — this is not "the port uses a default
where PROCESS uses a converged value", it is "PROCESS also uses the default".

The actual cause is **already documented and is not a cost defect**:
`mda_harness.EXPLAINED_DISAGREEMENTS[".heat_transport.p_plant_electric_base_total_mw"]` —
PROCESS's own converged `DataStructure` is internally inconsistent, because
`Stellarator.run(output=True)` reruns `st_build`/`st_coil` in the opposite order to the
solve pass, moving `.build.z_tf_inside_half` from `4.1556` to `7.3592` and hence
`.buildings.a_plant_floor_effective` from `563075.16` to `680433.44`, while
`p_plant_electric_base_total_mw` is never recomputed in the report pass. Every
disagreement in that chain is one `+17.604 MW` offset propagated linearly.

Confirmed independently here on the current harness run, since the chain runs through a
node this audit touches (`.power.acpow` also reads `.heat_transport.peakmva` and
`.pf_power.srcktpm`): `.heat_transport.tlvpmw` `got = 314.0510405901006` against
`expected = 296.4472995612409`, **difference `17.6037410288597`** — the same offset to six
figures, so `acpow`'s two PF reads contribute nothing to it (they are `0.0` on both
sides and their coefficients drop out of the difference).

**Two corrections to the existing records fall out of this**, both stale rather than
wrong-at-the-time:

- `costs.md` open question 7 says "the residual `.costs.coe` disagreement is `VacuumOld`'s
  … `rel_diff` between 1.7e-06 and 4.1e-04". That was true when written and is now
  superseded: `VacuumOld` still contributes (its `c224` delta is visible at `1.195e-04`),
  but it is now the *smaller* of two causes, ~140x below the `a_plant_floor_effective`
  offset. The sentence should be read as "the Account-224 component of the residual".
- `unit_registry.md`:113 records `.costs.coe` `rel_diff = 1.704e-06`. That number is from
  before the `p_plant_electric_base_total_mw` chain appeared; the current value is
  `1.733e-02`. `boundary_inputs_audit.md` §6.3 already flagged the move and correctly
  declined to attribute it.

Neither file was edited (both are records of a different wave's state, and one is
concurrently relevant to another session).

## 9. Vacuous agreement — the finding worth carrying forward

Of the fifteen outputs owned by the three interesting nodes, **fourteen are
`compare`'s `trivial_agreements`** (both sides exactly zero everywhere):

```
.costs.reactor_structure_cost           c2214                                  TRIVIAL-ZERO
.costs.pf_magnet_cost                   c2222 c22221 c22222 c22223 c22224      TRIVIAL-ZERO x5
.costs.pf_coil_power_conditioning_cost  c2252 c22521..c22527                   TRIVIAL-ZERO x8
```

against the seven small nodes' seven **real** agreements. This is not a new hole —
`ComparisonReport.trivial_agreements` exists precisely to keep the headline count honest,
and reports 73 such agreements out of 499 for the whole graph. What this audit adds is
the attribution: **at least 14 of those 73 are one cause** — PROCESS's stellarator
branch never running its PF coil or PF power subsystem — and no amount of harness work
will make them informative, because there is no PROCESS behaviour there to reproduce.
The right way to exercise this arithmetic is a tokamak configuration, not a better
tolerance.

The same cause reaches **four further nodes** that are not pure sources and so do not
appear in the 19, measured by asking the graph for every reader of a `.pf_coil.*`,
`.pf_power.*` or `.structure.gsmass`/`fncmass` variable:

| node | reads from the dead PF subsystem |
|---|---|
| `.power.acpow` | `.heat_transport.peakmva`, `.pf_power.srcktpm` |
| `.buildings.sizing` | `.pf_coil.m_pf_coil_max`, `.pf_coil.r_pf_coil_outer_max` |
| `.availability.electric_production` | `.pf_coil.p_pf_electric_supplies_mw` |
| `.power.cryo_q_loads_step` | `.pf_power.ensxpfm` |

All five of those fields are `0.0` in the converged run, so these four nodes are correct
here and are each carrying a term that would come alive on a tokamak. Listed so that a
future tokamak configuration has its blast radius already enumerated: **closing the PF gap
is 19 boundary variables read across 6 nodes** (12 owned by `PFCoil`/`CSCoil`, 7 by
`Power.pfpwr`), and it is a whole-subsystem port
(`PFCoil` ~3600 lines, `CSCoil`, `Power.pfpwr`), not a node.

## 10. The six non-cost sources with reads — one line each

The three zero-read ones (`.stellarator.machine_config`, `.physics.profiles.profile_grid`,
`.physics.profiles.parameterisation.l_mode_profile_reset`) are sources by construction and
were not examined; `machine_config`'s input-free design is documented at
`total_process.py`'s `not-a-switch` list and confirmed by the harness's
`no_backing_field` entry for it.

| node | reads | verdict |
|---|---|---|
| `.stellarator.heating` | `.current_drive.eta_ecrh_injector_wall_plug` (IN.DAT `:160`, `= 0.5`), `.current_drive.p_hcd_primary_extra_heat_mw` (IN.DAT `:161`, `= 0.0`) | **settled** — zero dotted writes to either field anywhere in `process/`; both are IN.DAT inputs the reference file sets explicitly |
| `.stellarator.coils.winding_pack_geometry` | `.tfcoil.dx_tf_turn_general` (IN.DAT `:239`), `.dx_tf_turn_insulation` (`:240`), `.dx_tf_turn_steel` (`:241`) | **settled** — all three set by the reference IN.DAT. `dx_tf_turn_general` *is* written by `process/models/tfcoil/superconducting.py:2371,2373,2439,3821,3823`, but the stellarator imports nothing from that module (only `tfcoil.base.TFConductorModel`) and `caller.py` never reaches it: category (d), and inert since the input file pins the value |
| `.stellarator.coils.coil_casing` | `.tfcoil.dr_tf_nose_case` (IN.DAT `:243`, `= 0.06`) | **settled** — zero writes anywhere |
| `.stellarator.beam_current` | `.current_drive.e_beam_kev` (no writes, default `1000.0` = converged), `.current_drive.p_hcd_beam_injected_total_mw` | **settled (d)** — the second is written only by `heating.py:74` (`isthtr == 3`) and by `CurrentDrive` (tokamak). The port **refuses** the `isthtr == 3` arm outright (`total_process.py:1493`), so no configuration this graph can be built for needs a producer; the field is `0.0` in both default and converged |
| `.stellarator.pulse_durations` | six `.times.*` | **settled, but by a different mechanism than the others** — `t_plant_pulse_dwell` and `t_plant_pulse_fusion_ramp` have zero writes; the other four are written by `st_init` (`process/models/stellarator/initialization.py:43-46`) to constants (`t_plant_pulse_burn = 3.15576e7`, the rest `0.0`). The port's cold `DataStructure` comes from `SingleRun.__init__`, which runs `init_process` and therefore `st_init` before any model runs (`sand_harness.ReferenceRun.cold`'s docstring; `test_mdf.cold_run`), so the seed already carries those values. `initialization.md` records the decision to leave `st_init`'s 16 literal writes unported. **Would be a real cold-start gap under any entry point that seeded from a bare `DataStructure()`** — `t_plant_pulse_burn`'s dataclass default is `1000.0` against `st_init`'s `3.15576e7`, a factor of 31558 |
| `.physics.profiles.parameterisation.ecrh_density_limit` | `.physics.b_plasma_toroidal_on_axis`, `.stellarator.max_gyrotron_frequency` | **settled, and not covered by `boundary_inputs_audit.md`** — `max_gyrotron_frequency` has zero writes (default `1e9` = converged). `b_plasma_toroidal_on_axis` has exactly one apparent write, `process/models/stellarator/density_limits.py:204`, and it is on a `copy.deepcopy(stellarator)` proxy (`:185`) that is discarded, so it never reaches `data`. The field is **iteration variable 2** — a genuine optimiser unknown, not a computed quantity |

Only the `pulse_durations` row is more than a formality, and only conditionally: the
verdict rests on the port's cold-start path running `init_process`, which was checked in
the two places that define it, not assumed.

## 11. What this audit did not determine

- **Whether any of the 30 "genuine input" verdicts is seeded from the *wrong* field.**
  This audit asked whether PROCESS writes the field, not whether the port's `In` binding
  points at the field the PROCESS call site actually uses. That is exactly the
  `.neoclassics.r_eff` defect class (`boundary_inputs_audit.md` §6.1), where a field with
  zero writes was a genuine `DataStructure` input *and* the wrong binding. The MDA harness
  cannot see it either. No such check was made here for any of the 51 variables.
- **Whether `.costs.c2214`'s being identically zero is a PROCESS modelling error.** The
  source comment (`# ? Not sure about this.`) invites the question; answering it needs a
  physics judgement about stellarator gravity supports, not a port audit.
- **Any configuration other than `REFERENCE_CONFIGURATION`.** Every "(d)" verdict is (d)
  *for `istell != 0`*. A tokamak reclassifies all 19 PF-subsystem rows from settled to
  gap in one step. The producer and its file:line are recorded per row so the re-run is
  mechanical, but it was not performed.
- **The three interesting nodes' arithmetic was never exercised.** Fourteen of their
  fifteen outputs are trivially zero, so this audit can say the port agrees with PROCESS
  and cannot say the port's formulas are right. `test_costs.py`'s Tier-1 cases
  (`_reference_reactor_structure_cost` and siblings, with legacy and fuzz samples at
  non-zero masses) are the only evidence that they are, and that evidence was read but not
  re-run here.
- **`.costs.coe`'s residual was attributed, not decomposed.** §8 shows these boundary
  inputs contribute exactly zero and points at the already-documented cause; it does not
  reproduce the full `17.604 MW -> 1.733e-02` propagation term by term. The one link
  checked arithmetically is `tlvpmw`, because that is the link running through a node in
  this audit's scope.

---

## 12. Acted on: three of the nodes are gone from the tree (2026-08-25)

*Added after the fact. §§1-11 above are the investigation as it stood, unedited; this
section records what was done with it and what a tokamak has to undo.*

`model_tree_design.md` §8 step 4c deleted three of the four all-zero cost slots from
`total_process.py`. **`Costs` is 40 slots, not 43, and `GRAPH` is 156 nodes, not 159.**

| slot deleted | account | why, in one line |
|---|---|---|
| `.costs.pf_magnet_cost` | 222.2 | a stellarator has no PF coils (§5); 21 of 27 declared reads dead, not merely zero (§5.1) |
| `.costs.pf_coil_power_conditioning_cost` | 225.2 | same subsystem, different producer — `Power.pfpwr` is never called (§6) |
| `.costs.reactor_structure_cost` | 221.4 | `st_strc` sets both masses to a literal `0.0` to avoid double-counting a tokamak account (§4) |

**`.costs.energy_storage_cost` was deliberately kept.** Its zeros come from
`i_pulsed_plant`, a switch, not from a subsystem this device lacks: a pulsed stellarator
would want the account, so it belongs to the switch-conversion work
(`model_tree_design.md` §8 step 6), not here. That distinction — *device fact* versus
*switch setting* — is the whole basis on which the three above were separated from the
fourth, and it is the question to ask of any future all-zero node.

The reasoning is §9's, promoted from a caveat to a decision. Fourteen outputs agreed
vacuously, and a node computing a value the configuration never computes is the
`EcrhDensityLimit` bug class and the reason `WardTaylorAvailability` is unregistered.
These three landed on zero — the right answer — by luck, which is exactly why no gate
saw them.

Deleted outright rather than converted to switched slots: there is no tokamak occupant
to switch *against*, so a variant mechanism here would be a paradigm invented for
something unused. **`Costs` splits when the tokamak arrives.**

### 12.1 What the harness said, since that is the safety argument

Measured against a `git archive HEAD` copy with `cottax` pinned to its own `HEAD`:

```
before   agreements: 499 (array-valued: 23, both-sides-exactly-zero: 73)
         disagreements: 34   unverifiable: 3   ungrounded: 0
         owned variables walked: 557, unaccounted: 0
         static switch kwargs checked: 61, mismatched: 0, not data-backed: 3

after    agreements: 485 (array-valued: 23, both-sides-exactly-zero: 59)
         disagreements: 34   unverifiable: 3   ungrounded: 0
         owned variables walked: 543, unaccounted: 0
         static switch kwargs checked: 57, mismatched: 0, not data-backed: 3
```

All 14 lost agreements came out of `both-sides-exactly-zero`, and **the `all
disagreements:` block is byte-identical** — `.costs.c22 = 4810.593820158084`,
`.costs.cdirt`, `.costs.concost`, `.costs.coe` and the rest unmoved. The 4 lost kwargs
are `PfMagnetCost`'s `n_cs_pf_coils`/`iohcl`/`i_pf_conductor`/`supercond_cost_model`.

`.costs.c2214`, `.costs.c2222` and `.costs.c2252` are unowned boundary inputs now, seeded
from their `cost_variables.py:129/145/195` defaults of `0.0` — the same value the nodes
produced, which is why `c221`/`c222`/`c225` and everything above them do not move. Their
fourteen sub-accounts (`c22221`-`c22224`, `c22521`-`c22527`) had no reader in this graph
and simply left it.

**The boundary shrank, 348 → 320.** Three added, 31 removed, and *what* went is the
finding: every `.pf_coil.*` and `.pf_power.*` read this audit classified (d), plus
`.structure.fncmass`/`.gsmass`, plus ten `.costs.ucpf*`/`cconshpf`/`ucfnc`/`ucwindpf`
unit-cost scalars and `.tfcoil.dcond`. §4b's category (d) — "correct for this
configuration and wrong for any other" — is **empty in the cost model now**, because the
nodes that made it non-empty are gone.

### 12.2 What a tokamak has to restore

Not a rewrite: three re-registrations plus their producers. Every file:line below is
already in §§4-6 above, which is what makes this mechanical.

| restore | its unported producers, from §§4-6 |
|---|---|
| `PfMagnetCost` (`models/costs/costs.py:3122`) with real `n_cs_pf_coils`/`iohcl` | `PFCoil.pfcoil` — `pfcoil.py:182,607,667,757,764,808,900,902,1054,1061,1079`; `PFCoil.waveform` — `:2891,2904,2918,3264,3274,3284`; `CSCoil.ohcalc` — `:3322,3548,3557,3622,3626` |
| `PfCoilPowerConditioningCost` (`:2632`) | `Power.pfpwr` — `power.py:352,411,562,569,571,572,575,576,590`. Not ported anywhere (§7); there is nothing to register, only something to write. |
| `ReactorStructureCost` (`:2486`) | `structure.py:47-52` (tokamak, tuple-unpacked) or `ife.py:1665-1666` (IFE), for `.structure.gsmass`/`.fncmass` |

All three port functions and all three `cottax` node classes **remain in
`models/costs/costs.py`, unregistered and tested** — `tests/functional_process/models/
costs/test_costs.py` keeps its Tier-1 cases for all three (`_reference_pf_magnet_cost`,
`_reference_pf_coil_power_conditioning_cost`, `_reference_reactor_structure_cost`) at
non-zero masses, which is the only evidence their formulas are right (§11) and is
unaffected by the deletion. Nothing was deleted except three slot declarations and their
imports.

Two things a restorer must not carry over unexamined. **`iohcl = 0` was a deliberate
deviation from `build_variables.py:177`'s default of `1`**, recorded in the deleted
slot's own comment: with `n_cs_pf_coils == 0`, `iohcl = 1` computes `npf = -1` and prices
a central solenoid out of `r_pf_coil_middle[-1]`. A tokamak has both values for real and
neither is a free choice. And **the three cost nodes must come back as occupants of a
switched slot, not unconditionally**, or the restoration reintroduces the defect in the
other direction.
