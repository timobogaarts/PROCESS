---
kind: model-unit
status: draft
confidence: medium
---

**Ported (partial by switch, minimal closure by function).** `pfcoil/masses.py` /
`test_masses.py`: `calculate_pf_coil_sizes`, `calculate_pf_coil_masses` — tier-1. Two
cottax nodes: `PFCoilSizes` (`.tokamak.pf_coil.sizes`) and `PFCoilMasses`
(`.tokamak.pf_coil.masses`).

**This is the file that produces the three variables the wave asked for**:
`.pf_coil.m_pf_coil_conductor_total` and `.pf_coil.m_pf_coil_structure_total` (read by
`functional_process/models/structure.py::Structure`) and `.pf_coil.r_pf_coil_outer`
(read by `functional_process/models/cryostat.py::Cryostat`). `tokamak_boundary.md`
attributes `.tokamak.pf_coil` **zero** boundary reads, which was true when it was written
— `model_tree_design.md` §8 step 4c had deleted this producer's only consumers. Those
three reads are this wave's new consumers, so the boundary table is now out of date for
this slot; recorded here rather than smoothed over, per the wave brief's evidence
discipline.

## source

`process/models/pfcoil.py`:

| lines | what |
|---|---|
| `737-845` | winding-pack geometry loop (the `else`, `i_pf_location != 1`, arm at `:796-837`) |
| `849-1026` | per-coil mass loop |
| `1028-1046` | `itr_sum` — **UNPORTED**, see below |
| `1053-1064` | mass and current summations |
| `1067-1079` | the plasma's slot in the per-coil arrays |
| `3237-3294` | `ohcalc`'s CS slot: edges, peak current, turns |
| `3504-3506`, `3526-3530` | CS steel area and case thickness |
| `3539-3583` | CS steel and conductor mass, and the `a_cs_cable_space` fudge |

UNPORTED, itemised:

| lines | what | why not |
|---|---|---|
| `744-794` | the `i_pf_location = 1` sizing arm | a different occupant; it sizes from `dr_cs`/`z_tf_inside_half`/`dr_tf_inboard` and **writes** `.pf_coil.j_pf_coil_wp_peak`, which this arm only reads |
| `871-904` | `superconpf` and `j_pf_wp_critical`/`j_crit_str_pf` for the PF coils | not in the mass closure — a coil's steel area comes from the JxB force, not from any critical current. The ported fits in `models/physics/superconductors.py` are deliberately **not** used here rather than re-ported |
| `917-936` | resistive-coil power sum | `i_pf_conductor = RESISTIVE`, a different occupant |
| `1028-1046` | `itr_sum` | see § "The stale-turn read" |
| `3296-3324` | `a_cs_turn`, `calculate_cs_turn_geometry_eu_demo`, `.cs_fatigue.*`, `f_a_pf_coil_void[6]` | feeds the CS fatigue chain only |
| `3326-3400` | CS peak fields (`b_cs_peak_flat_top_end`, `b_cs_peak_pulse_start`, `b_pf_coil_peak[6]`, `bpf2[6]`) | needs `calculate_cs_self_peak_magnetic_field`; no mass here depends on it |
| `3403-3499` | the whole CS stress and fatigue chain | the only part of `ohcalc` that calls `scipy.special.ellipk`/`ellipe` (`:4102`, `:4187`), which would need a custom JAX primitive. Not in the closure |
| `3585-3679` | `superconpf` for the CS, `j_cs_critical_*`, `temp_cs_superconductor_margin` | as above |
| `3684-3703` | the resistive CS power arm | different occupant |
| `3704` | `calculate_cs_self_midplane_axial_stress_time_profile` | stress chain |

## The stale-turn read

`.pf_coil.itr_sum` is computed at `:1030-1046`, and its CS term reads
`n_pf_coil_turns[n_cs_pf_coils - 1]` — but `ohcalc()`, which *writes* that entry
(`:3284-3294`), does not run until `:1050`. So `itr_sum` consumes the **previous
pipeline pass's** CS turn count.

Measured, not inferred. Driving PROCESS's own `pfcoil()` from a cold `DataStructure`
with the reference run's inputs and `n_pf_coil_turns[6]` seeded at PROCESS's own
bootstrap value of `100.0` gives `itr_sum = 6.412e8`, where the converged run reports
`1.0559e9`; the difference is exactly
`(dr_cs_bore + dr_cs/2) x (4652.995 - 100.0) x 4e4 = 4.147e8`.

Producing it faithfully would mean declaring a *second*, distinct loop-carried input
(the CS turn count from the previous pass) beside the six PF ones `CSFluxSwing` already
needs; producing it from the current turn count would be a silently different number. It
is not on this pass's boundary — the 2015 cost model that reads it is not in this graph —
so it is **UNPORTED** and no node owns it.

## data footprint

`PFCoilSizes`:

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_coil.c_pf_cs_coils_peak_ma` | read | explicit-arg | `:801`, `:809`, `:3287` — **one edge of the SCC**, see `currents.md` |
| `.pf_coil.j_pf_coil_wp_peak` | read | explicit-arg | `:803` |
| `.pf_coil.c_pf_coil_turn_peak_input` | read | explicit-arg | `:810`, `:3291` |
| `.pf_coil.r_pf_coil_middle` | read | explicit-arg | `:817`, `:820` |
| `.pf_coil.z_pf_coil_middle` | read | explicit-arg | `:824-836` |
| `.pf_coil.pf_current_safety_factor` | read | explicit-arg | `:805` |
| `.pf_coil.r_cs_inner`, `.pf_coil.r_cs_outer` | read | explicit-arg | `:3250-3255` |
| `.pf_coil.z_cs_upper`, `.pf_coil.z_cs_lower` | read | explicit-arg | `:3237-3242` |
| `.physics.rmajor`, `.physics.rminor`, `.physics.kappa` | read | explicit-arg | `:1067-1078` |
| `.pf_coil.n_pf_coil_turns` | write | explicit-arg | `:808`, `:3284`, `:1079` |
| `.pf_coil.r_pf_coil_inner` | write | explicit-arg | `:816`, `:3253`, `:1073` |
| `.pf_coil.r_pf_coil_outer` | write | explicit-arg | `:819`, `:3250`, `:1076` |
| `.pf_coil.z_pf_coil_upper` | write | explicit-arg | `:831`, `:3237`, `:1067` |
| `.pf_coil.z_pf_coil_lower` | write | explicit-arg | `:823`, `:3240`, `:1070` |
| `.pf_coil.r_pf_coil_outer_max` | write | explicit-arg | `:840-843` |
| `aturn`, `area`, `dx` | — | local | `:799-814`; `area` is recomputed in `PFCoilMasses` from the same declared inputs rather than carried between nodes, since neither has a `VarPath` |

`PFCoilMasses`:

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_coil.c_pf_cs_coils_peak_ma` | read | explicit-arg | `:801`, `:964`, `:1064` |
| `.pf_coil.j_pf_coil_wp_peak` | read | explicit-arg | `:803` |
| `.pf_coil.n_pf_coil_turns` | read | explicit-arg | `:812`, `:912` |
| `.pf_coil.r_pf_coil_middle` | read | explicit-arg | `:911`, `:965`, `:1010`, `:3543` |
| `.pf_coil.r_pf_coil_inner`, `.pf_coil.r_pf_coil_outer` | read | explicit-arg | `:989-990` |
| `.pf_coil.z_pf_coil_upper`, `.pf_coil.z_pf_coil_lower` | read | explicit-arg | `:992-993`, `:3529` |
| `.pf_coil.b_pf_coil_peak[0..5]` | read | explicit-arg, **per index** | `:963` |
| `.pf_coil.bpf2[0..5]` | read | explicit-arg, **per index** | `:963` |
| `.pf_coil.f_a_pf_coil_void` | read | explicit-arg | `:950` |
| `.pf_coil.pf_current_safety_factor` | read | explicit-arg | `:805` |
| `.pf_coil.sigpfcf`, `.pf_coil.sigpfcalw` | read | explicit-arg | `:977-979` |
| `.fwbs.den_steel` | read | explicit-arg | `:1011`, `:3544` |
| `.tfcoil.dcond[2]` | read | explicit-arg, **exact index** | `:947-948`, `i_pf_superconductor - 1` |
| `.tfcoil.dcond[0]` | read | explicit-arg, **exact index** | `:3571`, `i_cs_superconductor - 1` |
| `.pf_coil.a_cs_poloidal` | read | explicit-arg | `:3505`, `:3549` |
| `.pf_coil.f_a_cs_turn_steel` | read | explicit-arg | `:3505` |
| `.pf_coil.f_a_cs_void` | read | explicit-arg | `:3567` |
| `.pf_coil.m_pf_coil_conductor` | write | explicit-arg | `:945`, `:3563` |
| `.pf_coil.m_pf_coil_structure` | write | explicit-arg | `:1006`, `:3539` |
| `.pf_coil.pfcaseth` | write | explicit-arg | `:996`, `:3526` |
| `.pf_coil.m_pf_coil_conductor_total` | write | explicit-arg | `:1058` |
| `.pf_coil.m_pf_coil_structure_total` | write | explicit-arg | `:1061` |
| `.pf_coil.m_pf_coil_max` | write | explicit-arg | `:1016` |
| `.pf_coil.ricpf` | write | explicit-arg | `:1064` |
| `.pf_coil.a_cs_steel_poloidal` | write | explicit-arg | `:3504` |
| `.pf_coil.a_cs_cable_space` | write | explicit-arg | `:3548-3559` |
| `.pf_coil.p_pf_coil_resistive_total_flat_top` | write | **not declared** | `:851` initialises it to zero and `:928` adds to it only on the resistive arm. On this occupant it is a constant zero, not a computed value |

## proposed signature(s)

```python
def calculate_pf_coil_sizes(c_pf_cs_coils_peak_ma, j_pf_coil_wp_peak,
                            c_pf_coil_turn_peak_input, r_pf_coil_middle,
                            z_pf_coil_middle, pf_current_safety_factor,
                            r_cs_inner, r_cs_outer, z_cs_upper, z_cs_lower,
                            rmajor, rminor, kappa) -> tuple  # five (8,) + scalar
def calculate_pf_coil_masses(...20 args...) -> tuple  # three (7,) + six scalars
```

## cottax node

`PFCoilSizes` and `PFCoilMasses`, both `ExplicitFunction`, in
`functional_process/models/pfcoil/masses.py`. Both own their arrays **whole** at
`NGC2` width, which is why the CS's and the plasma's slots are folded in here rather
than left to `CSCoilGeometry`: `Cryostat` reads `.pf_coil.r_pf_coil_outer` as one array,
and an array with three partial owners has no single producer for that read.

`.pf_coil.b_pf_coil_peak`/`.pf_coil.bpf2` are read **per index** (`[0..5]`), matching
`fields.PFCoilPeakField`'s per-index `Output`s; index 6 is the CS's own self-field,
UNPORTED, and no mass here depends on it.

## tier signal

**Tier 1.** No iteration, no CoolProp, no external library on this arm.

**Sample provenance and the shape of the contract.** `test_masses.py` carries **one**
contract, `TestPFCoilChain`, whose reference is `PFCoil.pfcoil()` itself, run on a cold
`DataStructure` seeded with exactly the port's declared inputs, and whose ported side is
this package's public functions composed in `pfcoil()`'s own order. That is deliberate:
`pfcoil()` is a single 1023-line routine and most of what this package ports — the
position flattening, the two current blocks, the flux swing, the time-point currents, the
sizing and the mass sums — are inline stretches of it with no callable of their own.
Testing those against a hand-written "reference" would be testing a copy of the port
against the port. One contract with the real routine as oracle is a stronger check than
seven with fabricated ones, and it is also the only thing that validates the array
index bookkeeping, which is where this model's real difficulty lives.

The four units that *do* have a separable PROCESS callable
(`calculate_b_field_at_point`, `calculate_coil_current_waveform`,
`calculate_pf_coil_peak_fields`, `calculate_efc_currents`, plus geometry's three) keep
their own narrower contracts, so a failure there localises what a failure here only
detects.

The legacy point is a converged in-process `SingleRun` of `large_tokamak_eval.IN.DAT`;
re-running `pfcoil()` on it is idempotent, which is what makes a single-pass comparison
of a Gauss-Seidel step meaningful. Fuzzing is `±10 %` around it, with `zref` and
`f_z_cs_tf_internal` held fixed (both are exercised by `test_geometry.py`, which can
bound them without dragging the whole machine along).

**Value tolerance `rtol = 5e-11`.** The chain runs the SVD solve twice and propagates its
answer through the sizing, the Green's-function field and the mass sums, so the `~2e-13`
scipy-versus-jax LAPACK disagreement (see `currents.md`) is amplified rather than
absorbed. Measured worst component `3e-13` at the reference point. No solver iterates on
either side, so this is round-off, not convergence noise.

## switches touched

| switch | reachable values | live on `large_tokamak_eval` | decision | evidence |
|---|---|---|---|---|
| `.pf_coil.i_pf_conductor` | `0` `SUPERCONDUCTING`, `1` `RESISTIVE` | `0` (default, `pfcoil_variables.py:230`) | **split** | `:944-957` (conductor density), `:970-1002` (steel area and case thickness), `:917-936` (resistive power), `:3562-3583`, `:3681-3703`. Four separate bodies with different read sets |
| `.pf_coil.i_pf_superconductor` | `1`-`9` | `3` = NbTi (`large_tokamak_eval.IN.DAT:246`) | **split** | `:947-948` — it is an index into `.tfcoil.dcond`; a `FromExactly(tfcoil.dcond[2])`, the same shape `models/tfcoil/superconducting.py:1499` uses |
| `.pf_coil.i_cs_superconductor` | `1`-`9` | `1` = ITER Nb3Sn (`:245`) | **split**, same shape | `:3571` → `.tfcoil.dcond[0]` |
| `.pf_coil.i_pf_location[g]` | `1`-`4` | `(2, 2, 3, 3)` | **split** | `:744-794` vs `:796-837`: the `ABOVE_CS` arm sizes from the build and **writes** `j_pf_coil_wp_peak`, which this arm reads |
| `.build.iohcl` | `0`, `1` | `1` | **split** | `:1049-1050` gates `ohcalc()` entirely — with no CS there is no CS mass and index 6 stays zero |

**UNPORTED switch values** for `indat.py`'s `UNPORTED` table:
`i_pf_conductor = RESISTIVE`; `i_pf_superconductor != 3`; `i_cs_superconductor != 1`;
`i_pf_location` patterns containing `1` or `4`; `iohcl = 0`.

The two superconductor switches are worth a note: their *only* effect inside this closure
is which element of `.tfcoil.dcond` is read, and two of the nine values share a density
(`dcond[0] == dcond[1] == dcond[3] == dcond[4] == 6080`). Per the wave binding policy
that is still a different occupant, not a parameter — an `i_*` integer may not be a
static kwarg even when two arms' arithmetic coincides (`istore` precedent).

## calls into other models

None on this arm. `ohcalc()` calls `cs_fatigue.ncycle` (`:3492`) and
`materials.calculate_tresca_stress`/`calculate_von_mises_stress` (`:3508`, `:3514`), all
inside the UNPORTED stress chain.

## JAX-difficulty flags

- **`if z_pf_coil_middle[i] < 0`** (`:782`, `:791`, `:826`, `:834`) — four sign flips
  deciding which coil edge is called "upper". `jnp.where`; a kink at `z = 0`, where the
  two branches agree in value but not in derivative. Not smoothed.
- **Issue #97's `a_cs_cable_space` fudge** (`:3552-3559`) — `jnp.where` on
  `a < 1e-4`. The replacement `da*da / (2*da - a)` is continuous and `C^1` at the
  threshold (both value and first derivative match at `a = da`), so the kink is only in
  the second derivative. PROCESS's own comment says as much; verified rather than taken
  on trust.
- **`max()` reductions** (`:840`, `:1016`) → `jnp.max`, one-hot gradient at the
  argmax. Standard.
- **`math.sqrt(area)`** (`:814`) and `sqrt(drpdz**2 + 4*areaspf)` (`:997`) — both
  arguments are strictly positive on any in-domain point (`area` is a magnitude,
  `drpdz**2 + 4*areaspf` is a sum of a square and a positive area), so no `safe_sqrt`.
  If `sigpfcf` or a peak field went negative, `areaspf` could push the second negative;
  the fuzz bounds stay on the physical side rather than papering over it.
- No CoolProp. **No `scipy.special`** — that is precisely what excluding the CS stress
  chain buys, and it is the reason this closure is portable at all today.

## open questions

- **`.pf_coil.p_pf_coil_resistive_total_flat_top` is a constant zero on this occupant
  and is owned by nobody.** PROCESS initialises it at `:851` and only the resistive arm
  adds to it. A consumer in `pf_power` will read it as a `DataStructure` default. Either
  this occupant should own it as an explicit zero, or it belongs in
  `KNOWN_UNVERIFIABLE_OUTPUTS`. Not decided here.
- **`tokamak_boundary.md`'s `.tokamak.pf_coil` row says zero boundary reads and is now
  wrong** — three reads landed this wave (`structure.py`, `cryostat.py`). Reported to the
  orchestrator rather than edited, since that file is not this unit's to change.
- **Should `PFCoilSizes` and `PFCoilMasses` be one node?** They are split because
  `PFCoilMasses` reads five arrays `PFCoilSizes` owns, and merging them would hide the
  SCC edge (`c_pf_cs_coils_peak_ma` in, `n_pf_coil_turns` out) inside one node rather
  than exposing it to `Blocking`. Flagged so the choice reads as deliberate.
