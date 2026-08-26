---
kind: model-unit
status: draft
confidence: medium
---

**Ported (partial by switch, full by function).** `structure.py` / `test_structure.py`:
`calculate_structure_masses`, tier-1, the sole live occupant of `.tokamak.structure`.
`tokamak_boundary.md` lists three of this function's five outputs as boundary reads
(`aintmass`, `clgsmass`, `coldmass`); all five are ported and owned together since they
come from PROCESS's one `structure()` function via tuple unpacking (per the wave-1
brief's explicit instruction for this unit) -- `fncmass`/`gsmass` currently have no
reader in the graph, but nothing is saved by dropping them from one already-written
function.

## source

`process/models/structure.py`, 232 lines, full file in scope. `Structure.run()`
(31-74, the stateful shell) forwards into `Structure.structure()` (76-231, the
computation). No other model is called; no CoolProp; no internal iteration.

## data footprint

`Structure.run()`'s own body, before calling `structure()`:

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_coil.m_pf_coil_conductor_total` | read | explicit-arg | `structure.py:43` |
| `.pf_coil.m_pf_coil_structure_total` | read | explicit-arg | `structure.py:44` |

Summed into `total_weight_pf`, a local, then passed to `structure()` as `pfmass` --
folded into the port's `Structure.__call__` directly rather than given its own function
(one line, no branching).

`structure()`'s own reads (all `explicit-arg`; the `output` reporting block,
lines 197-229, reads nothing new and writes only to `self.outfile` -- dropped, pure
reporting):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.plasma_current` | read | explicit-arg | `ai`, `structure.py:54` |
| `.physics.rmajor` | read | explicit-arg | `r0`, `:55` |
| `.physics.rminor` | read | explicit-arg | `a`, `:56` |
| `.physics.kappa` | read | explicit-arg | `akappa`, `:57` |
| `.physics.b_plasma_toroidal_on_axis` | read | explicit-arg | `b0`, `:58` |
| `.tfcoil.i_tf_sup` | read | switch | `:59` -- see § switches touched |
| `.pf_coil.i_pf_conductor` | read | switch | `:60` -- see § switches touched |
| `.build.dr_tf_inner_bore` | read | explicit-arg | summed into `tf_h_width`, `:61-63` |
| `.build.dr_tf_outboard` | read | explicit-arg | same sum |
| `.build.dr_tf_inboard` | read | explicit-arg | same sum |
| `.build.z_tf_inside_half` | read | explicit-arg | `tfhmax`, `:64` |
| `.fwbs.whtshld` | read | explicit-arg | `shldmass`, `:65` |
| `.divertor.m_div_plate` | read | explicit-arg | `dvrtmass`, `:66` |
| `.tfcoil.m_tf_coils_total` | read | explicit-arg | `tfmass`, `:68` |
| `.fwbs.m_fw_total` | read | explicit-arg | `:69` |
| `.fwbs.m_blkt_total` | read | explicit-arg | `blmass`, `:70` |
| `.fwbs.m_fw_blkt_div_coolant_total` | read | explicit-arg | `:71` |
| `.fwbs.dewmkg` | read | explicit-arg | `dewmass`, `:72` — produced by `Cryostat.external_cryo_geometry` (`process/models/cryostat.py:83-85`), which this pass's `.tokamak.cryostat` occupant does **not** produce (out of that unit's minimal closure — only `r_cryostat_inboard` is ported there). A genuine cross-unit boundary read until/unless a future `.tokamak.cryostat` occupant is extended to own `dewmkg` too |
| `.structure.fncmass` | write | explicit-arg | `:48` |
| `.structure.aintmass` | write | explicit-arg | `:49` |
| `.structure.clgsmass` | write | explicit-arg | `:50` |
| `.structure.coldmass` | write | explicit-arg | `:51` |
| `.structure.gsmass` | write | explicit-arg | `:52` (source local `gsm`; the `DataStructure` field is `gsmass` -- `structure_variables.py:22`) |

## proposed signature(s)

```python
def calculate_structure_masses(
    ai, r0, a, akappa, b0, tf_h_width, tfhmax, shldmass, dvrtmass,
    pfmass, tfmass, m_fw_total, blmass, m_fw_blkt_div_coolant_total, dewmass,
) -> tuple[float, float, float, float, float]:  # fncmass, aintmass, clgsmass, coldmass, gsm
```

## cottax node

`Structure(ExplicitFunction)`, in `functional_process/models/structure.py`. Owns all
five `.structure.*` outputs; reads the fourteen `VarPath`s in the table above (`i_tf_sup`
and `i_pf_conductor` excluded — see next section).

## tier signal

**Tier 1.** No iteration, no calls into another model's method, no CoolProp. `output`'s
reporting arm is pure `po.ovarre` printing, dropped.

**Sample provenance.** `tests/unit/models/test_structure.py::TestStructure.test_structure`
provides one full legacy point with a hand-checked expected 5-tuple, `i_tf_sup=1`,
`i_pf_conductor=PFConductorModel.SUPERCONDUCTING` — exactly this occupant's live
combination — used verbatim as this unit's legacy sample.

## switches touched

| switch | reachable values | live on `large_tokamak_eval` | decision | evidence |
|---|---|---|---|---|
| `.tfcoil.i_tf_sup` | `0` (resistive), `1` (superconducting) | `1` (PROCESS default, `tfcoil_variables.py:261`, not set in the reference file) | **split** (per wave-1 policy: a switch read to branch selects an occupant, never a static kwarg) | `structure.py:165-166`: `if i_tf_sup == 1: coldmass += tfmass + aintmass + dewmass`. Changes `coldmass`'s reads-set (whether `tfmass`/`aintmass`/`dewmass` feed it at all) |
| `.pf_coil.i_pf_conductor` | `0` (`SUPERCONDUCTING`), `1` (`RESISTIVE`) | `0` (PROCESS default, `pfcoil_variables.py:230`, not set in the reference file) | **split**, same reasoning | `structure.py:167-168`: `if i_pf_conductor != PFConductorModel.RESISTIVE: coldmass += pfmass`. Changes whether `pfmass` feeds `coldmass` |

**One compound occupant, not four.** Both switches gate independent, additive terms of
the same one output (`coldmass`); on the reference run both conditions are simultaneously
true, so `calculate_structure_masses` bakes in *both* terms unconditionally rather than
accepting either switch as a parameter -- this is the occupant for the
`(i_tf_sup=1, i_pf_conductor=SUPERCONDUCTING)` cell of the 2x2 combination space. The
other three cells are **UNPORTED**:

- `(i_tf_sup=0, i_pf_conductor=SUPERCONDUCTING)` -- `coldmass = pfmass` only.
- `(i_tf_sup=1, i_pf_conductor=RESISTIVE)` -- `coldmass = tfmass + aintmass + dewmass` only.
- `(i_tf_sup=0, i_pf_conductor=RESISTIVE)` -- `coldmass = 0.0`.

Flagging this as a judgement call rather than a silent default: `traceability_policy.md`
records the exact same shape (two switches gating two 1-line additive terms of one
output, `confinement_time.md`'s `i_plasma_ignited`/`i_rad_loss`) as one of six deliberate
deviations from strict per-value splitting, on the ground that the differing body is a
handful of lines inside a much larger shared remainder. Here the "shared remainder" is
the whole rest of `calculate_structure_masses` (~30 lines) around two 1-line terms, so the
same reasoning applies; this record follows the wave-1 brief's stricter instruction ("no
switch is a static kwarg... even when two arms' reads are identical") by baking the *one*
live combination into a single occupant with **no switch parameter at all** (the switch
values are consumed at occupant-selection time, not passed in), rather than accepting
either integer as a kwarg on `calculate_structure_masses`. If a resistive-TF or
resistive-PF run is ever wanted, it needs its own occupant class, not a parameter on this
one.

## calls into other models

None.

## JAX-difficulty flags

- **`np.isinf(aintmass)` kludge** (`structure.py:156-158`) -- `needs-lax-cond-or-where`,
  severity `minor`. Ported as `jnp.where(jnp.isinf(aintmass_raw), 1e10, aintmass_raw)`;
  `jnp.isinf` is a plain boolean predicate on a traced value and traces cleanly. The
  `logger.error` call is pure reporting, dropped.
- No CoolProp, no other external library, no fractional-power zero-derivative traps
  (`safe_pow`/`safe_sqrt` not needed -- every exponent is an integer or `0.5` on an
  argument built from strictly positive physical masses/lengths on any in-domain point).

## open questions

- **Should `fncmass`/`gsmass` be dropped since nothing currently reads them?** Kept per
  the wave-1 brief's explicit instruction for this unit ("own all five if they're one
  function's outputs") — flagging only so the choice reads as deliberate, not an
  oversight, if a later boundary sweep asks why this node's declared outputs exceed its
  boundary-listed ones.
- **The `i_tf_sup`/`i_pf_conductor` compound-occupant decision above** is the same open
  "shared remainder" policy question `traceability_policy.md` and `plasma_geometry.md`
  already flag — not re-decided here, just another data point for whoever resolves it.
