---
kind: model-unit
status: reviewed
confidence: high
---

## source
`process/models/stellarator/initialization.py` (67 lines, full file in scope). One
function, `st_init(data)`.

## data footprint

`istell` gates the whole function (`if data.stellarator.istell == 0: return`) — not a
real read within stellarator scope: this audit only ever runs with `istell != 0` (see
`unit_registry.md`'s scope rule), so within the stellarator graph this function's body
always executes. It is the same top-level pipeline dispatch `switches.md`'s `istell`
entry already covers, not a second finding.

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.numerics.boundu[0]` | write | explicit-arg (see open question 1) | unconditional literal `40.0` — first real instance of `naming_convention.md`'s array-element `VarPath` pattern (`boundu` is the iteration-variable upper-bound array; index 0 is variable ID 1's bound, `aspect`'s upper bound per `iteration_variables.py`) |
| `.build.dr_cs` | write | explicit-arg | unconditional literal `0.0` |
| `.build.iohcl` | write | explicit-arg | unconditional literal `0` |
| `.pf_coil.f_z_cs_tf_internal` | write | explicit-arg | unconditional literal `0.0` |
| `.build.dr_cs_tf_gap` | write | explicit-arg | unconditional literal `0.0` |
| `.build.f_dr_tf_outboard_inboard` | write | explicit-arg | unconditional literal `1.0` |
| `.physics.i_plasma_pedestal` | write | explicit-arg | unconditional literal `0` — this is the very precondition `density_limits.md`'s `EcrhDensityLimit` requires; stellarator mode enforces it structurally at init, not accidentally |
| `.physics.beta_norm_max` | write | explicit-arg | unconditional literal `0.0` |
| `.physics.kappa95` | write | explicit-arg | unconditional literal `1.0` |
| `.physics.triang` | write | explicit-arg | unconditional literal `0.0` |
| `.physics.q95` | write | explicit-arg | unconditional literal `1.03` |
| `.current_drive.i_hcd_calculations` | write | explicit-arg | unconditional literal `0` |
| `.times.t_plant_pulse_coil_precharge` | write | explicit-arg | unconditional literal `0.0` |
| `.times.t_plant_pulse_plasma_current_ramp_up` | write | explicit-arg | unconditional literal `0.0` |
| `.times.t_plant_pulse_burn` | write | explicit-arg | unconditional literal `3.15576e7` (one year, in seconds) |
| `.times.t_plant_pulse_plasma_current_ramp_down` | write | explicit-arg | unconditional literal `0.0` |
| `.times.t_plant_pulse_fusion_ramp` | read | explicit-arg | **not written here** — pre-existing value (`times_variables.py` default `10.0`), read into the two sums below |
| `.times.t_plant_pulse_dwell` | read | explicit-arg | **not written here** — pre-existing value (default `1800.0`), read into the sum below |
| `.times.t_plant_pulse_plasma_present` | write | explicit-arg | sum of 4 of the literals above (`ramp_up + fusion_ramp + burn + ramp_down`) — a real function output, not `local-intermediate` (that label is for values that don't escape the function; this one does, into three real fields other PROCESS models consume) |
| `.times.t_plant_pulse_no_burn` | write | explicit-arg | sum of 5 terms (4 literals above + `dwell`) |
| `.times.t_plant_pulse_total` | write | explicit-arg | sum of 6 terms (5 literals above + `dwell`) |

## proposed signature(s)

**Not a computational node in the ordinary sense.** 16 of the 19 rows above are
unconditional literal assignments with no input at all — this function is a
stellarator-mode **device-preset/initial-condition table**, structurally identical to
`preset_config.py`'s `load_stellarator_config` (see that record) and to chunk 1D's
`fncmass`/`gsmass` constant-producer pattern, just at a larger scale (16 constants
instead of 2). Per that precedent, these belong in the ported graph as literal defaults
supplied at stellarator-mode graph-assembly time, not as a node any tier reads from.

The three summed pulse durations *are* a genuine (if trivial) pure function, and are
ported below since they're tier-1 and self-contained:

```python
def calculate_pulse_durations(
    t_plant_pulse_coil_precharge: float,
    t_plant_pulse_plasma_current_ramp_up: float,
    t_plant_pulse_burn: float,
    t_plant_pulse_plasma_current_ramp_down: float,
    t_plant_pulse_fusion_ramp: float,
    t_plant_pulse_dwell: float,
) -> tuple[float, float, float]:
    # returns (t_plant_pulse_plasma_present, t_plant_pulse_no_burn, t_plant_pulse_total)
    ...
```
Ported as `calculate_pulse_durations` in `initialization.py` (this module's `.py`
sibling), with a `PulseDurations` `ExplicitFunction` node. Its inputs are the *fields*,
not the literals — the stellarator-mode preset determines the values of four of them,
but the function itself doesn't know or care that they're constants in this mode; a
future non-stellarator caller (there is none today) could feed it different values.

## cottax node

**Actually written**, in `initialization.py`, registered in
`functional_process/total_process.py`:

```python
class PulseDurations(ExplicitFunction):
    t_plant_pulse_plasma_present = Output(lambda s: s.times.t_plant_pulse_plasma_present)
    t_plant_pulse_no_burn = Output(lambda s: s.times.t_plant_pulse_no_burn)
    t_plant_pulse_total = Output(lambda s: s.times.t_plant_pulse_total)

    def __call__(self, t_plant_pulse_coil_precharge=Input(...), ..., t_plant_pulse_dwell=Input(...)):
        return calculate_pulse_durations(t_plant_pulse_coil_precharge, ..., t_plant_pulse_dwell)
```

## tier signal
**Tier 1** for `calculate_pulse_durations` (plain arithmetic, no branches, no calls).
The other 16 writes are **not a tier at all** — device-preset literals, see above.

## switches touched
`istell` only, as the whole-function gate discussed above — no local branch, already
covered by `switches.md`.

## calls into other models
None.

## JAX-difficulty flags
None. Plain scalar arithmetic and literal assignment throughout.

## open questions
1. **`.numerics.boundu[0]` is a solver bound, not an ordinary physics value.** It
   overrides the upper bound of iteration variable ID 1 (`aspect`), which in cottax
   terms is metadata on a `DeclaredNode`'s unknown (an `Optimise` problem's bound), not a
   value any node reads or writes along a graph edge. Flagging rather than resolving:
   this audit has no established treatment yet for iteration-variable *bounds* as
   distinct from iteration-variable *values* (`conditional-ownership-by-run-config` in
   `schema.md` covers the latter). Worth a policy note wherever the `Optimise`
   problem/iteration-variable crosswalk is written up.
2. **Should the 16 device-preset literals be represented at all in the pure graph**, or
   simply hardcoded into whatever stellarator-mode default-input construction exists
   (analogous to `preset_config.py`'s machine presets)? Recommend the same answer for
   both files — see `preset_config.md`'s open questions, this is one decision, not two.
