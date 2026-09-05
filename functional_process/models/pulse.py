"""Pure-functional port of `process/models/pulse.py` (`Pulse`, `.tokamak.pulse`).

No `unit_registry.md` row yet and no `next_steps.md` edit -- registration is the
consolidation pass's job (wave-1 brief, "Registration instructions" below is written
for it).

## Scope

`Pulse.run()` (`process/models/pulse.py:142-162`) does exactly two things, both gated
by `self.data.pulse.i_pulsed_plant == 1` (a topology switch: `large_tokamak_eval.
IN.DAT:330` sets it to `1`; `!= 1` means neither call happens at all and this file
contributes nothing to the graph):

1. `self.tohswg(output=output)` (`:154`) -- computes `.constraints.
   t_current_ramp_up_min`, the lower bound consumed by constraint 41
   (`core/solver/constraints.py:1281-1299`, already ported and reading it as an
   ordinary declared input).
2. The burn-time calculation (`:158-162`) -- `self.data.times.t_plant_pulse_burn =
   self.calculate_burn_time(...)`.

**Only (2) is ported here.** (1) is scoped out, not silently dropped -- see
"Not ported: `tohswg`" below.

### `calculate_burn_time` -- ported unchanged

`process/models/pulse.py:275-316`, already a `@staticmethod` with no `self.data`
access -- the extraction seam is free, same shape as `density_limit.py`'s
`calculate_density_limit`. One deviation: PROCESS's `logger.error` call on a negative
burn time (`:306-314`) is dropped as pure reporting with no effect on the returned
value -- same precedent as `functional_process/cottax/structure.py`'s `aintmass`
comment ("PROCESS logs and kludges ... dropped here as pure reporting"). Here there
is not even a kludge: the negative value is returned as-is either way, so dropping the
log changes nothing about what the function computes.

**Why this closure matters beyond `Pulse.run()` itself**: `.times.t_plant_pulse_burn`
is a heavily-consumed declared input across the port already --
`functional_process/cottax/core/solver/constraints.py::constraint_13`'s own docstring names
`calculate_burn_time` as the "real *model* producer" it was written without, back when
this codebase scoped only the stellarator; `objectives.py`'s figures of merit 14/16/19;
`availability.py`'s three capacity-factor sites; `costs.py`'s fuel-cost terms; and
`stellarator/initialization.py::PulseDurations` (already registered for both devices,
per that unit's own record). All of them currently take `t_plant_pulse_burn` as a
boundary input with no producer on the tokamak path. This node is that producer.
`constraint_13` (`icc = 13`, "Burn time lower limit") is live on
`large_tokamak_eval.IN.DAT` (`icc` row 16), so this is not a hypothetical consumer --
it is the reference run's own active constraint.

### Not ported: `tohswg`

`process/models/pulse.py:164-273`. Left unported this pass, for three independent
reasons, any one of which would be enough on its own:

1. **Not live on the reference run.** `tohswg`'s sole output,
   `.constraints.t_current_ramp_up_min`, is read only by constraint 41
   (`core/solver/constraints.py:1281`), and `large_tokamak_eval.IN.DAT`'s `icc` list
   (lines 9-35) does not include `41` -- confirmed by grep, not inferred. Nothing on
   this reference arm reads `tohswg`'s output.
2. **Every read is PF-coil-owned, and PF coil is under concurrent edit.** `tohswg`
   reads `.pf_coil.c_pf_coil_turn`, `.pf_coil.n_cs_pf_coils`,
   `.pf_coil.i_pf_conductor`, `.pf_coil.p_cs_resistive_flat_top`,
   `.pf_coil.c_pf_cs_coils_peak_ma`, `.pf_coil.c_pf_coil_turn_peak_input`,
   `.pf_coil.rhopfbus`, `.pf_coil.ind_pf_cs_plasma_mutual`,
   `.pf_coil.n_pf_coil_turns`, and `.pf_power.vpfskv` -- every one of them a
   `functional_process/cottax/pfcoil/**` concern, which this wave's fencing assigns to
   a different agent. Declaring these reads now risks binding against a producer
   another agent is mid-rewrite on.
3. **Dynamic array indexing the naming convention does not cover.** Every PF-coil
   array read above is indexed by `self.data.pf_coil.n_cs_pf_coils - 1` (and
   `ind_pf_cs_plasma_mutual` additionally by `n_pf_cs_plasma_circuits - 1`) --
   `n_cs_pf_coils` is a run-topology count computed by `pfcoil.py`
   (`process/models/pfcoil.py:140-158`, incremented per coil group), not a literal
   constant the way `naming_convention.md`'s "Array elements" section describes
   (`f_nd_impurity_electron_array[2]`, a fixed `IterationVariable.array_index`). Turning
   this into a `VarPath` needs either treating `n_cs_pf_coils` as a build-time-known
   static count (plausible, since it is set once from input-derived coil-group sizes
   and never an iteration variable) or some other policy this file should not
   improvise. It also carries a genuine two-arm switch, `i_pf_conductor ==
   SUPERCONDUCTING` vs. resistive (`:190-202`), whose PROCESS-side resistive reads
   (`p_cs_resistive_flat_top`, `c_pf_cs_coils_peak_ma`) the superconducting arm does
   not have -- an ordinary split case per wave-1's binding policy, not itself a
   blocker, but one more reason this needs its own pass once the PF-coil producers it
   depends on exist.

Per the wave-1 brief's hard rule ("stop on that item, port everything else, and
report it") this is reported rather than improvised. See "Open questions" in the
audit record.
"""


def calculate_burn_time(
    vs_cs_pf_total_burn, v_plasma_loop_burn, t_plant_pulse_fusion_ramp
):
    """Burn time for a pulsed reactor. Ports `Pulse.calculate_burn_time`,
    `process/models/pulse.py:275-316`.

    PROCESS's `logger.error` on a negative result (`:306-314`) is not reproduced -- it
    is a diagnostic side effect with no bearing on the returned value, which is
    returned unclamped either way (module docstring, "`calculate_burn_time` -- ported
    unchanged").

    Parameters
    ----------
    vs_cs_pf_total_burn :
        Total volt-seconds in the CS and PF coils available for burn (V.s).
        `.pf_coil.vs_cs_pf_total_burn`.
    v_plasma_loop_burn :
        Plasma loop voltage during burn (V). `.physics.v_plasma_loop_burn`.
    t_plant_pulse_fusion_ramp :
        Time for the fusion ramp (s). `.times.t_plant_pulse_fusion_ramp`.

    Returns
    -------
    :
        Burn time (s), `.times.t_plant_pulse_burn`. May be negative -- PROCESS reports
        that condition but does not guard against it (see module docstring).
    """
    return (abs(vs_cs_pf_total_burn) / v_plasma_loop_burn) - t_plant_pulse_fusion_ramp
