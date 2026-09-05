"""Pure-functional port of S2 (`blanket_shield_tf_nuclear_power`), the `blktmodel` x
`ipowerflow` dispatch inside `Stellarator.st_fwbs` (registry unit #1's `stellarator.py`).

Audit record:
`functional_process/_audit/units/models/stellarator/stellarator_fwbs_s2.md` (read it
first) -- this file only implements the two of S2's three arms that are self-contained
tier-1 and do not touch the buggy `blanket_neutronics()` call site. The third arm
(`blktmodel == 1`, i.e. `blanket_neutronics()` + its `ipowerflow`-nested tail) is
audit-only, per the record's "arm 1" section -- it calls into `hcpb.py`'s already-ported
functions through a call site with two live PROCESS bugs, and porting it would mean
choosing how to route around both, not reproducing them.

Two ported arms, both `blktmodel != 1` (`stellarator.py:680`'s `else` branch):

- **Arm 2** (`ipowerflow == 0`, `stellarator.py:684-728`, the "old model"):
  `calculate_exponential_attenuation_blanket_shield_power` below. Small, self-contained,
  no bug. The arm also calls `self.sc_tf_coil_nuclear_heating_iter90()` (chunk 1F,
  `tf_nuclear_heating.py`'s `calculate_sc_tf_coil_nuclear_heating`,
  already ported) for `flu_tf_neutron_fast_peak`/`p_tf_nuclear_heat_mw` -- that call is
  a tier-3 composition edge onto an already-validated node, not reproduced here.
- **Arm 3** (`ipowerflow == 1`, `stellarator.py:730-1029`, the "new model"): split
  further by `.fwbs.i_p_coolant_pumping` into `_detailed_powerflow_core` plus
  `calculate_detailed_powerflow_blanket_shield_power` (`FRACTION_OF_HEAT`) and
  `calculate_detailed_powerflow_blanket_shield_power_user_input_pumping`
  (`USER_INPUT`) below -- see the next section. Self-contained (no
  cross-model calls), but two things are carved out of the port, both documented in the
  audit record: the CoolProp/`irefprop`-gated `temp_blkt_coolant_out` computation
  (803-823, not consumed anywhere else inside this arm, `non-traceable-external-call`),
  and the confirmed `p_div_rad_total_mw` bug (`.fwbs.p_div_rad_total_mw` is read at two
  sites, 792 and 1013, but never written anywhere on this call path -- deterministically
  the dataclass default `0.0` for the run's whole lifetime, per the record's "latent
  bugs" section) -- reproduced here as a literal `0.0`, not as an input, matching
  PROCESS's own actual runtime behaviour.

Both arms drop the trivial branch of `i_tf_sup` (`!= SUPERCONDUCTING` is "the absence
of the computation," not a second formula to port), matching the precedent
`tf_nuclear_heating.py` already set for that switch on a sibling field -- see the audit
record's "switches touched" section.

`.fwbs.i_p_coolant_pumping` used to be dropped on the same grounds and **that was a
defect**, fixed 2026-08-31. Arm 3's four `.heat_transport.p_*_coolant_pump_mw` writes
are gated on it (`stellarator.py:901-928` and `:1000-1013`), and the two values a
stellarator may legally take differ in *ownership*, not in formula:

- `FRACTION_OF_HEAT` (1) -- `stellarator_helias.IN.DAT:198`, the pinned reference
  machine -- computes all four as fractions of the incident thermal power.
- `USER_INPUT` (0) -- `helias_5b.IN.DAT:121` -- computes none of them; the input file
  supplies them (`120 + 56` MW FW+blanket, `24` MW divertor) and they are boundary
  inputs of the graph.
- `MECHANICAL` (2) and `MECHANICAL_WITH_PRESSURE_DROP` (3) are not reachable at all:
  `stellarator.py:924-928` raises `ProcessValueError("i_p_coolant_pumping = 0 or 1 only
  for stellarator")`. `indat.py` refuses those two arms rather than assembling one.

So the port carries **two occupants, not one node with a kwarg** (`_audit/next_steps.md`
§14.2): a node that owns a `VarPath` on one value of a switch and must not own it on
another is two nodes. Computing arm 3 unconditionally as if `FRACTION_OF_HEAT` made
`helias_5b` answer `15.6`/`16.8` MW where PROCESS reads `176.0`, and carried the error
downstream into `.costs.concost` and `.power.p_plant_electric_net_mw` -- the
`EcrhDensityLimit` bug class, on a machine that nonetheless reported "converged".
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    build,
    current_drive,
    first_wall,
    fwbs,
    heat_transport,
    physics,
)
from functional_process.models.stellarator.stellarator_fwbs_s2 import (
    calculate_detailed_powerflow_blanket_shield_power,
    calculate_detailed_powerflow_blanket_shield_power_user_input_pumping,
    calculate_exponential_attenuation_blanket_shield_power,
)


class ExponentialAttenuationBlanketShieldPower(ExplicitFunction):
    """cottax node: `calculate_exponential_attenuation_blanket_shield_power`.

    Not registered in `total_process.py` -- reserved for the consolidation pass, per
    this unit's boundary (see the audit record's switches-touched section: this arm is
    one of three `blktmodel`x`ipowerflow` alternatives sharing output fields with the
    other two arms, an `Alternative`/`Switch` design decision out of this audit's scope).
    """

    p_blkt_multiplication_mw = OutputInto(fwbs)
    p_blkt_nuclear_heat_total_mw = OutputInto(fwbs)
    p_shld_nuclear_heat_mw = OutputInto(fwbs)

    def __call__(
        self,
        p_neutron_total_mw=From(physics),
        pnucloss=From(fwbs),
        f_p_blkt_multiplication=From(fwbs),
        f_a_blkt_cooling_channels=From(fwbs),
        fblli2o=From(fwbs),
        fblbe=From(fwbs),
        dr_blkt_outboard=From(build),
    ):
        return calculate_exponential_attenuation_blanket_shield_power(
            p_neutron_total_mw,
            pnucloss,
            f_p_blkt_multiplication,
            f_a_blkt_cooling_channels,
            fblli2o,
            fblbe,
            dr_blkt_outboard,
        )


class DetailedPowerflowBlanketShieldPower(ExplicitFunction):
    """cottax node: `calculate_detailed_powerflow_blanket_shield_power`.

    `f_a_fw_coolant_inboard`/`f_a_fw_coolant_outboard` are given best-effort `VarPath`s
    under `.fwbs.*` (matching their PROCESS field names) even though the source never
    actually writes them there in this arm (they stay Python-locals, consumed by S4
    within the same call frame) -- same treatment `tf_nuclear_heating.py`
    gives its own best-effort output paths, flagged here for whoever wires S4.

    Not registered in `total_process.py` -- same reservation as the sibling arm above.
    """

    p_div_nuclear_heat_total_mw = OutputInto(fwbs)
    p_fw_hcd_nuclear_heat_mw = OutputInto(fwbs)
    p_fw_hcd_rad_total_mw = OutputInto(fwbs)
    pradloss = OutputInto(fwbs)
    p_fw_rad_total_mw = OutputInto(fwbs)
    f_a_fw_coolant_inboard = OutputInto(fwbs)
    f_a_fw_coolant_outboard = OutputInto(fwbs)
    p_fw_nuclear_heat_total_mw = OutputInto(fwbs)
    p_blkt_multiplication_mw = OutputInto(fwbs)
    p_blkt_nuclear_heat_total_mw = OutputInto(fwbs)
    p_fw_coolant_pump_mw = OutputInto(heat_transport)
    p_blkt_coolant_pump_mw = OutputInto(heat_transport)
    p_shld_nuclear_heat_mw = OutputInto(fwbs)
    p_shld_coolant_pump_mw = OutputInto(heat_transport)
    p_div_coolant_pump_mw = OutputInto(heat_transport)
    p_tf_nuclear_heat_mw = OutputInto(fwbs)

    def __call__(
        self,
        p_neutron_total_mw=From(physics),
        f_ster_div_single=From(fwbs),
        f_a_fw_outboard_hcd=From(fwbs),
        pnucloss=From(fwbs),
        a_fw_inboard=From(first_wall),
        a_fw_outboard=From(first_wall),
        a_fw_total=From(first_wall),
        p_plasma_rad_mw=From(physics),
        fhole=From(fwbs),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
        radius_fw_channel=From(fwbs),
        declfw=From(fwbs),
        dr_blkt_inboard=From(build),
        dr_blkt_outboard=From(build),
        declblkt=From(fwbs),
        f_p_fw_coolant_pump_total_heat=From(heat_transport),
        p_beam_orbit_loss_mw=From(current_drive),
        f_p_blkt_coolant_pump_total_heat=From(heat_transport),
        f_p_blkt_multiplication=From(fwbs),
        declshld=From(fwbs),
        dr_shld_inboard=From(build),
        dr_shld_outboard=From(build),
        f_p_shld_coolant_pump_total_heat=From(heat_transport),
        p_plasma_separatrix_mw=From(physics),
        f_p_div_coolant_pump_total_heat=From(heat_transport),
    ):
        return calculate_detailed_powerflow_blanket_shield_power(
            p_neutron_total_mw,
            f_ster_div_single,
            f_a_fw_outboard_hcd,
            pnucloss,
            a_fw_inboard,
            a_fw_outboard,
            a_fw_total,
            p_plasma_rad_mw,
            fhole,
            dr_fw_inboard,
            dr_fw_outboard,
            radius_fw_channel,
            declfw,
            dr_blkt_inboard,
            dr_blkt_outboard,
            declblkt,
            f_p_fw_coolant_pump_total_heat,
            p_beam_orbit_loss_mw,
            f_p_blkt_coolant_pump_total_heat,
            f_p_blkt_multiplication,
            declshld,
            dr_shld_inboard,
            dr_shld_outboard,
            f_p_shld_coolant_pump_total_heat,
            p_plasma_separatrix_mw,
            f_p_div_coolant_pump_total_heat,
        )


class DetailedPowerflowBlanketShieldPowerUserInputPumping(ExplicitFunction):
    """cottax node:
    `calculate_detailed_powerflow_blanket_shield_power_user_input_pumping`.

    The sibling above's arm at `.fwbs.i_p_coolant_pumping == USER_INPUT` (0), and it
    exists because the ownership genuinely differs: the four
    `.heat_transport.p_*_coolant_pump_mw` fields the `FRACTION_OF_HEAT` occupant owns
    are **run inputs** on this arm, written by nobody (`stellarator.py:904-906` is a
    bare `pass`). Registered by `indat.py`'s `BLANKET_SHIELD_POWER` under arm index 3.

    `f_a_fw_coolant_inboard`/`f_a_fw_coolant_outboard` get the same best-effort
    `.fwbs.*` `VarPath`s, and for the same reason, as the sibling above.
    """

    p_div_nuclear_heat_total_mw = OutputInto(fwbs)
    p_fw_hcd_nuclear_heat_mw = OutputInto(fwbs)
    p_fw_hcd_rad_total_mw = OutputInto(fwbs)
    pradloss = OutputInto(fwbs)
    p_fw_rad_total_mw = OutputInto(fwbs)
    f_a_fw_coolant_inboard = OutputInto(fwbs)
    f_a_fw_coolant_outboard = OutputInto(fwbs)
    p_fw_nuclear_heat_total_mw = OutputInto(fwbs)
    p_blkt_multiplication_mw = OutputInto(fwbs)
    p_blkt_nuclear_heat_total_mw = OutputInto(fwbs)
    p_shld_nuclear_heat_mw = OutputInto(fwbs)
    p_tf_nuclear_heat_mw = OutputInto(fwbs)

    def __call__(
        self,
        p_neutron_total_mw=From(physics),
        f_ster_div_single=From(fwbs),
        f_a_fw_outboard_hcd=From(fwbs),
        pnucloss=From(fwbs),
        a_fw_inboard=From(first_wall),
        a_fw_outboard=From(first_wall),
        a_fw_total=From(first_wall),
        p_plasma_rad_mw=From(physics),
        fhole=From(fwbs),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
        radius_fw_channel=From(fwbs),
        declfw=From(fwbs),
        dr_blkt_inboard=From(build),
        dr_blkt_outboard=From(build),
        declblkt=From(fwbs),
        f_p_blkt_coolant_pump_total_heat=From(heat_transport),
        f_p_blkt_multiplication=From(fwbs),
        declshld=From(fwbs),
        dr_shld_inboard=From(build),
        dr_shld_outboard=From(build),
    ):
        return calculate_detailed_powerflow_blanket_shield_power_user_input_pumping(
            p_neutron_total_mw,
            f_ster_div_single,
            f_a_fw_outboard_hcd,
            pnucloss,
            a_fw_inboard,
            a_fw_outboard,
            a_fw_total,
            p_plasma_rad_mw,
            fhole,
            dr_fw_inboard,
            dr_fw_outboard,
            radius_fw_channel,
            declfw,
            dr_blkt_inboard,
            dr_blkt_outboard,
            declblkt,
            f_p_blkt_coolant_pump_total_heat,
            f_p_blkt_multiplication,
            declshld,
            dr_shld_inboard,
            dr_shld_outboard,
        )
