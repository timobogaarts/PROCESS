"""Pure-functional port of the PF-coil power-supply sub-unit of
`process/models/power.py` (registry unit #14, chunk D).

Audit record: `functional_process/_audit/units/models/power/pf_coil_power.md`.
Covers `Power.pfpwr` (`power.py:300-604`, the computation; the output section
`:606-694` is out of scope as everywhere else in this package) and the four
`_pf_loss_*` staticmethods it calls (`:99-299`).

**This is the one subsystem of `power.py` a tokamak has and a stellarator does not.**
`total_process.TokamakProcess.power`'s docstring already measured that -- "Shared, 11
functions / 1522 lines; tokamak-new, `Power.pfpwr` and its four `_pf_loss_*` helpers --
the PF-coil power supply, which a stellarator has no PF coils to need" -- and
`stellarator.py:114-186` calls `tfpwr`, `component_thermal_powers`,
`calculate_cryo_loads`, `acpow` and `plant_electric_production` and never `Power.run`,
so it never reaches `pfpwr`. Hence the slot in `models/power/namespace.py` is
`PfCoilPowerSupplies | None` and the stellarator gets `None`.

**Why it was written: four missing producers, measured.** `boundary.
unproduced_but_computed` on `large_tokamak_nof` listed `.pf_power.srcktpm`,
`.pf_power.ensxpfm`, `.heat_transport.peakmva` and `.pf_coil.p_pf_electric_supplies_mw`
as boundary `input`s that PROCESS *writes* every pipeline pass -- all four frozen at
`0.0` in the port while PROCESS computed `1113.0075` kW, `17038.228` MJ, `134.98773` MVA
and `4.8813983` MW. `_audit/cost_boundary_inputs.md` §6 had recorded three of them as
"`Power.pfpwr`, unported" and settled them as category (d); §7 says outright that
`Power.pfpwr` "is **not ported anywhere** in `functional_process/` ... there is nothing
to register". There is now.

The consumers were already ported and already reading the zeros:
`electric_production.py::calculate_acpow` takes `srcktpm` and `peakmva` as arguments,
`calculate_plant_electric_production`/`power_profiles_over_time` take
`p_pf_electric_supplies_mw`, and `costs.py`'s Accounts 2252 and 2254 read `srcktpm`,
`ensxpfm` and `peakmva`. That is the silent-stale-read shape exactly: every one of those
nodes agreed with PROCESS to 1e-9 in Stage A, because Stage A seeds boundary inputs from
PROCESS's *converged* `DataStructure` and therefore handed each of them the right answer.

**The topology is baked, as it is throughout `models/pfcoil/`.** `pfpwr` loops over
`n_pf_coil_groups` (+1 for the CS), over `n_pf_coils_in_group[group]`, over
`n_pf_cs_plasma_circuits` and over `pulse_timings.n_pf_active_points_total`. All four
bounds are graph-assembly data on the reference topology -- `models/pfcoil/__init__.py`
records why (`_audit/naming_convention.md` § "Switches are not ports") -- so every loop
here is an ordinary Python `for` that unrolls at trace time, and every array has a
static shape. `PulseTimings.n_pf_active_points_total` is `len()` of a fixed 6-tuple, the
same argument `electric_production.py` makes for its own 7.

**Not ported, and why**: `pfbuspwr` (`power.py:353,410`), the summed busbar resistive
power in kW, is accumulated and then read only by `po.ovarre` in the output section --
a dead term outside reporting, dropped under the same convention chunk A used for
`tfreacmw`. The `9.9e9` sentinel `poloidalpower` takes on an interval shorter than one
second (`:513-515`) *is* reproduced, because unlike `pfbuspwr` it reaches a stored
field.
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.pfcoil import (
    N_COILS_IN_GROUP,
    N_PF_GROUPS,
    PLASMA_INDEX,
    REFERENCE_TOPOLOGY,
    PFCoilTopology,
)
from functional_process.paths import heat_transport, pf_coil, pf_power, physics, times
from functional_process.models.power.pf_coil_power import (
    COILS_IN_GROUP_WITH_CS,
    GROUP_CIRCUIT_INDEX,
    MIN_INTERVAL_S,
    N_PF_ACTIVE_INTERVALS,
    N_PF_ACTIVE_POINTS,
    N_PF_CS_PLASMA_CIRCUITS,
    N_PF_GROUPS_WITH_CS,
    PF_BUS_CURRENT_DENSITY_A_PER_CM2,
    PFCKTS_SPARE_CIRCUITS,
    POLOIDAL_POWER_SENTINEL_W,
    VPFSKV_KV,
    calculate_pf_coil_power_supplies,
    coils_in_group_with_cs,
    group_circuit_index,
)

# Step 2 of `_audit/formulas_split.md` moved the bodies below to
# `functional_process.models.power.pf_coil_power`; these names are imported above purely to
# keep every public name this module resolved before the move still resolving now
# (`models/**` modules must not lose any name -- see the split's invariant), not
# because the declaration below reads them itself.
__all__ = [
    "COILS_IN_GROUP_WITH_CS",
    "GROUP_CIRCUIT_INDEX",
    "MIN_INTERVAL_S",
    "N_COILS_IN_GROUP",
    "N_PF_ACTIVE_INTERVALS",
    "N_PF_ACTIVE_POINTS",
    "N_PF_CS_PLASMA_CIRCUITS",
    "N_PF_GROUPS",
    "N_PF_GROUPS_WITH_CS",
    "PFCKTS_SPARE_CIRCUITS",
    "PF_BUS_CURRENT_DENSITY_A_PER_CM2",
    "PLASMA_INDEX",
    "POLOIDAL_POWER_SENTINEL_W",
    "REFERENCE_TOPOLOGY",
    "VPFSKV_KV",
    "PFCoilTopology",
    "PfCoilPowerSupplies",
    "calculate_pf_coil_power_supplies",
    "coils_in_group_with_cs",
    "group_circuit_index",
    "jnp",
]


class PfCoilPowerSupplies(ExplicitFunction):
    """cottax node: `.power.pf_coil_power` -- `Power.pfpwr`, eleven owned fields.

    Landed 2026-08-30 to close four missing producers on `large_tokamak_nof`
    (`.pf_power.srcktpm`, `.pf_power.ensxpfm`, `.heat_transport.peakmva`,
    `.pf_coil.p_pf_electric_supplies_mw`); the other seven fields `pfpwr` writes come
    along because they come out of the same three blocks and leaving them out would
    make the node a subset of its own source for no reason.

    **Unswitched.** `Power.run` calls `pfpwr` on both of its own branches
    (`power.py:54,81`), and every switch inside it (`iohcl`, the coil topology,
    `i_pf_conductor`) is already part of `indat._pf_coil_system_arm`'s joint predicate,
    which this node inherits by depending on `models/pfcoil/`'s baked topology. What it
    is *not* is device-agnostic: see the module docstring and the `| None` slot.

    Sixteen reads, which is wide -- but eleven of them are the PF coil set's own
    geometry and currents, and the node is the point where the coil set meets the plant
    electrical system, so the width is the subsystem's and not the port's.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static, and the same object the PF coil package's own nodes carry: `pfpwr`'s four
    loop bounds are the coil topology's (module docstring), so a machine with no central
    solenoid loops over its groups and not over a fifth that does not exist. The read set
    does not move with it -- every array is read whole -- so one occupant serves both."""

    srcktpm = OutputInto(pf_power)
    poloidalpower = OutputInto(pf_power)
    ensxpfm = OutputInto(pf_power)
    peakpoloidalpower = OutputInto(pf_power)
    peakmva = OutputInto(heat_transport)
    vpfskv = OutputInto(pf_power)
    pfckts = OutputInto(pf_power)
    spfbusl = OutputInto(pf_power)
    acptmax = OutputInto(pf_power)
    spsmva = OutputInto(pf_power)
    p_pf_electric_supplies_mw = OutputInto(pf_coil)

    def __call__(
        self,
        rmajor=From(physics),
        c_pf_coil_turn_peak_input=From(pf_coil),
        rhopfbus=From(pf_coil),
        rho_pf_coil=From(pf_coil),
        r_pf_coil_middle=From(pf_coil),
        j_pf_coil_wp_peak=From(pf_coil),
        f_a_pf_coil_void=From(pf_coil),
        c_pf_cs_coils_peak_ma=From(pf_coil),
        c_pf_cs_coil_pulse_end_ma=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
        c_pf_coil_turn=From(pf_coil),
        ind_pf_cs_plasma_mutual=From(pf_coil),
        f_p_pf_energy_store_loss=From(pf_power),
        f_p_pf_psu_loss=From(pf_power),
        etapsu=From(pf_coil),
        p_plasma_ohmic_mw=From(physics),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
    ):
        return calculate_pf_coil_power_supplies(
            rmajor=rmajor,
            c_pf_coil_turn_peak_input=c_pf_coil_turn_peak_input,
            rhopfbus=rhopfbus,
            rho_pf_coil=rho_pf_coil,
            r_pf_coil_middle=r_pf_coil_middle,
            j_pf_coil_wp_peak=j_pf_coil_wp_peak,
            f_a_pf_coil_void=f_a_pf_coil_void,
            c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma,
            c_pf_cs_coil_pulse_end_ma=c_pf_cs_coil_pulse_end_ma,
            n_pf_coil_turns=n_pf_coil_turns,
            c_pf_coil_turn=c_pf_coil_turn,
            ind_pf_cs_plasma_mutual=ind_pf_cs_plasma_mutual,
            f_p_pf_energy_store_loss=f_p_pf_energy_store_loss,
            f_p_pf_psu_loss=f_p_pf_psu_loss,
            etapsu=etapsu,
            p_plasma_ohmic_mw=p_plasma_ohmic_mw,
            t_plant_pulse_coil_precharge=t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up=(t_plant_pulse_plasma_current_ramp_up),
            t_plant_pulse_fusion_ramp=t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn=t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down=(
                t_plant_pulse_plasma_current_ramp_down
            ),
            topology=self.topology,
        )
