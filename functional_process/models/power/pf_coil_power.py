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

from functional_process.models.pfcoil import (
    N_COILS_IN_GROUP,
    N_PF_GROUPS,
    PLASMA_INDEX,
    REFERENCE_TOPOLOGY,
    PFCoilTopology,
)
from functional_process.paths import heat_transport, pf_coil, pf_power, physics, times


def coils_in_group_with_cs(topology):
    """`n_pf_coils_in_group` as `pfpwr` reads it -- with the CS's own group appended.

    `pfpwr` increments `n_pf_coil_groups` by one when `iohcl != 0` (`power.py:346-347`)
    rather than reading a stored value, and `pfcoil()` separately writes
    `n_pf_coils_in_group[n_pf_coil_groups] = 1` (`pfcoil.py:155`) so the CS reads as a
    one-coil group. Both are guarded on the same switch, so with no solenoid the groups
    are exactly the coil set's own -- **not** four groups plus an empty fifth.
    """
    if topology.has_central_solenoid:
        return (*topology.n_pf_coils_in_group, 1)
    return topology.n_pf_coils_in_group


def group_circuit_index(coils_in_group):
    """`pf_group_circuit_index`, `pfpwr`'s running `ic` (`power.py:349-357`).

    The **last** coil of each group, which is the one every per-group quantity is
    evaluated at. `models/pfcoil/inductance.py`'s `topology.last_coil_of_group` is the
    same construction for the same reason; kept separate because that one stops at the
    PF groups and this one includes the CS's.
    """
    return tuple(
        sum(coils_in_group[: group + 1]) - 1 for group in range(len(coils_in_group))
    )


N_PF_GROUPS_WITH_CS = N_PF_GROUPS + 1
"""`n_pf_coil_groups` as `pfpwr` uses it on the reference topology: five, the four PF
groups plus the CS. An alias of `REFERENCE_TOPOLOGY`, kept for the harness cases."""

COILS_IN_GROUP_WITH_CS = (*N_COILS_IN_GROUP, 1)
"""`(1, 1, 2, 2, 1)` -- `N_COILS_IN_GROUP` with the CS's own one-coil group appended."""

GROUP_CIRCUIT_INDEX = group_circuit_index(COILS_IN_GROUP_WITH_CS)
"""`(0, 1, 3, 5, 6)` on the reference topology."""

N_PF_CS_PLASMA_CIRCUITS = PLASMA_INDEX + 1
"""`.pf_coil.n_pf_cs_plasma_circuits` = 8 -- six PF coils, the CS, and the plasma."""

N_PF_ACTIVE_POINTS = 6
"""`PulseTimings.n_pf_active_points_total`: `total_pulse_cumulative[:-1]`, i.e. the six
cumulative times `t0..t5`, dropping the end of dwell (`process/models/pulse.py:117`)."""

N_PF_ACTIVE_INTERVALS = N_PF_ACTIVE_POINTS - 1
"""Five intervals, and the length of `.pf_power.poloidalpower`
(`pf_power_variables.py:47`)."""

PF_BUS_CURRENT_DENSITY_A_PER_CM2 = 100.0
"""The aluminium bussing is sized at 100 A/cm^2 (`power.py:337-338`, `:360-361`)."""

VPFSKV_KV = 20.0
"""`.pf_power.vpfskv`, "PF coil voltage (kV)" -- a literal `20.0` in `pfpwr`
(`power.py:571`), not a read of anything."""

PFCKTS_SPARE_CIRCUITS = 6.0
"""`pfckts = (n_pf_cs_plasma_circuits - 2) + 6` (`power.py:572-574`). The `- 2` drops
the plasma circuit and one more; the `+ 6` is spares. Both literals of `pfpwr`."""

POLOIDAL_POWER_SENTINEL_W = 9.9e9
"""What `pfpwr` stores in `poloidalpower[k]` when interval `k` is shorter than one
second (`power.py:513-515`, "Flag when an interval is small or zero MDK 30/11/16"). A
flag value in a real field, so it is reproduced rather than smoothed."""

MIN_INTERVAL_S = 1.0
"""The one-second threshold both `poloidalpower`'s sentinel and the flat-top denominator
test against (`power.py:504-508`, `:545-548`)."""


def _pf_bus_and_coil_resistances(
    *,
    pfbusl,
    c_pf_coil_turn_peak_input,
    rhopfbus,
    rho_pf_coil,
    r_pf_coil_middle,
    j_pf_coil_wp_peak,
    f_a_pf_coil_void,
    c_pf_cs_coils_peak_ma,
    c_pf_cs_coil_pulse_end_ma,
    n_pf_coil_turns,
    topology,
):
    """Per-group busbar and coil resistance, and the peak resistive circuit power.

    `power.py:346-412`'s loop body, unrolled over the five groups. Every quantity is
    evaluated at that group's representative circuit `ic` (`GROUP_CIRCUIT_INDEX`) and
    the coil resistance is then multiplied by the number of coils in the group, which is
    PROCESS's own series-circuit model.

    Returns
    -------
    tuple
        `(res_pf_bus, res_pf_circuit_total, srcktpm)` -- two arrays (ohm), one entry
        per group `pfpwr` sees, and the summed peak resistive power in the PF circuits
        (kW), `.pf_power.srcktpm`.
    """
    coils_in_group = coils_in_group_with_cs(topology)
    circuit_index = group_circuit_index(coils_in_group)
    res_pf_bus = []
    res_pf_circuit_total = []
    p_pf_circuit_resistive_peak = []
    for group in range(len(coils_in_group)):
        ic = circuit_index[group]
        # Section area of the aluminium bussing (cm^2), then its resistance (ohm).
        # `/ 10000` is the cm^2 -> m^2 conversion PROCESS spells inline; the missing
        # factor 1.5 is folded into `rhopfbus` itself, as `power.py:363-366` says.
        albusa = (
            jnp.abs(c_pf_coil_turn_peak_input[ic]) / PF_BUS_CURRENT_DENSITY_A_PER_CM2
        )
        bus = rhopfbus * pfbusl / (albusa / 10000.0)

        coil = (
            rho_pf_coil
            * 2.0
            * jnp.pi
            * r_pf_coil_middle[ic]
            * jnp.abs(
                j_pf_coil_wp_peak[ic]
                / ((1.0 - f_a_pf_coil_void[ic]) * 1.0e6 * c_pf_cs_coils_peak_ma[ic])
            )
            * n_pf_coil_turns[ic] ** 2
            * coils_in_group[group]
        )
        total = coil + bus

        # Current per turn during burn, scaled from the peak by the end-of-pulse to
        # peak current ratio (`power.py:392-396`).
        cptburn = (
            c_pf_coil_turn_peak_input[ic]
            * c_pf_cs_coil_pulse_end_ma[ic]
            / c_pf_cs_coils_peak_ma[ic]
        )
        v_peak = jnp.abs(cptburn) * total
        res_pf_bus.append(bus)
        res_pf_circuit_total.append(total)
        p_pf_circuit_resistive_peak.append(1.0e-6 * v_peak * jnp.abs(cptburn))

    return (
        jnp.stack(res_pf_bus),
        jnp.stack(res_pf_circuit_total),
        1.0e3 * sum(p_pf_circuit_resistive_peak),
    )


def _pf_loss_power_supply_j(
    *, interval, c_pf_coil_turn, ind_pf_cs_plasma_mutual, f_p_pf_psu_loss, topology
):
    """Power-supply conversion loss over one interval (J). `power.py:122-176`.

    `sum_i (k_ps/2) * |(I_i[n+1] + I_i[n]) * sum_j M_ij (I_j[n+1] - I_j[n])|`, the
    plasma circuit excluded from the outer sum and included in the inner one.
    """
    n_circuits = topology.plasma_index + 1
    loss = 0.0
    for circuit in range(n_circuits - 1):
        c_sum = c_pf_coil_turn[circuit, interval + 1] + c_pf_coil_turn[circuit, interval]
        delta_flux = 0.0
        for coupled in range(n_circuits):
            delta_flux += ind_pf_cs_plasma_mutual[circuit, coupled] * (
                c_pf_coil_turn[coupled, interval + 1] - c_pf_coil_turn[coupled, interval]
            )
        loss += 0.5 * f_p_pf_psu_loss * jnp.abs(c_sum * delta_flux)
    return loss


def _pf_loss_busbar_j(
    *, interval, dt_pulse_phase_s, c_pf_coil_turn, res_pf_bus, topology
):
    """Busbar resistive loss over one interval (J). `power.py:178-222`.

    `dt * sum_groups I_mean^2 R_bus`, `I_mean` the average of the group's
    representative circuit current at the two ends of the interval.
    """
    circuit_index = group_circuit_index(coils_in_group_with_cs(topology))
    loss = 0.0
    for group in range(len(circuit_index)):
        ic = circuit_index[group]
        c_mean = 0.5 * (c_pf_coil_turn[ic, interval + 1] + c_pf_coil_turn[ic, interval])
        loss += dt_pulse_phase_s * c_mean**2 * res_pf_bus[group]
    return loss


def _pf_loss_interval_total_j(
    *,
    interval,
    dt_pulse_phase_s,
    poloidalenergy,
    f_p_pf_energy_store_loss,
    f_p_pf_psu_loss,
    c_pf_coil_turn,
    ind_pf_cs_plasma_mutual,
    res_pf_bus,
    topology,
):
    """Storage + power supply + busbar loss over one interval (J). `power.py:224-299`.

    **PROCESS's `if dt_pulse_phase_s <= 0: return 0` becomes a `jnp.where`**, and the
    two loss terms that would otherwise be evaluated on a zero-length interval are
    finite there anyway (the busbar term is proportional to `dt`, the storage and
    power-supply terms to current differences), so there is no guarded division to
    poison a tangent.
    """
    e_delta = poloidalenergy[interval + 1] - poloidalenergy[interval]
    total = (
        f_p_pf_energy_store_loss * jnp.abs(e_delta)
        + _pf_loss_power_supply_j(
            interval=interval,
            c_pf_coil_turn=c_pf_coil_turn,
            ind_pf_cs_plasma_mutual=ind_pf_cs_plasma_mutual,
            f_p_pf_psu_loss=f_p_pf_psu_loss,
            topology=topology,
        )
        + _pf_loss_busbar_j(
            interval=interval,
            dt_pulse_phase_s=dt_pulse_phase_s,
            c_pf_coil_turn=c_pf_coil_turn,
            res_pf_bus=res_pf_bus,
            topology=topology,
        )
    )
    return jnp.where(dt_pulse_phase_s <= 0.0, 0.0, total)


def calculate_pf_coil_power_supplies(
    rmajor,
    c_pf_coil_turn_peak_input,
    rhopfbus,
    rho_pf_coil,
    r_pf_coil_middle,
    j_pf_coil_wp_peak,
    f_a_pf_coil_void,
    c_pf_cs_coils_peak_ma,
    c_pf_cs_coil_pulse_end_ma,
    n_pf_coil_turns,
    c_pf_coil_turn,
    ind_pf_cs_plasma_mutual,
    f_p_pf_energy_store_loss,
    f_p_pf_psu_loss,
    etapsu,
    p_plasma_ohmic_mw,
    t_plant_pulse_coil_precharge,
    t_plant_pulse_plasma_current_ramp_up,
    t_plant_pulse_fusion_ramp,
    t_plant_pulse_burn,
    t_plant_pulse_plasma_current_ramp_down,
    *,
    topology=REFERENCE_TOPOLOGY,
):
    """`Power.pfpwr` -- the MVA, power and energy requirements of the PF coil system.

    Ports `process/models/power.py:300-604`. PROCESS's own summary of the routine:
    "The routine checks at the beginning of the flattop for the peak MVA, and at the
    end of flattop for the peak stored energy."

    Three blocks, in the source's order:

    1. **Resistive** (`:334-412`). Per-group busbar and coil resistances, and
       `.pf_power.srcktpm`, the summed peak resistive power in the circuits. See
       `_pf_bus_and_coil_resistances`.
    2. **Inductive** (`:414-499`). The double loop over coils and circuits: the
       per-circuit voltage `M_ij dI_j/dt` at the start of flat-top gives the inductive
       MVA and each circuit's supply rating, and `0.5 * (M I) . I` at each of the six
       waveform times gives the stored poloidal field energy, whose maximum is
       `.pf_power.ensxpfm`. `.heat_transport.peakmva` is the larger of the
       start-of-flat-top total (resistive + inductive) and the end-of-flat-top
       resistive-only figure -- which is why the routine looks at two times at all.
    3. **Dissipative** (`:501-604`). Per-interval storage/power-supply/busbar losses,
       averaged over the flat top because that is when electricity is generated, plus
       the ohmic-heating supply's own wall-plug loss. Their sum is
       `.pf_coil.p_pf_electric_supplies_mw`.

    **The plasma is a circuit but not a coil**, and the distinction drives the loop
    bounds: the outer loop runs over the seven coils (six PF plus the CS) that have a
    resistance and a stored energy, the inner over all eight circuits including the
    plasma, whose current couples into every coil's flux.

    Parameters
    ----------
    rmajor :
        Plasma major radius (m) -- the bus length is `8 * rmajor + 140`.
    c_pf_coil_turn_peak_input :
        Peak current per turn of each coil (A). `.pf_coil.c_pf_coil_turn_peak_input`.
    rhopfbus, rho_pf_coil :
        Resistivity of the aluminium bussing and of the coil conductor (ohm m).
    r_pf_coil_middle :
        Radius of each coil's centre (m).
    j_pf_coil_wp_peak :
        Peak winding-pack current density of each coil (A/m^2).
    f_a_pf_coil_void :
        Void (coolant) fraction of each coil's winding pack.
    c_pf_cs_coils_peak_ma, c_pf_cs_coil_pulse_end_ma :
        Peak and end-of-pulse current of each circuit (MA).
    n_pf_coil_turns :
        Turns in each coil.
    c_pf_coil_turn :
        Current per turn of each circuit at the six waveform times (A), `NGC2` x 6.
        `.pf_coil.c_pf_coil_turn`.
    ind_pf_cs_plasma_mutual :
        Mutual inductance matrix of the eight circuits (H).
    f_p_pf_energy_store_loss, f_p_pf_psu_loss :
        Fractions of the poloidal-field energy flow lost in the store and in the
        power supplies (M. Kovari, "PF power supplies accounting 2", issue #972).
    etapsu :
        Efficiency of the ohmic-heating power supply. `.pf_coil.etapsu`.
    p_plasma_ohmic_mw :
        Ohmic heating power delivered to the plasma (MW).
    t_plant_pulse_coil_precharge, t_plant_pulse_plasma_current_ramp_up,
    t_plant_pulse_fusion_ramp, t_plant_pulse_burn,
    t_plant_pulse_plasma_current_ramp_down :
        The five pulse phases the PF system is active over (s). `.times.t_plant_pulse_*`.
        The dwell is deliberately absent: `PulseTimings.pf_active_cumulative` drops it.

    Returns
    -------
    tuple
        `(srcktpm, poloidalpower, ensxpfm, peakpoloidalpower, peakmva, vpfskv, pfckts,
        spfbusl, acptmax, spsmva, p_pf_electric_supplies_mw)` -- `pfpwr`'s eleven
        written fields, in the source's own assignment order.
    """
    #  Bus length (m), `power.py:340`
    pfbusl = 8.0 * rmajor + 140.0

    res_pf_bus, res_pf_circuit_total, srcktpm = _pf_bus_and_coil_resistances(
        pfbusl=pfbusl,
        c_pf_coil_turn_peak_input=c_pf_coil_turn_peak_input,
        rhopfbus=rhopfbus,
        rho_pf_coil=rho_pf_coil,
        r_pf_coil_middle=r_pf_coil_middle,
        j_pf_coil_wp_peak=j_pf_coil_wp_peak,
        f_a_pf_coil_void=f_a_pf_coil_void,
        c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma,
        c_pf_cs_coil_pulse_end_ma=c_pf_cs_coil_pulse_end_ma,
        n_pf_coil_turns=n_pf_coil_turns,
        topology=topology,
    )
    coils_in_group = coils_in_group_with_cs(topology)
    n_circuits = topology.plasma_index + 1

    # ---- inductive MVA and stored energy (`power.py:414-499`) ----------------------
    # `delktim` is the ramp-up duration; every dI/dt in this block is taken across it.
    delktim = t_plant_pulse_plasma_current_ramp_up

    vpfi = [0.0] * n_circuits
    poloidalenergy = [0.0] * N_PF_ACTIVE_POINTS
    powpfi = 0.0
    powpfr = 0.0
    powpfr2 = 0.0
    coil = -1
    for group in range(len(coils_in_group)):
        for _ in range(coils_in_group[group]):
            coil += 1
            inductxcurrent = [0.0] * N_PF_ACTIVE_POINTS
            powpfii = 0.0
            for circuit in range(n_circuits):
                #  Voltage in `coil` due to the change of current in `circuit`
                vpfij = (
                    ind_pf_cs_plasma_mutual[coil, circuit]
                    * (c_pf_coil_turn[circuit, 2] - c_pf_coil_turn[circuit, 1])
                    / delktim
                )
                vpfi[coil] += vpfij
                powpfii += vpfij * c_pf_coil_turn[coil, 2] / 1.0e6
                for point in range(N_PF_ACTIVE_POINTS):
                    inductxcurrent[point] += (
                        ind_pf_cs_plasma_mutual[coil, circuit]
                        * c_pf_coil_turn[circuit, point]
                    )

            for point in range(N_PF_ACTIVE_POINTS):
                poloidalenergy[point] += (
                    0.5 * inductxcurrent[point] * c_pf_coil_turn[coil, point]
                )

            # Resistive power at the start (index 2) and end (index 4) of flat-top (MW)
            powpfr += (
                n_pf_coil_turns[coil]
                * c_pf_coil_turn[coil, 2]
                * res_pf_circuit_total[group]
                / 1.0e6
            )
            powpfr2 += (
                n_pf_coil_turns[coil]
                * c_pf_coil_turn[coil, 4]
                * res_pf_circuit_total[group]
                / 1.0e6
            )
            powpfi += powpfii

    poloidalenergy = jnp.stack([jnp.asarray(e) for e in poloidalenergy])

    # ---- dissipation over each interval (`power.py:501-566`) -----------------------
    t0 = 0.0
    t1 = t0 + t_plant_pulse_coil_precharge
    t2 = t1 + t_plant_pulse_plasma_current_ramp_up
    t3 = t2 + t_plant_pulse_fusion_ramp
    t4 = t3 + t_plant_pulse_burn
    t5 = t4 + t_plant_pulse_plasma_current_ramp_down
    pf_active_cumulative = [t0, t1, t2, t3, t4, t5]

    poloidalpower = []
    pfdissipation = []
    for interval in range(N_PF_ACTIVE_INTERVALS):
        dt = pf_active_cumulative[interval + 1] - pf_active_cumulative[interval]
        poloidalpower.append(
            jnp.where(
                jnp.abs(dt) > MIN_INTERVAL_S,
                (poloidalenergy[interval + 1] - poloidalenergy[interval])
                / jnp.where(dt == 0.0, 1.0, dt),
                POLOIDAL_POWER_SENTINEL_W,
            )
        )
        pfdissipation.append(
            _pf_loss_interval_total_j(
                interval=interval,
                dt_pulse_phase_s=dt,
                poloidalenergy=poloidalenergy,
                f_p_pf_energy_store_loss=f_p_pf_energy_store_loss,
                f_p_pf_psu_loss=f_p_pf_psu_loss,
                c_pf_coil_turn=c_pf_coil_turn,
                ind_pf_cs_plasma_mutual=ind_pf_cs_plasma_mutual,
                res_pf_bus=res_pf_bus,
                topology=topology,
            )
        )
    poloidalpower = jnp.stack(poloidalpower)

    # Mean dissipated power, over the flat top: "this is the time when electricity is
    # generated" (`power.py:543-556`). The `> 1 s` guard is PROCESS's own give-up.
    flat_top = pf_active_cumulative[4] - pf_active_cumulative[3]
    pfpowermw = (
        jnp.where(
            flat_top > MIN_INTERVAL_S,
            sum(pfdissipation) / jnp.where(flat_top == 0.0, 1.0, flat_top),
            0.0,
        )
        / 1.0e6
    )

    # ---- the stored-energy, MVA and rating summaries (`power.py:558-604`) ----------
    ensxpfm = 1.0e-6 * jnp.max(poloidalenergy)
    peakpoloidalpower = jnp.max(jnp.abs(poloidalpower)) / 1.0e6
    peakmva = jnp.maximum(powpfr + powpfi, powpfr2)

    pfckts = (n_circuits - 2) + PFCKTS_SPARE_CIRCUITS
    spfbusl = pfbusl * pfckts
    spsmva = 0.0
    acptmax = 0.0
    for circuit in range(n_circuits - 1):
        spsmva += 1.0e-6 * jnp.abs(vpfi[circuit] * c_pf_coil_turn_peak_input[circuit])
        acptmax += 1.0e-3 * jnp.abs(c_pf_coil_turn_peak_input[circuit]) / pfckts

    #  Wall-plug power dissipated in the ohmic-heating supply, additional to that
    #  required to move stored energy around (`power.py:596-604`, issue #713).
    wall_plug_ohmicmw = p_plasma_ohmic_mw * (1.0 / etapsu - 1.0)

    return (
        srcktpm,
        poloidalpower,
        ensxpfm,
        peakpoloidalpower,
        peakmva,
        VPFSKV_KV,
        pfckts,
        spfbusl,
        acptmax,
        spsmva,
        wall_plug_ohmicmw + pfpowermw,
    )


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
