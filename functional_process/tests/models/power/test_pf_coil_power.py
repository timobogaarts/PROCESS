"""Harness cases for `functional_process/cottax/power/pf_coil_power.py`.

Audit record: `functional_process/_audit/units/models/power/pf_coil_power.md`.

**Fuzz-only, for the same reason chunk A is**: `tests/unit/models/test_power.py` has no
automatically-generated case for `pfpwr` (unlike `cryo`/`acpow`/
`plant_electric_production`), so there is no legacy point to inherit. The bounds below
are anchored on `tests/regression/input_files/large_tokamak_nof.IN.DAT`'s converged
values rather than invented -- that run is where the four missing producers were
measured, and where the port was checked against PROCESS at `rtol=1e-12` on all eleven
outputs before this file existed (see the record's § "agreement").

**The arrays are eight-wide, not `NGC2`-wide, and that is the contract.** `pfpwr` reads
circuits `0..7` -- six PF coils, the CS, and the plasma -- and nothing above; the port's
signature therefore accepts any array at least that wide, and this case passes exactly
eight so that a regression which started reading past the plasma would fail here rather
than silently pick up a storage zero. `_reference_pfpwr` embeds each one into the
`NGC2`-wide storage PROCESS's `DataStructure` declares.

**Bounded away from zero where PROCESS divides.** `res_pf_coil` divides by
`(1 - f_a_pf_coil_void) * c_pf_cs_coils_peak_ma` and `cptburn` by
`c_pf_cs_coils_peak_ma` again (`power.py:376-396`), and the ramp-up duration is the
denominator of every `dI/dt` (`:414-419`), so those four are drawn strictly positive.
The signs of the coil currents are not physical here -- PROCESS takes `abs()` of every
one it divides by -- and a mixed-sign draw exercises the `abs` the way the real machine
does, where the CS current reverses.
"""

import numpy as np

from functional_process.cottax._harness import Tier1Contract, fuzz_samples
from functional_process.cottax.pfcoil import NGC2
from functional_process.cottax.power.pf_coil_power import (
    N_PF_ACTIVE_POINTS,
    N_PF_CS_PLASMA_CIRCUITS,
    calculate_pf_coil_power_supplies,
)
from process.core.model import DataStructure
from process.models.power import Power
from process.models.pulse import PulseTimings

_CIRCUITS = N_PF_CS_PLASMA_CIRCUITS
_POINTS = N_PF_ACTIVE_POINTS

_N_PF_COILS_IN_GROUP = (1, 1, 2, 2, 1)
"""`(1, 1, 2, 2)` from `large_tokamak_nof.IN.DAT` plus the CS's own one-coil group,
which `pfcoil()` writes at `pfcoil.py:155` when `iohcl != 0`."""


def _band(low, high, shape):
    """`(lower, upper)` arrays of `shape`, so `fuzz_samples` draws that shape."""
    return np.full(shape, low), np.full(shape, high)


def _reference_pfpwr(
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
):
    """Call PROCESS's `Power.pfpwr` through the port's signature.

    The topology fields are set here rather than fuzzed because they are exactly what
    `models/pfcoil/__init__.py` calls graph-assembly data: `iohcl = 1`,
    `n_pf_coil_groups = 4` (`pfpwr` adds the CS's own), `n_pf_coils_in_group` and
    `n_pf_cs_plasma_circuits = 8`. The port bakes the same values as module constants,
    so this is the one place the two spellings of the topology are compared.

    `t_plant_pulse_dwell` is passed as `0.0`: it is the only phase
    `PulseTimings.pf_active_cumulative` drops, so no arithmetic in `pfpwr` can see it.
    """
    data = DataStructure()
    data.build.iohcl = 1
    data.pf_coil.n_pf_coil_groups = len(_N_PF_COILS_IN_GROUP) - 1
    data.pf_coil.n_pf_coils_in_group = np.array([
        *_N_PF_COILS_IN_GROUP,
        *([0] * (NGC2 - len(_N_PF_COILS_IN_GROUP))),
    ])
    data.pf_coil.n_pf_cs_plasma_circuits = _CIRCUITS

    def wide(values):
        """One of this case's eight-wide arrays in `NGC2`-wide storage."""
        out = np.zeros(NGC2)
        out[: len(values)] = values
        return out

    data.physics.rmajor = rmajor
    data.physics.p_plasma_ohmic_mw = p_plasma_ohmic_mw
    data.pf_coil.c_pf_coil_turn_peak_input = wide(c_pf_coil_turn_peak_input)
    data.pf_coil.rhopfbus = rhopfbus
    data.pf_coil.rho_pf_coil = rho_pf_coil
    data.pf_coil.r_pf_coil_middle = wide(r_pf_coil_middle)
    data.pf_coil.j_pf_coil_wp_peak = wide(j_pf_coil_wp_peak)
    data.pf_coil.f_a_pf_coil_void = wide(f_a_pf_coil_void)
    data.pf_coil.c_pf_cs_coils_peak_ma = wide(c_pf_cs_coils_peak_ma)
    data.pf_coil.c_pf_cs_coil_pulse_end_ma = wide(c_pf_cs_coil_pulse_end_ma)
    data.pf_coil.n_pf_coil_turns = wide(n_pf_coil_turns)
    data.pf_coil.etapsu = etapsu

    turns = np.zeros((NGC2, _POINTS))
    turns[:_CIRCUITS, :] = c_pf_coil_turn
    data.pf_coil.c_pf_coil_turn = turns

    mutual = np.zeros((NGC2, NGC2))
    mutual[:_CIRCUITS, :_CIRCUITS] = ind_pf_cs_plasma_mutual
    data.pf_coil.ind_pf_cs_plasma_mutual = mutual

    data.pf_power.f_p_pf_energy_store_loss = f_p_pf_energy_store_loss
    data.pf_power.f_p_pf_psu_loss = f_p_pf_psu_loss

    power = Power()
    power.data = data
    power.pfpwr(
        output=False,
        pulse_timings=PulseTimings(
            t_plant_pulse_coil_precharge=t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up=(t_plant_pulse_plasma_current_ramp_up),
            t_plant_pulse_fusion_ramp=t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn=t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down=(
                t_plant_pulse_plasma_current_ramp_down
            ),
            t_plant_pulse_dwell=0.0,
        ),
    )

    return (
        data.pf_power.srcktpm,
        np.asarray(data.pf_power.poloidalpower),
        data.pf_power.ensxpfm,
        data.pf_power.peakpoloidalpower,
        data.heat_transport.peakmva,
        data.pf_power.vpfskv,
        data.pf_power.pfckts,
        data.pf_power.spfbusl,
        data.pf_power.acptmax,
        data.pf_power.spsmva,
        data.pf_coil.p_pf_electric_supplies_mw,
    )


class TestPfCoilPowerSupplies(Tier1Contract):
    """`Power.pfpwr` -> `calculate_pf_coil_power_supplies`, all eleven outputs."""

    audit_record = "models/power/pf_coil_power.md"
    reference = _reference_pfpwr
    ported = calculate_pf_coil_power_supplies

    fuzz_bounds = {
        "rmajor": (5.0, 12.0),
        "c_pf_coil_turn_peak_input": _band(-4.5e4, 4.5e4, (_CIRCUITS,)),
        "rhopfbus": (1.0e-8, 5.0e-8),
        "rho_pf_coil": (1.0e-8, 5.0e-8),
        "r_pf_coil_middle": _band(2.0, 18.0, (_CIRCUITS,)),
        "j_pf_coil_wp_peak": _band(1.0e6, 3.0e7, (_CIRCUITS,)),
        "f_a_pf_coil_void": _band(0.1, 0.5, (_CIRCUITS,)),
        # Strictly positive: three of PROCESS's divisions land on this array.
        "c_pf_cs_coils_peak_ma": _band(1.0, 40.0, (_CIRCUITS,)),
        "c_pf_cs_coil_pulse_end_ma": _band(-40.0, 40.0, (_CIRCUITS,)),
        "n_pf_coil_turns": _band(50.0, 800.0, (_CIRCUITS,)),
        "c_pf_coil_turn": _band(-5.0e4, 5.0e4, (_CIRCUITS, _POINTS)),
        "ind_pf_cs_plasma_mutual": _band(-2.0, 6.0, (_CIRCUITS, _CIRCUITS)),
        "f_p_pf_energy_store_loss": (0.02, 0.3),
        "f_p_pf_psu_loss": (0.02, 0.3),
        "etapsu": (0.5, 0.95),
        "p_plasma_ohmic_mw": (0.0, 20.0),
        # Every phase longer than the one-second guards `pfpwr` tests against, so the
        # `9.9e9` sentinel and the "give up" flat-top branch are not the case under
        # test here; the ramp-up is additionally the denominator of every `dI/dt`.
        "t_plant_pulse_coil_precharge": (10.0, 60.0),
        "t_plant_pulse_plasma_current_ramp_up": (50.0, 600.0),
        "t_plant_pulse_fusion_ramp": (5.0, 50.0),
        "t_plant_pulse_burn": (100.0, 1.0e4),
        "t_plant_pulse_plasma_current_ramp_down": (50.0, 600.0),
    }
    samples = fuzz_samples(fuzz_bounds, count=20, seed=20260830)
