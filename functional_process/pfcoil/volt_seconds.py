"""Pure functions for the PF/CS volt-second accounting: per-turn current waveforms
and the CS/PF volt-seconds, extracted from
`functional_process/models/pfcoil/volt_seconds.py`.

That module still holds the graph declarations (`ExplicitFunction` occupants) that wire
these functions to `VarPath`s; read its module docstring for the SCC these nodes close.
The audit record is `functional_process/_audit/units/models/pfcoil/volt_seconds.md`
and mirrors these functions, not the declarations that call them.
"""

import jax.numpy as jnp

from functional_process.models.pfcoil import (
    CS_INDEX,
    N_PF_COILS,
    NGC2,
    PLASMA_INDEX,
    REFERENCE_TOPOLOGY,
    SPHERICAL_TOKAMAK_TOPOLOGY,
)


def calculate_pf_coil_turn_currents(
    f_c_pf_cs_peak_time_array,
    c_pf_coil_turn_peak_input,
    c_pf_cs_coils_peak_ma,
    plasma_current,
    *,
    topology=REFERENCE_TOPOLOGY,
):
    """Per-turn current of every circuit at the six waveform time points (A).

    Ports the tail of `PFCoil.pfcoil()`, `process/models/pfcoil.py:1082-1111`,
    unchanged: for the six PF coils and the CS (circuits `0..6`), the waveform
    fraction times `copysign(c_pf_coil_turn_peak_input[i], c_pf_cs_coils_peak_ma[i])`;
    the plasma circuit (row 7) is `[0, 0, I_p, I_p, I_p, 0]`. Rows `8..21` stay the
    storage default `0.0`, exactly as PROCESS leaves them.

    Parameters
    ----------
    f_c_pf_cs_peak_time_array :
        Waveform fraction per circuit per time point.
        `.pf_coil.f_c_pf_cs_peak_time_array` (`NGC2` x 6).
    c_pf_coil_turn_peak_input :
        Input peak current per turn for each PF coil / the CS (A).
        `.pf_coil.c_pf_coil_turn_peak_input`.
    c_pf_cs_coils_peak_ma :
        Peak current of each circuit (MA) -- read for its *sign* only.
        `.pf_coil.c_pf_cs_coils_peak_ma`.
    plasma_current :
        Plasma current (A). `.physics.plasma_current`.

    Returns
    -------
    :
        `.pf_coil.c_pf_coil_turn` (`NGC2` x 6), A.
    """
    plasma = topology.plasma_index
    per_turn = (
        f_c_pf_cs_peak_time_array[:plasma, :]
        * jnp.copysign(
            c_pf_coil_turn_peak_input[:plasma],
            c_pf_cs_coils_peak_ma[:plasma],
        )[:, None]
    )
    plasma_row = plasma_current * jnp.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    return (
        jnp.zeros((NGC2, 6)).at[:plasma, :].set(per_turn).at[plasma, :].set(plasma_row)
    )


def calculate_pf_volt_seconds_no_central_solenoid(
    ind_pf_cs_plasma_mutual,
    c_pf_coil_turn,
    *,
    topology=SPHERICAL_TOKAMAK_TOPOLOGY,
):
    """`calculate_pf_cs_volt_seconds` on a machine with no central solenoid.

    Ports `PFCoil.vsec`, `process/models/pfcoil.py:1615-1720`, at `iohcl = 0`. Two
    differences, both from the same `if`:

    - `nef = n_pf_cs_plasma_circuits - 1` rather than `- 2` (`:1622-1626`), so the PF
      loop covers **every** circuit but the plasma -- there is no CS circuit to leave
      out of it.
    - `vs_cs_ramp` and `vs_cs_burn` are **never assigned** on this arm (`:1647`,
      `:1677`, both guarded), so PROCESS's totals are the PF sums plus whatever those
      two fields already hold -- their `pfcoil_variables.py` storage default, `0.0`.
      Reproduced by leaving them out of the sums entirely rather than adding a zero,
      because the two are the same number and only one of them is the same *statement*.

    Returns
    -------
    tuple
        `(vs_cs_pf_total_burn, vs_cs_pf_total_pulse)`, Wb.
    """
    n = topology.n_pf_coils
    ind_plasma = ind_pf_cs_plasma_mutual[topology.plasma_index, :]

    vs_ramp = jnp.sum(ind_plasma[:n] * (c_pf_coil_turn[:n, 2] - c_pf_coil_turn[:n, 1]))
    vs_burn = jnp.sum(ind_plasma[:n] * (c_pf_coil_turn[:n, 4] - c_pf_coil_turn[:n, 2]))
    return vs_burn, vs_ramp + vs_burn


def calculate_pf_cs_volt_seconds(ind_pf_cs_plasma_mutual, c_pf_coil_turn):
    """Volt-second capability of the PF/CS system linked to the plasma (Wb).

    Ports `PFCoil.vsec`, `process/models/pfcoil.py:1615-1720`, `iohcl != 0` arm, with
    the topology baked the way the whole package bakes it (`__init__.py`): `nef =
    n_pf_cs_plasma_circuits - 2 = 6` PF circuits, the CS at index 6, the plasma row of
    the mutual-inductance matrix at index 7. The `vsdum` scratch array is the loops'
    common subexpression and is not materialised.

    Per circuit `i` (the source's `vsdum` columns spelled out; time-point columns
    `1`/`2`/`4` are the only ones `vsec` reads):

        vs_ramp_i = M[plasma, i] * (c_turn[i, 2] - c_turn[i, 1])
        vs_burn_i = M[plasma, i] * (c_turn[i, 4] - c_turn[i, 2])

    summed over the six PF circuits plus the CS; `pulse = ramp + burn`.

    Parameters
    ----------
    ind_pf_cs_plasma_mutual :
        Mutual-inductance matrix (H). `.pf_coil.ind_pf_cs_plasma_mutual`
        (`NGC2` x `NGC2`); only the plasma row is read, as in the source.
    c_pf_coil_turn :
        Per-turn circuit currents at the six time points (A).
        `.pf_coil.c_pf_coil_turn`.

    Returns
    -------
    tuple
        `(vs_cs_pf_total_burn, vs_cs_pf_total_pulse)`, Wb --
        `.pf_coil.vs_cs_pf_total_burn` (negative on this machine;
        `calculate_burn_time` takes `abs()`), `.pf_coil.vs_cs_pf_total_pulse`.
    """
    ind_plasma = ind_pf_cs_plasma_mutual[PLASMA_INDEX, :]

    vs_pf_coils_total_ramp = jnp.sum(
        ind_plasma[:N_PF_COILS]
        * (c_pf_coil_turn[:N_PF_COILS, 2] - c_pf_coil_turn[:N_PF_COILS, 1])
    )
    vs_cs_ramp = ind_plasma[CS_INDEX] * (
        c_pf_coil_turn[CS_INDEX, 2] - c_pf_coil_turn[CS_INDEX, 1]
    )
    vs_cs_pf_total_ramp = vs_cs_ramp + vs_pf_coils_total_ramp

    vs_cs_burn = ind_plasma[CS_INDEX] * (
        c_pf_coil_turn[CS_INDEX, 4] - c_pf_coil_turn[CS_INDEX, 2]
    )
    vs_pf_coils_total_burn = jnp.sum(
        ind_plasma[:N_PF_COILS]
        * (c_pf_coil_turn[:N_PF_COILS, 4] - c_pf_coil_turn[:N_PF_COILS, 2])
    )
    vs_cs_pf_total_burn = vs_cs_burn + vs_pf_coils_total_burn

    vs_cs_pf_total_pulse = vs_cs_pf_total_ramp + vs_cs_pf_total_burn
    return vs_cs_pf_total_burn, vs_cs_pf_total_pulse
