"""The PF/CS volt-second accounting: per-turn current waveforms and `PFCoil.vsec`.

Audit record: `functional_process/_audit/units/models/pfcoil/volt_seconds.md`.

Added 2026-08-27 (`cold_boundary.md` producer 4). `.pf_coil.vs_cs_pf_total_burn` was
one of the six cold boundary zeros: with it and `.physics.res_plasma` both `0.0`, the
cold `pulse.burn_time` computed `abs(0)/0 - 10 = nan` -- the last of the 11 non-finite
roots. The package `__init__.py` scoped both of this module's blocks out of the sizing
wave by name ("everything reachable only through ... `vsec()`, `waveform`'s downstream
`c_pf_coil_turn` ... is UNPORTED"); this module is that debt, paid.

Two nodes, mirroring PROCESS's own split:

- `PFCoilTurnCurrents` -- the tail of `PFCoil.pfcoil()`
  (`process/models/pfcoil.py:1082-1111`): each circuit's per-turn current at the six
  waveform time points, `.pf_coil.c_pf_coil_turn` owned whole (`NGC2` x 6). Owned as a
  `data` field rather than kept a local of `vsec`, because PROCESS's CS stress-profile
  chain reads it computationally (`pfcoil.py:4235`, unported) and `outvolt`/`outpf`
  report it -- the same producer-side whole-array argument as
  `inductance.py::PFCoilInductance`.
- `PFCoilVoltSeconds` -- `PFCoil.vsec` (`process/models/pfcoil.py:1615-1720`), the
  `iohcl != 0` arm (the package's joint predicate refuses `iohcl = 0` before assembly).
  Owns the two outputs with computational readers: `.pf_coil.vs_cs_pf_total_burn`
  (`pulse.py:159` -> `calculate_burn_time`) and `.pf_coil.vs_cs_pf_total_pulse`
  (constraint 12, `core/solver/constraints.py:582`). The seven other stores of `vsec`
  (`nef`, `vsdum`, `vs_pf_coils_total_ramp`, `vs_cs_ramp`, `vs_cs_pf_total_ramp`,
  `vs_cs_burn`, `vs_pf_coils_total_burn`, `vs_pf_coils_total_pulse`,
  `vs_cs_total_pulse`) have reporting-only readers (`outvolt`, mfile comparison, plot
  summaries -- measured by grep over `process/`) and stay local intermediates, per the
  package's minimal-closure discipline.

**Registering these two closes a measured cycle merge.** `burn_time` reads
`vs_cs_pf_total_burn`; `vsec` reads the inductance matrix and the turn currents; the
turn currents read `waveform`'s arrays; and `flux_swing` reads
`.physics.vs_plasma_ramp_required` from `plasma_inductance.volt_seconds`, which reads
`.times.t_plant_pulse_burn` back from `burn_time`. So the five-node PF coil ring and
the two-node volt-second/burn-time ring become **one nine-node SCC**. Its cut is
measured in `mda.CUTS` (the existing three entries -- `t_plant_pulse_burn`,
`ind_pf_cs_plasma_mutual`, `n_pf_coil_turns` -- remain jointly sufficient and each
necessary; `test_mda.py` re-derives the table).
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.pfcoil import (
    CS_INDEX,
    N_PF_COILS,
    NGC2,
    PLASMA_INDEX,
)
from functional_process.paths import pf_coil, physics


def calculate_pf_coil_turn_currents(
    f_c_pf_cs_peak_time_array,
    c_pf_coil_turn_peak_input,
    c_pf_cs_coils_peak_ma,
    plasma_current,
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
    per_turn = (
        f_c_pf_cs_peak_time_array[:PLASMA_INDEX, :]
        * jnp.copysign(
            c_pf_coil_turn_peak_input[:PLASMA_INDEX],
            c_pf_cs_coils_peak_ma[:PLASMA_INDEX],
        )[:, None]
    )
    plasma_row = plasma_current * jnp.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    return (
        jnp
        .zeros((NGC2, 6))
        .at[:PLASMA_INDEX, :]
        .set(per_turn)
        .at[PLASMA_INDEX, :]
        .set(plasma_row)
    )


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


class PFCoilTurnCurrents(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.turn_currents`. Occupant for the package's one
    joint configuration (`iohcl = 1`, the `(2, 2, 3, 3)` topology); no switch of its
    own -- `pfcoil()` computes the block unconditionally.
    """

    c_pf_coil_turn = OutputInto(pf_coil)

    def __call__(
        self,
        f_c_pf_cs_peak_time_array=From(pf_coil),
        c_pf_coil_turn_peak_input=From(pf_coil),
        c_pf_cs_coils_peak_ma=From(pf_coil),
        plasma_current=From(physics),
    ):
        return calculate_pf_coil_turn_currents(
            f_c_pf_cs_peak_time_array=f_c_pf_cs_peak_time_array,
            c_pf_coil_turn_peak_input=c_pf_coil_turn_peak_input,
            c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma,
            plasma_current=plasma_current,
        )


class PFCoilVoltSeconds(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.volt_seconds`. Occupant for `iohcl = 1` (the
    `iohcl == 0` arm drops the CS terms and one circuit -- a different reads-set,
    refused with the rest of the package by `indat._pf_coil_system_arm`).
    """

    vs_cs_pf_total_burn = OutputInto(pf_coil)
    vs_cs_pf_total_pulse = OutputInto(pf_coil)

    def __call__(
        self,
        ind_pf_cs_plasma_mutual=From(pf_coil),
        c_pf_coil_turn=From(pf_coil),
    ):
        return calculate_pf_cs_volt_seconds(
            ind_pf_cs_plasma_mutual=ind_pf_cs_plasma_mutual,
            c_pf_coil_turn=c_pf_coil_turn,
        )
