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

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.pfcoil import (
    REFERENCE_TOPOLOGY,
    SPHERICAL_TOKAMAK_TOPOLOGY,
    PFCoilTopology,
)
from functional_process.paths import pf_coil, physics
from functional_process.pfcoil.volt_seconds import (
    calculate_pf_coil_turn_currents,
    calculate_pf_cs_volt_seconds,
    calculate_pf_volt_seconds_no_central_solenoid,
)


class PFCoilTurnCurrents(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.turn_currents`. Occupant for the package's one
    joint configuration (`iohcl = 1`, the `(2, 2, 3, 3)` topology); no switch of its
    own -- `pfcoil()` computes the block unconditionally.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static, and the only thing that changes between the two machines: which row is
    the plasma's. The read set does not move -- `pfcoil()`'s tail reads the three arrays
    whole and branches on nothing -- so one node serves both topologies."""

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
            topology=self.topology,
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


class PFCoilVoltSecondsNoCentralSolenoid(PFCoilVoltSeconds):
    """cottax node: `.tokamak.pf_coil.volt_seconds`, the `iohcl = 0` occupant.

    The same two reads and the same two outputs as `PFCoilVoltSeconds` -- `vsec` reads
    the inductance matrix and the turn currents whole on both arms -- so this is a
    subclass with a different body rather than a different signature. What differs is
    which circuits the sums run over; see
    `calculate_pf_volt_seconds_no_central_solenoid`.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)

    def __call__(
        self,
        ind_pf_cs_plasma_mutual=From(pf_coil),
        c_pf_coil_turn=From(pf_coil),
    ):
        return calculate_pf_volt_seconds_no_central_solenoid(
            ind_pf_cs_plasma_mutual=ind_pf_cs_plasma_mutual,
            c_pf_coil_turn=c_pf_coil_turn,
            topology=self.topology,
        )
