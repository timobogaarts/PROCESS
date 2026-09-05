"""What current each PF coil group carries, at each of the three time points.

Audit record: `functional_process/_audit/units/models/pfcoil/currents.md`.

This is the half of `PFCoil.pfcoil()` between the coil placement and the coil sizing:

- `calculate_efc_currents` -- `PFCoil.efc` (`process/models/pfcoil.py:1403-1506`) with
  its three helpers `fixb` (`:5133-5183`), `mtrx` (`:5186-5285`) and `PFCoil.solv`
  (`:1567-1613`), plus the residual norm `rsid` (`:5063-5130`). A damped least-squares
  fit, by SVD, of the group currents that reproduce a wanted field at a set of points.
- `calculate_plasma_initiation_currents` -- `pfcoil()`'s "Flux swing coils" block
  (`:366-405`): the currents that null the field across the plasma midplane at
  breakdown, `.pf_coil.ccl0`.
- `calculate_equilibrium_currents` -- `pfcoil()`'s `i_pf_current = 1`, conventional
  aspect ratio arm (`:456-598`): the divertor-coil currents are fixed analytically and
  the outside-TF groups' currents are solved for against the required vertical field.
- `calculate_cs_flux_swing` -- `:600-661`, the CS current-density ratio that supplies
  whatever volt-seconds the PF set does not.
- `calculate_time_point_currents` -- `:663-728`, each coil's current at beginning of
  pulse, beginning of flat-top and end of flat-top.

**A three-node cycle lives here, and it is real.** `calculate_cs_flux_swing` reads
`.pf_coil.n_pf_coil_turns`, which `masses.py`'s `PFCoilSizes` owns; `PFCoilSizes` reads
`.pf_coil.c_pf_cs_coils_peak_ma`, which `fields.py`'s `PFCoilCurrentWaveform` owns; and
that reads `.pf_coil.c_pf_cs_coil_*_ma`, which `calculate_time_point_currents` owns from
`.pf_coil.f_j_cs_start_end_flat_top`. PROCESS closes the loop by *bootstrapping*: the
first visit to `pfcoil()` sets `ind_pf_cs_plasma_mutual` to all ones and
`n_pf_coil_turns` to a flat 100 (`:605-608`, `first_call`) and then relies on
`Caller.call_models` re-running the whole pipeline up to ten times until nothing moves.
That is a Gauss-Seidel iteration over an undeclared SCC, and it is exactly what this
port is meant to make visible. **Nothing here is a `FixedPointFunction`**: no node reads
a `VarPath` it owns, so this is a genuine multi-node SCC for `Blocking` to find and for
a `Drive` to solve, not a self-loop. See `currents.md` § "The cycle" and this wave's
report -- the assembler has to decide which algorithm drives it.

`.pf_coil.ind_pf_cs_plasma_mutual` **was** a boundary input here and is not one any
more: `inductance.py::PFCoilInductance` ports its producer, `PFCoil.induct`
(`:1721-1984`). The cycle above therefore has four nodes, not three, and the matrix is
an internal edge of the block rather than something supplied from outside -- which also
means PROCESS's `first_call` seeding of it is the iteration's initial guess. See
`inductance.md` § "The cycle, one node larger".
"""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.pfcoil import (
    REFERENCE_TOPOLOGY,
    SPHERICAL_TOKAMAK_TOPOLOGY,
    PFCoilTopology,
    PFLocation,
)
from functional_process.cottax.paths import build, pf_coil, physics
from functional_process.models.pfcoil.currents import (
    calculate_cs_flux_swing,  # noqa: F401 -- re-exported for tests
    calculate_cs_flux_swing_for_topology,
    calculate_efc_currents,  # noqa: F401 -- re-exported for tests
    calculate_equilibrium_currents,  # noqa: F401 -- re-exported for tests
    calculate_equilibrium_currents_for_topology,
    calculate_plasma_initiation_currents,  # noqa: F401 -- re-exported for tests
    calculate_plasma_initiation_currents_for_topology,
    calculate_plasma_initiation_currents_no_central_solenoid,  # noqa: F401 -- re-exported for tests
    calculate_plasma_initiation_currents_no_central_solenoid_for_topology,
    calculate_time_point_currents,  # noqa: F401 -- re-exported for tests
    calculate_time_point_currents_for_topology,
    calculate_time_point_currents_no_central_solenoid,  # noqa: F401 -- re-exported for tests
    calculate_time_point_currents_no_central_solenoid_for_topology,
)

FIXED_CURRENT_GROUPS = REFERENCE_TOPOLOGY.groups_at(PFLocation.ABOVE_TF)
"""The `i_pf_location = 2` divertor-coil groups of the reference topology, whose current
PROCESS fixes analytically and then hands to the equilibrium solve as fixed-current
filaments (`pfcoil.py:485-511`). Each holds one coil on that run, so `nfxf0 = 2`."""

EQUILIBRIUM_GROUPS = REFERENCE_TOPOLOGY.groups_at(PFLocation.OUTSIDE_TF)
"""The reference topology's `i_pf_location = 3` groups, whose current the SVD solves
for, in `pcls0`'s order (`pfcoil.py:519-532`)."""


class CSCurrentDensityPulseStart(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.current_density_pulse_start`.

    One line of `pfcoil()` (`process/models/pfcoil.py:161-164`), given its own node
    because three separate downstream nodes read it and none of them owns it.
    """

    j_cs_pulse_start = OutputInto(pf_coil)

    def __call__(
        self,
        j_cs_flat_top_end=From(pf_coil),
        f_j_cs_start_pulse_end_flat_top=From(pf_coil),
    ):
        return j_cs_flat_top_end * f_j_cs_start_pulse_end_flat_top


class PFCoilInitiationCurrents(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.initiation_currents`. Occupant for `iohcl = 1`.

    Owns `.pf_coil.ccl0` and `.pf_coil.ssq0`. The `iohcl = 0` arm (no CS, so no filaments
    and `c_cs_flat_top_end = 0`, `pfcoil.py:202-204`) is UNPORTED.

    The CS filament placement is computed inside rather than read: PROCESS stores the
    filament arrays in `.pf_coil.r/z/c_pf_cs_current_filaments`, but *overwrites* parts
    of them twice more within the same routine (see `fields.py`'s module docstring), so
    those three `VarPath`s have no single owner and are deliberately not claimed by any
    node in this package.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static. How many groups the solve has and how many coils each holds -- the shape
    of the least-squares problem, not a value in it."""

    ssq0 = OutputInto(pf_coil)
    ccl0 = OutputInto(pf_coil)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        r_pf_coil_middle_group_array=From(pf_coil),
        z_pf_coil_middle_group_array=From(pf_coil),
        r_cs_middle=From(pf_coil),
        dz_cs_full=From(pf_coil),
        a_cs_poloidal=From(pf_coil),
        j_cs_flat_top_end=From(pf_coil),
        f_j_cs_start_pulse_end_flat_top=From(pf_coil),
        alfapf=From(pf_coil),
    ):
        return calculate_plasma_initiation_currents_for_topology(
            rmajor=rmajor,
            rminor=rminor,
            r_pf_coil_middle_group_array=r_pf_coil_middle_group_array,
            z_pf_coil_middle_group_array=z_pf_coil_middle_group_array,
            r_cs_middle=r_cs_middle,
            dz_cs_full=dz_cs_full,
            a_cs_poloidal=a_cs_poloidal,
            j_cs_flat_top_end=j_cs_flat_top_end,
            f_j_cs_start_pulse_end_flat_top=f_j_cs_start_pulse_end_flat_top,
            alfapf=alfapf,
            topology=self.topology,
        )


class PFCoilInitiationCurrentsNoCentralSolenoid(PFCoilInitiationCurrents):
    """cottax node: `.tokamak.pf_coil.initiation_currents`, the `iohcl = 0` occupant.

    Occupant for `spherical_tokamak_eval.IN.DAT:69` / `st_regression.IN.DAT:1485`.
    **Five reads fewer**, not five zeros: `r_cs_middle`, `dz_cs_full`, `a_cs_poloidal`
    and `j_cs_flat_top_end` all exist only to place and charge the CS filaments, and
    `pfcoil.py:202-204` sets `nfxf = 0` before any of them is used. See
    `calculate_plasma_initiation_currents_no_central_solenoid`.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        r_pf_coil_middle_group_array=From(pf_coil),
        z_pf_coil_middle_group_array=From(pf_coil),
        alfapf=From(pf_coil),
    ):
        return calculate_plasma_initiation_currents_no_central_solenoid_for_topology(
            rmajor=rmajor,
            rminor=rminor,
            r_pf_coil_middle_group_array=r_pf_coil_middle_group_array,
            z_pf_coil_middle_group_array=z_pf_coil_middle_group_array,
            alfapf=alfapf,
            topology=self.topology,
        )


class PFCoilEquilibriumCurrents(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.equilibrium_currents`.

    Occupant for `i_pf_current = 1` with `not (itart == 1 and itartpf == 0)`. Owns
    `.pf_coil.ccls` and `.physics.b_plasma_vertical_required` -- the latter is written
    by both arms of the `itart` branch (`pfcoil.py:444-454` and `:575`) from the same
    expression, so it belongs to whichever occupant is instantiated.

    **One node for both topologies, and the reads are why.** Which groups are fixed and
    which are solved for follows from `i_pf_location`, which is `topology`'s; the read
    set does not change with it, because both arms read the same two group arrays whole.
    That is the difference between this slot and `placement`, where the ST arm gains
    `rref` and so needs an occupant of its own.

    UNPORTED arms: `i_pf_current = 0` (currents read from `ccls_ma` instead of solved
    for) and `itart = 1, itartpf = 0` (the spherical-tokamak scaling that bypasses the
    SVD entirely, `:411-454`; **neither tracked ST file takes it**, both setting
    `itartpf = 1`).
    """

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static. Which groups are fixed-current divertor coils and which are equilibrium
    coils -- the shape of the reduced least-squares problem."""

    ccls = OutputInto(pf_coil)
    b_plasma_vertical_required = OutputInto(physics)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        kappa=From(physics),
        aspect=From(physics),
        plasma_current=From(physics),
        beta_poloidal_vol_avg=From(physics),
        ind_plasma_internal_norm=From(physics),
        r_pf_coil_middle_group_array=From(pf_coil),
        z_pf_coil_middle_group_array=From(pf_coil),
        alfapf=From(pf_coil),
    ):
        return calculate_equilibrium_currents_for_topology(
            rmajor=rmajor,
            rminor=rminor,
            kappa=kappa,
            aspect=aspect,
            plasma_current=plasma_current,
            beta_poloidal_vol_avg=beta_poloidal_vol_avg,
            ind_plasma_internal_norm=ind_plasma_internal_norm,
            r_pf_coil_middle_group_array=r_pf_coil_middle_group_array,
            z_pf_coil_middle_group_array=z_pf_coil_middle_group_array,
            alfapf=alfapf,
            topology=self.topology,
        )


class CSFluxSwing(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.flux_swing`. Occupant for `iohcl = 1`.

    **One edge of the SCC** described in this module's docstring: it reads
    `.pf_coil.n_pf_coil_turns`, which `masses.py`'s `PFCoilSizes` owns. Checked against
    the brief's Shape-A/Shape-B rule before declaring it -- this is not an apparent
    self-loop that dissolves on inspection. The producer really is a different node, the
    value really is the sizing pass's output, and PROCESS really does bootstrap it
    (`pfcoil.py:605-608`).
    """

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static, and necessarily a topology *with* a central solenoid -- this node is the
    solenoid's flux-swing balance, and `iohcl = 0` deletes it rather than changing it."""

    f_j_cs_start_end_flat_top = OutputInto(pf_coil)

    def __call__(
        self,
        ccls=From(pf_coil),
        ind_pf_cs_plasma_mutual=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
        vs_plasma_ramp_required=From(physics),
        dr_cs_bore=From(build),
        dr_cs=From(build),
        dz_cs_full=From(pf_coil),
        a_cs_poloidal=From(pf_coil),
        j_cs_flat_top_end=From(pf_coil),
        f_j_cs_start_pulse_end_flat_top=From(pf_coil),
    ):
        return calculate_cs_flux_swing_for_topology(
            ccls=ccls,
            ind_pf_cs_plasma_mutual=ind_pf_cs_plasma_mutual,
            n_pf_coil_turns=n_pf_coil_turns,
            vs_plasma_ramp_required=vs_plasma_ramp_required,
            dr_cs_bore=dr_cs_bore,
            dr_cs=dr_cs,
            dz_cs_full=dz_cs_full,
            a_cs_poloidal=a_cs_poloidal,
            j_cs_flat_top_end=j_cs_flat_top_end,
            f_j_cs_start_pulse_end_flat_top=f_j_cs_start_pulse_end_flat_top,
            topology=self.topology,
        )


class PFCoilTimePointCurrents(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.time_point_currents`. Occupant for
    `i_pf_current != 0` and `iohcl = 1`.

    Owns the three `.pf_coil.c_pf_cs_coil_*_ma` arrays at full `NGC2` width, plus
    `.pf_coil.ccl0_ma`/`.pf_coil.ccls_ma`, which on this arm are a pure unit conversion
    of `ccl0`/`ccls` (`pfcoil.py:678-680`) rather than inputs.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static. Which slot each coil's three currents land in."""

    c_pf_cs_coil_pulse_start_ma = OutputInto(pf_coil)
    c_pf_cs_coil_flat_top_ma = OutputInto(pf_coil)
    c_pf_cs_coil_pulse_end_ma = OutputInto(pf_coil)
    ccl0_ma = OutputInto(pf_coil)
    ccls_ma = OutputInto(pf_coil)

    def __call__(
        self,
        ccl0=From(pf_coil),
        ccls=From(pf_coil),
        a_cs_poloidal=From(pf_coil),
        j_cs_flat_top_end=From(pf_coil),
        f_j_cs_start_pulse_end_flat_top=From(pf_coil),
        f_j_cs_start_end_flat_top=From(pf_coil),
    ):
        return calculate_time_point_currents_for_topology(
            ccl0=ccl0,
            ccls=ccls,
            a_cs_poloidal=a_cs_poloidal,
            j_cs_flat_top_end=j_cs_flat_top_end,
            f_j_cs_start_pulse_end_flat_top=f_j_cs_start_pulse_end_flat_top,
            f_j_cs_start_end_flat_top=f_j_cs_start_end_flat_top,
            topology=self.topology,
        )


class PFCoilTimePointCurrentsNoCentralSolenoid(PFCoilTimePointCurrents):
    """cottax node: `.tokamak.pf_coil.time_point_currents`, the `iohcl = 0` occupant.

    Occupant for `i_pf_current != 0` with no central solenoid. Two differences from
    `PFCoilTimePointCurrents`:

    - `a_cs_poloidal` and `j_cs_flat_top_end` are not read. They exist here only to form
      `c_cs_flat_top_end`, which `pfcoil.py:203` sets to zero outright on this arm.
    - **It owns `.pf_coil.f_j_cs_start_end_flat_top` instead of reading it.** On the
      conventional arm that ratio comes from `.tokamak.cs_coil.flux_swing`; with no
      solenoid `pfcoil.py:658-661` assigns the constant `1.0` and there is no
      flux-swing node to assign it. The field's storage default is `0.0`, not `1.0`, so
      this is a value that has to be produced rather than left to the boundary --
      see `F_J_CS_START_END_FLAT_TOP_NO_CS`.

    **This breaks the package's four-node cycle on this machine**, which is a finding
    rather than a convenience: with `flux_swing` gone the edge
    `sizes -> flux_swing -> time_point_currents` has no middle, and
    `.pf_coil.f_j_cs_start_end_flat_top` stops being loop-carried. Whether the remaining
    edges still close a ring is `Blocking`'s question, not this class's.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)

    f_j_cs_start_end_flat_top = OutputInto(pf_coil)

    def __call__(
        self,
        ccl0=From(pf_coil),
        ccls=From(pf_coil),
        f_j_cs_start_pulse_end_flat_top=From(pf_coil),
    ):
        return calculate_time_point_currents_no_central_solenoid_for_topology(
            ccl0=ccl0,
            ccls=ccls,
            f_j_cs_start_pulse_end_flat_top=f_j_cs_start_pulse_end_flat_top,
            topology=self.topology,
        )
