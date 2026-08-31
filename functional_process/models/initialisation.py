"""The seed's own writes, as nodes: `process/core/init.py` and `st_init`.

`_audit/init_audit.md` is the record. `init.py` writes **35** `DataStructure` fields and
`st_init` a further **18**, and not one of them has a physics formula on its right-hand
side -- which was read for a while as "so it is just defaults", and is not what follows.
Eight of the 35 are *sentinel resolutions* (a dataclass default that is not a value),
four are *presence flags*, and the rest are derivations, a switch-keyed literature
material table among them. Thirteen of those writes land on a value that neither the
input file nor the dataclass default supplies, and those thirteen are exactly the
`off` rows of every `reference_provider_*.txt` pin: the paths where believing the file
and the defaults gives a different machine from the one PROCESS solves.

**Why they are nodes and not a defaults table** (`next_steps.md` §24.1): a derivation
ported as a node *removes* a boundary input, because a node owns its output and cottax's
containment check enforces it. "Port `init.py`'s derivations" and "raise the provider's
independence ratio" are one job counted from two ends. A table would instead have to be
consulted by the provider -- a second copy of `init.py`, in a file this port cannot make
the graph depend on.

**Why they are `raw -> resolved` and not self-loops** (§24.2 item 2). `eff_tf_cryo =
-1.0 -> 0.13` reads and writes one path, which is not a node. The raw value comes from
`importer.read_indat` -- `Imported.raw_values()`'s `.raw.<area>.<field>` namespace exists
for exactly this -- and is threaded here as a **static field on the node**, resolved at
assembly by `indat.resolve_*`, rather than as a traced graph read. That narrowing is
deliberate and it is the same argument `indat._quench_helium_table` makes for the helium
property table:

- The raw value is the input file's text. No model writes any of these fields, so the
  only way one could move during a solve is by being an iteration variable -- and that
  is **checked, not assumed**: `indat` refuses a machine whose `ixc` names a field one of
  these nodes owns, because the alternative is a constant silently overwriting an
  unknown the optimiser is moving.
- A `.raw.*` root would have to be *seeded*, and `mdf.seed` grounds an unrecognised
  `VarPath` at `0.0` (`mdf.py:411-414`) rather than raising. A raw read spelled under a
  root no `DataStructure` has would therefore resolve every sentinel against `0.0` in
  silence. Static is the spelling that cannot do that.

**One home, because the finding was that there wasn't one.** `init_audit.md` §6 counted
six places in this port that had each independently rediscovered an `init.py` rule, each
with its own comment saying so, none registered anywhere -- and a seventh that had
rediscovered one *wrongly*. The nodes here are that shared home; `indat.resolve_*` is the
shared home for the resolutions that decide a *switch* rather than a value, and
`resolve_i_tf_bucking` is the precedent this file follows.

**Faithfulness includes the gaps.** `init.py:933-940` resolves `eff_tf_cryo` for a
superconducting and for a cryo-aluminium magnet and *not* for a water-cooled copper one,
which is left at the `-1.0` sentinel. That is reproduced, not repaired: a defect the
record names is a defect the port carries (`traceability_policy.md`).
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    ModelNamespace,
    OutputInto,
)

from functional_process.paths import (
    build,
    buildings,
    pf_coil,
    physics,
    tfcoil,
    times,
)


class TfCryoplantEfficiency(ExplicitFunction):
    """cottax node: `.tfcoil.eff_tf_cryo`, `init.py:933-940`'s sentinel resolved.

    A node with no reads. That is not a degenerate case here, it is the shape of the
    thing: the value is a *literature constant selected by a switch* -- the ITER
    cryoplant's 0.13 for a superconducting magnet, a Strawbridge-plot extrapolation's
    0.40 for cryo-aluminium -- and a switch is resolved at assembly in this port, never
    read as a graph value (`indat.machine_from_indat`'s docstring says why: no switch in
    PROCESS is ever an iteration or a scan variable). What reaches the graph is one
    number, and what the node buys is that the number has an owner, so the boundary
    provider stops answering `.tfcoil.eff_tf_cryo` from a `-1.0` that is not a value.

    `-1.0` is what a defaults table supplies today, on all seven tracked
    configurations, and `.power.thermal_cryo`'s cryoplant divides by it.
    """

    eff_tf_cryo = OutputInto(tfcoil)

    value: float = dataclasses.field(kw_only=True)
    """The resolved efficiency, from `indat.resolve_eff_tf_cryo`. A fraction in [0, 1]
    -- except on the copper arm PROCESS does not resolve, where the `-1.0` stands."""

    def __call__(self):
        return self.value


class TfInsulationYoungsModulus(ExplicitFunction):
    """cottax node: `.tfcoil.eyoung_ins`, `init.py:961-975`'s material table.

    20 GPa (ITER, DDD11-2 v2 2 2009) for a copper or superconducting magnet, 2.5 GPa
    (Kapton) for cryo-aluminium; `indat.resolve_eyoung_ins` holds the table and the test.
    Read by the TF stress model, which is why the slot is empty on a stellarator: this
    port's stellarator has no TF stress node, so an occupant there would be a write with
    no next use.
    """

    eyoung_ins = OutputInto(tfcoil)

    value: float = dataclasses.field(kw_only=True)
    """Insulation Young's modulus (Pa), from `indat.resolve_eyoung_ins`."""

    def __call__(self):
        return self.value


class TfConductorYoungsModulus(ExplicitFunction):
    """cottax node: `.tfcoil.eyoung_cond_axial` and `.tfcoil.eyoung_cond_trans`,
    `init.py:992-1034`.

    **One node for both, because `init.py` writes both in one branch** whose arms are
    not separable: `i_tf_cond_eyoung_axial == 0` zeroes the pair, `== 2` takes the axial
    modulus from the literature table (Nb3Sn 32 GPa / Bi-2212 80 GPa / NbTi 6.8 GPa /
    REBCO 145 GPa, each with a DOI in `indat.EYOUNG_COND_AXIAL_LITERATURE`) and makes
    the transverse one either zero or a copy, and `== 1` writes neither. Splitting it
    would mint a place for an intermediate that does not exist.

    Every tracked file takes the `== 0` arm -- none of the seven names the switch -- so
    both outputs are `0.0` today, against a `6.6e8` Pa default that reads as an answer.
    """

    eyoung_cond_axial = OutputInto(tfcoil)
    eyoung_cond_trans = OutputInto(tfcoil)

    axial: float = dataclasses.field(kw_only=True)
    transverse: float = dataclasses.field(kw_only=True)
    """Named for the outputs rather than spelling the field names twice: a
    `NodalDeclaration` field is data, not a port, and `eyoung_cond_axial` as both an
    output name and a field name would read as though the two were connected."""

    def __call__(self):
        return self.axial, self.transverse


class PfCoilResistivity(ExplicitFunction):
    """cottax node: `.pf_coil.rho_pf_coil`, `init.py:1140`.

    A **physical consistency rule**: a superconducting PF coil has no resistivity, so the
    field's `2.5e-8` ohm-m default -- copper's, and a real number -- is wrong for the
    conductor every tracked file chooses. `indat.resolve_rho_pf_coil` holds it.
    """

    rho_pf_coil = OutputInto(pf_coil)

    value: float = dataclasses.field(kw_only=True)
    """PF coil winding resistivity (ohm-m), from `indat.resolve_rho_pf_coil`."""

    def __call__(self):
        return self.value


class BeamElectronDensityFraction(ExplicitFunction):
    """cottax node: `.physics.f_nd_beam_electron`, `init.py:1145-1147`.

    The second physical consistency rule: a machine with no neutral beam has no hot beam
    density, whatever the file says. `indat.resolve_f_nd_beam_electron` holds the
    condition.

    **This field can be an iteration variable** -- PROCESS declares one for it -- and a
    node that owns it would then be a constant overwriting an unknown the optimiser is
    moving. `indat._initialisation` refuses such a machine rather than assembling one.
    """

    f_nd_beam_electron = OutputInto(physics)

    value: float = dataclasses.field(kw_only=True)
    """Hot beam density as a fraction of the electron density."""

    def __call__(self):
        return self.value


class EnergyStorageBuildingVolume(ExplicitFunction):
    """cottax node: `.buildings.esbldgm3`, `init.py:827`.

    A steady-state plant stores no pulse energy, so it has no energy storage building,
    and PROCESS says so by zeroing its volume. The slot is empty on a *pulsed* plant --
    there `init.py` writes nothing and the file's value (or
    `buildings_variables.py:143`'s `1.0e3` m^3) is the answer, so a node would be an
    identity with no rule in it.
    """

    esbldgm3 = OutputInto(buildings)

    value: float = dataclasses.field(kw_only=True)
    """Energy storage building volume (m^3). `0.0` wherever this node exists."""

    def __call__(self):
        return self.value


class DoubleNullUpperBuild(ExplicitFunction):
    """cottax node: the upper vertical build forced to match the lower,
    `init.py:610-612`.

    **A build-geometry identity, and it belongs to a build node rather than to a parse
    step** (`init_audit.md` §1.2): under a double-null plasma the machine is up-down
    symmetric, so the upper shield and vacuum vessel take the lower ones' thicknesses.
    PROCESS logs a warning when it does this, because one of the three writes *destroys*
    a value the input file set -- `.build.dz_shld_upper` is an `off` row on both
    double-null configurations, where the file says `0.3` and PROCESS solves with `0.6`.

    **Two of the three writes are here and the third is not.** `init.py:610` also sets
    `dz_fw_plasma_gap = dz_xpoint_divertor`, and on both double-null configurations
    `.build.dz_fw_plasma_gap` is not a boundary path at all -- nothing in the assembled
    graph reads it -- so porting it would be a write with no next use, which this port
    does not do (`next_steps.md` §14.2's rule, applied to a write rather than a read).
    It joins the moment a reader does.

    The slot is empty on a single-null machine, where none of the three fires, and on a
    stellarator, whose own build node already owns `.build.dz_shld_upper`.
    """

    dz_shld_upper = OutputInto(build)
    dz_vv_upper = OutputInto(build)

    def __call__(self, dz_shld_lower=From(build), dz_vv_lower=From(build)):
        return dz_shld_lower, dz_vv_lower


class StellaratorSolenoidAbsent(ExplicitFunction):
    """cottax node: `.build.dr_cs` and `.build.dr_cs_tf_gap`, `st_init:23,26`.

    **A stellarator has no central solenoid**, and `st_init` says so by zeroing its
    thickness and the gap between it and the TF coil -- alongside `iohcl = 0`, which is
    the switch form of the same statement and is answered at assembly
    (`indat._pf_coil_topology`). Both are `off` rows on both stellarator pins, where a
    defaults table supplies `build_variables.py`'s `0.811 m` solenoid and `0.08 m` gap
    for a machine that has neither.

    The right answer to "how thick is the solenoid" on a stellarator is *absence*, not
    zero, and this port's rule is to declare fewer reads rather than read a field as
    zero (`next_steps.md` §20.2). This node exists because the readers are still there:
    `.build.dr_cs` is a boundary path of the assembled stellarator graph today. It is
    the faithful port of what `st_init` does, and it is the thing to delete when those
    reads go.
    """

    dr_cs = OutputInto(build)
    dr_cs_tf_gap = OutputInto(build)

    def __call__(self):
        return 0.0, 0.0


class StellaratorPulseTimes(ExplicitFunction):
    """cottax node: the four pulse phase durations `st_init:43-46` forces.

    **A stellarator runs steady state**, so there is no precharge, no plasma current
    ramp and no ramp-down, and the burn is one year (3.15576e7 s) rather than the
    `times_variables.py` default's 1000 s. All four are `off` rows on both stellarator
    pins -- and they are the four that make the difference most visible, since believing
    the defaults gives a machine that burns for a thousand seconds and stops.

    The three sums built from them are a separate node
    (`models/stellarator/initialization.PulseDurations`), which is `st_init`'s own split:
    the four here are literals and the three there are a function of them.
    """

    t_plant_pulse_coil_precharge = OutputInto(times)
    t_plant_pulse_plasma_current_ramp_up = OutputInto(times)
    t_plant_pulse_burn = OutputInto(times)
    t_plant_pulse_plasma_current_ramp_down = OutputInto(times)

    def __call__(self):
        # `st_init:45`'s own comment: one year.
        return 0.0, 0.0, 3.15576e7, 0.0


class Initialisation(ModelNamespace):
    """The seed's writes, one slot per resolved field.

    Device-agnostic on purpose: `init.py` runs on every machine, and the fields it
    resolves belong to no subsystem in common (`.tfcoil`, `.pf_coil`, `.physics`,
    `.buildings`, `.build`, `.times`). Grouping them by *what writes them* rather than by
    the area they land in is the one grouping that matches the thing being ported, and
    `init_audit.md` §6's finding -- six independent rediscoveries with no shared home --
    is the argument for having a home at all.

    A slot is `None` where the machine's own switches say the seed writes nothing. That
    is `model_tree_design.md`'s "second occupant, not a kwarg" applied to absence, the
    same way `SuperconductingTfCoil.dx_tf_side_case_min` is `None` on a file that sets
    the width itself.
    """

    tf_cryoplant_efficiency: TfCryoplantEfficiency | None = dataclasses.field(
        kw_only=True
    )
    """`init.py:933-940`. Never `None` today -- the resolution has an answer for every
    conductor, the copper arm's answer being the unresolved sentinel."""

    tf_insulation_youngs_modulus: TfInsulationYoungsModulus | None = dataclasses.field(
        kw_only=True
    )
    """`init.py:961-975`. `None` on a stellarator, which has no TF stress node to read
    it."""

    tf_conductor_youngs_modulus: TfConductorYoungsModulus | None = dataclasses.field(
        kw_only=True
    )
    """`init.py:992-1034`. `None` on a stellarator, for the same reason."""

    pf_coil_resistivity: PfCoilResistivity | None = dataclasses.field(kw_only=True)
    """`init.py:1140`. `None` on a stellarator, whose PF nodes do not read it."""

    beam_electron_density_fraction: BeamElectronDensityFraction | None = (
        dataclasses.field(kw_only=True)
    )
    """`init.py:1145-1147`. `None` on a stellarator, where `st_init` turns heating and
    current drive off entirely and no ported node reads the fraction."""

    energy_storage_building_volume: EnergyStorageBuildingVolume | None = (
        dataclasses.field(kw_only=True)
    )
    """`init.py:827`. `None` on a pulsed plant, which keeps its own volume."""

    double_null_upper_build: DoubleNullUpperBuild | None = dataclasses.field(
        kw_only=True
    )
    """`init.py:610-612`. `None` on a single-null machine and on a stellarator."""

    stellarator_solenoid_absent: StellaratorSolenoidAbsent | None = dataclasses.field(
        kw_only=True
    )
    """`st_init:23,26`. `None` on a tokamak, which `st_init` returns from at its first
    line."""

    stellarator_pulse_times: StellaratorPulseTimes | None = dataclasses.field(
        kw_only=True
    )
    """`st_init:43-46`. `None` on a tokamak, where `.times.t_plant_pulse_burn` has a
    real producer (`.tokamak.pulse.burn_time`) and is the variable `mda.CUTS` breaks the
    four-node schedule cycle on."""


__all__ = [
    "BeamElectronDensityFraction",
    "DoubleNullUpperBuild",
    "EnergyStorageBuildingVolume",
    "Initialisation",
    "PfCoilResistivity",
    "StellaratorPulseTimes",
    "StellaratorSolenoidAbsent",
    "TfConductorYoungsModulus",
    "TfCryoplantEfficiency",
    "TfInsulationYoungsModulus",
]
