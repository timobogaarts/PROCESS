"""PROCESS's input encoding, and the one place this port reads it.

`total_process.py` is the tree -- typed slots and the models that fill them, and not one
`i_*` integer anywhere in it. This module is the adapter between that tree and the
legacy IN.DAT format: `switches_from_indat` reads the integers, the registries below map
each one to the occupant it selects, `UNPORTED` records why a real PROCESS value has
none, and `machine_from_indat` assembles the `StellaratorProcess` an input file
describes. Everything switch-shaped is PROCESS's input encoding, not the machine, which
is why it is all here and none of it is beside a subsystem.

`machine_from_indat`'s own docstring is where the argument lives for why assembly time
is the only correct place to resolve a switch (short version: no switch in PROCESS is
ever an iteration variable or a scan variable, so no switch can change between two
evaluations of one assembled graph).

`GRAPH` and `graph_for` live here too, and that is the honest filing rather than a
convenience: `graph_for()` with no argument *is* `REFERENCE_MACHINE`, the graph of one
particular legacy input file, and every caller in this package calls it that way. The
alternative -- keeping them in `total_process.py` -- is not merely churn-minimising, it
is a genuine import cycle, since this module must import `StellaratorProcess` from
there. Run this module directly for a smoke check (builds the graph, prints its
node/port counts and each machine's cycles); `render_xdsm.py`, `mda.py`, `mdf.py`,
`sand_harness.py` and `run_mda_harness.py` import `GRAPH`/`graph_for` from here.
"""

import functools
import re
from pathlib import Path

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import to_graph

from functional_process.models.availability.availability import CplifeAvail
from functional_process.models.availability.namespace import Availability
from functional_process.models.buildings.buildings import (
    Bldgs,
    BldgsSizes,
)
from functional_process.models.buildings.namespace import Buildings
from functional_process.models.costs.costs import (
    CostOfElectricity,
)
from functional_process.models.costs.namespace import Costs
from functional_process.models.physics.confinement_time import (
    StellaratorConfinementTime,
)
from functional_process.models.physics.namespace import (
    Physics,
    PhysicsConfinementTime,
    PhysicsProfiles,
    ProfileParameterisationParabolic,
    ProfileParameterisationPedestal,
)
from functional_process.models.power.electric_production import (
    PlantElectricProductionReactor,
    PowerProfilesOverTime,
)
from functional_process.models.power.namespace import Power
from functional_process.models.power.tf_coil_power import (
    TfPowerResistive,
    TfPowerSuperconducting,
)
from functional_process.models.power.thermal_cryo import (
    CryoLoads,
    CryoQLoadsStep,
    CryoQNucStep,
)
from functional_process.models.stellarator.build import (
    AFwTotalNoPowerflow,
    AFwTotalWithPowerflow,
)
from functional_process.models.stellarator.heating import (
    EcrhHeating,
    LowhybHeating,
)
from functional_process.models.stellarator.namespace import (
    BlanketShieldPowerExponential,
    Stellarator,
    StellaratorFwbs,
)
from functional_process.models.stellarator.plasma_physics import (
    NeutronWallLoad,
    RadiatedWallLoadAndFraction,
)
from functional_process.models.stellarator.preset_config import (
    StellaratorMachineConfig,
    read_stellarator_config_file,
)
from functional_process.models.stellarator.stellarator_fwbs_s2 import (
    DetailedPowerflowBlanketShieldPower,
)
from functional_process.models.stellarator.stellarator_fwbs_s4 import (
    BlanketComponentMasses,
)
from functional_process.models.switch_enums import (
    BlanketDualCoolantModel,
    CoilNuclearHeatingModel,
    CostOfElectricityModel,
    IFEModel,
    NetElectricPowerModel,
    NeutronWallLoadModel,
    PowerFlowModel,
    SphericalTokamakModel,
)
from functional_process.total_process import StellaratorProcess
from process.data_structure.pfcoil_variables import PFConductorModel
from process.data_structure.physics_variables import (
    ConfinementRadiationLossModel,
    ConfinementTimeModel,
    PlasmaIgnitionModel,
)
from process.models.physics.current_drive import CurrentDriveModel
from process.models.power import PumpingPowerModelTypes
from process.models.tfcoil.base import TFConductorModel

REFERENCE_STELLA_CONF = (
    Path(__file__).resolve().parent.parent
    / "tests/regression/input_files/stellarator_helias.stella_conf.json"
)
"""`REFERENCE_INPUT_FILE`'s `istell == 6` machine-config companion.

`Stellarator.st_new_config()` opens `f"{data.globals.output_prefix}stella_conf.json"`
before anything else runs, so for the reference run this file *is* the machine being
designed. Read here, at assembly time, and handed to `StellaratorMachineConfig` as static
data -- the whole point of unit #8's shape decision (`preset_config.md`): `istell == 6`'s
file I/O is a `non-traceable-external-call` that never has to enter a traced body,
because which machine is being designed cannot change during a solve.

Named beside `REFERENCE_INPUT_FILE` below rather than next to the tree, because the
`.stellarator.istell` switch needs it; the two must stay companions (same
stem, same directory), which is what PROCESS's own `output_prefix` convention enforces
for a real run."""


REFERENCE_INPUT_FILE = "tests/regression/input_files/stellarator_helias.IN.DAT"
"""The run this whole port is validated against -- `mda_harness.py`, `mda_constraint_
harness.py` and every number in `_audit/next_steps.md` \u00a7 8 use it. Named here so
`REFERENCE_CONFIGURATION` can be checked against it mechanically instead of by eye."""

_ISTELL_PRESET_REASON = (
    "`istell` in 1..5 selects one of five hardcoded machine presets (Helias 5/4/3, "
    "W7-X 30/50) copied onto `StellaratorConfigData` by `preset_config.py`'s reflective "
    "`hasattr`/`setattr` loop; only `istell == 6` (config read from file) is in scope. "
    "See `core/solver/switches.md` § `data.stellarator.istell` -- second role, whose "
    "disposition is still open in `_audit/next_steps.md` § 2. The confinement-time "
    "binding itself would be identical to the `istell == 6` occupant; it is the "
    "surrounding preset data that is unported"
)
"""Shared by all five refused `istell` presets -- one reason, five values."""

REFERENCE_MACHINE_SWITCHES = {
    "istell": 6,  # `stellarator_helias.IN.DAT:137`
    "isthtr": 1,  # `:139` -- equals PROCESS's own default, listed anyway
    "i_plasma_pedestal": 0,  # `:118`
    "i_cost_model": 0,  # `:248`
    "ireactor": 1,  # `:245` -- equals PROCESS's own default, listed anyway
}
"""The switch values `REFERENCE_INPUT_FILE` actually sets, as a faithful transcription.

**Every switch the file sets explicitly is listed, including ones whose value happens to
equal PROCESS's own default** (`isthtr`, `ireactor`). Listing them regardless makes this
a transcription of the file rather than a diff against PROCESS's defaults, and means a
future change to a default cannot silently move the reference run.
`test_machine.py` parses the file and checks this dict against it, both ways, so
the two cannot drift.

This exists as data, not as behaviour: `machine_from_indat` reads the file itself. It is
here so the check has something to compare against.
"""

UNPORTED = {
    ("istell", 0): (
        "istell == 0 is a tokamak, and this tree has no tokamak: `Stellarator` has no "
        "counterpart namespace, so assembling it would give stellarator geometry, "
        "stellarator coils and stellarator FWBS driven by a tokamak confinement "
        "scaling -- a device nobody has built and this port has never tested. Refused "
        "rather than absent, because it is the kind where assembling anyway hands you a "
        "graph that looks complete and is wrong. Consequence, stated rather than "
        "papered over: `istell` has no usable default here, so an IN.DAT that does not "
        "set `istell = 6` is refused"
    ),
    ("istell", 1): _ISTELL_PRESET_REASON,
    ("istell", 2): _ISTELL_PRESET_REASON,
    ("istell", 3): _ISTELL_PRESET_REASON,
    ("istell", 4): _ISTELL_PRESET_REASON,
    ("istell", 5): _ISTELL_PRESET_REASON,
    ("isthtr", 3): (
        "the NBI branch of `st_heat` calls `current_drive.culnbi()`, a model that is "
        "not audited yet (registry unit #5)"
    ),
    ("blktmodel_ipowerflow", 0): (
        "S2's blktmodel == 1 arm is `blanket_neutronics()`, which calls "
        "`self.hcpb.nuclear_heating_blanket()`/`nuclear_heating_shield()` with zero "
        "arguments against 2-/7-keyword-argument @staticmethods -- a live PROCESS bug "
        "that would TypeError the moment this arm actually executes (unit_registry.md "
        "row 13, next_steps.md §3). hcpb.py's own 3 ported nodes "
        "(NuclearHeatingBlanket/Shield/Magnets) exist but are not usable here until "
        "that call site has a resolution."
    ),
    ("blktmodel_blkttype", 0): (
        "the blktmodel != 0 blanket-mass arm (stellarator.py:1093-1181) computes "
        "m_blkt_steel_total/m_blkt_beryllium from six .build.bl{u,m,p}{i,o}th "
        "sub-assembly thicknesses, additionally writes .fwbs.whtblbreed and "
        ".fwbs.f_a_blkt_cooling_channels, and writes neither .fwbs.m_blkt_li2o nor "
        ".fwbs.m_blkt_vanadium at all -- a different node with a different port set, "
        "not written yet. Refused rather than assembled empty: BlanketCost reads all "
        "four masses unconditionally, so an empty arm would silently hand it boundary "
        "values for fields PROCESS does compute on that arm."
    ),
    ("blktmodel_blkttype", 1): (
        "the liquid-breeder sub-arm (blkttype in {1, 2}, WCLL/HCLL, "
        "stellarator.py:1058-1066) writes .fwbs.wtbllipb and .fwbs.m_blkt_lithium in "
        "place of .fwbs.m_blkt_li2o/.m_blkt_beryllium -- different fields, not a "
        "different formula for the same ones. Not ported: neither replacement field "
        "has a reader in this graph, and stellarator_helias.IN.DAT leaves blkttype at "
        "its default of 3. Values 1 and 2 select the identical formula, so this one "
        "entry covers both; there is no separate value=2 entry because there is no "
        "separate behaviour to name."
    ),
    ("i_tf_sup", 2): (
        "aluminium TF (i_tf_sup == 2) runs the identical calculate_tf_power_resistive "
        "branch as i_tf_sup == 0 -- `Power.tfpwr` dispatches on `i_tf_sup != 1` only, "
        "one formula for both. Request `.tfcoil.i_tf_sup == 0` instead; it fills the "
        "slot with the same occupant. Kept as a refused value rather than a second "
        "registry entry pointing at TfPowerResistive so the claim stays visible."
    ),
    ("i_cost_model", 1): (
        "KOVARI_2014 (i_cost_model == 1) is PROCESS's own default cost model and is "
        "unported: costs_2015.py has no cottax nodes, so on that arm this port computes "
        "no cost of electricity at all and .costs.coe/.costs.concost would surface as "
        "unowned boundary inputs. Filling the slot with the 1990 model instead would "
        "compute *a different number for the same field* -- worse than the "
        "EcrhDensityLimit bug class, which merely computed a value the configuration "
        "never asks for. This used to be spelled as a slot holding None; it is a "
        "refusal now, because a tree with no optional slots cannot say 'absent'"
    ),
    ("i_cost_model", 2): (
        "i_cost_model == 2 injects a user-supplied Model instance at runtime "
        "(process/main.py's `costs` setter, lines 766-768) -- there is no PROCESS-side "
        "subgraph to port at all, so no occupant can exist here. Refused rather than "
        "left absent: unlike KOVARI_2014, a caller asking for this has a model in mind "
        "that this graph has never seen."
    ),
}
"""Why a known PROCESS value has no occupant, verbatim from the `Alternative(unported=)`
declarations this replaced.

**Refusal, and nothing else.** A value in here raises `NotImplementedError` naming the
reason. Its quieter sibling -- a slot holding `None`, meaning *"this configuration's
graph does not compute these values"* -- lives in `COST_OF_ELECTRICITY` and nowhere else.
The four `| None`s that used to be here all left, two because they were unreachable
(every joint key outside `BLANKET_MASSES`/`BLANKET_SHIELD_POWER` already raised) and two
because the configurations they stood for, `i_cost_model == 1` and `istell == 0`, are
ones this port cannot honestly assemble; the distinction between the two kinds survives
in the reasons: `i_cost_model == 1` would hand you a graph that computes no cost of
electricity, `== 2` and `istell == 0` would hand you one that looks complete and is
wrong.

**When a value belongs here and when it belongs in a registry as `None`.** Refuse where
*this port* has not written the arm, or has written something that would be wrong on it.
Assemble absence where **PROCESS itself computes nothing** -- `ireactor != 1 or
ipnet != 0` is the only such case in the tree, and there the six `.costs.coe`-chain
fields keep their entering values in PROCESS exactly as they surface as boundary inputs
here. Refusing that one instead would have made `PowerProfilesOverTime`, a ported and
registered occupant, unreachable through this factory.

Keyed by `(field, value)`. For the two dispatches that read two integers at once the
`field` is the joint name `blktmodel_ipowerflow` / `blktmodel_blkttype` and the `value`
is an **arm index**, not a switch value -- see `_blanket_shield_power_arm` /
`_blanket_mass_arm`, whose docstrings are the mapping.

One of those arms, `("blktmodel_blkttype", 0)`, is unreachable through
`machine_from_indat` and kept anyway: `blktmodel == 1` selects arm 0 of *both*
dispatches, and the shield-power slot is resolved first, so the reason that surfaces is
the `blanket_neutronics()` one. The mass-arm reason is still the correct record of what
`stellarator.py:1093-1181` does, and it is what a future occupant of that arm has to
answer; it is not deleted merely because a sibling refusal masks it.
"""


def _slot_occupant(field, value, registry, *, build=None):
    """One registry lookup, with both failure modes spelled out.

    A miss on the registry *and* on `UNPORTED` means a value PROCESS has never had, or a
    typo -- reported with the values that do exist, which is the "a typo'd value fails
    loudly" property the old `Switch.choose` had and is worth keeping.

    Raises
    ------
    NotImplementedError
        The value is a real PROCESS branch this port has not written an occupant for;
        the recorded reason is in the message.
    ValueError
        The value is not one PROCESS has, or is a typo.
    """
    if value in registry:
        occupant = registry[value]
        return build(occupant) if build is not None else occupant()
    if (field, value) in UNPORTED:
        raise NotImplementedError(
            f"{field} == {value} is a real PROCESS branch but is not ported: "
            f"{UNPORTED[field, value]}"
        )
    raise ValueError(
        f"{field} == {value} is not a known value; this port has occupants for "
        f"{sorted(registry)} and records why it has none for "
        f"{sorted(v for f, v in UNPORTED if f == field)}"
    )


CONFINEMENT_TIME = {6: StellaratorConfinementTime}
"""`.stellarator.istell` -> the confinement-time occupant.

One entry, because this tree has one device. The tokamak arm (`istell == 0`, the bare
`ConfinementTime`) is in `UNPORTED`: it was never a real configuration here, only a
tokamak scaling law bolted onto stellarator geometry, coils and FWBS. `ConfinementTime`
remains the annotation on `PhysicsConfinementTime.model`, since
`StellaratorConfinementTime` subclasses it -- the family is real, and has one member.
"""

HEATING = {1: EcrhHeating, 2: LowhybHeating}
""".stellarator.isthtr` -> the auxiliary-heating occupant."""

FW_AREA = {0: AFwTotalNoPowerflow, 1: AFwTotalWithPowerflow}
"""`.heat_transport.ipowerflow` -> the first-wall-area occupant."""

PROFILE_PARAMETERISATION = {
    0: ProfileParameterisationParabolic,
    1: ProfileParameterisationPedestal,
}
"""`.physics.i_plasma_pedestal` -> the profile-shape occupant.

Both arms are real occupants and both assemble; only one of them is reachable through
`machine_from_indat`, and that is `ST_INIT_I_PLASMA_PEDESTAL`'s doing, not this
registry's. The pedestal arm stays registered for the same reason
`("blktmodel_blkttype", 0)` stays in `UNPORTED`: it is the correct record of a real
PROCESS branch, and a structural what-if reaches it through `eqx.tree_at` the way every
other unreachable arm is reached.
"""

ST_INIT_I_PLASMA_PEDESTAL = 0
"""What `.physics.i_plasma_pedestal` is on a stellarator run, whatever the IN.DAT says.

`process/models/stellarator/initialization.py:31` -- `st_init`, which runs on every
`istell != 0` run -- assigns `data.physics.i_plasma_pedestal = 0` unconditionally, in
the same block that zeroes the central solenoid (`data.build.iohcl = 0`, `:24`). So the
file's value is **dead** on this device: an IN.DAT saying `istell = 6,
i_plasma_pedestal = 1` runs parabolic profiles in PROCESS, and the factory used to read
that `1` and assemble `ProfileParameterisationPedestal` for it -- a configuration
PROCESS cannot produce.

**Read from the forcing rather than from the file, and not refused.** The two honest
options were to pin the arm or to reject a file whose value `st_init` will overwrite;
pinning is what reproduces PROCESS. Refusing would make this port decline an input file
PROCESS runs happily, and the factory's job is to model the run, not to police the file.
That the file's value is ignored is said here, in the docstring of the constant that
ignores it, and pinned by
`test_switch_coverage.test_a_process_forced_switch_cannot_move_the_machine`.

`switch_kwarg_survey.md` §7 records the same shape for `iohcl`, which no test that
compares against the input file can see at all, because neither the file nor the factory
ever mentions it.
"""

BUILDING_SIZING = {
    0: Bldgs,
    1: functools.partial(BldgsSizes, i_hcd_primary=CurrentDriveModel.ITER_NEUTRAL_BEAM),
}
"""`.buildings.i_bldgs_size` -> the building-size occupant."""

TF_POWER = {0: TfPowerResistive, 1: TfPowerSuperconducting}
"""`.tfcoil.i_tf_sup` -> the TF-power occupant."""


def _power_profiles_over_time(i_tf_sup):
    """The `.costs.ireactor == 0` occupant: the power *profiles* only.

    Takes `i_tf_sup` and uses it for nothing, because every occupant of one slot is
    built the same way and this arm has no TF-conductor dependence to carry --
    `PowerProfilesOverTime` declares no static field at all.
    """
    return PowerProfilesOverTime()


def _plant_electric_production_reactor(i_tf_sup):
    """The `.costs.ireactor == 1` occupant: net electric power as well as the profiles.

    `i_tf_sup` is a parameter and not a constant because it is `.tfcoil.i_tf_sup`, the
    switch the `power.tf_power` slot is keyed on; `machine_from_indat` resolves it once
    and threads it here. It used to be written `SUPERCONDUCTING` in a
    `functools.partial`, where an `i_tf_sup = 0` machine kept it.
    """
    return PlantElectricProductionReactor(
        itart=SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO,
        i_tf_sup=i_tf_sup,
        i_blkt_dual_coolant=BlanketDualCoolantModel.SINGLE_COOLANT_SOLID_BREEDER,
        i_p_coolant_pumping=PumpingPowerModelTypes.FRACTION_OF_HEAT,
    )


ELECTRIC_PRODUCTION = {
    0: _power_profiles_over_time,
    1: _plant_electric_production_reactor,
}
"""`.costs.ireactor` -> the electric-production occupant, as a builder taking the
`.tfcoil.i_tf_sup` this occupant needs. Built through `_slot_occupant(..., build=)`, so
the threading is one expression in `machine_from_indat` and not a fifth transcription."""


def _no_cost_of_electricity():
    """The absent occupant of `costs.cost_of_electricity`: `ireactor != 1 or ipnet != 0`.

    `None`, and nothing else. PROCESS does not call `coelc()` on this arm at all, so
    `.costs.coe` and its five companions keep their entering values and surface as
    boundary inputs -- see that slot's own docstring for why absence is the honest
    occupant here and a refusal is not.
    """
    return None  # noqa: RET501 -- the returned `None` is the occupant, not a fall-off


def _cost_of_electricity_calculated():
    """The present occupant: `ireactor == 1 and ipnet == 0`, the only pair `coelc()`
    runs on. The two statics restate the arm that selected this builder.
    """
    return CostOfElectricity(
        ife=IFEModel.MAGNETIC_CONFINEMENT,
        itart=SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO,
        ireactor=CostOfElectricityModel.CALCULATED,
        ipnet=NetElectricPowerModel.SCALED_POSITIVE,
    )


COST_OF_ELECTRICITY = {0: _no_cost_of_electricity, 1: _cost_of_electricity_calculated}
"""`_cost_of_electricity_arm(ireactor, ipnet)` -> the cost-of-electricity occupant, or
`None`. Keyed by the **arm index** that function documents, never by a switch value --
the same discipline the two blanket dispatches follow."""


def _cost_of_electricity_arm(ireactor: int, ipnet: int) -> int:
    """Which arm of `Costs.run()`'s cost-of-electricity dispatch a pair of switches
    selects.

    `process/models/costs/costs.py:82-83`, transcribed:

    ```
    if ireactor == 1 and ipnet == 0:  -> arm 1   CostOfElectricity
    else:                             -> arm 0   nothing is computed
    ```

    Two switches, one condition, so one arm index rather than two keys: `ireactor == 0`
    ("do not calculate MW(electric) or c-o-e", `cost_variables.py:521-525`) and
    `ipnet == 1` ("let go < 0 (no c-o-e)", `:515-519`) are two ways of saying the same
    thing to the same `if`, and neither PROCESS nor this port distinguishes them
    downstream. Arm 1 is PROCESS's own default (`ireactor = 1`, `ipnet = 0`) and the
    reference run.
    """
    return 1 if (ireactor == 1 and ipnet == 0) else 0


BLANKET_SHIELD_POWER = {
    1: BlanketShieldPowerExponential,
    2: DetailedPowerflowBlanketShieldPower,
}
"""`_blanket_shield_power_arm(blktmodel, ipowerflow)` -> the blanket/shield-power
occupant. Keyed by the **arm index** that function documents, never by a switch value."""

BLANKET_MASSES = {2: BlanketComponentMasses}
"""`_blanket_mass_arm(blktmodel, blkttype)` -> the blanket-mass occupant, same kind of
key."""


def _blanket_shield_power_arm(blktmodel: int, ipowerflow: int) -> int:
    """Which arm of `st_fwbs`'s blanket/shield-power dispatch a pair of switches selects.

    `stellarator.py:608-...`, transcribed:

    ```
    if blktmodel == 1:              -> arm 0   blanket_neutronics(); UNPORTED
    else:                           # blktmodel == 0
        if ipowerflow == 0:         -> arm 1   BlanketShieldPowerExponential
        else:                       -> arm 2   DetailedPowerflowBlanketShieldPower
    ```

    So `blktmodel` is the **outer** test and `ipowerflow` only distinguishes the two
    arms *inside* `blktmodel == 0`. Arm 2 is PROCESS's own default (`blktmodel = 0`,
    `ipowerflow = 1`) and the reference run.
    """
    if blktmodel == 1:
        return 0
    return 2 if ipowerflow == 1 else 1


def _blanket_mass_arm(blktmodel: int, blkttype: int) -> int:
    """Which arm of `st_fwbs`'s blanket-mass dispatch a pair of switches selects.

    `stellarator.py:1056-1091`, transcribed:

    ```
    if blktmodel == 0:
        if blkttype in {1, 2}:      -> arm 1   liquid breeder (WCLL/HCLL); UNPORTED
        else:                       -> arm 2   BlanketComponentMasses (solid breeder)
    else:                           # blktmodel == 1
                                    -> arm 0   sub-assembly thicknesses; UNPORTED
    ```

    Again `blktmodel` is the outer test; `blkttype` is consulted only inside
    `blktmodel == 0`. Arm 2 is PROCESS's own default (`blktmodel = 0`, `blkttype = 3`)
    and the reference run. `blkttype`'s values 1 and 2 select the identical formula, so
    they share arm 1.
    """
    if blktmodel != 0:
        return 0
    return 1 if blkttype in {1, 2} else 2


COST_MODEL = {0: Costs}
"""`.costs.i_cost_model` -> the cost-model occupant. `1` (KOVARI_2014, PROCESS's own
default) and `2` are both refused, with their reasons in `UNPORTED`; the slot used to
default to `None` for the first of them and no longer can."""

_INDAT_INTEGER = re.compile(r"\s*([A-Za-z_]\w*)\s*=\s*(-?\d+)\s*(\*.*)?$")


def switches_from_indat(input_file):
    """Every `name = <integer>` this input file sets, as a plain dict.

    Deliberately not a full IN.DAT parser: the only thing a machine is built from is
    integer switches, and PROCESS's own `SingleRun` is what reads everything else.
    A name the file never mentions is simply absent, which is what "falls through to the
    default" means.
    """
    text = Path(input_file).read_text()
    found = {}
    for line in text.splitlines():
        match = _INDAT_INTEGER.match(line)
        if match:
            found[match.group(1)] = int(match.group(2))
    return found


def machine_from_indat(input_file, stella_conf=None):
    r"""The `StellaratorProcess` an IN.DAT describes -- the only thing that builds one.

    Every slot below is passed explicitly because every slot below has no default: there
    is no `StellaratorProcess()` to apply deltas to any more. A switch the file does not
    mention still falls through to PROCESS's own default, but it falls through *here*,
    as the second argument to `switches.get`, where it is visible and cited.

    **The only place in this port an `i_*` integer is ever read.** Everything downstream
    sees a tree of model instances; nothing else has to know that `6` means Helias or
    that `blktmodel` and `ipowerflow` are consulted together.

    **Why assembly time is the only correct place for this, and not a preference.** Every
    switch PROCESS has is a constant for the whole solve:

        grep -n "\"i_\|'i_" process/core/solver/iteration_variables.py   -> no matches
        grep -n "\"i_\|'i_\|istell" process/core/scan.py                -> no matches

    No switch is an iteration variable and none is a scan variable, so no switch can
    change between two evaluations of one assembled graph, and `Scan` re-solves from
    scratch per point anyway. A switch therefore carries no derivative, participates in
    no edge, and has nothing to contribute to a `Graph` -- which is cottax's position
    that graph structure is decided by the caller once, not re-read per evaluation.

    The rejected alternative was one node owning the union of every variant's ports and
    branching internally. That would make a node read `eta_ecrh_injector_wall_plug` *and*
    `eta_lowhyb_injector_wall_plug` regardless of which is live, inventing graph edges
    that do not exist in the run being modelled, and would put a non-differentiable
    integer on a port. It also loses the result `Stellarator.fw_area` records: **a switch
    can decide whether the graph has a cycle**, which is not a fact any single fused node
    could express.

    Joint dispatch is ordinary code here rather than a mechanism: `blktmodel` is read
    together with `ipowerflow` for one slot and with `blkttype` for another, and
    `ireactor` together with `ipnet` for a third, by `_blanket_shield_power_arm` /
    `_blanket_mass_arm` / `_cost_of_electricity_arm`, each of which turns a tuple of
    **legal switch values** into the **arm index** its registry is keyed on. No switch
    value is ever used as a key, and no switch has a default outside its own declared
    domain. So is cross-slot coherence -- `istell == 6` sets both the machine config and
    the confinement binding, because they are two consequences of one choice, which is
    why the two are resolved together, into named locals, before anything else.
    `i_tf_sup`, `ipowerflow` and `ireactor` are read into locals for the same reason and
    a second one: each also has to reach a *static field* of some occupant that branches
    on it internally, and until step 4d each of those fields carried its own hardcoded
    copy of the answer. **A switch is resolved once, here, and threaded; it is never
    also written into a constructor kwarg** -- that is what
    `test_switch_coverage.test_no_slot_contradicts_a_factory_switch` now checks
    mechanically over the whole assembled tree, at every value each switch can take.

    One switch reaches the tree *without* being read from the file at all:
    `i_plasma_pedestal`, which `st_init` overwrites on every stellarator run. See
    `ST_INIT_I_PLASMA_PEDESTAL`.

    Raises
    ------
    NotImplementedError
        The file asks for a real PROCESS branch this port has no occupant for. **A file
        that sets nothing at all raises here**, on `istell`: PROCESS's own default is
        `0`, a tokamak, and this tree has none. `istell` has no usable default and that
        is the intent.
    """
    switches = switches_from_indat(input_file)

    def pick(field, registry, default, **kw):
        return _slot_occupant(field, switches.get(field, default), registry, **kw)

    # The device is resolved first, on its own, and both its consequences together:
    # PROCESS's own default is `istell = 0`, a tokamak, which is in `UNPORTED`, so a
    # file that never mentions `istell` is refused naming `istell` rather than whichever
    # slot the constructor happened to evaluate first.
    istell = switches.get("istell", 0)
    machine_config = _slot_occupant(
        "istell",
        istell,
        {6: StellaratorMachineConfig},
        build=lambda cls: cls(
            machine_config=read_stellarator_config_file(
                REFERENCE_STELLA_CONF if stella_conf is None else stella_conf
            )
        ),
    )
    confinement_time = _slot_occupant(
        "istell",
        istell,
        CONFINEMENT_TIME,
        build=lambda cls: cls(
            i_confinement_time=ConfinementTimeModel.ISS04_STELLARATOR,
            i_rad_loss=ConfinementRadiationLossModel.CORE_ONLY,
            i_plasma_ignited=PlasmaIgnitionModel.IGNITED,
        ),
    )
    # The two joint dispatches, resolved into named locals before the constructor call
    # for the same reason `istell` is: so the *first* thing a refused combination
    # reports is the one the caller asked for, not whichever slot Python evaluated
    # first. Every default here is PROCESS's own -- `fwbs_variables.py:479` for
    # `blktmodel`, `:494` for `blkttype`, `heat_transport_variables.py:94` for
    # `ipowerflow` -- and the switch *values* are turned into **arm indices** by the two
    # named functions above, which is the only thing the registries are keyed on.
    #
    # This used to read `blktmodel = switches.get("blktmodel", 2)` and pass that value
    # through where an arm index was wanted. `2` is not a legal `blktmodel` at all: it
    # was a sentinel meaning "not set", picked so the reference run happened to land on
    # arm 2. The consequence was an inverted mapping -- stating PROCESS's own default
    # `blktmodel = 0` was refused, while `blktmodel = 1` (KIT HCPB neutronics) silently
    # assembled `BlanketShieldPowerExponential`, a node written for a different switch's
    # arm. That is the `ScTfCoilNuclearHeating` bug class, reintroduced by a key
    # derivation instead of by a registration.
    blktmodel = switches.get("blktmodel", 0)
    ipowerflow = switches.get("ipowerflow", 1)
    blanket_shield_power = _slot_occupant(
        "blktmodel_ipowerflow",
        _blanket_shield_power_arm(blktmodel, ipowerflow),
        BLANKET_SHIELD_POWER,
    )
    blanket_masses = _slot_occupant(
        "blktmodel_blkttype",
        _blanket_mass_arm(blktmodel, switches.get("blkttype", 3)),
        BLANKET_MASSES,
    )
    # `i_tf_sup` and `ipowerflow` each decide a slot *and* were each transcribed onto
    # nodes that branch on them internally -- five sites for the first, two for the
    # second -- so a machine could be resistive at `power.tf_power` and superconducting
    # at the five, or pre-2014 at `fw_area` and comprehensive-2014 at the two.
    # Resolved into a local once, here, and threaded below; the nodes lost their
    # constructor kwarg (`switch_kwarg_survey.md` §4.1/§4.3, band (a) items 1 and 3).
    #
    # The slot is resolved *before* the value is threaded, deliberately: `i_tf_sup == 2`
    # is an `UNPORTED` refusal, so no unported value ever reaches an occupant's field.
    i_tf_sup = switches.get("i_tf_sup", 1)
    tf_power = _slot_occupant("i_tf_sup", i_tf_sup, TF_POWER)
    i_tf_sup = TFConductorModel(i_tf_sup)
    ipowerflow = PowerFlowModel(ipowerflow)
    # `ireactor` decides two slots, not one: which electric-production occupant runs,
    # and -- jointly with `ipnet` -- whether `costs.cost_of_electricity` exists at all.
    # `cost_variables.py:521`/`:515` for both defaults.
    ireactor = switches.get("ireactor", 1)
    cost_of_electricity = _slot_occupant(
        "ireactor_ipnet",
        _cost_of_electricity_arm(ireactor, switches.get("ipnet", 0)),
        COST_OF_ELECTRICITY,
    )
    return StellaratorProcess(
        costs=pick(
            "i_cost_model",
            COST_MODEL,
            1,
            build=lambda cls: cls(cost_of_electricity=cost_of_electricity),
        ),
        stellarator=Stellarator(
            machine_config=machine_config,
            heating=pick("isthtr", HEATING, 1),
            fw_area=_slot_occupant("ipowerflow", int(ipowerflow), FW_AREA),
            fwbs=StellaratorFwbs(
                blanket_shield_power=blanket_shield_power,
                blanket_masses=blanket_masses,
            ),
            neutron_wall_load=NeutronWallLoad(
                i_pflux_fw_neutron=NeutronWallLoadModel.SCALED_PLASMA_SURFACE_AREA,
                ipowerflow=ipowerflow,
            ),
            radiated_wall_load_and_fraction=RadiatedWallLoadAndFraction(
                i_pflux_fw_neutron=NeutronWallLoadModel.SCALED_PLASMA_SURFACE_AREA,
                ipowerflow=ipowerflow,
            ),
        ),
        physics=Physics(
            profiles=PhysicsProfiles(
                # Not `switches.get("i_plasma_pedestal", 1)`: `st_init` overwrites the
                # file's value on every stellarator run, so the file cannot decide this
                # slot and this port must not pretend it does. See
                # `ST_INIT_I_PLASMA_PEDESTAL`.
                parameterisation=_slot_occupant(
                    "i_plasma_pedestal",
                    ST_INIT_I_PLASMA_PEDESTAL,
                    PROFILE_PARAMETERISATION,
                ),
            ),
            confinement_time=PhysicsConfinementTime(model=confinement_time),
        ),
        power=Power(
            tf_power=tf_power,
            cryo_q_nuc_step=CryoQNucStep(
                i_tf_sup=i_tf_sup,
                inuclear=CoilNuclearHeatingModel.FRANCES_FOX,
            ),
            cryo_q_loads_step=CryoQLoadsStep(
                i_tf_sup=i_tf_sup,
                i_pf_conductor=PFConductorModel.SUPERCONDUCTING,
            ),
            cryo_loads=CryoLoads(
                i_tf_sup=i_tf_sup,
                i_pf_conductor=PFConductorModel.SUPERCONDUCTING,
            ),
        ),
        buildings=Buildings(sizing=pick("i_bldgs_size", BUILDING_SIZING, 0)),
        availability=Availability(
            electric_production=_slot_occupant(
                "ireactor",
                ireactor,
                ELECTRIC_PRODUCTION,
                build=lambda make: make(i_tf_sup),
            ),
            cplife_avail=CplifeAvail(
                i_tf_sup=i_tf_sup,
                itart=SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO,
            ),
        ),
    )


REFERENCE_MACHINE = machine_from_indat(REFERENCE_INPUT_FILE)
"""The machine `stellarator_helias.IN.DAT` describes -- the run this port is validated
against (`istell = 6`, `i_plasma_pedestal = 0`, `i_cost_model = 0`; every other switch at
PROCESS's own default)."""


def graph_for(machine=None):
    """The assembled graph for one machine; `REFERENCE_MACHINE` if unstated.

    **There is no bare form to fall back to any more, and that is the point.** Five
    registration bugs here shared one root cause: a value copied from PROCESS's bare
    `*_variables.py` defaults rather than from the run modelled (`i_confinement_time` 34
    vs 38, `i_thermal_electric_conversion` 0 vs 2, `i_p_coolant_pumping` 2 vs 1,
    `i_plasma_ignited` 0 vs 1, and `i_cost_model` 1 vs 0, which left `.costs.coe` with no
    producer and 43 nodes unregistered), each found by the MDA harness after the fact.
    The first fix was to stop *defaulting* to the silent-IN.DAT graph; this argument's
    default has been `REFERENCE_MACHINE` since. The second is that the silent-IN.DAT
    graph can no longer be built at all -- `StellaratorProcess()` raises, because every
    switched slot lost its default -- so a machine comes from an IN.DAT or from an
    explicit `eqx.tree_at` on one, and from nowhere else.
    """
    return to_graph(REFERENCE_MACHINE if machine is None else machine)


GRAPH = graph_for()
"""`REFERENCE_MACHINE`'s graph -- the `stellarator_helias.IN.DAT` run this port is
validated against."""

if __name__ == "__main__":
    n_vars = sum(
        len(node.inputs) + len(node.outputs) for node in GRAPH.definitions.values()
    )
    print(f"{len(GRAPH.definitions)} nodes, {n_vars} ports (inputs + outputs, unmerged)")
    for name, node in GRAPH.definitions.items():
        print(f"  {name.path_str()}: {len(node.inputs)} in, {len(node.outputs)} out")

    print("\ncycles, per machine:")
    for label, machine in (
        ("the reference machine", REFERENCE_MACHINE),
        (
            # Both slots `ipowerflow` decides, not just `fw_area`: it also picks arm 1
            # of the blanket/shield-power dispatch. Swapping one and not the other used
            # to be the only spelling available, because arm 1 was unreachable through
            # `machine_from_indat` at all -- the joint key was derived from an illegal
            # `blktmodel` sentinel. It is reachable now, and this what-if says
            # `ipowerflow = 0` coherently.
            "ipowerflow = 0",
            eqx.tree_at(
                lambda m: (
                    m.stellarator.fw_area,
                    m.stellarator.fwbs.blanket_shield_power,
                ),
                REFERENCE_MACHINE,
                (AFwTotalNoPowerflow(), BlanketShieldPowerExponential()),
            ),
        ),
    ):
        cycles = graph_for(machine).cycles
        print(f"  {label}: {[[n.path_str() for n in c] for c in cycles] or 'acyclic'}")
