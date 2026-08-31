"""`models/initialisation` against PROCESS's own seed, on all seven configurations.

**This module is the oracle those nodes took away.** Until they landed, every one of the
thirteen fields was a *boundary* path, and `provider.disagreements` compared the port's
answer with `init_process`'s on every run -- that is what the `off` rows in each
`reference_provider_*.txt` were. A path a node produces is no longer a boundary path, so
the provider stops checking it: the pins going to zero `off` rows is the removal of a
check as much as it is the closing of a gap. The check has to come back somewhere, and
this is where.

Two halves, and they fail differently:

- **The resolvers** (`indat.resolve_*`) are checked arm by arm against `process/core/
  init.py`'s own branches, including the arms no tracked file takes and the one arm
  `init.py` *does not* have (a water-cooled copper magnet's cryoplant efficiency, which
  is left at the sentinel). No PROCESS run; these are table lookups.
- **The assembled machines** are checked against a real `init_process`, field by field,
  for all seven configurations. That is the half that would catch a resolver wired to
  the wrong switch, or a slot occupied on a machine PROCESS does not write.

`cold_start.cold_state` is cached on disk, so the seven runs cost a first-time price
(~6 s per stellarator, under a second per tokamak) and nothing after.
"""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from functional_process import indat  # noqa: E402
from functional_process.cold_start import cold_state  # noqa: E402
from functional_process.indat import graph_for, machine_from_indat  # noqa: E402
from functional_process.provider import CONFIGURATIONS, stem  # noqa: E402
from functional_process.vocabulary import (  # noqa: E402
    SuperconductorModel,
    TFConductorModel,
)

# --------------------------------------------------------------- the resolvers alone


@pytest.mark.parametrize(
    ("i_tf_sup", "expected"),
    [
        (TFConductorModel.SUPERCONDUCTING, 0.13),
        (TFConductorModel.HELIUM_COOLED_ALUMINIUM, 0.40),
        # `init.py:935-940` has no copper arm, so the sentinel survives and
        # `.power.thermal_cryo` divides by `-1.0`. A defect the record names is a defect
        # the port carries.
        (TFConductorModel.WATER_COOLED_COPPER, -1.0),
    ],
)
def test_the_cryoplant_efficiency_sentinel_resolves_by_conductor(i_tf_sup, expected):
    assert indat.resolve_eff_tf_cryo(indat.EFF_TF_CRYO_UNSET, i_tf_sup) == expected


def test_a_stated_cryoplant_efficiency_is_not_touched():
    """The sentinel is a neighbourhood of `-1.0`, not the literal, and everything else
    is the user's -- including a `0.0` no `abs(x + 1) < 1e-6` test would catch.
    """
    for stated in (0.0, 0.13, 0.5, 1.0):
        assert (
            indat.resolve_eff_tf_cryo(stated, TFConductorModel.SUPERCONDUCTING) == stated
        )


@pytest.mark.parametrize(
    ("i_tf_sup", "expected"),
    [
        (TFConductorModel.SUPERCONDUCTING, 20.0e9),
        # `init.py:962-967`: copper has no insulation material defined and borrows the
        # ITER design value, so it shares the superconductor's arm here where it has
        # none in `resolve_eff_tf_cryo`.
        (TFConductorModel.WATER_COOLED_COPPER, 20.0e9),
        (TFConductorModel.HELIUM_COOLED_ALUMINIUM, 2.5e9),
    ],
)
def test_the_insulation_modulus_sentinel_resolves_by_conductor(i_tf_sup, expected):
    assert indat.resolve_eyoung_ins(indat.EYOUNG_INS_UNSET, i_tf_sup) == expected


def test_the_insulation_modulus_sentinel_is_an_inequality():
    """`init.py:961` tests `<= 1e8`, so *any* modulus at or below the sentinel is unset
    -- a file asking for a genuinely soft insulator gets ITER's 20 GPa instead. Pinned
    rather than repaired: it is PROCESS's own test, transcribed.
    """
    assert indat.resolve_eyoung_ins(5.0e7, TFConductorModel.SUPERCONDUCTING) == 20.0e9  # noqa: RUF069 -- exact by intent; see the module docstring
    assert indat.resolve_eyoung_ins(2.0e8, TFConductorModel.SUPERCONDUCTING) == 2.0e8  # noqa: RUF069 -- exact by intent; see the module docstring


def test_the_conductor_moduli_are_zeroed_when_stiffness_is_not_considered():
    """`i_tf_cond_eyoung_axial == 0`, which is the default and every tracked file's arm.
    Both outputs go to zero and neither raw value survives -- including the `6.6e8` Pa
    axial default that reads as an answer.
    """
    assert indat.resolve_eyoung_cond(
        6.6e8, 1.0e9, 0, 1, SuperconductorModel.ITER_NB3SN
    ) == (0.0, 0.0)


def test_the_conductor_moduli_are_the_user_s_at_arm_one():
    """`i_tf_cond_eyoung_axial == 1`: `init.py` writes neither, so the node is the
    identity and the file's numbers reach the stress model untouched.
    """
    assert indat.resolve_eyoung_cond(
        1.0e10, 2.0e10, 1, 1, SuperconductorModel.ITER_NB3SN
    ) == (1.0e10, 2.0e10)


@pytest.mark.parametrize(
    ("i_tf_sc_mat", "axial"),
    [
        (SuperconductorModel.ITER_NB3SN, 32e9),
        (SuperconductorModel.WST_NB3SN, 32e9),
        (SuperconductorModel.BI2212, 80e9),
        (SuperconductorModel.OLD_LUBELL_NBTI, 6.8e9),
        (SuperconductorModel.CROCO_REBCO, 145e9),
    ],
)
def test_the_conductor_axial_modulus_is_keyed_on_the_material_not_the_model(
    i_tf_sc_mat, axial
):
    """`init.py:1002-1027` keys on `SuperconductorModel(...).material`, so two models
    naming one material get one modulus -- the `dcond[]` shape, and the reason
    `EYOUNG_COND_AXIAL_LITERATURE` is keyed on `SuperconductorMaterial`.
    """
    assert indat.resolve_eyoung_cond(6.6e8, 0.0, 2, 1, i_tf_sc_mat)[0] == axial


def test_the_transverse_modulus_copies_the_axial_one_or_is_zero():
    """`init.py:1029-1034`, the inner branch of the `== 2` arm."""
    assert indat.resolve_eyoung_cond(
        6.6e8, 0.0, 2, 1, SuperconductorModel.ITER_NB3SN
    ) == (32e9, 32e9)
    assert indat.resolve_eyoung_cond(
        6.6e8, 0.0, 2, 0, SuperconductorModel.ITER_NB3SN
    ) == (32e9, 0.0)


def test_a_superconducting_pf_coil_has_no_resistivity():
    assert indat.resolve_rho_pf_coil(2.5e-8, 0) == 0.0  # noqa: RUF069 -- exact by intent; see the module docstring
    assert indat.resolve_rho_pf_coil(2.5e-8, 1) == 2.5e-8  # noqa: RUF069 -- exact by intent; see the module docstring


@pytest.mark.parametrize(
    ("i_hcd_calculations", "i_hcd_primary", "expected"),
    [
        (1, 5, 0.005),  # NBI, and the fraction is the user's
        (1, 8, 0.005),  # the second NBI value
        (1, 10, 0.0),  # heating, but not a beam
        (0, 5, 0.0),  # a beam named, but no HCD calculation at all
    ],
)
def test_no_beam_means_no_beam_density(i_hcd_calculations, i_hcd_primary, expected):
    """`init.py:1145-1147`'s nested `if`, written as the conjunction it is."""
    assert (
        indat.resolve_f_nd_beam_electron(0.005, i_hcd_calculations, i_hcd_primary)
        == expected
    )


def test_a_steady_state_plant_has_no_energy_storage_building():
    assert indat.resolve_esbldgm3(1.0e3, 0) == 0.0  # noqa: RUF069 -- exact by intent; see the module docstring
    assert indat.resolve_esbldgm3(1.0e3, 1) == 1.0e3  # noqa: RUF069 -- exact by intent; see the module docstring


# ----------------------------------------------------- the machines against the seed

SEED_FIELDS = {
    "tf_cryoplant_efficiency": {"value": ("tfcoil", "eff_tf_cryo")},
    "tf_insulation_youngs_modulus": {"value": ("tfcoil", "eyoung_ins")},
    "tf_conductor_youngs_modulus": {
        "axial": ("tfcoil", "eyoung_cond_axial"),
        "transverse": ("tfcoil", "eyoung_cond_trans"),
    },
    "pf_coil_resistivity": {"value": ("pf_coil", "rho_pf_coil")},
    "beam_electron_density_fraction": {"value": ("physics", "f_nd_beam_electron")},
    "energy_storage_building_volume": {"value": ("buildings", "esbldgm3")},
}
"""`{slot: {node field: (area, DataStructure field)}}` for every occupant that carries
its answer as data. The two nodes that carry theirs as literals in the body
(`StellaratorSolenoidAbsent`, `StellaratorPulseTimes`) are checked separately, and the
node that reads the graph (`DoubleNullUpperBuild`) is checked by its inputs."""

ST_INIT_LITERALS = {
    "stellarator_solenoid_absent": {
        ("build", "dr_cs"): 0.0,
        ("build", "dr_cs_tf_gap"): 0.0,
    },
    "stellarator_pulse_times": {
        ("times", "t_plant_pulse_coil_precharge"): 0.0,
        ("times", "t_plant_pulse_plasma_current_ramp_up"): 0.0,
        ("times", "t_plant_pulse_burn"): 3.15576e7,
        ("times", "t_plant_pulse_plasma_current_ramp_down"): 0.0,
    },
}


@pytest.mark.parametrize(
    "input_file", CONFIGURATIONS, ids=[stem(c) for c in CONFIGURATIONS]
)
def test_every_occupied_slot_agrees_with_init_process(input_file):
    """The oracle. An occupant's value is PROCESS's post-`init_process` value, exactly.

    Exact equality and not a tolerance: every one of these is a literal or a copy on
    both sides, so a difference in the last digits would mean the two are not the same
    number rather than that one of them rounded.
    """
    seed = cold_state(input_file).seed
    machine = machine_from_indat(input_file)
    for slot, fields in SEED_FIELDS.items():
        occupant = getattr(machine.initialisation, slot)
        if occupant is None:
            continue
        for attribute, (area, field) in fields.items():
            assert getattr(occupant, attribute) == getattr(getattr(seed, area), field), (
                f"{slot}.{attribute} against .{area}.{field}"
            )


@pytest.mark.parametrize(
    "input_file", CONFIGURATIONS, ids=[stem(c) for c in CONFIGURATIONS]
)
def test_an_empty_slot_is_a_field_init_process_leaves_alone(input_file):
    """The other half of the oracle, and the one a coverage count cannot give.

    A slot is empty where `init.py`'s branch does not fire, and "does not fire" means
    the seed holds the value the *file* holds. Checked for the two slots whose emptiness
    is decided by a switch rather than by the device: a machine wrongly classified as
    pulsed, or as single-null, fails here rather than silently keeping a boundary input
    the solve then answers from a stale default.
    """
    seed = cold_state(input_file).seed
    machine = machine_from_indat(input_file)
    if machine.initialisation.energy_storage_building_volume is None:
        assert int(seed.pulse.i_pulsed_plant) == 1
        assert seed.buildings.esbldgm3 != 0.0  # noqa: RUF069 -- exact by intent; see the module docstring
    if machine.initialisation.double_null_upper_build is None:
        # A stellarator has no such slot to fill; a single-null tokamak keeps the file's
        # own upper shield thickness.
        assert int(seed.physics.i_single_null) == 1 or int(seed.stellarator.istell) != 0
    else:
        assert seed.build.dz_shld_upper == seed.build.dz_shld_lower
        assert seed.build.dz_vv_upper == seed.build.dz_vv_lower


@pytest.mark.parametrize(
    "input_file", CONFIGURATIONS, ids=[stem(c) for c in CONFIGURATIONS]
)
def test_st_init_s_literals_are_the_seed_s_on_a_stellarator_and_absent_otherwise(
    input_file,
):
    """`st_init` returns at its first line on `istell == 0`, so both slots are the
    stellarator arm of one dispatch -- and on a tokamak the *same* fields must not be
    those literals, or the port would be reproducing a forcing PROCESS did not apply.
    """
    seed = cold_state(input_file).seed
    machine = machine_from_indat(input_file)
    stellarator = int(seed.stellarator.istell) != 0
    for slot, expected in ST_INIT_LITERALS.items():
        occupant = getattr(machine.initialisation, slot)
        assert (occupant is not None) is stellarator, slot
        if not stellarator:
            continue
        for (area, field), value in expected.items():
            assert getattr(getattr(seed, area), field) == value, f".{area}.{field}"


def test_the_seed_owned_field_list_covers_every_output_these_nodes_declare():
    """`SEED_OWNED_FIELDS` is what `_refuse_seed_owned_unknowns` checks an `ixc` against,
    and a field that a node owns but the list omits is a check that silently does not
    happen. Derived from the assembled graphs rather than from the list, so adding an
    occupant without listing its field fails here.
    """
    owned = set()
    for input_file in CONFIGURATIONS:
        graph = graph_for(machine_from_indat(input_file))
        for path, node in graph.definitions.items():
            if not path.path_str().startswith(".initialisation"):
                continue
            owned |= {out.var.path_str().rsplit(".", 1)[-1] for out in node.outputs}
    assert owned <= set(indat.SEED_OWNED_FIELDS), owned - set(indat.SEED_OWNED_FIELDS)
    # And nothing in the list is dead: every name is a field some machine's seed nodes
    # actually own, across the seven.
    assert set(indat.SEED_OWNED_FIELDS) == owned
