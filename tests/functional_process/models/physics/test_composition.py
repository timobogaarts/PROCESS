"""Harness cases for `composition.py`.

Both reference adapters build a real `Physics` instance (most of its constructor
sub-models are `None` -- neither `plasma_composition` nor
`calculate_effective_charge_ionisation_profiles` touch them, confirmed by the audit
record's data-footprint table) and a real `DataStructure`, run
`impurity_radiation.initialise_imprad` to populate the real L(Z, Te)/atomic-mass tables
(the same move `test_radiation_power.py` makes), then call PROCESS's own method. Nothing
here is a transcription of expected numbers -- if the audit's read set were wrong, the
adapter would leave a field at its `DataStructure` default and the two sides would
disagree.
"""

import inspect

import numpy as np
from cottax.interfaces.pytree_namespace_module import resolve, to_graph
from cottax.spec import VarPath

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.indat import PLASMA_COMPOSITION
from functional_process.models.physics.composition import (
    CalculateEffectiveChargeIonisationProfiles,
    PlasmaCompositionNonIgnited,
    calculate_effective_charge_ionisation_profiles,
    plasma_composition,
)
from functional_process.paths import impurity_radiation
from process.core.model import DataStructure
from process.data_structure.physics_variables import PlasmaIgnitionModel
from process.models.physics import impurity_radiation as impurity
from process.models.physics.physics import Physics
from process.models.physics.profiles import NeProfile, TeProfile


def _physics(data):
    """A `Physics` instance carrying only `.data` -- every sub-model is `None`."""
    model = Physics(
        plasma_profile=None,
        current_drive=None,
        plasma_beta=None,
        plasma_inductance=None,
        plasma_density_limit=None,
        plasma_exhaust=None,
        plasma_bootstrap_current=None,
        plasma_confinement=None,
        plasma_transition=None,
        plasma_current=None,
        plasma_fields=None,
        plasma_dia_current=None,
        plasma_geometry=None,
        scrape_off_layer=None,
    )
    model.data = data
    return model


def _reference_plasma_composition(
    nd_plasma_electrons_vol_avg,
    f_nd_alpha_thermal_electron,
    fusden_alpha_total,
    f_nd_protium_electrons,
    proton_rate_density,
    f_nd_beam_electron,
    f_nd_impurity_electron_array,
    temp_plasma_electron_vol_avg_kev,
    temp_impurity_keV_array,
    impurity_arr_zav,
    f_plasma_fuel_deuterium,
    f_plasma_fuel_tritium,
    f_plasma_fuel_helium3,
    f_temp_plasma_electron_density_vol_avg,
    f_beam_tritium,
    m_impurity_amu_array,
    i_plasma_ignited,
):
    """Call PROCESS's `Physics.plasma_composition` through the port's signature.

    `.physics.first_call` is deliberately forced to `0` here regardless of what a
    `DataStructure()` default would give -- it must select PROCESS's real-profile `pc`
    branch (not the parabolic-estimate bootstrap) to match what the port always
    computes now. See `plasma_composition`'s own docstring for why.
    """
    data = DataStructure()
    impurity.initialise_imprad(data)

    data.physics.nd_plasma_electrons_vol_avg = nd_plasma_electrons_vol_avg
    data.physics.f_nd_alpha_thermal_electron = f_nd_alpha_thermal_electron
    data.physics.fusden_alpha_total = fusden_alpha_total
    data.physics.f_nd_protium_electrons = f_nd_protium_electrons
    data.physics.proton_rate_density = proton_rate_density
    data.physics.f_nd_beam_electron = f_nd_beam_electron
    data.impurity_radiation.f_nd_impurity_electron_array = np.array(
        f_nd_impurity_electron_array, dtype=float
    )
    data.physics.temp_plasma_electron_vol_avg_kev = temp_plasma_electron_vol_avg_kev
    data.impurity_radiation.temp_impurity_keV_array = np.asarray(
        temp_impurity_keV_array, dtype=float
    )
    data.impurity_radiation.impurity_arr_zav = np.asarray(impurity_arr_zav, dtype=float)
    data.physics.f_plasma_fuel_deuterium = f_plasma_fuel_deuterium
    data.physics.f_plasma_fuel_tritium = f_plasma_fuel_tritium
    data.physics.f_plasma_fuel_helium3 = f_plasma_fuel_helium3
    data.physics.first_call = 0  # forces PROCESS's real-profile `pc` branch, see above
    data.physics.f_temp_plasma_electron_density_vol_avg = (
        f_temp_plasma_electron_density_vol_avg
    )
    data.current_drive.f_beam_tritium = f_beam_tritium
    data.impurity_radiation.m_impurity_amu_array = np.asarray(
        m_impurity_amu_array, dtype=float
    )
    data.physics.i_plasma_ignited = PlasmaIgnitionModel(i_plasma_ignited)

    physics = _physics(data)
    physics.plasma_composition()

    return (
        data.physics.nd_plasma_alphas_thermal_vol_avg,
        data.physics.nd_plasma_protons_vol_avg,
        data.physics.nd_beam_ions,
        data.physics.nd_plasma_fuel_ions_vol_avg,
        data.impurity_radiation.f_nd_impurity_electron_array,
        data.physics.nd_plasma_impurities_vol_avg,
        data.physics.nd_plasma_ions_total_vol_avg,
        data.physics.f_nd_plasma_carbon_electron,
        data.physics.f_nd_plasma_oxygen_electron,
        data.physics.f_nd_plasma_iron_argon_electron,
        data.physics.n_charge_plasma_effective_vol_avg,
        data.physics.f_alpha_electron,
        data.physics.f_alpha_ion,
        data.physics.m_fuel_amu,
        data.physics.m_beam_amu,
        data.physics.m_ions_total_amu,
        data.physics.n_charge_plasma_effective_mass_weighted_vol_avg,
    )


class TestPlasmaComposition(Tier1Contract):
    """`Physics.plasma_composition` -> `plasma_composition`."""

    audit_record = "models/physics/composition.md"
    reference = _reference_plasma_composition
    ported = plasma_composition
    # `temp_impurity_keV_array`/`impurity_arr_zav` are the same 14x200 compile-time
    # tables `test_radiation_power.py`/`test_impurity_radiation.py` already declare
    # static for the identical reason (tabulated atomic-physics constants, not values a
    # solver perturbs -- differentiating componentwise over 2800 entries each would also
    # make this contract's gradient check impractically slow). `m_impurity_amu_array` is
    # the same kind of constant. `f_nd_impurity_electron_array` is deliberately *not*
    # static -- entries 2/3 are iteration variables 125/126 elsewhere in the graph (see
    # `test_radiation_power.py::TestImpurityRadiationTotals`), so its gradient matters.
    static_argnames = (
        "i_plasma_ignited",
        "temp_impurity_keV_array",
        "impurity_arr_zav",
        "m_impurity_amu_array",
    )

    # Both points are lifted verbatim from
    # tests/unit/models/physics/test_physics.py::test_plasma_composition
    # (generated from large_tokamak_nof.IN.DAT) -- the first exercises
    # `fusden_alpha_total == 0` ("not yet calculated" branch), the second the opposite.
    # Both originally also exercised `first_call`'s two branches in PROCESS itself; the
    # port no longer has a `first_call` branch to exercise (see `plasma_composition`'s
    # docstring), so both points now go through the reference adapter's forced
    # `first_call = 0` (real-profile `pc`) path -- the sample names are kept as-is since
    # they still identify the underlying PROCESS legacy points, not the branch tested.
    samples = [
        legacy_sample(
            "large_tokamak_nof-first_call",
            nd_plasma_electrons_vol_avg=7.5e19,
            f_nd_alpha_thermal_electron=0.10000000000000001,
            fusden_alpha_total=0.0,
            f_nd_protium_electrons=0.0,
            proton_rate_density=0.0,
            f_nd_beam_electron=0.0,
            f_nd_impurity_electron_array=np.array([
                0.90000000000000002,
                0.10000000000000001,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.00038000000000000008,
                5.0000000000000021e-06,
            ]),
            temp_plasma_electron_vol_avg_kev=12,
            temp_impurity_keV_array=None,  # filled in by the sample-list postprocess
            impurity_arr_zav=None,
            f_plasma_fuel_deuterium=0.5,
            f_plasma_fuel_tritium=0.5,
            f_plasma_fuel_helium3=0.0,
            f_temp_plasma_electron_density_vol_avg=0.0,
            f_beam_tritium=9.9999999999999995e-07,
            m_impurity_amu_array=np.array([
                1.01,
                4.0030000000000001,
                9.0099999999999998,
                12.01,
                14.01,
                15.999000000000001,
                20.18,
                28.09,
                39.950000000000003,
                55.850000000000001,
                58.700000000000003,
                83.799999999999997,
                131.30000000000001,
                183.84999999999999,
            ]),
            i_plasma_ignited=PlasmaIgnitionModel.NON_IGNITED,
        ),
        legacy_sample(
            "large_tokamak_nof-steady",
            nd_plasma_electrons_vol_avg=7.5e19,
            f_nd_alpha_thermal_electron=0.10000000000000001,
            fusden_alpha_total=1.973996644759543e17,
            f_nd_protium_electrons=0.0,
            proton_rate_density=540072280299564.38,
            f_nd_beam_electron=0.0,
            f_nd_impurity_electron_array=np.array([
                0.78128900936605694,
                0.10000000000000001,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.00038000000000000008,
                5.0000000000000021e-06,
            ]),
            temp_plasma_electron_vol_avg_kev=12,
            temp_impurity_keV_array=None,
            impurity_arr_zav=None,
            f_plasma_fuel_deuterium=0.5,
            f_plasma_fuel_tritium=0.5,
            f_plasma_fuel_helium3=0.0,
            f_temp_plasma_electron_density_vol_avg=1.0521775929921553,
            f_beam_tritium=9.9999999999999995e-07,
            m_impurity_amu_array=np.array([
                1.01,
                4.0030000000000001,
                9.0099999999999998,
                12.01,
                14.01,
                15.999000000000001,
                20.18,
                28.09,
                39.950000000000003,
                55.850000000000001,
                58.700000000000003,
                83.799999999999997,
                131.30000000000001,
                183.84999999999999,
            ]),
            i_plasma_ignited=PlasmaIgnitionModel.NON_IGNITED,
        ),
        # A hand-built ignited point (`IGNITED`) -- neither PROCESS legacy point
        # exercises this branch (`large_tokamak_nof.IN.DAT` is non-ignited). Values are
        # a scaled-down variant of the two points above, not a validated operating
        # point -- realistic magnitudes only, per `test_harness.md`'s fuzz-domain intent.
        legacy_sample(
            "ignited-hand-built",
            nd_plasma_electrons_vol_avg=7.5e19,
            f_nd_alpha_thermal_electron=0.1,
            fusden_alpha_total=1.9e17,
            f_nd_protium_electrons=0.0,
            proton_rate_density=5.4e14,
            f_nd_beam_electron=0.0,
            f_nd_impurity_electron_array=np.array([
                0.85,
                0.1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0004,
                5e-6,
            ]),
            temp_plasma_electron_vol_avg_kev=12,
            temp_impurity_keV_array=None,
            impurity_arr_zav=None,
            f_plasma_fuel_deuterium=0.5,
            f_plasma_fuel_tritium=0.5,
            f_plasma_fuel_helium3=0.0,
            f_temp_plasma_electron_density_vol_avg=1.05,
            f_beam_tritium=1e-6,
            m_impurity_amu_array=np.array([
                1.01,
                4.003,
                9.01,
                12.01,
                14.01,
                15.999,
                20.18,
                28.09,
                39.95,
                55.85,
                58.70,
                83.80,
                131.30,
                183.85,
            ]),
            i_plasma_ignited=PlasmaIgnitionModel.IGNITED,
        ),
    ]


def _fill_tables(samples):
    """Fill each sample's `temp_impurity_keV_array`/`impurity_arr_zav` with the real
    L(Z, Te)/<Z>(Te) tables, loaded once from PROCESS's shipped data files.

    Loading them is expensive enough (28 files) that it should not happen once per
    sample at class-body-evaluation time; done as a single post-process instead, the
    same reasoning `test_radiation_power.py::_impurity_tables` gives.
    """
    data = DataStructure()
    impurity.initialise_imprad(data)
    temp_table = np.array(data.impurity_radiation.temp_impurity_keV_array, dtype=float)
    zav_table = np.array(data.impurity_radiation.impurity_arr_zav, dtype=float)
    for sample in samples:
        kwargs = dict(sample.kwargs)
        kwargs["temp_impurity_keV_array"] = temp_table
        kwargs["impurity_arr_zav"] = zav_table
        object.__setattr__(sample, "kwargs", type(sample.kwargs)(kwargs))
    return samples


TestPlasmaComposition.samples = _fill_tables(TestPlasmaComposition.samples)


def _reference_calculate_effective_charge_ionisation_profiles(
    temp_electron_profile_kev,
    f_nd_impurity_electron_array,
    temp_impurity_keV_array,
    impurity_arr_zav,
):
    """Call PROCESS's `Physics.calculate_effective_charge_ionisation_profiles`."""
    data = DataStructure()
    impurity.initialise_imprad(data)
    data.impurity_radiation.f_nd_impurity_electron_array = np.asarray(
        f_nd_impurity_electron_array, dtype=float
    )
    data.impurity_radiation.temp_impurity_keV_array = np.asarray(
        temp_impurity_keV_array, dtype=float
    )
    data.impurity_radiation.impurity_arr_zav = np.asarray(impurity_arr_zav, dtype=float)

    te_profile = TeProfile()
    te_profile.data = data
    te_profile.run = lambda: None
    te_profile.profile_y = np.asarray(temp_electron_profile_kev, dtype=float)

    ne_profile = NeProfile()
    ne_profile.data = data
    ne_profile.run = lambda: None

    physics = _physics(data)
    physics.plasma_profile = type(
        "StubPlasmaProfile", (), {"teprofile": te_profile, "neprofile": ne_profile}
    )()
    physics.calculate_effective_charge_ionisation_profiles()

    return (
        data.physics.n_charge_plasma_effective_profile,
        data.impurity_radiation.n_charge_impurity_profile,
    )


_N_POINTS = 11
_TE_PROFILE = 15.0 * (1.0 - 0.85 * np.linspace(0.0, 1.0, _N_POINTS) ** 2) + 0.5


class TestCalculateEffectiveChargeIonisationProfiles(Tier1Contract):
    """`Physics.calculate_effective_charge_ionisation_profiles` ->
    `calculate_effective_charge_ionisation_profiles`.
    """

    audit_record = "models/physics/composition.md"
    reference = _reference_calculate_effective_charge_ionisation_profiles
    ported = calculate_effective_charge_ionisation_profiles
    static_argnames = ("temp_impurity_keV_array", "impurity_arr_zav")

    samples = _fill_tables([
        legacy_sample(
            "large_tokamak_nof-profile",
            temp_electron_profile_kev=_TE_PROFILE,
            f_nd_impurity_electron_array=np.array([
                0.78128900936605694,
                0.10000000000000001,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.00038000000000000008,
                5.0000000000000021e-06,
            ]),
            temp_impurity_keV_array=None,
            impurity_arr_zav=None,
        ),
    ])


# ---------------------------------------------------------------- the Shape B split
#
# `.impurity_radiation.f_nd_impurity_electron_array` is both read (indices 2:13) and
# written (indices 0/1) by `plasma_composition` -- a self-loop shape that would raise
# `cottax.spec`'s "reads what it also owns" construction error if the array were
# addressed as one `VarPath` (see `_audit/next_steps.md` §5, "Shape B"). PROCESS's
# other apparent self-loop in this unit, `.physics.first_call`, was never a genuine
# cycle at all -- see `plasma_composition`'s own docstring -- so it is not ported and
# there is no `FixedPointFunction`/`Cut` here for it.
# `.physics.first_call` above, flagged but *not* resolved by an earlier pass (see
# `composition.md`'s "cottax node" section). It is resolved here, not via
# `FixedPointFunction` like `first_call`, but by addressing the field at index
# granularity: the read range (2:13) and the write range (0/1) are disjoint, so once
# each index is its own `VarPath` (`SequenceKey`-addressed, matching the real
# `DataStructure` field's own `list[float]` storage) there is no overlap left to
# conflict. `PlasmaComposition` now owns indices 0/1 outright.


def _pure_function_kwargs_to_node_kwargs(kwargs):
    """Pack/unpack adapter: `plasma_composition`'s one-array-argument kwargs ->
    `PlasmaComposition.__call__`'s fourteen-individual-index kwargs (minus
    `i_plasma_ignited`, which is the node's static field, not a port).

    Same idiom `coils/calculate.py`'s minted `coilcurrent` adapters use elsewhere in this
    codebase: a small, explicit translation between the pure function's array-shaped
    signature and the node's per-index `VarPath`s, not a new pattern.
    """
    node_kwargs = {k: v for k, v in kwargs.items() if k != "i_plasma_ignited"}
    array = node_kwargs.pop("f_nd_impurity_electron_array")
    for i in range(2, 14):
        node_kwargs[f"f_nd_impurity_electron_array_{i}"] = array[i]
    return node_kwargs


def test_plasma_composition_owns_h_and_he_fractions():
    """The actual point of this split: `PlasmaComposition` now *owns*
    `.impurity_radiation.f_nd_impurity_electron_array[0]`/`[1]` (the `H_`/`He`
    fractions it computes), and reads only indices 2-13 -- never the whole array, and
    never indices 0/1 as a read. `to_graph` succeeding at all is only half the proof;
    checking exactly which `VarPath`s are owned/read is the other half (a node could
    assemble while silently reading or owning the wrong indices).
    """
    node = PlasmaCompositionNonIgnited()
    graph = to_graph(node)
    assert graph.definitions

    owned = {out.var for out in node.outputs}
    read = {inp.var for inp in node.inputs}

    h_path = resolve(impurity_radiation.f_nd_impurity_electron_array[0], VarPath)
    he_path = resolve(impurity_radiation.f_nd_impurity_electron_array[1], VarPath)
    assert h_path in owned
    assert he_path in owned
    assert h_path not in read
    assert he_path not in read

    for i in range(2, 14):
        idx_path = resolve(
            lambda s, i=i: s.impurity_radiation.f_nd_impurity_electron_array[i],
            VarPath,
        )
        assert idx_path in read
        assert idx_path not in owned


def test_calculate_effective_charge_ionisation_profiles_assembles_alone():
    """`CalculateEffectiveChargeIonisationProfiles` reads all fourteen indices as
    individual `FromExactly`s (no whole-array read left anywhere in this file) and must still
    assemble on its own.
    """
    graph = to_graph(CalculateEffectiveChargeIonisationProfiles())
    assert graph.definitions


def test_calculate_effective_charge_ionisation_profiles_depends_on_plasma_composition():
    """Combining the two nodes in one graph is now a real dependency, not a coincidence
    of two boundary inputs: `CalculateEffectiveChargeIonisationProfiles` reads indices
    0/1 of `f_nd_impurity_electron_array`, and `PlasmaComposition` owns exactly those
    two -- so the two nodes together have one fewer unowned boundary input than either
    has alone (12 fewer than the naive count, since indices 0/1 move from "both
    boundary" to "one boundary, one produced").
    """
    plasma_composition_node = PlasmaCompositionNonIgnited()
    profiles_node = CalculateEffectiveChargeIonisationProfiles()
    graph = to_graph(plasma_composition_node, profiles_node)
    assert graph.definitions
    assert len(graph.definitions) == 2

    h_path = resolve(impurity_radiation.f_nd_impurity_electron_array[0], VarPath)
    he_path = resolve(impurity_radiation.f_nd_impurity_electron_array[1], VarPath)
    boundary_inputs = set(graph.unowned_inputs)
    assert h_path not in boundary_inputs
    assert he_path not in boundary_inputs


def test_plasma_composition_node_matches_pure_function():
    """The node's per-index reassembly must be numerically exact against the pure
    function's own array-shaped call, for both `PlasmaComposition`'s ordinary outputs
    and the two indices it now owns -- not merely "assembles without raising."

    Uses the same legacy sample points `TestPlasmaComposition` already validates against
    PROCESS, so this is a second, independent check on top of the value/gradient
    contract: the node wiring around an already-verified pure function.
    """
    for sample in TestPlasmaComposition.samples:
        kwargs = dict(sample.kwargs)
        i_plasma_ignited = kwargs["i_plasma_ignited"]

        want = plasma_composition(
            **{k: v for k, v in kwargs.items() if k != "i_plasma_ignited"},
            i_plasma_ignited=i_plasma_ignited,
        )

        node = PLASMA_COMPOSITION[PlasmaIgnitionModel(int(i_plasma_ignited))]()
        node_kwargs = _pure_function_kwargs_to_node_kwargs(kwargs)
        # The ignited occupant does not declare `.physics.f_nd_beam_electron` at all --
        # that is the read the split removed -- so the shared kwarg dict is filtered by
        # the occupant's own signature rather than by a hardcoded name list.
        declared = inspect.signature(type(node).__call__).parameters
        got = node(**{k: v for k, v in node_kwargs.items() if k in declared})

        # `want` is the pure function's 17-tuple in its own order: results[:4],
        # results[4] (the full post-update array -- index into it for H_/He_),
        # results[5:] (12 more). Splitting index 4 into two elements makes `expected`
        # 18 long, matching the node's 18 `Output`s.
        expected = (
            *want[:4],
            want[4][0],
            want[4][1],
            *want[5:],
        )
        assert len(got) == len(expected) == 18
        for g, e in zip(got, expected, strict=True):
            np.testing.assert_allclose(np.asarray(g), np.asarray(e), rtol=0, atol=0)
