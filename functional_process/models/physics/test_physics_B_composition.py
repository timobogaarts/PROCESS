"""Harness cases for `physics_B_composition.py`.

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

import numpy as np

from cottax.interfaces.pytree_namespace_module import path_of, to_graph
from cottax.spec import VarPath
from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.physics.physics_B_composition import (
    NextFirstCall,
    PlasmaComposition,
    calculate_effective_charge_ionisation_profiles,
    next_first_call,
    plasma_composition,
)
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
    first_call,
    alphan,
    alphat,
    f_temp_plasma_electron_density_vol_avg,
    f_beam_tritium,
    m_impurity_amu_array,
    is_ignited,
):
    """Call PROCESS's `Physics.plasma_composition` through the port's signature."""
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
    data.physics.first_call = first_call
    data.physics.alphan = alphan
    data.physics.alphat = alphat
    data.physics.f_temp_plasma_electron_density_vol_avg = (
        f_temp_plasma_electron_density_vol_avg
    )
    data.current_drive.f_beam_tritium = f_beam_tritium
    data.impurity_radiation.m_impurity_amu_array = np.asarray(
        m_impurity_amu_array, dtype=float
    )
    data.physics.i_plasma_ignited = (
        PlasmaIgnitionModel.IGNITED if is_ignited else PlasmaIgnitionModel.NON_IGNITED
    )

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
        data.physics.first_call,
        data.physics.f_alpha_electron,
        data.physics.f_alpha_ion,
        data.physics.m_fuel_amu,
        data.physics.m_beam_amu,
        data.physics.m_ions_total_amu,
        data.physics.n_charge_plasma_effective_mass_weighted_vol_avg,
    )


class TestPlasmaComposition(Tier1Contract):
    """`Physics.plasma_composition` -> `plasma_composition`."""

    audit_record = "models/physics/physics_B_composition.md"
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
    # `first_call` is also static: it is a discrete `{0, 1}` bookkeeping flag, not a
    # continuous physical quantity, and `jnp.where(first_call == 1, ...)` is a step
    # function evaluated *exactly at* its own jump for the `first_call-1` sample --
    # PROCESS's own finite difference there estimates the slope of the *neighbouring*
    # branch (the `first_call != 1` pass-through), not this branch's true (zero)
    # analytic derivative, so the two are answering different questions at that point.
    # No sample value moves it off the boundary the way `fast_alpha_beta`'s
    # `low-deuterium-no-alphas` sample does for its own continuous threshold, because
    # `first_call` has no off-boundary value to hold it at -- 0 and 1 are its only
    # meaningful values.
    static_argnames = (
        "is_ignited",
        "first_call",
        "temp_impurity_keV_array",
        "impurity_arr_zav",
        "m_impurity_amu_array",
    )

    # Both points are lifted verbatim from
    # tests/unit/models/physics/test_physics.py::test_plasma_composition
    # (generated from large_tokamak_nof.IN.DAT) -- the first exercises `first_call == 1`
    # and `fusden_alpha_total == 0` (both "not yet calculated" branches at once), the
    # second exercises the opposite of both.
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
            first_call=1,
            alphan=1,
            alphat=1.45,
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
            is_ignited=False,
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
            first_call=0,
            alphan=1,
            alphat=1.45,
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
            is_ignited=False,
        ),
        # A hand-built ignited point (`is_ignited=True`) -- neither PROCESS legacy point
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
            first_call=0,
            alphan=1,
            alphat=1.45,
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
            is_ignited=True,
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

    audit_record = "models/physics/physics_B_composition.md"
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
# `plasma_composition`'s `.physics.first_call` self-loop cannot be a plain node --
# `to_graph(Avail(...))` proved that shape raises `ValueError: reads [...], which it
# also owns` directly from `cottax.spec`'s `__check_init__` (see
# `_audit/next_steps.md` §5, "Shape B"). `NextFirstCall`/`PlasmaComposition` split that
# self-loop out via `FixedPointFunction`. These checks are the actual point of the
# split: prove the shape is now legal, don't just assert it.


def test_next_first_call_matches_the_shared_helper():
    """`NextFirstCall.step` and `plasma_composition`'s own `first_call_next` return
    value both call `next_first_call` -- not two independent reimplementations of the
    same formula. This just confirms the node's `step` body actually is that call.
    """
    for first_call in (0, 1, 0.0, 1.0):
        node = NextFirstCall()
        got = node.step(first_call)
        want = next_first_call(first_call)
        assert got == want


def test_next_first_call_assembles_as_a_fixed_point_node():
    """The actual point of this split: `to_graph(NextFirstCall())` must succeed.

    Before this split, `plasma_composition`'s `.physics.first_call` self-loop had no
    node written for it at all, precisely because it was known in advance (from
    `Avail`'s precedent) that declaring it as a plain node would hit `cottax.spec`'s
    "reads what it also owns" construction error. `FixedPointFunction` mints the cut
    internally (`^cond.physics.first_call`), so the body node reads the real
    `.physics.first_call` and the separate `FixedPoint` problem node owns it --
    `to_graph` must build both without raising.
    """
    graph = to_graph(NextFirstCall())
    assert graph.definitions
    # Two nodes: the `step` body and the `FixedPoint` problem it feeds -- exactly the
    # pair `FixedPointFunction.node_definitions_and_names` documents.
    assert len(graph.definitions) == 2


def test_plasma_composition_node_assembles_and_does_not_own_first_call():
    """`PlasmaComposition` (the ordinary node) must also assemble on its own, and must
    not itself own `.physics.first_call` -- that `VarPath` belongs to `NextFirstCall`'s
    `FixedPoint` problem node alone, not this one.
    """
    node = PlasmaComposition(is_ignited=False)
    graph = to_graph(node)
    assert graph.definitions
    owned = {out.var for out in node.outputs}
    # Built the same way the node itself resolves an `Output`'s `where`, rather than
    # hand-rolling the internal key representation.
    first_call_path = path_of(lambda s: s.physics.first_call, VarPath)
    assert first_call_path not in owned


def test_next_first_call_and_plasma_composition_assemble_together():
    """The two halves of the split coexist in one graph with no naming collision:
    `NextFirstCall`'s `FixedPoint` problem owns `.physics.first_call`;
    `PlasmaComposition` only reads it. This is the shape a later consolidation pass
    (not this one) would register into `total_process.py`.
    """
    graph = to_graph(NextFirstCall(), PlasmaComposition(is_ignited=False))
    assert graph.definitions
    assert len(graph.definitions) == 3  # NextFirstCall's 2 + PlasmaComposition's 1
