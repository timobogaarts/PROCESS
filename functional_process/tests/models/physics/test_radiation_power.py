"""Harness cases for the ported radiation-power arithmetic (registry unit #20).

Every reference adapter here exists to close one of the two back doors in
`radiation_power.calculate_radiation_powers(plasma_profile, ..., data_structure)`. The
adapters build a **real** `DataStructure` and a **real** `PlasmaProfile` with real
`NeProfile`/`TeProfile` sub-models, populate exactly the fields the audit record says are
read, and call the **real** PROCESS functions. Nothing is transcribed.

That construction is the test. If the audit's read set were incomplete -- if
`ImpurityRadiation` reached for some `data` field the record does not list -- these
adapters would leave it at its zero default and disagree loudly with the port, which is
fed only the listed values. `TestRadiationPowers` makes that argument end to end, against
PROCESS's own entry point with both object arguments intact, and
`TestCombineRadiationPowers` is the only thing it cannot isolate.

The two sub-model `run()` methods are stubbed to no-ops and `profile_y` set directly,
exactly as `test_plasma_profiles.py` does and for the same reason: they belong to
`process/models/physics/profiles.py`, registry unit #21, which is unported, and they
would otherwise overwrite the profile arrays a case needs to control.
"""

import numpy as np
from cottax.interfaces.pytree_namespace_module import resolve, to_graph
from cottax.spec import VarPath

from functional_process.cottax._harness import Tier1Contract, legacy_sample
from functional_process.cottax.physics.radiation_power import (
    ImpurityRadiationTotals,
    PlasmaRadiationPowers,
    SynchrotronRadiationPower,
    calculate_impurity_radiation_power_density,
    calculate_impurity_radiation_totals,
    calculate_radiation_powers,
    psync_albajar_fidone,
)
from functional_process.cottax.paths import impurity_radiation
from process.core.model import DataStructure
from process.models.physics import impurity_radiation as impurity
from process.models.physics import radiation_power as reference_module
from process.models.physics.plasma_profiles import PlasmaProfile
from process.models.physics.profiles import NeProfile, TeProfile

# 11 points -> 10 intervals (even), so `scipy.integrate.simpson` uses the plain composite
# rule and the `_simpson` this port borrows from `plasma_profiles.py` matches it.
# PROCESS's own default is 201; 11 keeps `jacfwd` over the three profile arrays cheap
# without changing which rule applies.
_N_POINTS = 11
_RHO = np.linspace(0.0, 1.0, _N_POINTS)

_NE = 8.0e19 * (1.0 - 0.8 * _RHO**2)
_TE = 20.0 * (1.0 - 0.9 * _RHO**2) + 0.5
"""A profile pair whose temperatures stay inside the 0.001-40 keV table range.

Deliberate: outside it, `calculate_impurity_radiation_power_density`'s clamps are both
discontinuous *and* wrong (see the port's docstring), so a finite-difference comparison
there would be measuring the edge of a bug rather than the port.
"""

_CORE_RADIUS = 0.65
"""`radius_plasma_core_norm`, kept **off** the `_RHO` grid on purpose.

`create_f_rad_core_profile` is a step function of both this and each `profile_x[i]`.
PROCESS's default is 0.6, which with `_RHO` lands exactly on a grid point -- PROCESS's
own 1e-3 *relative* perturbation of either side would then flip the mask and the finite
difference would report a spurious 1e3-sized derivative for a quantity whose true
derivative is zero. 0.65 is the same physics with the discontinuity away from every
sample, and it is why `radius_plasma_core_norm` is held fixed under fuzzing rather than
drawn.
"""


def _impurity_tables():
    """The real L(Z, Te) tables, loaded from PROCESS's shipped data files once.

    `initialise_imprad` reads 28 files; doing that per reference evaluation would
    dominate a gradient run. The arrays it produces are compile-time constants of the
    graph (see the port's module docstring), so caching them is not a shortcut.
    """
    data = DataStructure()
    impurity.initialise_imprad(data)
    return (
        np.array(data.impurity_radiation.temp_impurity_keV_array, dtype=float),
        np.array(data.impurity_radiation.pden_impurity_lz_nd_temp_array, dtype=float),
    )


_TEMP_TABLE, _LZ_TABLE = _impurity_tables()
_N_TABLE = _TEMP_TABLE.shape[1]

_HYDROGEN, _ARGON = 0, 8
_SPECIES_TEMP_TABLE = _TEMP_TABLE[[_HYDROGEN, _ARGON]]
_SPECIES_LZ_TABLE = _LZ_TABLE[[_HYDROGEN, _ARGON]]
_SPECIES_FRACTIONS = np.array([1.0, 1.8e-3])
"""Hydrogen at the fraction `initialise_imprad` gives it, plus a seeded argon.

Both terms matter at these temperatures (`L_H ~ 3e-36`, `f_Ar L_Ar ~ 3e-36` W m^3), so a
port that dropped a species, or double-counted one, shows up rather than being masked by
a single dominant term.
"""


def _data_with_tables(temp_impurity_kev_array, pden_impurity_lz_nd_temp_array, **fields):
    """A `DataStructure` carrying the given L(Z, Te) tables at rows `0..k-1`.

    The selected species are placed at the *front* of the 14-row arrays so that
    `ImpurityRadiation.__init__`'s `np.nonzero(f_nd > 1e-30)[0]` selects exactly them, in
    the same order as the port's stacked arguments. PROCESS's arithmetic depends on a
    species only through its two table rows and its fraction, never on its index, so this
    is a relabelling and not a change of case.
    """
    data = DataStructure()
    n_species, n_table = np.shape(temp_impurity_kev_array)

    full_temp = np.zeros((14, n_table))
    full_lz = np.zeros((14, n_table))
    full_temp[:n_species] = temp_impurity_kev_array
    full_lz[:n_species] = pden_impurity_lz_nd_temp_array

    data.impurity_radiation.temp_impurity_keV_array = full_temp
    data.impurity_radiation.pden_impurity_lz_nd_temp_array = full_lz
    data.impurity_radiation.impurity_arr_len_tab = np.full(14, n_table)
    data.physics.n_plasma_profile_elements = _N_POINTS

    for name, value in fields.items():
        setattr(data.impurity_radiation, name, value)
    return data


def _plasma_profile(data, profile_x, nd_electron_profile, temp_electron_profile_kev):
    """A real `PlasmaProfile` carrying the three arrays this unit reads off it.

    Those three -- `neprofile.profile_x`, `neprofile.profile_y`, `teprofile.profile_y` --
    are the entire `plasma_profile` read set of `calculate_radiation_powers`. Nothing else
    is set on either sub-model except `profile_dx`, which is included precisely to show it
    makes no difference: `scipy.integrate.simpson` ignores `dx` whenever `x` is given, so
    it is set to a deliberately wrong value here and every case still agrees to
    `rtol=1e-12`. That is the evidence for leaving it out of the port's signature.
    """
    ne_profile, te_profile = NeProfile(), TeProfile()
    for obj in (ne_profile, te_profile):
        obj.data = data
        obj.run = lambda: None
        obj.profile_x = np.asarray(profile_x, dtype=float)
        # Wrong on purpose -- see the docstring.
        obj.profile_dx = 1.0

    ne_profile.profile_y = np.asarray(nd_electron_profile, dtype=float)
    te_profile.profile_y = np.asarray(temp_electron_profile_kev, dtype=float)

    model = PlasmaProfile(ne_profile, te_profile)
    model.data = data
    return model


def _fractions_at_front(f_nd_impurity_electron_array):
    """The 14-entry `f_nd_impurity_electron_array` with the selected species first."""
    fractions = np.zeros(14)
    fractions[: len(f_nd_impurity_electron_array)] = f_nd_impurity_electron_array
    return fractions


def _reference_impurity_radiation_power_density(
    nd_electron_profile,
    temp_electron_profile_kev,
    f_nd_impurity_electron,
    temp_impurity_kev,
    pden_impurity_lz_nd_temp,
):
    """`impurity_radiation.calculate_impurity_radiation_power_density`, one species."""
    data = _data_with_tables(
        np.atleast_2d(temp_impurity_kev),
        np.atleast_2d(pden_impurity_lz_nd_temp),
        f_nd_impurity_electron_array=_fractions_at_front([f_nd_impurity_electron]),
    )
    return impurity.calculate_impurity_radiation_power_density(
        imp_element_index=0,
        nd_electron_profile=np.asarray(nd_electron_profile, dtype=float),
        temp_electron_profile_kev=np.asarray(temp_electron_profile_kev, dtype=float),
        data=data,
    )


def _reference_impurity_radiation_totals(
    profile_x,
    nd_electron_profile,
    temp_electron_profile_kev,
    f_nd_impurity_electron_array,
    temp_impurity_kev_array,
    pden_impurity_lz_nd_temp_array,
    radius_plasma_core_norm,
    f_p_plasma_core_rad_reduction,
):
    """`ImpurityRadiation(...).calculate_imprad()`, both totals read off the instance.

    Those two results have no `data` storage in PROCESS -- they live on the object
    `calculate_radiation_powers` constructs and throws away, which is why the port mints
    `VarPath`s for them.
    """
    data = _data_with_tables(
        temp_impurity_kev_array,
        pden_impurity_lz_nd_temp_array,
        f_nd_impurity_electron_array=_fractions_at_front(f_nd_impurity_electron_array),
        radius_plasma_core_norm=radius_plasma_core_norm,
        f_p_plasma_core_rad_reduction=f_p_plasma_core_rad_reduction,
    )
    plasma_profile = _plasma_profile(
        data, profile_x, nd_electron_profile, temp_electron_profile_kev
    )

    model = impurity.ImpurityRadiation(plasma_profile, data)
    model.calculate_imprad()
    return model.pden_impurity_rad_total_mw, model.pden_impurity_core_rad_total_mw


def _reference_radiation_powers(
    profile_x,
    nd_electron_profile,
    temp_electron_profile_kev,
    f_nd_impurity_electron_array,
    temp_impurity_kev_array,
    pden_impurity_lz_nd_temp_array,
    radius_plasma_core_norm,
    f_p_plasma_core_rad_reduction,
    nd_plasma_electron_on_axis,
    rminor,
    b_plasma_toroidal_on_axis,
    aspect,
    alphan,
    alphat,
    tbeta,
    temp_plasma_electron_on_axis_kev,
    f_sync_reflect,
    rmajor,
    kappa,
    vol_plasma,
):
    """PROCESS's entry point, both object arguments intact, `RadpwrData` as a tuple.

    This is the case that proves the audit's read set. PROCESS is handed a
    `PlasmaProfile` and a whole `DataStructure`; the port is handed the eight values the
    record says are read off them. Anything else being read would break the agreement.
    """
    data = _data_with_tables(
        temp_impurity_kev_array,
        pden_impurity_lz_nd_temp_array,
        f_nd_impurity_electron_array=_fractions_at_front(f_nd_impurity_electron_array),
        radius_plasma_core_norm=radius_plasma_core_norm,
        f_p_plasma_core_rad_reduction=f_p_plasma_core_rad_reduction,
    )
    plasma_profile = _plasma_profile(
        data, profile_x, nd_electron_profile, temp_electron_profile_kev
    )

    result = reference_module.calculate_radiation_powers(
        plasma_profile,
        nd_plasma_electron_on_axis,
        rminor,
        b_plasma_toroidal_on_axis,
        aspect,
        alphan,
        alphat,
        tbeta,
        temp_plasma_electron_on_axis_kev,
        f_sync_reflect,
        rmajor,
        kappa,
        vol_plasma,
        data,
    )
    return (
        result.pden_plasma_sync_mw,
        result.pden_plasma_core_rad_mw,
        result.pden_plasma_outer_rad_mw,
        result.pden_plasma_rad_mw,
    )


class TestSynchrotronRadiationPower(Tier1Contract):
    """`psync_albajar_fidone` -> the same, unchanged.

    The one function in the unit that was already pure and already explicit, so the
    reference is PROCESS's function itself with no adapter at all. Tested separately from
    the composite because the impurity term there is the same order of magnitude and
    would hide a moderate error in this one.
    """

    audit_record = "models/physics/radiation_power.md"
    reference = reference_module.psync_albajar_fidone
    ported = psync_albajar_fidone

    samples = [
        legacy_sample(
            # Large-tokamak-shaped point; `pden_plasma_sync_mw` in
            # `tests/unit/data/large_tokamak_MFILE.DAT` is 7.6e-3 MW/m^3, the order this
            # lands on.
            "large-tokamak-shaped",
            nd_plasma_electron_on_axis=1.6e20,
            rminor=2.9,
            b_plasma_toroidal_on_axis=5.7,
            aspect=3.1,
            alphan=1.0,
            alphat=1.45,
            tbeta=2.0,
            temp_plasma_electron_on_axis_kev=29.4,
            f_sync_reflect=0.6,
            rmajor=9.0,
            kappa=1.85,
            vol_plasma=2500.0,
        ),
        legacy_sample(
            # Compact, high field, low reflectivity: `dum` and the two
            # `(1 - f_sync_reflect)` powers are all far from the point above.
            "compact-high-field",
            nd_plasma_electron_on_axis=4.0e20,
            rminor=0.6,
            b_plasma_toroidal_on_axis=12.0,
            aspect=3.4,
            alphan=0.6,
            alphat=2.0,
            tbeta=2.0,
            temp_plasma_electron_on_axis_kev=15.0,
            f_sync_reflect=0.1,
            rmajor=2.0,
            kappa=1.6,
            vol_plasma=20.0,
        ),
        legacy_sample(
            # Stellarator-shaped: high aspect ratio, so `g_function`'s
            # `exp(-0.82 aspect)` term is essentially off.
            "high-aspect-stellarator",
            nd_plasma_electron_on_axis=1.6e20,
            rminor=1.8,
            b_plasma_toroidal_on_axis=5.0,
            aspect=10.0,
            alphan=1.0,
            alphat=1.45,
            tbeta=2.0,
            temp_plasma_electron_on_axis_kev=29.4,
            f_sync_reflect=0.6,
            rmajor=18.0,
            kappa=1.0,
            vol_plasma=900.0,
        ),
    ]

    # PROCESS's own iteration-variable bounds where they exist (`b_plasma_toroidal_on_axis`
    # is 2, `rmajor` 3, `alphat` 5, `alphan` 6, `kappa` 70, `f_sync_reflect` 108);
    # plausible operating ranges otherwise. `tbeta` is kept at or above 1 so that
    # `tbeta**1.53 + 1.87 alphat - 0.16` stays positive -- `k_function` raises that to a
    # negative power and is non-finite below, which PROCESS does not guard either.
    fuzz_bounds = {
        "nd_plasma_electron_on_axis": (5.0e19, 5.0e20),
        "rminor": (0.5, 4.0),
        "b_plasma_toroidal_on_axis": (2.0, 15.0),
        "aspect": (1.5, 12.0),
        "alphan": (0.1, 1.0),
        "alphat": (0.5, 2.5),
        "tbeta": (1.0, 3.0),
        "temp_plasma_electron_on_axis_kev": (5.0, 60.0),
        "f_sync_reflect": (0.05, 0.9),
        "rmajor": (1.5, 20.0),
        "kappa": (1.0, 2.5),
        "vol_plasma": (10.0, 3000.0),
    }


class TestImpurityRadiationPowerDensity(Tier1Contract):
    """`impurity_radiation.calculate_impurity_radiation_power_density`, one species.

    The two 200-entry table arguments are `static_argnames`. They are not switches; they
    are tabulated atomic-physics constants read from
    `process/data/lz_non_corona_14_elements/*.dat` at startup, never written by any model
    and never an iteration variable, so a derivative with respect to one of them is not a
    quantity any solver can consume. Differentiating them would cost 1600 reference
    evaluations per sample to check nothing. The interpolation's own derivative is still
    checked, through `temp_electron_profile_kev`.
    """

    audit_record = "models/physics/radiation_power.md"
    reference = _reference_impurity_radiation_power_density
    ported = calculate_impurity_radiation_power_density
    static_argnames = ("temp_impurity_kev", "pden_impurity_lz_nd_temp")

    samples = [
        legacy_sample(
            # `tests/unit/models/physics/test_impurity_radiation.py::test_pimpden`'s
            # point verbatim, on hydrogen (the species that test uses, index 0).
            "test_pimpden-hydrogen",
            nd_electron_profile=np.array([
                9.42593370e19,
                9.37237672e19,
                9.21170577e19,
                8.94392086e19,
                8.56902197e19,
                8.08700913e19,
                7.49788231e19,
                6.80164153e19,
                5.99828678e19,
                3.28986749e19,
            ]),
            temp_electron_profile_kev=np.array([
                27.73451868,
                27.25167194,
                25.82164396,
                23.50149071,
                20.39190536,
                16.64794796,
                12.50116941,
                8.31182764,
                4.74643357,
                0.1,
            ]),
            f_nd_impurity_electron=1.0,
            temp_impurity_kev=_TEMP_TABLE[_HYDROGEN],
            pden_impurity_lz_nd_temp=_LZ_TABLE[_HYDROGEN],
        ),
        legacy_sample(
            # Argon, whose L(Z, Te) varies by four decades across the profile -- the
            # log-log interpolation is doing real work here, unlike for hydrogen.
            "argon-seeded",
            nd_electron_profile=_NE,
            temp_electron_profile_kev=_TE,
            f_nd_impurity_electron=1.8e-3,
            temp_impurity_kev=_TEMP_TABLE[_ARGON],
            pden_impurity_lz_nd_temp=_LZ_TABLE[_ARGON],
        ),
    ]

    # Temperatures are drawn strictly inside the table's 0.001-40 keV span so no draw
    # lands on a clamp, where the reference is discontinuous.
    fuzz_bounds = {
        "nd_electron_profile": (
            np.full(_N_POINTS, 1.0e19),
            np.full(_N_POINTS, 3.0e20),
        ),
        "temp_electron_profile_kev": (
            np.full(_N_POINTS, 0.05),
            np.full(_N_POINTS, 35.0),
        ),
        "f_nd_impurity_electron": (1.0e-4, 1.0),
    }
    fuzz_fixed = {
        "temp_impurity_kev": _TEMP_TABLE[_ARGON],
        "pden_impurity_lz_nd_temp": _LZ_TABLE[_ARGON],
    }


class TestImpurityRadiationTotals(Tier1Contract):
    """`ImpurityRadiation.calculate_imprad()` -> `calculate_impurity_radiation_totals`.

    The case that closes the `plasma_profile` back door: the reference builds a real
    `PlasmaProfile` (with a deliberately wrong `profile_dx`, see `_plasma_profile`) and
    the port gets three bare arrays.

    Same `static_argnames` reasoning as above. `f_nd_impurity_electron_array` is
    emphatically *not* static -- entries 2 and 3 of PROCESS's full 14-entry array are
    iteration variables 125/126, so its gradient is one the optimiser really consumes.

    `profile_x` and `radius_plasma_core_norm` are held fixed under fuzzing.
    `radius_plasma_core_norm` for the step-function reason in `_CORE_RADIUS`; `profile_x`
    because a random grid is not monotone, and while both `_simpson` and `scipy` handle
    that identically, a near-duplicate pair of abscissae divides by an interval width of
    almost zero and would test floating-point luck rather than the port.
    """

    audit_record = "models/physics/radiation_power.md"
    reference = _reference_impurity_radiation_totals
    ported = calculate_impurity_radiation_totals
    static_argnames = ("temp_impurity_kev_array", "pden_impurity_lz_nd_temp_array")

    samples = [
        legacy_sample(
            "hydrogen-plus-argon",
            profile_x=_RHO,
            nd_electron_profile=_NE,
            temp_electron_profile_kev=_TE,
            f_nd_impurity_electron_array=_SPECIES_FRACTIONS,
            temp_impurity_kev_array=_SPECIES_TEMP_TABLE,
            pden_impurity_lz_nd_temp_array=_SPECIES_LZ_TABLE,
            radius_plasma_core_norm=_CORE_RADIUS,
            f_p_plasma_core_rad_reduction=1.0,
        ),
        legacy_sample(
            # `f_p_plasma_core_rad_reduction < 1` and a smaller core, so the two totals
            # are genuinely independent rather than one scaling with the other.
            "reduced-core-fraction",
            profile_x=_RHO,
            nd_electron_profile=_NE,
            temp_electron_profile_kev=_TE,
            f_nd_impurity_electron_array=_SPECIES_FRACTIONS,
            temp_impurity_kev_array=_SPECIES_TEMP_TABLE,
            pden_impurity_lz_nd_temp_array=_SPECIES_LZ_TABLE,
            radius_plasma_core_norm=0.35,
            f_p_plasma_core_rad_reduction=0.6,
        ),
    ]

    fuzz_bounds = {
        "nd_electron_profile": (
            np.full(_N_POINTS, 1.0e19),
            np.full(_N_POINTS, 3.0e20),
        ),
        "temp_electron_profile_kev": (
            np.full(_N_POINTS, 0.05),
            np.full(_N_POINTS, 35.0),
        ),
        "f_nd_impurity_electron_array": (
            np.array([0.5, 1.0e-4]),
            np.array([1.0, 1.0e-2]),
        ),
        "f_p_plasma_core_rad_reduction": (0.1, 1.0),
    }
    fuzz_fixed = {
        "profile_x": _RHO,
        "temp_impurity_kev_array": _SPECIES_TEMP_TABLE,
        "pden_impurity_lz_nd_temp_array": _SPECIES_LZ_TABLE,
        "radius_plasma_core_norm": _CORE_RADIUS,
    }


class TestRadiationPowers(Tier1Contract):
    """`radiation_power.calculate_radiation_powers` -> the composite port.

    **The unit's headline case.** PROCESS's side is called with its real signature --
    `PlasmaProfile` object, `DataStructure` object and all -- and the port's side with the
    eight values the audit says are read off those two, plus the thirteen scalars PROCESS
    already passed explicitly. Agreement to `rtol=1e-12` on all four outputs is the
    evidence that the record's read set is complete: a missed read would leave the
    reference using a zero default the port never sees.

    It is also the only case that exercises `combine_radiation_powers`, which has no case
    of its own -- there is no way to drive the impurity model to two prescribed totals
    without replacing it, and a reference that replaced PROCESS's callees would no longer
    be PROCESS. The second sample varies the core/outer split independently of the total,
    which is what an isolated case would have been for.
    """

    audit_record = "models/physics/radiation_power.md"
    reference = _reference_radiation_powers
    ported = calculate_radiation_powers
    static_argnames = ("temp_impurity_kev_array", "pden_impurity_lz_nd_temp_array")

    samples = [
        legacy_sample(
            "st_phys-shaped-operating-point",
            profile_x=_RHO,
            nd_electron_profile=_NE,
            temp_electron_profile_kev=_TE,
            f_nd_impurity_electron_array=_SPECIES_FRACTIONS,
            temp_impurity_kev_array=_SPECIES_TEMP_TABLE,
            pden_impurity_lz_nd_temp_array=_SPECIES_LZ_TABLE,
            radius_plasma_core_norm=_CORE_RADIUS,
            f_p_plasma_core_rad_reduction=1.0,
            nd_plasma_electron_on_axis=1.6e20,
            rminor=1.8,
            b_plasma_toroidal_on_axis=5.0,
            aspect=10.0,
            alphan=1.0,
            alphat=1.45,
            tbeta=2.0,
            temp_plasma_electron_on_axis_kev=29.4,
            f_sync_reflect=0.6,
            rmajor=18.0,
            kappa=1.0,
            vol_plasma=900.0,
        ),
        legacy_sample(
            # Core/outer split moved without moving the total: `pden_plasma_core_rad_mw`
            # and `pden_plasma_outer_rad_mw` are the two the three additions could swap.
            "shifted-core-outer-split",
            profile_x=_RHO,
            nd_electron_profile=_NE,
            temp_electron_profile_kev=_TE,
            f_nd_impurity_electron_array=_SPECIES_FRACTIONS,
            temp_impurity_kev_array=_SPECIES_TEMP_TABLE,
            pden_impurity_lz_nd_temp_array=_SPECIES_LZ_TABLE,
            radius_plasma_core_norm=0.35,
            f_p_plasma_core_rad_reduction=0.6,
            nd_plasma_electron_on_axis=1.6e20,
            rminor=1.8,
            b_plasma_toroidal_on_axis=5.0,
            aspect=10.0,
            alphan=1.0,
            alphat=1.45,
            tbeta=2.0,
            temp_plasma_electron_on_axis_kev=29.4,
            f_sync_reflect=0.6,
            rmajor=18.0,
            kappa=1.0,
            vol_plasma=900.0,
        ),
    ]

    # Not fuzzed. Its two halves are fuzzed separately above and their composition is
    # three additions; fuzzing the whole thing would re-pay for 21 differentiable
    # arguments' worth of finite differences to check the same arithmetic twice.


# ---------------------------------------------------- ImpurityRadiationTotals's node
#
# `calculate_impurity_radiation_totals` (the pure function, tested above) is unchanged.
# What changes here is only `ImpurityRadiationTotals.__call__`'s *signature*:
# `.impurity_radiation.f_nd_impurity_electron_array` is now fourteen individually
# `SequenceKey`-addressed reads (one per species index) instead of one whole-array
# read, matching `composition.py`'s identical treatment of the same field.
# `imp_indices` still selects a static gather over them before forwarding to the pure
# function -- these checks confirm the node still assembles and still computes the
# identical answer, not just that the signature changed shape.


def _node_kwargs_from_totals_kwargs(kwargs, full_fractions):
    """Pack/unpack adapter: `calculate_impurity_radiation_totals`'s pre-selected,
    already-gathered array kwargs -> `ImpurityRadiationTotals.__call__`'s fourteen
    individual per-index kwargs plus the two whole (unselected) 14-row tables.

    `full_fractions` supplies all fourteen species' fractions (only `imp_indices`'
    positions need to agree with `kwargs["f_nd_impurity_electron_array"]`, which is
    already the pre-gathered subset) -- same small, explicit pack/unpack idiom
    `composition.py`'s equivalent adapter uses, itself following
    `coils/calculate.py`'s `coilcurrent` precedent.
    """
    node_kwargs = {
        "radius_plasma_profile_norm": kwargs["profile_x"],
        "nd_plasma_electron_profile": kwargs["nd_electron_profile"],
        "temp_plasma_electron_profile_kev": kwargs["temp_electron_profile_kev"],
        "temp_impurity_keV_array": _TEMP_TABLE,
        "pden_impurity_lz_nd_temp_array": _LZ_TABLE,
        "radius_plasma_core_norm": kwargs["radius_plasma_core_norm"],
        "f_p_plasma_core_rad_reduction": kwargs["f_p_plasma_core_rad_reduction"],
    }
    for i in range(14):
        node_kwargs[f"f_nd_impurity_electron_array_{i}"] = full_fractions[i]
    return node_kwargs


def test_impurity_radiation_totals_assembles_alone():
    """`ImpurityRadiationTotals` must assemble on its own with the new per-index
    signature -- fourteen per-index reads of `f_nd_impurity_electron_array`, none of
    them the whole-array `VarPath`.
    """
    node = ImpurityRadiationTotals(imp_indices=(_HYDROGEN, _ARGON))
    graph = to_graph(node)
    assert graph.definitions

    read = {inp.var for inp in node.inputs}
    for i in range(14):
        idx_path = resolve(
            lambda s, i=i: s.impurity_radiation.f_nd_impurity_electron_array[i],
            VarPath,
        )
        assert idx_path in read
    whole_array_path = resolve(impurity_radiation.f_nd_impurity_electron_array, VarPath)
    assert whole_array_path not in read


def test_impurity_radiation_totals_assembles_with_all_three_nodes():
    """The three `radiation_power.py` nodes still assemble together after the signature
    change -- `PlasmaRadiationPowers` cannot be registered before
    `ImpurityRadiationTotals` is (see that class's own docstring), so this is the shape
    a later consolidation pass would register.
    """
    graph = to_graph(
        SynchrotronRadiationPower(),
        ImpurityRadiationTotals(imp_indices=(_HYDROGEN, _ARGON)),
        PlasmaRadiationPowers(),
    )
    assert graph.definitions
    assert len(graph.definitions) == 3


def test_impurity_radiation_totals_node_matches_pure_function():
    """The node's per-index reassembly + gather must be numerically exact against the
    pure function's own pre-selected-array call, for the same sample
    `TestImpurityRadiationTotals` already validates against PROCESS.

    `full_fractions` places `_SPECIES_FRACTIONS` at indices `_HYDROGEN`/`_ARGON` (real
    indices, not "at the front" the way the PROCESS reference adapter above relabels
    them) and every other index at 0 -- `imp_indices=(_HYDROGEN, _ARGON)` then gathers
    exactly the two rows `calculate_impurity_radiation_totals` receives directly.
    """
    kwargs = dict(TestImpurityRadiationTotals.samples[0].kwargs)

    full_fractions = np.zeros(14)
    full_fractions[_HYDROGEN] = kwargs["f_nd_impurity_electron_array"][0]
    full_fractions[_ARGON] = kwargs["f_nd_impurity_electron_array"][1]

    want = calculate_impurity_radiation_totals(**kwargs)

    node = ImpurityRadiationTotals(imp_indices=(_HYDROGEN, _ARGON))
    node_kwargs = _node_kwargs_from_totals_kwargs(kwargs, full_fractions)
    got = node(**node_kwargs)

    assert len(got) == len(want) == 2
    for g, e in zip(got, want, strict=True):
        np.testing.assert_allclose(np.asarray(g), np.asarray(e), rtol=1e-12, atol=0)
