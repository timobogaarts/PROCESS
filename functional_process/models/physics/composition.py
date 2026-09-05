"""Pure physics functions extracted from `models/physics/composition.py`.

See that file for the declarations wiring these into the graph, and
`_audit/units/models/physics/composition.md` for the audit record. No
graph-framework import belongs in this module -- see `_audit/formulas_split.md`.
"""

import jax
import jax.numpy as jnp

from functional_process.models.physics.impurity_radiation import (
    calculate_average_charge_at_temp,
)
from functional_process.vocabulary import PlasmaIgnitionModel, constants

CARBON_INDEX = 3

OXYGEN_INDEX = 5

IRON_INDEX = 9

ARGON_INDEX = 8
"""`element2index` results for the four species `plasma_composition` reports
separately (`f_nd_plasma_carbon_electron` etc.) -- same justification as `H_INDEX`.
"""

IMPURITY_SLICE = slice(2, 14)
"""`impurity_arr_z > 2` (PROCESS's "impurities, not fuel ions" mask), resolved to a
static slice.

`impurity_arr_z`'s 14 entries are atomic numbers (`1, 2, 4, 6, 7, 8, 10, 14, 18, 26, 28,
36, 54, 74` -- `initialise_imprad`'s hardcoded per-species `z=` arguments), fixed at
`initialise_imprad`, never written anywhere else in PROCESS. `z > 2` is therefore always
exactly "every species except H_ (index 0, z=1) and He (index 1, z=2)" -- i.e. indices
2 through 13 -- for every `DataStructure` `plasma_composition` could run against, so the
mask collapses to a static slice rather than a traced boolean array. This is the same
move `radiation_power.ImpurityRadiationTotals.imp_indices` makes for its own
species-selection mask, though for a different underlying reason: that one is
data-dependent (`f_nd > 1e-30`) and hoisted to a static field per run configuration; this
one is provably constant across *every* configuration, so it needs no field at all.
"""

H_INDEX = 0
"""`element2index("H_", ...)` -- always 0, see module docstring."""

HE_INDEX = 1
"""`element2index("He", ...)` -- always 1, see module docstring."""


def plasma_composition(
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
    """Plasma component fractional makeup -- density, charge and mass bookkeeping.

    Ports `Physics.plasma_composition` (`physics.py:1166-1491`). Every `self.data`
    read/write in the source becomes an explicit argument/return value, in the order
    PROCESS computes them; two things could not be ported as straight translations,
    each documented in full in the audit record's own section:

    - **`.physics.first_call` is not ported at all -- deliberately, not an oversight.**
      PROCESS reads it to pick between two formulas for one intermediate (`pc`): a
      crude parabolic-profile estimate on the very first call, or the real
      profile-derived value (`.physics.f_temp_plasma_electron_density_vol_avg`,
      produced by `physics/plasma_profiles.py`, registry unit #12) on every call after.
      An earlier draft of this port represented that as a genuine `FixedPointFunction`
      self-loop (`NextFirstCall`, since removed) -- technically valid, `to_graph()`
      assembled it, but it was solving the wrong problem. Checking `f_temp_plasma_
      electron_density_vol_avg`'s *real* producer (`plasma_profiles.py:126`, same
      formula, same two inputs `alphan`/`alphat`) shows it has **no dependency on
      anything `plasma_composition` produces** -- so, same shape as this session's
      `Divertor`/`beta_fast_alpha`/`beta_beam` findings, the "first call" branch exists
      only because PROCESS's own imperative call order happens to run
      `plasma_composition` before `plasma_profiles` the first time through the
      pipeline, not because of a genuine cycle. Ordered correctly in a real graph, the
      bootstrap branch is unreachable dead code. This function therefore always takes
      the real `f_temp_plasma_electron_density_vol_avg` and uses it directly as `pc` --
      `first_call`/`alphan`/`alphat` are not parameters here at all, and neither is a
      `first_call_next` return value.
    - **the `znfuel < 0` domain check**: PROCESS raises `ProcessValueError`. The
      quantity itself (`znfuel`, and hence `nd_plasma_fuel_ions_vol_avg`) is a
      well-defined finite (if unphysical) negative float when this fires, not a NaN or
      Inf produced by an invalid math operation -- so "return non-finite instead of
      raising", `test_harness.md`'s standard domain-guard shape, does not apply
      cleanly. The check is simply **not ported**; a negative `nd_plasma_fuel_ions_vol_avg`
      flows through unchanged rather than raising or becoming NaN. See the audit
      record's open questions.
    - **`i_plasma_ignited`**: PROCESS's `PlasmaIgnitionModel(i_plasma_ignited) ==
      PlasmaIgnitionModel.NON_IGNITED` compare selects between two `nd_beam_ions`
      formulas with genuinely different read sets (`traceability_policy.md`'s default
      would be "split"). Kept as a static, resolved-before-tracing parameter instead,
      because it is two lines deep inside an otherwise-shared 328-line function --
      splitting would duplicate the other ~95% of the body across two top-level
      functions. Flagged as a policy deviation, not silently applied; see the audit
      record's switches section.

      **It is spelled and typed as PROCESS spells it**: the `PlasmaIgnitionModel`
      enum, under PROCESS's own field name. An earlier draft restated it as a plain
      `bool` named `is_ignited`, which `_audit/switch_elimination_design.md` §3
      classifies as kind (d), "alias / noise -- delete, don't rename": it forced
      `mda_harness.STATIC_KWARG_ALIASES` to carry a hand-written
      `bool(v == 1)` reconstruction of the mapping purely so `switch_audit` could
      check the field at all. With the enum there is nothing to reconstruct.

    Every impurity-species sum (`znimp`, the total-impurity-density loop, the
    charge-weighted accumulators) is a `for imp in range(14): if z[imp] > 2: ...` in the
    source. All four collapse to a single `jax.vmap` over `IMPURITY_SLICE` (or, for
    `n_charge_plasma_effective_vol_avg`, over all 14 species -- see below, it is the one
    accumulator PROCESS does *not* gate by `z > 2`) followed by plain array
    reductions -- `traceability_policy.md`'s "in-place sequential mutation ... often
    signals ... a reduction that a vectorised rewrite would simplify", found exactly as
    predicted.

    Parameters
    ----------
    nd_plasma_electrons_vol_avg, f_nd_alpha_thermal_electron :
        Electron density (m^-3); alpha-ash fraction of it.
    fusden_alpha_total :
        Alpha particle production rate (m^-3 s^-1); `< 1e-6` selects the
        "not calculated yet" proton-density estimate.
    f_nd_protium_electrons, proton_rate_density :
        User-input protium override fraction; proton production rate.
    f_nd_beam_electron :
        Beam ion fraction of electron density (non-ignited case only).
    f_nd_impurity_electron_array :
        `(14,)` relative impurity densities (`n_imp/n_e`). Indices 0 (`H_`) and 1
        (`He`) are **overwritten** by this function; the rest are read only.
    temp_plasma_electron_vol_avg_kev :
        Volume-averaged electron temperature (keV), the single point every average
        charge is evaluated at.
    temp_impurity_keV_array, impurity_arr_zav :
        `(14, 200)` L(Z, Te) temperature abscissae and average-charge tables --
        compile-time constants, see `impurity_radiation.py`'s own port.
    f_plasma_fuel_deuterium, f_plasma_fuel_tritium, f_plasma_fuel_helium3 :
        Fuel mix fractions.
    f_temp_plasma_electron_density_vol_avg :
        Profile-derived `pc` value, used directly -- see the docstring section above
        for why the `first_call` bootstrap this replaces is not ported. Produced by
        `physics/plasma_profiles.py` (registry unit #12, already ported).
    f_beam_tritium :
        Tritium fraction of the neutral beam (`.current_drive.f_beam_tritium`).
    m_impurity_amu_array :
        `(14,)` impurity atomic masses (amu) -- compile-time constant.
    i_plasma_ignited :
        `PlasmaIgnitionModel.IGNITED` selects the ignited case (`nd_beam_ions = 0`);
        `PlasmaIgnitionModel.NON_IGNITED` (PROCESS's default,
        `physics_variables.py:881`) selects the non-ignited case. Static -- see the
        docstring section above.

    Returns
    -------
    tuple
        `(nd_plasma_alphas_thermal_vol_avg, nd_plasma_protons_vol_avg, nd_beam_ions,
        nd_plasma_fuel_ions_vol_avg, f_nd_impurity_electron_array,
        nd_plasma_impurities_vol_avg, nd_plasma_ions_total_vol_avg,
        f_nd_plasma_carbon_electron, f_nd_plasma_oxygen_electron,
        f_nd_plasma_iron_argon_electron, n_charge_plasma_effective_vol_avg,
        f_alpha_electron, f_alpha_ion, m_fuel_amu, m_beam_amu,
        m_ions_total_amu, n_charge_plasma_effective_mass_weighted_vol_avg)`, matching
        the order PROCESS computes them in (`first_call_next` omitted -- see above).
    """
    if PlasmaIgnitionModel(i_plasma_ignited) is PlasmaIgnitionModel.IGNITED:
        return plasma_composition_ignited(
            nd_plasma_electrons_vol_avg,
            f_nd_alpha_thermal_electron,
            fusden_alpha_total,
            f_nd_protium_electrons,
            proton_rate_density,
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
        )
    return plasma_composition_non_ignited(
        nd_plasma_electrons_vol_avg,
        f_nd_alpha_thermal_electron,
        fusden_alpha_total,
        f_nd_protium_electrons,
        proton_rate_density,
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
        f_nd_beam_electron=f_nd_beam_electron,
    )


def plasma_composition_ignited(
    nd_plasma_electrons_vol_avg,
    f_nd_alpha_thermal_electron,
    fusden_alpha_total,
    f_nd_protium_electrons,
    proton_rate_density,
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
):
    """`plasma_composition` at `i_plasma_ignited == IGNITED` (1) -- the reference run's.

    An ignited plasma has no beam ions, so `.physics.f_nd_beam_electron` is **not read
    at all**: that is the single edge one node carrying the switch invented
    (`switch_kwarg_survey.md` §3, `live (1)`). Parameters and returns are the
    composite's, less `i_plasma_ignited` and `f_nd_beam_electron`.
    """
    return _plasma_composition(
        lambda nd_plasma_electrons_vol_avg: jnp.zeros_like(nd_plasma_electrons_vol_avg),
        nd_plasma_electrons_vol_avg,
        f_nd_alpha_thermal_electron,
        fusden_alpha_total,
        f_nd_protium_electrons,
        proton_rate_density,
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
    )


def plasma_composition_non_ignited(
    nd_plasma_electrons_vol_avg,
    f_nd_alpha_thermal_electron,
    fusden_alpha_total,
    f_nd_protium_electrons,
    proton_rate_density,
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
    *,
    f_nd_beam_electron,
):
    """`plasma_composition` at `i_plasma_ignited == NON_IGNITED` (0) -- PROCESS's own
    default (`physics_variables.py:881`) and the conventional tokamak's.

    Parameters and returns are the composite's, less `i_plasma_ignited`.
    """
    return _plasma_composition(
        lambda nd_plasma_electrons_vol_avg: (
            nd_plasma_electrons_vol_avg * f_nd_beam_electron
        ),
        nd_plasma_electrons_vol_avg,
        f_nd_alpha_thermal_electron,
        fusden_alpha_total,
        f_nd_protium_electrons,
        proton_rate_density,
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
    )


def _plasma_composition(
    beam_ion_density,
    nd_plasma_electrons_vol_avg,
    f_nd_alpha_thermal_electron,
    fusden_alpha_total,
    f_nd_protium_electrons,
    proton_rate_density,
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
):
    """Everything both `i_plasma_ignited` arms share, given the arm's own beam-ion
    density law.

    `beam_ion_density` is a function of the electron density, not a switch: an ignited
    plasma's is `zeros_like` and a beam-heated one's is `nd * f_nd_beam_electron`, and
    which one a node gets follows from which arm function it called. Splitting here
    rather than keeping the two lines behind a static kwarg is
    `_audit/next_steps.md` §14.2; the note in this module's docstring calling the kwarg
    a deliberate policy deviation -- "two lines deep inside an otherwise-shared
    328-line function" -- is withdrawn with the policy that allowed it.
    """
    nd_plasma_alphas_thermal_vol_avg = (
        nd_plasma_electrons_vol_avg * f_nd_alpha_thermal_electron
    )

    protons_not_yet_calculated = fusden_alpha_total < 1.0e-6
    protons_early = jnp.maximum(
        f_nd_protium_electrons * nd_plasma_electrons_vol_avg,
        nd_plasma_alphas_thermal_vol_avg * (f_plasma_fuel_helium3 + 1.0e-3),
    )
    safe_fusden_alpha_total = jnp.where(
        protons_not_yet_calculated, 1.0, fusden_alpha_total
    )
    protons_later = jnp.maximum(
        f_nd_protium_electrons * nd_plasma_electrons_vol_avg,
        nd_plasma_alphas_thermal_vol_avg * proton_rate_density / safe_fusden_alpha_total,
    )
    nd_plasma_protons_vol_avg = jnp.where(
        protons_not_yet_calculated, protons_early, protons_later
    )

    nd_beam_ions = beam_ion_density(nd_plasma_electrons_vol_avg)

    # <Z>(T_e) for every species at the single bulk electron temperature -- one vmap
    # over the species axis replaces 14 (or, in the source's other loop, up to 14)
    # separate per-species calls.
    zav_all = jax.vmap(calculate_average_charge_at_temp, in_axes=(None, 0, 0))(
        temp_plasma_electron_vol_avg_kev, temp_impurity_keV_array, impurity_arr_zav
    )

    znimp = (
        jnp.sum(zav_all[IMPURITY_SLICE] * f_nd_impurity_electron_array[IMPURITY_SLICE])
        * nd_plasma_electrons_vol_avg
    )

    znfuel = (
        nd_plasma_electrons_vol_avg
        - 2.0 * nd_plasma_alphas_thermal_vol_avg
        - nd_plasma_protons_vol_avg
        - nd_beam_ions
        - znimp
    )
    # PROCESS raises ProcessValueError here if znfuel < 0.0. Not ported -- see the
    # function's own docstring and the audit record's open questions.

    nd_plasma_fuel_ions_vol_avg = znfuel / (1.0 + f_plasma_fuel_helium3)

    # `.at[].set()` requires a jax array -- a plain numpy array (e.g. straight off a
    # `DataStructure` field) has no `.at`.
    f_nd_impurity_electron_array = jnp.asarray(f_nd_impurity_electron_array)
    f_nd_impurity_electron_array = f_nd_impurity_electron_array.at[H_INDEX].set(
        (
            nd_plasma_protons_vol_avg
            + (f_plasma_fuel_deuterium + f_plasma_fuel_tritium)
            * nd_plasma_fuel_ions_vol_avg
            + nd_beam_ions
        )
        / nd_plasma_electrons_vol_avg
    )
    f_nd_impurity_electron_array = f_nd_impurity_electron_array.at[HE_INDEX].set(
        f_plasma_fuel_helium3 * nd_plasma_fuel_ions_vol_avg / nd_plasma_electrons_vol_avg
        + f_nd_alpha_thermal_electron
    )

    nd_plasma_impurities_vol_avg = (
        jnp.sum(f_nd_impurity_electron_array[IMPURITY_SLICE])
        * nd_plasma_electrons_vol_avg
    )

    nd_plasma_ions_total_vol_avg = (
        nd_plasma_fuel_ions_vol_avg
        + nd_plasma_alphas_thermal_vol_avg
        + nd_plasma_protons_vol_avg
        + nd_beam_ions
        + nd_plasma_impurities_vol_avg
    )

    f_nd_plasma_carbon_electron = f_nd_impurity_electron_array[CARBON_INDEX]
    f_nd_plasma_oxygen_electron = f_nd_impurity_electron_array[OXYGEN_INDEX]
    f_nd_plasma_iron_argon_electron = (
        f_nd_impurity_electron_array[IRON_INDEX]
        + f_nd_impurity_electron_array[ARGON_INDEX]
    )

    # Unlike every other species sum in this function, PROCESS does *not* gate this one
    # by `impurity_arr_z > 2` -- all 14 species (including the just-updated H_/He_
    # entries) contribute. Confirmed against the source (physics.py:1359-1369): the
    # `for imp in range(N_IMPURITIES):` loop here has no `if` inside it.
    n_charge_plasma_effective_vol_avg = jnp.sum(
        f_nd_impurity_electron_array * zav_all**2
    )

    # `pc`: always the real profile-derived value. PROCESS's own `first_call`-gated
    # parabolic-estimate fallback is not reproduced -- see the function's own
    # docstring for why (an ordering artefact of PROCESS's imperative call sequence,
    # not a genuine dependency this port needs to bootstrap around).
    pc = f_temp_plasma_electron_density_vol_avg

    f_alpha_electron = 0.88155 * jnp.exp(
        -temp_plasma_electron_vol_avg_kev * pc / 67.4036
    )
    f_alpha_ion = 1.0 - f_alpha_electron

    m_fuel_amu = (
        constants.M_DEUTERON_AMU * f_plasma_fuel_deuterium
        + constants.M_TRITON_AMU * f_plasma_fuel_tritium
        + constants.M_HELION_AMU * f_plasma_fuel_helium3
    )

    m_beam_amu = (
        constants.M_DEUTERON_AMU * (1.0 - f_beam_tritium)
        + constants.M_TRITON_AMU * f_beam_tritium
    )

    m_ions_total_amu = (
        m_fuel_amu * nd_plasma_fuel_ions_vol_avg
        + constants.M_ALPHA_AMU * nd_plasma_alphas_thermal_vol_avg
        + nd_plasma_protons_vol_avg * constants.M_PROTON_AMU
        + m_beam_amu * nd_beam_ions
        + nd_plasma_electrons_vol_avg
        * jnp.sum(
            f_nd_impurity_electron_array[IMPURITY_SLICE]
            * m_impurity_amu_array[IMPURITY_SLICE]
        )
    ) / nd_plasma_ions_total_vol_avg

    n_charge_plasma_effective_mass_weighted_vol_avg = (
        f_plasma_fuel_deuterium * nd_plasma_fuel_ions_vol_avg / constants.M_DEUTERON_AMU
        + f_plasma_fuel_tritium * nd_plasma_fuel_ions_vol_avg / constants.M_TRITON_AMU
        + 4.0
        * f_plasma_fuel_helium3
        * nd_plasma_fuel_ions_vol_avg
        / constants.M_HELION_AMU
        + 4.0 * nd_plasma_alphas_thermal_vol_avg / constants.M_ALPHA_AMU
        + nd_plasma_protons_vol_avg / constants.M_PROTON_AMU
        + (1.0 - f_beam_tritium) * nd_beam_ions / constants.M_DEUTERON_AMU
        + f_beam_tritium * nd_beam_ions / constants.M_TRITON_AMU
    ) / nd_plasma_electrons_vol_avg + jnp.sum(
        f_nd_impurity_electron_array[IMPURITY_SLICE]
        * zav_all[IMPURITY_SLICE] ** 2
        / m_impurity_amu_array[IMPURITY_SLICE]
    )

    return (
        nd_plasma_alphas_thermal_vol_avg,
        nd_plasma_protons_vol_avg,
        nd_beam_ions,
        nd_plasma_fuel_ions_vol_avg,
        f_nd_impurity_electron_array,
        nd_plasma_impurities_vol_avg,
        nd_plasma_ions_total_vol_avg,
        f_nd_plasma_carbon_electron,
        f_nd_plasma_oxygen_electron,
        f_nd_plasma_iron_argon_electron,
        n_charge_plasma_effective_vol_avg,
        f_alpha_electron,
        f_alpha_ion,
        m_fuel_amu,
        m_beam_amu,
        m_ions_total_amu,
        n_charge_plasma_effective_mass_weighted_vol_avg,
    )


def calculate_effective_charge_ionisation_profiles(
    temp_electron_profile_kev,
    f_nd_impurity_electron_array,
    temp_impurity_keV_array,
    impurity_arr_zav,
):
    """Effective-charge and per-species charge profiles across the plasma radius.

    Ports `Physics.calculate_effective_charge_ionisation_profiles`
    (`physics.py:1749-1781`). The source is a double Python loop (species x radial
    point), each iteration calling `calculate_average_charge_at_temp` on a
    length-1 array. Both dimensions vectorise away at once: `calculate_average_charge_at_temp`
    already accepts an array `temp_electron_kev` (its `jnp.interp` is elementwise), so a
    single call per species handles every radial point together, and `jax.vmap` over
    the species axis handles the rest -- the identical move `plasma_composition` above
    makes for `zav_all`, just over `(14, n_points)` instead of `(14,)`.

    Parameters
    ----------
    temp_electron_profile_kev :
        Electron temperature profile (keV), `(n_points,)`. PROCESS's own source reads
        this off `self.plasma_profile.teprofile.profile_y`; the audit record notes this
        is the same array `radiation_power.py`'s port already minted as
        `.physics.temp_plasma_electron_profile_kev` (registry units #20/#21) -- reused,
        not re-minted, see `CalculateEffectiveChargeIonisationProfiles` below.
    f_nd_impurity_electron_array :
        `(14,)` relative impurity densities.
    temp_impurity_keV_array, impurity_arr_zav :
        `(14, 200)` L(Z, Te) tables, as in `plasma_composition`.

    Returns
    -------
    tuple
        `(n_charge_plasma_effective_profile, n_charge_impurity_profile)`:
        `(n_points,)` effective-charge profile and `(14, n_points)` per-species charge
        profile.
    """
    n_charge_impurity_profile = jax.vmap(
        calculate_average_charge_at_temp, in_axes=(None, 0, 0)
    )(temp_electron_profile_kev, temp_impurity_keV_array, impurity_arr_zav)

    n_charge_plasma_effective_profile = jnp.sum(
        f_nd_impurity_electron_array[:, None] * n_charge_impurity_profile**2, axis=0
    )

    return n_charge_plasma_effective_profile, n_charge_impurity_profile


def effective_charge_ionisation_profiles_from_indexed_impurities(
    temp_plasma_electron_profile_kev,
    f_nd_impurity_electron_array_0,
    f_nd_impurity_electron_array_1,
    f_nd_impurity_electron_array_2,
    f_nd_impurity_electron_array_3,
    f_nd_impurity_electron_array_4,
    f_nd_impurity_electron_array_5,
    f_nd_impurity_electron_array_6,
    f_nd_impurity_electron_array_7,
    f_nd_impurity_electron_array_8,
    f_nd_impurity_electron_array_9,
    f_nd_impurity_electron_array_10,
    f_nd_impurity_electron_array_11,
    f_nd_impurity_electron_array_12,
    f_nd_impurity_electron_array_13,
    temp_impurity_keV_array,
    impurity_arr_zav,
):
    """Restack the fourteen per-index impurity fractions, then delegate.

    The graph reads `f_nd_impurity_electron_array` as fourteen individually-addressed
    ports (see `CalculateEffectiveChargeIonisationProfiles`'s docstring for why); this
    reassembles the `(14,)` array `calculate_effective_charge_ionisation_profiles`
    itself takes, unchanged.
    """
    f_nd_impurity_electron_array = jnp.stack([
        f_nd_impurity_electron_array_0,
        f_nd_impurity_electron_array_1,
        f_nd_impurity_electron_array_2,
        f_nd_impurity_electron_array_3,
        f_nd_impurity_electron_array_4,
        f_nd_impurity_electron_array_5,
        f_nd_impurity_electron_array_6,
        f_nd_impurity_electron_array_7,
        f_nd_impurity_electron_array_8,
        f_nd_impurity_electron_array_9,
        f_nd_impurity_electron_array_10,
        f_nd_impurity_electron_array_11,
        f_nd_impurity_electron_array_12,
        f_nd_impurity_electron_array_13,
    ])
    return calculate_effective_charge_ionisation_profiles(
        temp_plasma_electron_profile_kev,
        f_nd_impurity_electron_array,
        temp_impurity_keV_array,
        impurity_arr_zav,
    )
