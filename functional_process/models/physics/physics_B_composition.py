"""Pure-functional port of `Physics.plasma_composition` and
`Physics.calculate_effective_charge_ionisation_profiles`
(`process/models/physics/physics.py`).

Registry unit #9, chunk B. Audit record:
`functional_process/models/physics/physics_B_composition.md` -- read it first,
especially "the `first_call` self-loop" and "the `znfuel` raise" sections. `first_call`
is not ported at all -- it turned out to be an ordering artifact of PROCESS's own
imperative call sequence, not a genuine cycle; see `plasma_composition`'s own docstring
for the full account. The `znfuel` domain check remains deliberately unported -- see
that section.

Both functions were flagged blocked in `unit_registry.md` row #9 (they call
`impurity_radiation.calculate_average_charge_at_temp`/`element2index`, registry unit
#23, audited by a parallel agent). **That block cleared during this session**: unit #23
landed in this same directory (`impurity_radiation.py`/`.md`/`test_impurity_radiation.py`)
before this chunk was written, with exactly the two functions needed, already pure and
JAX-traceable. This port imports them directly rather than re-deriving them -- see the
audit record's "scope correction: unit #23 unblocked" section for the two things that
makes possible: closing the `data` back door completely, and collapsing PROCESS's
per-species Python loops into `jax.vmap` over the species axis.

**`element2index`'s indices are resolved to Python `int` literals here, not called at
trace time.** `impurity_radiation.py`'s own port already establishes that
`element2index` is a graph-assembly-time lookup, not a traced node (its docstring: "not
a JAX-traced code path... resolved once, on Python values"). `initialise_imprad`
populates `impurity_arr_label`/`impurity_arr_z` in a fixed order every real PROCESS run
(`imp_label`'s literal default: H_, He, Be, C_, N_, O_, Ne, Si, Ar, Fe, Ni, Kr, Xe, W_ --
`impurity_radiation_variables.py:61-74`, copied verbatim by `initialise_imprad`), so
`element2index("H_", ...) == 0` and `element2index("He", ...) == 1` for every
`DataStructure` this function could ever run against -- verified against the source, not
assumed; see the audit record's data-footprint table for the full index table this port
hardcodes.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FromExactly,
    Output,
)

from functional_process.models.physics.impurity_radiation import (
    calculate_average_charge_at_temp,
)
from process.core import constants

H_INDEX = 0
"""`element2index("H_", ...)` -- always 0, see module docstring."""

HE_INDEX = 1
"""`element2index("He", ...)` -- always 1, see module docstring."""

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
    is_ignited,
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
    - **`is_ignited`**: PROCESS's `PlasmaIgnitionModel(i_plasma_ignited) ==
      PlasmaIgnitionModel.NON_IGNITED` compare selects between two `nd_beam_ions`
      formulas with genuinely different read sets (`traceability_policy.md`'s default
      would be "split"). Kept as a plain Python `bool` parameter (static, resolved
      before tracing) instead, because it is two lines deep inside an otherwise-shared
      328-line function -- splitting would duplicate the other ~95% of the body across
      two top-level functions. Flagged as a policy deviation, not silently applied; see
      the audit record's switches section.

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
    is_ignited :
        `True` selects the ignited case (`nd_beam_ions = 0`); `False` (PROCESS's
        default) selects the non-ignited case. Static -- see the docstring section
        above.

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

    if is_ignited:
        nd_beam_ions = jnp.zeros_like(nd_plasma_electrons_vol_avg)
    else:
        nd_beam_ions = nd_plasma_electrons_vol_avg * f_nd_beam_electron

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


class PlasmaComposition(ExplicitFunction):
    """cottax node: `plasma_composition`'s outputs.

    `.physics.first_call` is not a port here at all -- earlier this session it was
    (`NextFirstCall`, a `FixedPointFunction`, since removed): technically valid,
    `to_graph()` assembled it, but on reflection it was the wrong fix. Checking the
    real producer of the value `first_call` exists to bootstrap
    (`.physics.f_temp_plasma_electron_density_vol_avg`, from `plasma_profiles.py`)
    shows it has no dependency back on anything `plasma_composition` produces -- so,
    same shape as this session's `Divertor`/`beta_fast_alpha`/`beta_beam` findings,
    `first_call` was never a genuine cycle, just an artefact of PROCESS's own
    imperative call order. `plasma_composition` (the ported function) now always uses
    the real value directly; see its own docstring for the full account. There is
    nothing left to declare a `FixedPointFunction` for.

    **A second, previously unflagged Shape-B self-loop was found while wiring this
    node -- and is now resolved, not merely worked around.** The earlier draft found
    that `.impurity_radiation.f_nd_impurity_electron_array` is both read (for
    `znimp`/`nd_plasma_impurities_vol_avg`/the per-species fraction outputs -- all read
    only the `z > 2` slice, indices 2:13, which `plasma_composition`'s H_/He_ writes
    never touch) and, inside `plasma_composition`, *written* (indices 0/1). Declaring
    the *whole array* as both an `FromExactly` and an `Output` reproduces the identical
    `ValueError: reads [...], which it also owns` `Avail`/`NextFirstCall` hit -- but the
    two ranges that read and write are disjoint at index granularity (2:13 read, 0/1
    written, no overlap), so the self-loop was an artefact of addressing the field as
    one `VarPath` rather than fourteen. Per-element addressing
    (`~/jaxgraph`'s `_Recorder.__getitem__`: an `int` index becomes a `SequenceKey`
    component, so `s.impurity_radiation.f_nd_impurity_electron_array[i]` is a real,
    distinct `VarPath` for each `i`, matching the real `DataStructure` field's own
    `list[float]` storage -- see `impurity_radiation_variables.py`) removes the
    conflict entirely: this node reads indices 2-13 (`IMPURITY_SLICE = slice(2, 14)`,
    twelve entries -- `2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13`) as twelve ordinary
    `FromExactly`s and **owns** indices 0 (`H_`, `f_nd_impurity_electron_array_h`) and 1
    (`He`, `f_nd_impurity_electron_array_he`) as two ordinary `Output`s -- twelve read
    plus two owned is the whole fourteen-entry array, with no `FixedPointFunction`/`Cut`
    machinery needed at all. `to_graph(PlasmaComposition(...))` confirms this
    empirically (`test_plasma_composition_owns_h_and_he_fractions`).

    Indices 0/1's *old* values are never read by `plasma_composition` before being
    overwritten (`.at[H_INDEX].set(...)`/`.at[HE_INDEX].set(...)` -- confirmed by
    reading the function body, not assumed), so `__call__` below assembles the full
    14-array it hands to `plasma_composition` with placeholder zeros at those two
    positions; the result is numerically exact, not an approximation -- the two
    placeholders are unconditionally overwritten before anything downstream reads them.

    `is_ignited` is a graph-assembly-time switch, not a port -- `eqx.field(static=True)`,
    same move `ConfinementTime`/`EcrhDensityLimit` already make for their own switches
    (`physics_B_composition.md`'s "switches touched" section: the two branches'
    reads-sets genuinely differ, but the differing part is two lines inside an
    otherwise-shared 328-line function, so it stays a static kwarg rather than an
    `Alternative` split).
    """

    is_ignited: bool = eqx.field(static=True)

    nd_plasma_alphas_thermal_vol_avg = Output(
        lambda s: s.physics.nd_plasma_alphas_thermal_vol_avg
    )
    nd_plasma_protons_vol_avg = Output(lambda s: s.physics.nd_plasma_protons_vol_avg)
    nd_beam_ions = Output(lambda s: s.physics.nd_beam_ions)
    nd_plasma_fuel_ions_vol_avg = Output(
        lambda s: s.physics.nd_plasma_fuel_ions_vol_avg
    )
    f_nd_impurity_electron_array_h = Output(
        lambda s: s.impurity_radiation.f_nd_impurity_electron_array[H_INDEX]
    )
    """`f_nd_impurity_electron_array[0]` (PROCESS display label
    `f_nd_impurity_electrons(01)` per `naming_convention.md` § "Array elements" --
    record both, they are not the same thing). The `H_` fraction `plasma_composition`
    computes and writes."""
    f_nd_impurity_electron_array_he = Output(
        lambda s: s.impurity_radiation.f_nd_impurity_electron_array[HE_INDEX]
    )
    """`f_nd_impurity_electron_array[1]` (display label `f_nd_impurity_electrons(02)`).
    The `He` fraction `plasma_composition` computes and writes."""
    nd_plasma_impurities_vol_avg = Output(
        lambda s: s.physics.nd_plasma_impurities_vol_avg
    )
    nd_plasma_ions_total_vol_avg = Output(
        lambda s: s.physics.nd_plasma_ions_total_vol_avg
    )
    f_nd_plasma_carbon_electron = Output(
        lambda s: s.physics.f_nd_plasma_carbon_electron
    )
    f_nd_plasma_oxygen_electron = Output(
        lambda s: s.physics.f_nd_plasma_oxygen_electron
    )
    f_nd_plasma_iron_argon_electron = Output(
        lambda s: s.physics.f_nd_plasma_iron_argon_electron
    )
    n_charge_plasma_effective_vol_avg = Output(
        lambda s: s.physics.n_charge_plasma_effective_vol_avg
    )
    f_alpha_electron = Output(lambda s: s.physics.f_alpha_electron)
    f_alpha_ion = Output(lambda s: s.physics.f_alpha_ion)
    m_fuel_amu = Output(lambda s: s.physics.m_fuel_amu)
    m_beam_amu = Output(lambda s: s.physics.m_beam_amu)
    m_ions_total_amu = Output(lambda s: s.physics.m_ions_total_amu)
    n_charge_plasma_effective_mass_weighted_vol_avg = Output(
        lambda s: s.physics.n_charge_plasma_effective_mass_weighted_vol_avg
    )

    def __call__(
        self,
        nd_plasma_electrons_vol_avg=FromExactly(
            lambda s: s.physics.nd_plasma_electrons_vol_avg
        ),
        f_nd_alpha_thermal_electron=FromExactly(
            lambda s: s.physics.f_nd_alpha_thermal_electron
        ),
        fusden_alpha_total=FromExactly(lambda s: s.physics.fusden_alpha_total),
        f_nd_protium_electrons=FromExactly(lambda s: s.physics.f_nd_protium_electrons),
        proton_rate_density=FromExactly(lambda s: s.physics.proton_rate_density),
        f_nd_beam_electron=FromExactly(lambda s: s.physics.f_nd_beam_electron),
        f_nd_impurity_electron_array_2=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[2]
        ),
        f_nd_impurity_electron_array_3=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[3]
        ),
        f_nd_impurity_electron_array_4=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[4]
        ),
        f_nd_impurity_electron_array_5=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[5]
        ),
        f_nd_impurity_electron_array_6=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[6]
        ),
        f_nd_impurity_electron_array_7=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[7]
        ),
        f_nd_impurity_electron_array_8=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[8]
        ),
        f_nd_impurity_electron_array_9=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[9]
        ),
        f_nd_impurity_electron_array_10=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[10]
        ),
        f_nd_impurity_electron_array_11=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[11]
        ),
        f_nd_impurity_electron_array_12=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[12]
        ),
        f_nd_impurity_electron_array_13=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[13]
        ),
        temp_plasma_electron_vol_avg_kev=FromExactly(
            lambda s: s.physics.temp_plasma_electron_vol_avg_kev
        ),
        temp_impurity_keV_array=FromExactly(
            lambda s: s.impurity_radiation.temp_impurity_keV_array
        ),
        impurity_arr_zav=FromExactly(lambda s: s.impurity_radiation.impurity_arr_zav),
        f_plasma_fuel_deuterium=FromExactly(lambda s: s.physics.f_plasma_fuel_deuterium),
        f_plasma_fuel_tritium=FromExactly(lambda s: s.physics.f_plasma_fuel_tritium),
        f_plasma_fuel_helium3=FromExactly(lambda s: s.physics.f_plasma_fuel_helium3),
        f_temp_plasma_electron_density_vol_avg=FromExactly(
            lambda s: s.physics.f_temp_plasma_electron_density_vol_avg
        ),
        f_beam_tritium=FromExactly(lambda s: s.current_drive.f_beam_tritium),
        m_impurity_amu_array=FromExactly(
            lambda s: s.impurity_radiation.m_impurity_amu_array
        ),
    ):
        # `plasma_composition` (the pure function) is unchanged -- still one 14-array
        # parameter, physics untouched. Indices 0/1 are placeholders: the function
        # never reads the *old* values there (see the class docstring), only
        # overwrites them via `.at[H_INDEX].set(...)`/`.at[HE_INDEX].set(...)`, so
        # zeros are numerically exact, not an approximation.
        placeholder = jnp.zeros_like(f_nd_impurity_electron_array_2)
        f_nd_impurity_electron_array = jnp.stack([
            placeholder,  # index 0 (H_) -- owned by this node's own Output, not read
            placeholder,  # index 1 (He) -- ditto
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

        results = plasma_composition(
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
            self.is_ignited,
        )
        # `results[4]` is the post-update 14-array (index 4 of `plasma_composition`'s
        # return tuple, see its own docstring); this node owns its two updated entries
        # individually (`f_nd_impurity_electron_array_h`/`_he`) rather than the whole
        # array. Order must match the `Output` declarations above: results[:4] (4), the
        # two extracted H_/He_ entries (2), results[5:] (12) -- 18 total.
        return (
            *results[:4],
            results[4][H_INDEX],
            results[4][HE_INDEX],
            *results[5:],
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


class CalculateEffectiveChargeIonisationProfiles(ExplicitFunction):
    """cottax node: `calculate_effective_charge_ionisation_profiles`, ports declared.

    `temp_electron_profile_kev` is bound to `.physics.temp_plasma_electron_profile_kev`
    -- **reused, not re-minted**: `radiation_power.py`'s `ImpurityRadiationTotals`
    (registry unit #20) already mints exactly this name for exactly this array
    (`teprofile.profile_y`, off the same `PlasmaProfile` instance), and this node is a
    second, independent consumer of it -- the same kind of confirming evidence
    `radiation_power.md`/`fusion_reactions.md` already record for their own reused
    mints.

    `plasma_composition` is ported as a single node class above, `PlasmaComposition`,
    with 17 ordinary outputs including the two per-index impurity-fraction outputs
    below -- see that class's own docstring for why `.physics.first_call` is not a
    port at all.

    `f_nd_impurity_electron_array` is read here as fourteen individual `FromExactly`s
    (`SequenceKey`-addressed, one per index, matching `PlasmaComposition`'s own
    per-index treatment of the same field -- see that class's docstring) rather than
    one whole-array `FromExactly`. **This is a real dependency, not cosmetic**: two of the
    fourteen -- indices 0 (`H_`) and 1 (`He`) -- are exactly the entries
    `PlasmaComposition` owns, so a graph containing both nodes now has a genuine edge
    from `PlasmaComposition`'s `f_nd_impurity_electron_array_h`/`_he` outputs into this
    node's own reads, where previously the whole-array `FromExactly` was an unproduced
    boundary variable no matter what else was in the graph
    (`test_calculate_effective_charge_ionisation_profiles_depends_on_plasma_composition`).
    """

    n_charge_plasma_effective_profile = Output(
        lambda s: s.physics.n_charge_plasma_effective_profile
    )
    n_charge_impurity_profile = Output(
        lambda s: s.impurity_radiation.n_charge_impurity_profile
    )

    def __call__(
        self,
        temp_electron_profile_kev=FromExactly(
            lambda s: s.physics.temp_plasma_electron_profile_kev
        ),
        f_nd_impurity_electron_array_0=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[0]
        ),
        f_nd_impurity_electron_array_1=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[1]
        ),
        f_nd_impurity_electron_array_2=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[2]
        ),
        f_nd_impurity_electron_array_3=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[3]
        ),
        f_nd_impurity_electron_array_4=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[4]
        ),
        f_nd_impurity_electron_array_5=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[5]
        ),
        f_nd_impurity_electron_array_6=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[6]
        ),
        f_nd_impurity_electron_array_7=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[7]
        ),
        f_nd_impurity_electron_array_8=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[8]
        ),
        f_nd_impurity_electron_array_9=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[9]
        ),
        f_nd_impurity_electron_array_10=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[10]
        ),
        f_nd_impurity_electron_array_11=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[11]
        ),
        f_nd_impurity_electron_array_12=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[12]
        ),
        f_nd_impurity_electron_array_13=FromExactly(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array[13]
        ),
        temp_impurity_keV_array=FromExactly(
            lambda s: s.impurity_radiation.temp_impurity_keV_array
        ),
        impurity_arr_zav=FromExactly(lambda s: s.impurity_radiation.impurity_arr_zav),
    ):
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
            temp_electron_profile_kev,
            f_nd_impurity_electron_array,
            temp_impurity_keV_array,
            impurity_arr_zav,
        )
