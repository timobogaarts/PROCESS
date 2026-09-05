"""Pure-functional port of `Physics.plasma_composition` and
`Physics.calculate_effective_charge_ionisation_profiles`
(`process/models/physics/physics.py`).

Registry unit #9, chunk B. Audit record:
`functional_process/_audit/units/models/physics/composition.md` -- read it
first, especially "the `first_call` self-loop" and "the `znfuel` raise" sections.
`first_call` is not ported at all -- it turned out to be an ordering artifact of
PROCESS's own imperative call sequence, not a genuine cycle; see `plasma_composition`'s
own docstring for the full account. The `znfuel` domain check remains deliberately
unported -- see that section.

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

import functools

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    Output,
    OutputInto,
)

from functional_process.paths import current_drive, impurity_radiation, physics
from functional_process.models.physics.composition import (
    H_INDEX,
    HE_INDEX,
    calculate_effective_charge_ionisation_profiles,
    effective_charge_ionisation_profiles_from_indexed_impurities,
    plasma_composition,
    plasma_composition_ignited,
    plasma_composition_non_ignited,
)

__all__ = [
    "calculate_effective_charge_ionisation_profiles",
    "plasma_composition",
]


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

    **`i_plasma_ignited` was an `eqx.field(static=True)` here and is gone**
    (`_audit/next_steps.md` §14.2): this is the family base, and one occupant per value
    sits below it. `composition.md`'s "switches touched" section recorded the kwarg as a
    deliberate policy deviation -- the two branches' reads-sets genuinely differ, but
    the differing part is two lines inside an otherwise-shared 328-line function -- and
    that deviation is withdrawn with the policy that allowed it. The shared 328 lines
    did not have to be duplicated: they are `_plasma_composition`, and the two lines are
    the arm's own beam-ion-density law.
    """

    nd_plasma_alphas_thermal_vol_avg = OutputInto(physics)
    nd_plasma_protons_vol_avg = OutputInto(physics)
    nd_beam_ions = OutputInto(physics)
    nd_plasma_fuel_ions_vol_avg = OutputInto(physics)
    f_nd_impurity_electron_array_h = Output(
        impurity_radiation.f_nd_impurity_electron_array[H_INDEX]
    )
    """`f_nd_impurity_electron_array[0]` (PROCESS display label
    `f_nd_impurity_electrons(01)` per `naming_convention.md` § "Array elements" --
    record both, they are not the same thing). The `H_` fraction `plasma_composition`
    computes and writes."""
    f_nd_impurity_electron_array_he = Output(
        impurity_radiation.f_nd_impurity_electron_array[HE_INDEX]
    )
    """`f_nd_impurity_electron_array[1]` (display label `f_nd_impurity_electrons(02)`).
    The `He` fraction `plasma_composition` computes and writes."""
    nd_plasma_impurities_vol_avg = OutputInto(physics)
    nd_plasma_ions_total_vol_avg = OutputInto(physics)
    f_nd_plasma_carbon_electron = OutputInto(physics)
    f_nd_plasma_oxygen_electron = OutputInto(physics)
    f_nd_plasma_iron_argon_electron = OutputInto(physics)
    n_charge_plasma_effective_vol_avg = OutputInto(physics)
    f_alpha_electron = OutputInto(physics)
    f_alpha_ion = OutputInto(physics)
    m_fuel_amu = OutputInto(physics)
    m_beam_amu = OutputInto(physics)
    m_ions_total_amu = OutputInto(physics)
    n_charge_plasma_effective_mass_weighted_vol_avg = OutputInto(physics)

    def _composition(
        self,
        arm,
        nd_plasma_electrons_vol_avg,
        f_nd_alpha_thermal_electron,
        fusden_alpha_total,
        f_nd_protium_electrons,
        proton_rate_density,
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
        """The array assembly and result reshaping both `i_plasma_ignited`
        occupants share, given the arm function that occupant is for.

        Not a port surface: `_params` reads `__call__`'s signature only
        (`ExplicitFunction._signature_of`), so what each occupant declares is still
        its own parameter list -- which is the point of the split, since the ignited
        one does not declare `.physics.f_nd_beam_electron`.
        """
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

        results = arm(
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


class PlasmaCompositionIgnited(PlasmaComposition):
    """`i_plasma_ignited == IGNITED` (1) -- the reference run's.

    **One read leaves with this occupant**: `.physics.f_nd_beam_electron`. An
    ignited plasma has no beam ions, so `nd_beam_ions` is `zeros_like` and the
    beam fraction is never consulted.
    """

    def __call__(
        self,
        nd_plasma_electrons_vol_avg=From(physics),
        f_nd_alpha_thermal_electron=From(physics),
        fusden_alpha_total=From(physics),
        f_nd_protium_electrons=From(physics),
        proton_rate_density=From(physics),
        f_nd_impurity_electron_array_2=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[2]
        ),
        f_nd_impurity_electron_array_3=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[3]
        ),
        f_nd_impurity_electron_array_4=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[4]
        ),
        f_nd_impurity_electron_array_5=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[5]
        ),
        f_nd_impurity_electron_array_6=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[6]
        ),
        f_nd_impurity_electron_array_7=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[7]
        ),
        f_nd_impurity_electron_array_8=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[8]
        ),
        f_nd_impurity_electron_array_9=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[9]
        ),
        f_nd_impurity_electron_array_10=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[10]
        ),
        f_nd_impurity_electron_array_11=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[11]
        ),
        f_nd_impurity_electron_array_12=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[12]
        ),
        f_nd_impurity_electron_array_13=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[13]
        ),
        temp_plasma_electron_vol_avg_kev=From(physics),
        temp_impurity_keV_array=From(impurity_radiation),
        impurity_arr_zav=From(impurity_radiation),
        f_plasma_fuel_deuterium=From(physics),
        f_plasma_fuel_tritium=From(physics),
        f_plasma_fuel_helium3=From(physics),
        f_temp_plasma_electron_density_vol_avg=From(physics),
        f_beam_tritium=From(current_drive),
        m_impurity_amu_array=From(impurity_radiation),
    ):
        return self._composition(
            plasma_composition_ignited,
            nd_plasma_electrons_vol_avg,
            f_nd_alpha_thermal_electron,
            fusden_alpha_total,
            f_nd_protium_electrons,
            proton_rate_density,
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


class PlasmaCompositionNonIgnited(PlasmaComposition):
    """`i_plasma_ignited == NON_IGNITED` (0) -- PROCESS's own default
    (`physics_variables.py:881`) and the conventional tokamak's.

    Reads `.physics.f_nd_beam_electron`, which its sibling does not: beam ions
    are that fraction of the electron density.
    """

    def __call__(
        self,
        nd_plasma_electrons_vol_avg=From(physics),
        f_nd_alpha_thermal_electron=From(physics),
        fusden_alpha_total=From(physics),
        f_nd_protium_electrons=From(physics),
        proton_rate_density=From(physics),
        f_nd_beam_electron=From(physics),
        f_nd_impurity_electron_array_2=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[2]
        ),
        f_nd_impurity_electron_array_3=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[3]
        ),
        f_nd_impurity_electron_array_4=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[4]
        ),
        f_nd_impurity_electron_array_5=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[5]
        ),
        f_nd_impurity_electron_array_6=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[6]
        ),
        f_nd_impurity_electron_array_7=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[7]
        ),
        f_nd_impurity_electron_array_8=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[8]
        ),
        f_nd_impurity_electron_array_9=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[9]
        ),
        f_nd_impurity_electron_array_10=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[10]
        ),
        f_nd_impurity_electron_array_11=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[11]
        ),
        f_nd_impurity_electron_array_12=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[12]
        ),
        f_nd_impurity_electron_array_13=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[13]
        ),
        temp_plasma_electron_vol_avg_kev=From(physics),
        temp_impurity_keV_array=From(impurity_radiation),
        impurity_arr_zav=From(impurity_radiation),
        f_plasma_fuel_deuterium=From(physics),
        f_plasma_fuel_tritium=From(physics),
        f_plasma_fuel_helium3=From(physics),
        f_temp_plasma_electron_density_vol_avg=From(physics),
        f_beam_tritium=From(current_drive),
        m_impurity_amu_array=From(impurity_radiation),
    ):
        return self._composition(
            functools.partial(
                plasma_composition_non_ignited,
                f_nd_beam_electron=f_nd_beam_electron,
            ),
            nd_plasma_electrons_vol_avg,
            f_nd_alpha_thermal_electron,
            fusden_alpha_total,
            f_nd_protium_electrons,
            proton_rate_density,
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


class CalculateEffectiveChargeIonisationProfiles(ExplicitFunction):
    """cottax node: `calculate_effective_charge_ionisation_profiles`, ports declared.

    `temp_plasma_electron_profile_kev` is read `From(physics)`
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

    n_charge_plasma_effective_profile = OutputInto(physics)
    n_charge_impurity_profile = OutputInto(impurity_radiation)

    def __call__(
        self,
        temp_plasma_electron_profile_kev=From(physics),
        f_nd_impurity_electron_array_0=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[0]
        ),
        f_nd_impurity_electron_array_1=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[1]
        ),
        f_nd_impurity_electron_array_2=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[2]
        ),
        f_nd_impurity_electron_array_3=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[3]
        ),
        f_nd_impurity_electron_array_4=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[4]
        ),
        f_nd_impurity_electron_array_5=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[5]
        ),
        f_nd_impurity_electron_array_6=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[6]
        ),
        f_nd_impurity_electron_array_7=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[7]
        ),
        f_nd_impurity_electron_array_8=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[8]
        ),
        f_nd_impurity_electron_array_9=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[9]
        ),
        f_nd_impurity_electron_array_10=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[10]
        ),
        f_nd_impurity_electron_array_11=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[11]
        ),
        f_nd_impurity_electron_array_12=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[12]
        ),
        f_nd_impurity_electron_array_13=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[13]
        ),
        temp_impurity_keV_array=From(impurity_radiation),
        impurity_arr_zav=From(impurity_radiation),
    ):
        return effective_charge_ionisation_profiles_from_indexed_impurities(
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
        )
