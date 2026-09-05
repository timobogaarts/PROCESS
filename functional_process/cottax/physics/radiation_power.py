"""Pure-functional port of `process/models/physics/radiation_power.py`.

Registry unit #20. Audit record:
`functional_process/_audit/units/models/physics/radiation_power.md`.

PROCESS's entry point is

    calculate_radiation_powers(plasma_profile, <13 physics scalars>, data_structure)

-- two opaque object arguments, one of which is the whole `DataStructure`. Closing those
two back doors is the whole point of this unit, and the answer turns out to be small:

- **off `plasma_profile`**: three arrays and nothing else --
  `neprofile.profile_y` (electron density profile), `teprofile.profile_y` (electron
  temperature profile) and `neprofile.profile_x` (the normalised radius grid). A fourth,
  `neprofile.profile_dx`, is passed to `scipy.integrate.simpson` but **ignored**, because
  `x=` is given alongside it -- so it is not a dependency at all.
- **off `data_structure`**: six `.impurity_radiation` fields and one `.physics` one, all
  reads. `calculate_radiation_powers` writes **nothing** to `data_structure` --
  `ImpurityRadiation` accumulates only onto `self`, and the caller assigns the returned
  `RadpwrData` itself. The `data` argument is a pure read-through.

### Scope finding

Following that read set means this unit's real source is `radiation_power.py` (243 LOC)
**plus the `ImpurityRadiation` half of `process/models/physics/impurity_radiation.py`**
(756 LOC), which is **not in `unit_registry.md`**. See the record's § Scope correction.
Everything on the impurity path is ported here rather than deferred, because
`calculate_radiation_powers` is 6 lines of arithmetic wrapped around it -- porting the
wrapper alone would have ported nothing.

### What is deliberately not ported

`initialise_imprad` / `init_imp_element` / `read_impurity_file` -- the file readers that
populate the L(Z,Te) tables from `process/data/lz_non_corona_14_elements/*.dat`. They are
I/O run once at startup; their product is a compile-time constant of the graph, not a
value flowing along an edge. `calculate_average_charge_at_temp` is not on this path at
all (nothing in `calculate_radiation_powers` reaches it).

### A latent PROCESS bug, ported faithfully

`calculate_impurity_radiation_power_density`'s two out-of-table clamps assign the *loss
function* `L(Z, Te)` (units W m^3, ~1e-33) directly into the *power density* array (units
W/m^3, ~1e6) instead of clamping `L` and then multiplying by `f_nd * ne^2`. Out of range
the answer is therefore ~38 orders of magnitude too small -- effectively zero. See
`calculate_impurity_radiation_power_density`'s docstring and the record's § open
questions. Reproduced exactly here; **not** silently fixed.
"""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.paths import impurity_radiation, physics
from functional_process.models.physics.radiation_power import (
    calculate_impurity_radiation_power_density,
    calculate_impurity_radiation_totals,
    calculate_radiation_powers,
    combine_radiation_powers,
    impurity_radiation_totals_from_indexed_impurities,
    psync_albajar_fidone,
)

__all__ = [
    "calculate_impurity_radiation_power_density",
    "calculate_impurity_radiation_totals",
    "calculate_radiation_powers",
]


class SynchrotronRadiationPower(ExplicitFunction):
    """cottax node: `psync_albajar_fidone`, unchanged, ports declared.

    Every input is an ordinary `.physics` field and the single output is
    `.physics.pden_plasma_sync_mw`, a real `physics_variables.py` field. Nothing minted,
    no switch, no alternative arm -- this one is ready for `total_process.COMMON`.

    Note the output is *only* the synchrotron term. `stellarator.py:2147` and
    `physics.py` both assign it to `.physics.pden_plasma_sync_mw` verbatim, so the node's
    output and PROCESS's field are the same quantity with no post-processing.
    """

    pden_plasma_sync_mw = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_on_axis=From(physics),
        rminor=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        aspect=From(physics),
        alphan=From(physics),
        alphat=From(physics),
        tbeta=From(physics),
        temp_plasma_electron_on_axis_kev=From(physics),
        f_sync_reflect=From(physics),
        rmajor=From(physics),
        kappa=From(physics),
        vol_plasma=From(physics),
    ):
        return psync_albajar_fidone(
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
        )


class ImpurityRadiationTotals(ExplicitFunction):
    """cottax node: `calculate_impurity_radiation_totals`, ports declared.

    **Blocked from `total_process.py` on one thing** -- see `imp_indices` below and the
    record's § open questions 2. Everything else about the node is settled.

    **`f_nd_impurity_electron_array` is read as fourteen individual, index-addressed
    `FromExactly`s**, not one whole-array `FromExactly` -- the same per-index treatment
    `composition.PlasmaComposition`/`CalculateEffectiveChargeIonisationProfiles`
    now give the identical field (`SequenceKey`-addressed, matching the real
    `DataStructure` field's own `list[float]` storage, per `naming_convention.md` §
    "Array elements"). This is what this class's own `__call__` docstring used to flag
    as the open reason the array stayed whole: "the number of parameters would then vary
    with the configuration." **That is resolved, not just worked around**: the signature
    now declares all fourteen indices unconditionally -- the array's length is fixed at
    14 for every real `DataStructure` (`initialise_imprad`'s hardcoded per-species
    arguments never vary the count, only which entries are near-zero), so the *port
    count* no longer depends on `imp_indices` at all. `imp_indices` still selects which
    of the fourteen feed `calculate_impurity_radiation_totals`, exactly as before -- only
    now that selection happens on fourteen individually-named `VarPath`s instead of one
    array `VarPath`, via `jnp.stack` + a static gather inside `__call__`.

    This narrows, but does not close, § open questions 2: the residual blocker there is
    whether `imp_indices` can silently disagree with which fractions are actually
    near-zero *during a solve* (the `f_plasma_fuel_helium3 == 0 and
    f_nd_alpha_thermal_electron == 0` case) -- a data-dependent shape-stability question,
    not a signature-shape one. Per-index addressing does not touch that; see the record's
    updated § open questions 2 for exactly what remains.

    Three minted `VarPath`s, same justification as `ProfileFactors`' two in
    `plasma_profiles.py`:

    - `.physics.nd_plasma_electron_profile`, `.physics.temp_plasma_electron_profile_kev`
      -- **reused, not re-minted**: `plasma_profiles.ProfileFactors` already mints exactly
      these two for exactly these two arrays (`neprofile.profile_y`,
      `teprofile.profile_y`). This unit is the second consumer, which is the evidence
      those spellings were the right call.
    - `.physics.radius_plasma_profile_norm` -- **new here.** The normalised radius grid
      (`neprofile.profile_x`) likewise has no PROCESS storage; it is built by
      `Profile.initialise_profile_x`/`normalise_profile_x` (`profiles.py:60-84`) as
      `arange(n) / (n - 1)` and read straight off the object. Named as a sibling of the
      existing `.impurity_radiation.radius_plasma_core_norm`, which it is compared
      against. Its producer is unit #21 (`profiles.py`), still unported.

    `neprofile.profile_dx` is **not** an input, although PROCESS passes it to both
    `integrate.simpson` calls: `scipy` ignores `dx` whenever `x` is given. Declaring it
    would assert a dependency the computation does not have.

    The two outputs are minted too. `pden_impurity_rad_total_mw` /
    `pden_impurity_core_rad_total_mw` exist only as attributes of the `ImpurityRadiation`
    instance that `calculate_radiation_powers` constructs and discards; they are never
    stored on `data`. Minted into `.impurity_radiation` because that is the area whose
    tables and fractions produce them.
    """

    imp_indices: tuple[int, ...] = eqx.field(static=True)
    """Which of the 14 species are present -- a graph-assembly-time fact.

    PROCESS derives this per evaluation as
    `np.nonzero(f_nd_impurity_electron_array > 1e-30)[0]`
    (`impurity_radiation.py:650-652`). That is a data-dependent gather: neither the
    result's *shape* nor its contents are known to a tracer, so it cannot stay inside the
    node. Hoisting it to a static field is the same move `naming_convention.md` § "switches
    are not ports" prescribes for a topology-changing switch, and it is honest -- which
    impurity species a machine has is a design decision, not a state variable.

    **This is what blocks registration**, though for a narrower reason than it looks.
    Iteration variables 125-136 cover array indices 2-13 with bounds `(1e-8, 0.01)`, 22
    orders above the threshold, so the optimiser cannot deselect a species. But
    `physics.plasma_composition()` recomputes indices 0 (`H_`) and 1 (`He`) every
    evaluation -- on the stellarator path too, `stellarator.py:1910` -- and `He` is
    exactly `0.0` when `f_plasma_fuel_helium3` and `f_nd_alpha_thermal_electron` both are.
    That is a per-run configuration fact, so `imp_indices` *can* be resolved at assembly
    time; nothing in `configuration.py` does it or checks it yet. See the record's § open
    questions 2.

    **Not a signature-shape blocker any more.** Before per-index reads, this field's
    docstring on `__call__` also worried that addressing individual species one at a
    time would make the node's own *parameter count* vary with `imp_indices` -- it does
    not: `__call__` now always declares fourteen (one per index, unconditionally),
    and `imp_indices` only selects a static gather over them, exactly as it selected a
    gather over the one whole-array `FromExactly` before. Only the shape-*during-a-solve*
    question above remains open.
    """

    pden_impurity_rad_total_mw = OutputInto(impurity_radiation)
    pden_impurity_core_rad_total_mw = OutputInto(impurity_radiation)

    def __call__(
        self,
        radius_plasma_profile_norm=From(physics),
        nd_plasma_electron_profile=From(physics),
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
        pden_impurity_lz_nd_temp_array=From(impurity_radiation),
        radius_plasma_core_norm=From(impurity_radiation),
        f_p_plasma_core_rad_reduction=From(impurity_radiation),
    ):
        """Reassembles the fourteen individually-addressed fractions and selects
        `imp_indices` before forwarding.

        Each species' fraction is its own read (`SequenceKey`-addressed `VarPath`, one
        per index) rather than one whole-array read -- fourteen ports, always,
        regardless of `imp_indices`, so the signature stays fixed as
        `NodalDeclaration` requires. `imp_indices` only changes which of the fourteen
        are gathered out before being handed to `calculate_impurity_radiation_totals`
        -- the selection is a gather on the (now reassembled) read, not a second home
        for the arithmetic, same as before this change. The two 200-wide table
        arguments stay whole-array `FromExactly`s: they are compile-time constants, not
        per-species runtime values a caller would ever address individually.
        """
        return impurity_radiation_totals_from_indexed_impurities(
            self.imp_indices,
            radius_plasma_profile_norm,
            nd_plasma_electron_profile,
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
            pden_impurity_lz_nd_temp_array,
            radius_plasma_core_norm,
            f_p_plasma_core_rad_reduction,
        )


class PlasmaRadiationPowers(ExplicitFunction):
    """cottax node: `combine_radiation_powers`, unchanged, ports declared.

    Three real `.physics` fields out, one real `.physics` field plus the two
    `ImpurityRadiationTotals` mints in. Structurally ready; it can only be registered
    once `ImpurityRadiationTotals` is, since without that node its two impurity inputs
    have no producer.

    Note what this node does **not** do: `stellarator.py:2152-2159` clips
    `pden_plasma_core_rad_mw` and `pden_plasma_outer_rad_mw` at zero *after* assigning
    them. That `max(..., 0)` is in `st_phys` (unit #1, chunk 1B), not in
    `calculate_radiation_powers`, so it belongs to the caller's node; ported here it would
    double-own the fields. **The other caller,
    `physics.PhysicsCalculations.physics()` (`physics.py:750-753`), does not clip at
    all** -- the two call sites disagree, which is the whole reason the clip cannot live
    in this node: it is not a property of the radiation model, it is a property of one
    of its two callers.

    **So this node's two clipped outputs are minted `_unclipped` names**, and
    `plasma_physics.py`'s `ClippedRadiationPowers` owns the real
    `.physics.pden_plasma_core_rad_mw`/`pden_plasma_outer_rad_mw` fields. Before that
    split this node claimed the real fields while computing the *pre*-clip value -- a
    latent divergence from PROCESS, invisible only because the clip happens to be
    inactive on this run (measured: core `0.0575`, outer `0.0553`, both comfortably
    positive). `pden_plasma_rad_mw` keeps its real name: PROCESS never clips it
    (`stellarator.py:2151`), which is also what makes `KNOWN_MINT_VALUES`'s
    `pden_impurity_rad_total_mw` inversion sound where the core one is not.
    """

    pden_plasma_core_rad_mw_unclipped = OutputInto(physics)
    pden_plasma_outer_rad_mw_unclipped = OutputInto(physics)
    pden_plasma_rad_mw = OutputInto(physics)

    def __call__(
        self,
        pden_impurity_rad_total_mw=From(impurity_radiation),
        pden_impurity_core_rad_total_mw=From(impurity_radiation),
        pden_plasma_sync_mw=From(physics),
    ):
        return combine_radiation_powers(
            pden_impurity_rad_total_mw,
            pden_impurity_core_rad_total_mw,
            pden_plasma_sync_mw,
        )
