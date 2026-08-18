"""Pure-functional port of `Physics.plasma_composition` and
`Physics.calculate_effective_charge_ionisation_profiles`
(`process/models/physics/physics.py`).

Registry unit #9, chunk B. Audit record:
`functional_process/models/physics/physics_B_composition.md` -- read it first,
especially "the `first_call` self-loop" and "the `znfuel` raise" sections. The
`first_call` self-loop's *node* shape (`NextFirstCall`/`PlasmaComposition` below) is now
resolved via `FixedPointFunction`; the `znfuel` domain check remains deliberately
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

import equinox as eqx
import jax
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FixedPointFunction,
    Input,
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


def next_first_call(first_call):
    """The next value of `.physics.first_call`, after one call to `plasma_composition`.

    Extracted from `plasma_composition`'s body (not reimplemented) because it is the
    one piece of that function's computation that reads a `VarPath` it also writes --
    `.physics.first_call` itself -- the Shape-B self-loop `physics_B_composition.md`'s
    "the `first_call` self-loop" section describes. `plasma_composition` below calls
    this helper for its own `first_call_next` return value, and `NextFirstCall` (the
    `FixedPointFunction` node further down this module) calls it again for `step` --
    same formula, same helper, two different node bindings of the one `VarPath`.

    PROCESS writes `.physics.first_call = 0` *only inside* the `first_call == 1`
    branch (`physics.py:1387`) -- not unconditionally. The other branch leaves the
    field alone, so this is a genuine pass-through there
    (`d(next_first_call)/d(first_call) == 1`), not a constant reset
    (`test_gradient_agreement` caught an earlier draft that got this wrong).

    Note what this does *not* depend on: nothing else in `plasma_composition`. Unlike
    `pc`/`pc_bootstrap` (which shares the same `first_call == 1` branch condition but
    also needs `alphan`/`alphat`), `next_first_call` is a pure function of `first_call`
    alone -- confirmed by reading `physics.py:1381-1387` directly, not assumed. So
    isolating it into its own `FixedPointFunction` node costs nothing extra in inputs;
    it is not a partial view into a larger coupled computation, just this one `VarPath`.
    """
    return jnp.where(first_call == 1, 0, first_call)


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
    first_call,
    alphan,
    alphat,
    f_temp_plasma_electron_density_vol_avg,
    f_beam_tritium,
    m_impurity_amu_array,
    is_ignited,
):
    """Plasma component fractional makeup -- density, charge and mass bookkeeping.

    Ports `Physics.plasma_composition` (`physics.py:1166-1491`). Every `self.data`
    read/write in the source becomes an explicit argument/return value, in the order
    PROCESS computes them; three things could not be ported as straight translations,
    each documented in full in the audit record's own section:

    - **the `first_call` self-loop**: PROCESS reads `.physics.first_call`, uses it to
      pick between two formulas for one intermediate (`pc`), and *only on the
      `first_call == 1` branch* writes it back as `0` -- the other branch leaves it
      untouched (a real pass-through, not a no-op: `d(first_call_next)/d(first_call) ==
      1` there, caught by `test_gradient_agreement` when an earlier draft of this port
      wrote an unconditional `0`). A node cannot read a `VarPath` it also owns
      (`~/jaxgraph/CLAUDE.md`, "The graph" -- "a node may not read what it owns"), so
      this function takes `first_call` as an ordinary input and returns the next value
      (`first_call_next`, via the `next_first_call` helper below) as an ordinary
      return value under a different name -- **this module now splits that pair
      across two node classes**, `NextFirstCall` (`FixedPointFunction`, owns
      `.physics.first_call`) and `PlasmaComposition` (`ExplicitFunction`, everything
      else, reads `first_call` as a plain cross-node `Input`); see both classes' own
      docstrings and the audit record.
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
    first_call :
        `1` on PROCESS's very first call to this function in a run, else `0`. See the
        docstring section above.
    alphan, alphat :
        Density/temperature profile indices, used only by the `first_call` branch's `pc`
        estimate.
    f_temp_plasma_electron_density_vol_avg :
        Profile-derived `pc` value, used only by the non-`first_call` branch. Produced
        by `physics/plasma_profiles.py` (registry unit #12, already ported).
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
        first_call_next, f_alpha_electron, f_alpha_ion, m_fuel_amu, m_beam_amu,
        m_ions_total_amu, n_charge_plasma_effective_mass_weighted_vol_avg)`, matching
        the order PROCESS computes them in.
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

    first_call_is_bootstrap = first_call == 1
    pc_bootstrap = (1.0 + alphan) * (1.0 + alphat) / (1.0 + alphan + alphat)
    pc = jnp.where(
        first_call_is_bootstrap, pc_bootstrap, f_temp_plasma_electron_density_vol_avg
    )
    # PROCESS writes `.physics.first_call = 0` *only inside* the `first_call == 1`
    # branch (physics.py:1387) -- not unconditionally, as an earlier draft of this
    # module's docstring claimed (caught by `test_gradient_agreement`: an unconditional
    # `0` gives d(first_call_next)/d(first_call) == 0 everywhere, but PROCESS's own
    # branch structure makes the *other* branch pass `first_call` through unchanged, so
    # the true derivative there is 1). Ported faithfully as a pass-through on the
    # non-bootstrap branch. Same formula as `next_first_call` above -- called here,
    # not duplicated, so the two node bindings below (`plasma_composition`'s own
    # ordinary return, and `NextFirstCall.step`) can never drift apart.
    first_call_next = next_first_call(first_call)

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
        first_call_next,
        f_alpha_electron,
        f_alpha_ion,
        m_fuel_amu,
        m_beam_amu,
        m_ions_total_amu,
        n_charge_plasma_effective_mass_weighted_vol_avg,
    )


class NextFirstCall(FixedPointFunction):
    """cottax node: the `.physics.first_call` self-loop, split out of
    `plasma_composition` -- see that function's own docstring and the module
    docstring's "the `first_call` self-loop" cross-reference, and
    `physics_B_composition.md`'s dedicated section.

    `.physics.first_call` is read by `plasma_composition` (to pick between two `pc`
    formulas) and, on that same read's `first_call == 1` branch only, overwritten with
    `0` -- one `VarPath`, read and written by the same underlying PROCESS call. A
    single `cottax` node cannot own a `VarPath` it also reads
    (`~/jaxgraph/CLAUDE.md`, "The graph": *"a node may not read what it owns"*) --
    confirmed the hard way this session by `to_graph(Avail(...))` raising exactly this
    error for an unrelated unit's identical shape. `FixedPointFunction` is the
    structural admission requirement this shape needs: `step` reads the real
    `.physics.first_call` and mints `^cond.physics.first_call`; the `FixedPoint`
    problem node this class also declares (`node_definitions_and_names`'s second
    element, bound at `^problem.NextFirstCall`) reads that minted copy and owns the
    real `.physics.first_call`.

    This is a **structural admission only** -- declaring the shape so it can sit in a
    `Graph` at all, not a decision to ever drive it. No solver/step algorithm is
    assigned here; `FixedPoint.declared`/`declared_outside_cycles` exist for exactly
    this ("perfectly valid to sit undriven in the graph", `next_steps.md` §5).

    Deliberately its own node, not folded into `PlasmaComposition` below: bundling the
    two would force `plasma_composition`'s other 17 ordinary outputs to be owned by
    this `FixedPoint` problem node as well, even though none of them need iterating --
    only `.physics.first_call` does. `step` reuses `next_first_call` rather than
    reimplementing its formula, so this node and `PlasmaComposition`'s own
    `first_call_next` (computed but not owned there, see below) can never disagree.
    """

    first_call = Output(lambda s: s.physics.first_call)

    def step(self, first_call=Input(lambda s: s.physics.first_call)):
        return next_first_call(first_call)


class PlasmaComposition(ExplicitFunction):
    """cottax node: `plasma_composition`'s ordinary outputs -- everything **except**
    `.physics.first_call`, which `NextFirstCall` above owns instead.

    `first_call` is bound here as a perfectly ordinary `Input` -- the *current* value,
    read like any other cross-node input, with no write-back. `plasma_composition`
    (the ported function) still computes and returns `first_call_next` internally
    (needed for `pc`'s branch selection, computed as a side effect either way), but
    this node's `__call__` drops that element before returning, since this node does
    not declare `.physics.first_call` as one of its `Output`s -- `NextFirstCall`
    declares it, once, alone.

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
    f_nd_impurity_electron_array = Output(
        lambda s: s.impurity_radiation.f_nd_impurity_electron_array
    )
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
        nd_plasma_electrons_vol_avg=Input(
            lambda s: s.physics.nd_plasma_electrons_vol_avg
        ),
        f_nd_alpha_thermal_electron=Input(
            lambda s: s.physics.f_nd_alpha_thermal_electron
        ),
        fusden_alpha_total=Input(lambda s: s.physics.fusden_alpha_total),
        f_nd_protium_electrons=Input(lambda s: s.physics.f_nd_protium_electrons),
        proton_rate_density=Input(lambda s: s.physics.proton_rate_density),
        f_nd_beam_electron=Input(lambda s: s.physics.f_nd_beam_electron),
        f_nd_impurity_electron_array=Input(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array
        ),
        temp_plasma_electron_vol_avg_kev=Input(
            lambda s: s.physics.temp_plasma_electron_vol_avg_kev
        ),
        temp_impurity_keV_array=Input(
            lambda s: s.impurity_radiation.temp_impurity_keV_array
        ),
        impurity_arr_zav=Input(lambda s: s.impurity_radiation.impurity_arr_zav),
        f_plasma_fuel_deuterium=Input(lambda s: s.physics.f_plasma_fuel_deuterium),
        f_plasma_fuel_tritium=Input(lambda s: s.physics.f_plasma_fuel_tritium),
        f_plasma_fuel_helium3=Input(lambda s: s.physics.f_plasma_fuel_helium3),
        first_call=Input(lambda s: s.physics.first_call),
        alphan=Input(lambda s: s.physics.alphan),
        alphat=Input(lambda s: s.physics.alphat),
        f_temp_plasma_electron_density_vol_avg=Input(
            lambda s: s.physics.f_temp_plasma_electron_density_vol_avg
        ),
        f_beam_tritium=Input(lambda s: s.current_drive.f_beam_tritium),
        m_impurity_amu_array=Input(
            lambda s: s.impurity_radiation.m_impurity_amu_array
        ),
    ):
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
            first_call,
            alphan,
            alphat,
            f_temp_plasma_electron_density_vol_avg,
            f_beam_tritium,
            m_impurity_amu_array,
            self.is_ignited,
        )
        # Drop `first_call_next` (index 11 of `plasma_composition`'s 18-tuple) -- this
        # node does not own `.physics.first_call`, `NextFirstCall` does.
        return results[:11] + results[12:]


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

    `plasma_composition` is split across two node classes above -- `PlasmaComposition`
    (its 17 ordinary outputs) and `NextFirstCall` (the `.physics.first_call` Shape-B
    self-loop, admitted via `FixedPointFunction`) -- see the audit record's "the
    `first_call` self-loop" section and those two classes' own docstrings.
    """

    n_charge_plasma_effective_profile = Output(
        lambda s: s.physics.n_charge_plasma_effective_profile
    )
    n_charge_impurity_profile = Output(
        lambda s: s.impurity_radiation.n_charge_impurity_profile
    )

    def __call__(
        self,
        temp_electron_profile_kev=Input(
            lambda s: s.physics.temp_plasma_electron_profile_kev
        ),
        f_nd_impurity_electron_array=Input(
            lambda s: s.impurity_radiation.f_nd_impurity_electron_array
        ),
        temp_impurity_keV_array=Input(
            lambda s: s.impurity_radiation.temp_impurity_keV_array
        ),
        impurity_arr_zav=Input(lambda s: s.impurity_radiation.impurity_arr_zav),
    ):
        return calculate_effective_charge_ionisation_profiles(
            temp_electron_profile_kev,
            f_nd_impurity_electron_array,
            temp_impurity_keV_array,
            impurity_arr_zav,
        )
