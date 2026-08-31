"""Pure-functional port of the two functions from
`process/models/physics/impurity_radiation.py` that unit #20 (`radiation_power.py`) did
not need.

Registry unit #23. Audit record:
`functional_process/_audit/units/models/physics/impurity_radiation.md`.

Most of this file's source range -- `ImpurityRadiation`, `create_f_rad_core_profile`,
`calculate_impurity_radiation_power_density` (roughly L379-755) -- is **already ported**,
in `functional_process/models/physics/radiation_power.py`
(`ImpurityRadiationTotals`/`PlasmaRadiationPowers` and the two helper functions they
wrap). It was found and closed while unit #20 audited `calculate_radiation_powers`'s own
`ImpurityRadiation(...).calculate_imprad()` call -- see that module's docstring and
`radiation_power.md`'s "Scope correction" section. **Not duplicated here**; nothing in
this file re-defines it.

What is left, and what this file ports:

- `calculate_average_charge_at_temp` -- `_calculate_average_charge_at_temp_compiled`'s
  `@njit` body merged into its shell (the same move
  `calculate_impurity_radiation_power_density`'s port already made for its sibling) --
  the temperature-dependent average ionic charge <Z>(T_e), per species.
- `element2index` -- a species-label lookup.

Both are reached from `physics.py`'s `plasma_composition`/
`calculate_effective_charge_ionisation_profiles` (registry unit #9), not from
`radiation_power.calculate_radiation_powers`'s own call path -- the scope correction
`unit_registry.md` row #23 records (`calculate_average_charge_at_temp` is not on unit
#20's path at all; `radiation_power.md` says so explicitly). Neither function gets a
cottax node here: their outputs are consumed *inside* loops in those two unit-#9 methods,
never returned as a single `data.<area>.<field>` value by the function alone, so node
ownership belongs to unit #9's own audit, not this one -- see `impurity_radiation.md`'s
"cottax node" section. This file closes the `data` back door and ports the arithmetic;
wiring either function into a node is unit #9's decision to make.

### `initialise_imprad`'s four tables -- vendored constants, not nodes

`initialise_imprad` / `init_imp_element` / `read_impurity_file` (L27-376) are still not
*ported*: they are one-time startup I/O, and their outputs are graph constants, not
values flowing along a runtime edge. That was unit #20's and this record's original
reasoning and it has not changed.

What *has* changed is that a native solve environment has to hold their results
(`_audit/next_steps.md` §22.8): `initialise_imprad` runs at `process/main.py:430`, ahead
of `init_process`, and it is `init_audit.md` §5's "fifth initialisation source". Four
`data.impurity_radiation` fields come out of it and out of nothing else, and a
`NativeState` that answered them with the dataclass default answered four all-zero
tables -- which cost every tokamak its solve.

**They are vendored constants here, and each of the four for its own reason:**

- `temp_impurity_keV_array`, `pden_impurity_lz_nd_temp_array`, `impurity_arr_zav` --
  atomic data read off `process/data/lz_non_corona_14_elements/*_lz_tau.dat` and
  `*_z_tau.dat`. They depend on **nothing**: not on the input file, not on any switch,
  not on any iteration variable. There is no argument for which they could be the output
  of a node, because a node with no inputs whose value never changes *is* a constant.
  Vendored as `functional_process/data/impurity_tables.npz` rather than as a re-port of
  `read_impurity_file`, so the port needs neither the `process` package's data directory
  nor its ad-hoc `C `-comment/regex parser at runtime -- §23.2's standing rule, and
  `tests/functional_process/test_provider.py` asserts every element equal to what a live
  `initialise_imprad` writes.
- `m_impurity_amu_array` -- fourteen atomic masses from `process/core/constants.py`,
  vendored as a literal below because fourteen numbers read better as fourteen numbers.

**What is deliberately *not* answered from this module**:
`f_nd_impurity_electron_array`. `init_imp_element` writes it too (`1.0` for hydrogen, `0`
for the rest), but `init.py:381-384`'s alias loop overwrites all fourteen elements from
`f_nd_impurity_electrons` *afterwards*, so `initialise_imprad`'s value is dead on every
run. It is `native.py`'s alias derivation that answers it, from a field the state
already holds. Likewise `impurity_arr_label`, `impurity_arr_z` and
`impurity_arr_len_tab`: `composition.py` already resolved the two of those it needs to
constants, and nothing reads the third.
"""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np

N_IMPURITIES = 14
"""Species in the impurity arrays (`impurity_radiation_variables.N_IMPURITIES`)."""

IMPURITY_LABELS = (
    "H_", "He", "Be", "C_", "N_", "O_", "Ne", "Si", "Ar", "Fe", "Ni", "Kr", "Xe", "W_",
)  # fmt: skip
"""`impurity_arr_label`, in `initialise_imprad`'s own fixed order -- the order every
other table on this page is indexed by. Equal to `imp_label`'s dataclass default, which
is what `initialise_imprad` copies it from."""

M_IMPURITY_AMU_ARRAY = (
    1.00782503223,  # M_PROTIUM_AMU        1H
    4.002602,  # M_HELIUM_AMU        (3He, 4He) average
    9.0121831,  # M_BERYLLIUM_AMU      9Be
    12.0096,  # M_CARBON_AMU        (12C, 13C, 14C) average
    14.00643,  # M_NITROGEN_AMU      (14N, 15N) average
    15.99903,  # M_OXYGEN_AMU        (16O, 17O, 18O) average
    20.1797,  # M_NEON_AMU          (20Ne, 21Ne, 22Ne) average
    28.084,  # M_SILICON_AMU
    39.948,  # M_ARGON_AMU
    55.845,  # M_IRON_AMU
    58.6934,  # M_NICKEL_AMU
    83.798,  # M_KRYPTON_AMU
    131.293,  # M_XENON_AMU
    183.84,  # M_TUNGSTEN_AMU
)
"""`.impurity_radiation.m_impurity_amu_array`, as `initialise_imprad` fills it.

Vendored from `process/core/constants.py`'s `M_*_AMU` names, one per species in
`IMPURITY_LABELS` order -- the fourteen `m_species_amu=` arguments of
`initialise_imprad`'s fourteen `init_imp_element` calls, in call order. Asserted equal
to PROCESS's own in `test_provider.py`."""

_TABLES_NPZ = Path(__file__).resolve().parents[2] / "data" / "impurity_tables.npz"


@lru_cache(maxsize=1)
def impurity_tables() -> dict[str, np.ndarray]:
    """The three `(14, 200)` atomic-data tables `initialise_imprad` reads from disk.

    Returns
    -------
    dict
        `temp_impurity_keV_array` (species temperature abscissae, keV),
        `pden_impurity_lz_nd_temp_array` (L(Z, Te)) and `impurity_arr_zav` (<Z>(Te)),
        each `(14, 200)` and indexed by `IMPURITY_LABELS`' order.

    The temperature axis is stored once rather than fourteen times: `init_imp_element`
    reads a `Te[eV]` row per species and all fourteen files carry the *same* 200
    temperatures, so PROCESS's `(14, 200)` array holds one row repeated -- measured, and
    re-asserted every time this loads. `_audit/next_steps.md` §22.8 records why these
    are vendored rather than re-read.
    """
    with np.load(_TABLES_NPZ) as loaded:
        temperature = np.asarray(loaded["temp_impurity_keV"], dtype=float)
        lz = np.asarray(loaded["pden_impurity_lz_nd_temp"], dtype=float)
        zav = np.asarray(loaded["impurity_arr_zav"], dtype=float)
    return {
        "temp_impurity_keV_array": np.broadcast_to(
            temperature, (N_IMPURITIES, temperature.size)
        ).copy(),
        "pden_impurity_lz_nd_temp_array": lz,
        "impurity_arr_zav": zav,
    }


def calculate_average_charge_at_temp(
    temp_electron_kev, temp_impurity_kev, impurity_arr_zav
):
    """One species' temperature-dependent average ionic charge, <Z>(T_e).

    Ports `impurity_radiation._calculate_average_charge_at_temp_compiled`
    (`impurity_radiation.py:437-510`) for a single species, with its table row passed in
    rather than indexed out of `data.impurity_radiation` -- the same move
    `radiation_power.calculate_impurity_radiation_power_density` already made for its
    sibling function.

    Log-*x*, linear-*y* interpolation -- unlike the sibling function, which is log-log:
    <Z> is interpolated linearly, only the temperature axis is logged.

    **Two things in the source are dropped, both dead code, both already found for the
    sibling function** (see `radiation_power.md`'s open question 3):

    - the `np.digitize` block computes `indices` and never uses the result;
    - `impurity_arr_len_tab[i] - 1` is used as the top-of-table index, but the
      interpolation always uses the full 200-wide row regardless, so a
      shorter-than-200 table was never actually supported by this function. This port
      drops `len_tab` from the signature and indexes `[-1]`.

    **The explicit boundary clamps are also dropped, but for a different reason than
    the sibling's -- they are not dead code here, they are provably redundant.**
    `jnp.interp` (like `np.interp`) already clamps `x` outside `xp`'s range to the
    corresponding endpoint `fp` value; the source's explicit
    `n_charge_impurity_average[mask] = impurity_arr_zav[i, 0 or -1]` assignments
    overwrite the interpolation's result with *exactly the value it already produced*.
    This is the opposite situation from
    `calculate_impurity_radiation_power_density`'s clamps, which overwrite a
    log-log-interpolated-then-exponentiated value with the raw, un-exponentiated,
    wrong-unit table entry -- a real bug (see that function's docstring). Here the
    interpolation is already linear in the clamped quantity, so the clamp is a no-op by
    construction -- confirmed, not just argued, by this port's value-agreement test
    passing with the assignments omitted.

    Parameters
    ----------
    temp_electron_kev :
        Electron temperature(s) (keV) to evaluate <Z> at. PROCESS's own callers
        (`physics.py`) always pass a length-1 array and `.squeeze()` the result --
        despite the source's `np.array | float` type hint, a true scalar would raise on
        the boolean-mask assignment PROCESS's own version performs. This port accepts
        whatever shape is given; the boundary behaviour documented above holds either
        way.
    temp_impurity_kev :
        The species' L(Z, Te) table temperatures (keV), ascending.
    impurity_arr_zav :
        The species' average-charge table, same length, same abscissae.

    Returns
    -------
    :
        <Z>(T_e), same shape as `temp_electron_kev`.
    """
    return jnp.interp(
        jnp.log(temp_electron_kev),
        jnp.log(temp_impurity_kev),
        impurity_arr_zav,
    )


def element2index(element, impurity_arr_label):
    """Index of a species label in the 14-entry impurity species array.

    Ports `impurity_radiation.element2index` (`impurity_radiation.py:605-627`), with the
    label array passed in rather than indexed out of `data.impurity_radiation`.

    Not a numerical computation -- a lookup over `impurity_arr_label`, which is a
    compile-time constant (`initialise_imprad` populates it from the fixed 14-species
    order in `ImpurityRadiationData.imp_label`'s default; see the module docstring's
    "out of scope" section). Both arguments are declared `static_argnames` in this
    unit's harness case: neither is a continuous quantity a solver could perturb, so a
    derivative with respect to either is not something that exists.

    Parameters
    ----------
    element :
        Species label, e.g. `"H_"`, `"Ar"`.
    impurity_arr_label :
        The 14-entry label array (`data.impurity_radiation.impurity_arr_label`).

    Returns
    -------
    int
        The species' index, `0`-`13`.

    Raises
    ------
    ValueError
        If `element` is not one of `impurity_arr_label`'s entries. PROCESS raises
        `ProcessValueError` for the same condition; this port raises a plain
        `ValueError` instead, to avoid importing `process.core.exceptions` into a
        pure-port module (this package's ported files depend only on `cottax`/`jax`).
        Not a JAX-traced code path either way: `element2index` is resolved once, on
        Python values, at the same graph-assembly-time status
        `naming_convention.md` gives a topology switch -- it is a lookup a caller does
        before tracing starts, not a node in the traced graph.
    """
    labels = [str(label) for label in impurity_arr_label]
    try:
        return labels.index(str(element))
    except ValueError as exc:
        raise ValueError(
            f"element {element!r} is not found in impurity_arr_label"
        ) from exc
