"""Harness cases for the ported tokamak bootstrap-current chain.

Port: `functional_process/cottax/physics/bootstrap_current.py`. Audit record:
`functional_process/_audit/units/models/physics/bootstrap_current.md`.

**Three kinds of oracle here, and the split is the point.**

- Five of the Sauter scaling's internals -- the trapped-particle fraction, the two local
  poloidal betas and the two collisionalities -- are already pure `@staticmethod`s or
  `data`-free instance methods in `process/`, so `reference` is the PROCESS callable
  itself (bound to a throwaway `SauterBootstrapCurrent()` where PROCESS made it an
  instance method for no reason).
- `bootstrap_fraction_sauter` is diffed against **the real stateful PROCESS body**:
  `_reference_bootstrap_fraction_sauter` builds a real `DataStructure` and a real
  `PlasmaProfile` with real `NeProfile`/`TeProfile` sub-models, populates exactly the
  fields the audit record lists, and calls `SauterBootstrapCurrent.
  bootstrap_fraction_sauter(plasma_profile)` -- the "close the `data` back-door"
  technique. If the audit's read set were incomplete, the omitted field would sit at its
  dataclass default and the value test would disagree loudly. The two sub-model `run()`
  methods are stubbed to no-ops and `profile_y` set directly, exactly as
  `test_radiation_power.py` does and for the same reason: the profiles are
  `models/physics/profiles.py`'s outputs, produced upstream in the graph, and re-running
  them here would overwrite the arrays a case controls.
- The two `physics.py` bookkeeping functions have **no PROCESS callable at all** -- they
  are written inline in `Physics.run()`, a 7000-line method that cannot be entered
  without the whole plasma model. Their `reference` is therefore a *second reading of
  the source*, which is a weaker oracle than everything else in this file and is labelled
  as such on each contract. `test_reference_arm_matches_recorded_mfile` supplies the
  oracle those contracts cannot: `large_tokamak_eval`'s own recorded MFILE numbers, from
  a real PROCESS run, through the whole capped-fraction-to-auxiliary-fraction chain.

**`triang` is pinned, not fuzzed, and that is a read-set claim under test.** Four PROCESS
functions here take `triang` and the port takes it nowhere: it reaches only
`_trapped_particle_fraction_sauter`'s `fit == 2` branch, which no call site selects. Each
adapter pins it to `large_tokamak_eval`'s `+0.5`, and
`test_reference_is_invariant_to_triangularity` moves it across zero and asserts the
PROCESS reference does not notice -- which is what "the live chain does not read
`.physics.triang`" means operationally.

**Six registered boundary defects, all one site.** The three `_calculate_l*` contracts
fail `test_gradient_finite_at_zero` on `inverse_q` (all three), `sqeps` (`l31`,
`l31_32`) and `rmajor` (`l31` alone), because `_electron_collisionality_sauter` divides
by `|inverse_q * sqeps**3 * sqrt(tempe) * 1.875e7|` (`bootstrap_current.py:1728-1733`):
zero the denominator and the collisionality is `inf`, while every consumer has the shape
`f/(a + b*inf)` -- a finite `0` with a `nan` tangent. That is
`_harness/boundary.py`'s unguarded-division class, recorded there rather than repaired,
and each contract's docstring says which arguments it registers and why. The same three
functions are also exercised transitively, in value **and** in gradient, by
`TestBootstrapFractionSauter`, whose own zero-boundary probe is clean -- the scalars that
would zero `inverse_q` or `sqeps` there (`q0`, `q95`, `rminor`) make the whole value
non-finite, which the check steps aside for.
"""

import numpy as np
import pytest
from cottax.interfaces.pytree_namespace_module import resolve, to_graph
from cottax.spec import VarPath

from functional_process.cottax._harness import Tier1Contract, legacy_sample
from functional_process.cottax.physics.bootstrap_current import (
    NoDiamagneticCurrent,
    NoPfirschSchluterCurrent,
    PlasmaCurrentFractions,
    SauterBootstrapCurrentFraction,
    _beta_poloidal_sauter,
    _beta_poloidal_total_sauter,
    _calculate_l31_32_coefficient,
    _calculate_l31_coefficient,
    _calculate_l34_alpha_31_coefficient,
    _electron_collisionality_sauter,
    _ion_collisionality_sauter,
    _trapped_particle_fraction_sauter,
    bootstrap_fraction_sauter,
    calculate_plasma_current_fractions,
    diamagnetic_fraction_scene,
    enforce_bootstrap_current_fraction_max,
    ps_fraction_scene,
)
from functional_process.cottax.paths import physics
from process.core.model import DataStructure
from process.models.physics.bootstrap_current import SauterBootstrapCurrent
from process.models.physics.physics import ps_fraction_scene as process_ps_fraction_scene
from process.models.physics.plasma_current import PlasmaDiamagneticCurrent
from process.models.physics.plasma_profiles import PlasmaProfile
from process.models.physics.profiles import NeProfile, TeProfile

_AUDIT_RECORD = "models/physics/bootstrap_current.md"

_N_POINTS = 11
"""Profile length for every case in this file. PROCESS's default is 201.

Nothing in this unit's integral is parity-sensitive -- `bootstrap_fraction_sauter`
integrates with a plain rectangle sum over `arange(2, n)` annuli
(`bootstrap_current.py:1608`), not with Simpson's rule -- so `n` only sets how many
components `jacfwd` and PROCESS's finite difference have to walk. 11 keeps a gradient run
over three profile arrays affordable; `test_bootstrap_fraction_sauter_at_process_default`
checks the same code at `n = 501` in value.
"""

_TRIANG = 0.5
"""`large_tokamak_eval.IN.DAT`'s triangularity, pinned in every PROCESS adapter.

The port reads it nowhere; see the module docstring and
`test_reference_is_invariant_to_triangularity`.
"""

# `large_tokamak_nof.IN.DAT`'s converged point, lifted verbatim from
# `tests/unit/models/physics/test_physics.py::test_bootstrap_fraction_sauter`'s only
# parametrisation -- the same PROCESS unit test whose recorded expectation
# `_LEGACY_EXPECTED_BFS` below pins.
_LEGACY_PHYSICS = {
    "nd_plasma_ions_total_vol_avg": 7.1297522422781575e19,
    "rminor": 2.6666666666666665,
    "temp_plasma_separatrix_kev": 0.10000000000000001,
    "temp_plasma_ion_vol_avg_kev": 12.570861186498382,
    "q0": 1.0,
    "m_ions_total_amu": 2.5,
    "n_charge_plasma_effective_vol_avg": 2.5211399464385624,
    "radius_plasma_pedestal_density_norm": 0.9400000000000001,
    "b_plasma_toroidal_on_axis": 5.326133750416047,
    "plasma_current": 16528278.760008096,
    "a_plasma_poloidal": 38.39822223637151,
    "f_plasma_fuel_helium3": 0.0,
    "temp_plasma_pedestal_kev": 5.5,
    "nd_plasma_electrons_vol_avg": 8.016748468651018e19,
    "temp_plasma_electron_vol_avg_kev": 12.570861186498382,
    "rmajor": 8.0,
    "q95": 3.5,
    "nd_plasma_separatrix_electron": 3.6992211545476006e19,
    "temp_plasma_electron_on_axis_kev": 25.986118047669795,
    "nd_plasma_pedestal_electron": 6.2886759627309195e19,
    "tbeta": 2.0,
    "nd_plasma_electron_on_axis": 1.054474759840606e20,
    "alphan": 1.0,
    "radius_plasma_pedestal_temp_norm": 0.9400000000000001,
    "alphat": 1.45,
}

_LEGACY_EXPECTED_BFS = 0.4052168782500341
"""The recorded expectation of `tests/unit/models/physics/test_physics.py::
test_bootstrap_fraction_sauter`, at `n_plasma_profile_elements = 501`.

Unlike `plasma_current.md`'s **D5**, this one is not stale: PROCESS reproduces it to the
last bit today (measured in `process_port`), so it is usable as an answer and not only as
a sample point. `test_bootstrap_fraction_sauter_at_process_default` asserts against it.
"""


def _profile_arrays(n_plasma_profile_elements):
    """The three profile arrays PROCESS itself produces at `_LEGACY_PHYSICS`.

    Generated by running the real `PlasmaProfile` (`i_plasma_pedestal = 1`, the value
    `large_tokamak_eval.IN.DAT:291` sets) rather than transcribed as literals, so the
    sample is a point on PROCESS's own pedestal profile and stays one if that model
    changes. This is the ported profile chain's output in the assembled graph
    (`profiles.py`'s `ProfileGrid` for the grid, `plasma_profiles.py`'s
    `PedestalProfileValues` for the pair), which is why this unit declares reads on it
    instead of re-deriving it.
    """
    data = DataStructure()
    for name, value in _LEGACY_PHYSICS.items():
        setattr(data.physics, name, value)
    data.physics.triang = _TRIANG
    data.physics.n_plasma_profile_elements = n_plasma_profile_elements

    ne_profile, te_profile = NeProfile(), TeProfile()
    ne_profile.data = data
    te_profile.data = data
    model = PlasmaProfile(ne_profile, te_profile)
    model.data = data
    model.run()

    return (
        np.asarray(ne_profile.profile_x, dtype=float),
        np.asarray(ne_profile.profile_y, dtype=float),
        np.asarray(te_profile.profile_y, dtype=float),
    )


_RHO_NORM, _NE_PROFILE, _TE_PROFILE = _profile_arrays(_N_POINTS)

_RADIAL_ELEMENTS = np.arange(2, _N_POINTS)
"""`bootstrap_current.py:1532`'s index array, from 2 because the coefficient functions
should return 0 at `j == 1`. Static everywhere it appears: it is a loop bound, not a
value."""

_RHO = np.sqrt(_LEGACY_PHYSICS["a_plasma_poloidal"] / np.pi) * _RHO_NORM
_SQEPS = np.sqrt(_RHO_NORM * (_LEGACY_PHYSICS["rminor"] / _LEGACY_PHYSICS["rmajor"]))
_NE_19 = _NE_PROFILE * 1e-19
_NI_19 = (
    _LEGACY_PHYSICS["nd_plasma_ions_total_vol_avg"]
    / _LEGACY_PHYSICS["nd_plasma_electrons_vol_avg"]
) * _NE_19
_TEMPE = _TE_PROFILE
_TEMPI = (
    _LEGACY_PHYSICS["temp_plasma_ion_vol_avg_kev"]
    / _LEGACY_PHYSICS["temp_plasma_electron_vol_avg_kev"]
) * _TE_PROFILE
_ZEFF = np.full(_N_POINTS, _LEGACY_PHYSICS["n_charge_plasma_effective_vol_avg"])
_AMAIN = np.full(_N_POINTS, _LEGACY_PHYSICS["m_ions_total_amu"])
_ZMAIN = np.full(_N_POINTS, 1.0 + _LEGACY_PHYSICS["f_plasma_fuel_helium3"])
_INVERSE_Q = 1.0 / (
    _LEGACY_PHYSICS["q0"]
    + (_LEGACY_PHYSICS["q95"] - _LEGACY_PHYSICS["q0"]) * _RHO_NORM**2
)
"""The intermediate profiles `bootstrap_fraction_sauter` builds at `:1494-1527`, rebuilt
here so the five sub-function contracts sample the same physical point the whole-function
one does rather than an invented neighbourhood."""

_SAUTER = SauterBootstrapCurrent()
"""A `SauterBootstrapCurrent` with no `data` bound, on purpose.

Four of the six PROCESS callables used as references below are instance methods that
touch no `self` state at all (`bootstrap_current.py:1653`, `:1684`, `:1779`, `:1828`);
binding no `DataStructure` is what proves it -- an accidental `self.data` read would
raise `AttributeError` rather than quietly succeed.
"""


def _band(profile, low=0.85, high=1.15):
    """Per-component fuzz bounds as a multiplicative band around a real profile.

    Drawing each point independently inside +-15% keeps positivity (every `log` in this
    unit needs it) and keeps the profile monotone enough that `_gradient`'s stencil sees
    a physical shape, without pinning the port to one profile family.
    """
    return (profile * low, profile * high)


# ---------------------------------------------------------------------------
# Reference adapters
# ---------------------------------------------------------------------------


def _reference_trapped_particle_fraction_sauter(radial_elements, sqeps):
    """`SauterBootstrapCurrent._trapped_particle_fraction_sauter` at its default `fit`.

    `triang` is pinned and `fit` left at its `0` default -- the ASTRA branch, the one
    any call site in `process/` selects. The port takes neither argument; see the module
    docstring.
    """
    return SauterBootstrapCurrent._trapped_particle_fraction_sauter(
        radial_elements, _TRIANG, sqeps
    )


def _reference_calculate_l31_coefficient(
    radial_elements,
    number_of_elements,
    rmajor,
    b_plasma_toroidal_on_axis,
    ne,
    ni,
    tempe,
    tempi,
    inverse_q,
    rho,
    zeff,
    sqeps,
):
    """`SauterBootstrapCurrent._calculate_l31_coefficient` with `triang` pinned.

    PROCESS's fifth positional parameter is `triang` (`bootstrap_current.py:1833`),
    which the port does not take. Pinning it here and fuzzing everything else is the
    read-set claim under test: if this arm read it, the value test would disagree the
    moment a fuzz draw moved the eleven arguments that remain.
    """
    return _SAUTER._calculate_l31_coefficient(
        radial_elements,
        number_of_elements,
        rmajor,
        b_plasma_toroidal_on_axis,
        _TRIANG,
        ne,
        ni,
        tempe,
        tempi,
        inverse_q,
        rho,
        zeff,
        sqeps,
    )


def _reference_calculate_l31_32_coefficient(
    radial_elements,
    number_of_elements,
    rmajor,
    b_plasma_toroidal_on_axis,
    ne,
    ni,
    tempe,
    tempi,
    inverse_q,
    rho,
    zeff,
    sqeps,
):
    """`SauterBootstrapCurrent._calculate_l31_32_coefficient`, `triang` pinned.

    Same shape as `_reference_calculate_l31_coefficient`; this one additionally calls
    `_calculate_l31_coefficient` internally (`bootstrap_current.py:2086-2100`), so the
    contract covers the composition as well as the `F32_ee`/`F32_ei` polynomials.
    """
    return _SAUTER._calculate_l31_32_coefficient(
        radial_elements,
        number_of_elements,
        rmajor,
        b_plasma_toroidal_on_axis,
        _TRIANG,
        ne,
        ni,
        tempe,
        tempi,
        inverse_q,
        rho,
        zeff,
        sqeps,
    )


def _reference_calculate_l34_alpha_31_coefficient(
    radial_elements,
    number_of_elements,
    rmajor,
    b_plasma_toroidal_on_axis,
    inverse_q,
    sqeps,
    tempi,
    tempe,
    amain,
    zmain,
    ni,
    ne,
    rho,
    zeff,
):
    """`SauterBootstrapCurrent._calculate_l34_alpha_31_coefficient`, `triang` pinned.

    The one contract that reaches `_ion_collisionality_sauter` through the caller that
    mislabels its argument -- `zmain` is passed into a parameter PROCESS's signature
    calls `zeff` (audit record **D1**) -- so both names appear in this signature, with
    `zmain` carrying `1 + f_plasma_fuel_helium3` and `zeff` the effective charge.
    """
    return _SAUTER._calculate_l34_alpha_31_coefficient(
        radial_elements,
        number_of_elements,
        rmajor,
        b_plasma_toroidal_on_axis,
        _TRIANG,
        inverse_q,
        sqeps,
        tempi,
        tempe,
        amain,
        zmain,
        ni,
        ne,
        rho,
        zeff,
    )


def _reference_bootstrap_fraction_sauter(
    n_plasma_profile_elements,
    radius_plasma_profile_norm,
    nd_plasma_electron_profile,
    temp_plasma_electron_profile_kev,
    a_plasma_poloidal,
    rminor,
    rmajor,
    nd_plasma_ions_total_vol_avg,
    nd_plasma_electrons_vol_avg,
    temp_plasma_ion_vol_avg_kev,
    temp_plasma_electron_vol_avg_kev,
    n_charge_plasma_effective_vol_avg,
    q0,
    q95,
    m_ions_total_amu,
    f_plasma_fuel_helium3,
    b_plasma_toroidal_on_axis,
    plasma_current,
    triang=_TRIANG,
):
    """`SauterBootstrapCurrent.bootstrap_fraction_sauter` on a real `DataStructure`.

    The sixteen `self.data.physics.*` reads the PROCESS body makes off its back door are
    exactly the arguments above (plus `triang`, which it threads to a branch that never
    runs). Nothing else is set on the `DataStructure`, so an unlisted read lands on a
    dataclass default and the value test would see it.

    `triang` is a keyword with a default so that
    `test_reference_is_invariant_to_triangularity` can vary it; the harness never passes
    it, because the port has no such parameter.
    """
    data = DataStructure()
    data.physics.n_plasma_profile_elements = int(n_plasma_profile_elements)
    data.physics.a_plasma_poloidal = a_plasma_poloidal
    data.physics.rminor = rminor
    data.physics.rmajor = rmajor
    data.physics.nd_plasma_ions_total_vol_avg = nd_plasma_ions_total_vol_avg
    data.physics.nd_plasma_electrons_vol_avg = nd_plasma_electrons_vol_avg
    data.physics.temp_plasma_ion_vol_avg_kev = temp_plasma_ion_vol_avg_kev
    data.physics.temp_plasma_electron_vol_avg_kev = temp_plasma_electron_vol_avg_kev
    data.physics.n_charge_plasma_effective_vol_avg = n_charge_plasma_effective_vol_avg
    data.physics.q0 = q0
    data.physics.q95 = q95
    data.physics.m_ions_total_amu = m_ions_total_amu
    data.physics.f_plasma_fuel_helium3 = f_plasma_fuel_helium3
    data.physics.b_plasma_toroidal_on_axis = b_plasma_toroidal_on_axis
    data.physics.plasma_current = plasma_current
    data.physics.triang = triang

    ne_profile, te_profile = NeProfile(), TeProfile()
    for obj in (ne_profile, te_profile):
        obj.data = data
        obj.run = lambda: None
        obj.profile_x = np.asarray(radius_plasma_profile_norm, dtype=float)
    ne_profile.profile_y = np.asarray(nd_plasma_electron_profile, dtype=float)
    te_profile.profile_y = np.asarray(temp_plasma_electron_profile_kev, dtype=float)

    plasma_profile = PlasmaProfile(ne_profile, te_profile)
    plasma_profile.data = data

    model = SauterBootstrapCurrent()
    model.data = data
    return model.bootstrap_fraction_sauter(plasma_profile)


def _reference_enforce_bootstrap_current_fraction_max(
    f_c_plasma_bootstrap, f_c_plasma_bootstrap_max
):
    """`process/models/physics/physics.py:546-556`, **transcribed**.

    The weakest oracle in this file, and deliberately labelled: the PROCESS original is
    four lines written inline in `Physics.run()`, which has no callable sub-shell, so
    there is nothing to import and call. What this contract checks is that the port
    agrees with a second reading of those four lines -- not that it agrees with PROCESS's
    own execution. `test_reference_arm_matches_recorded_mfile` is what supplies the
    second kind of evidence.

    The `i_bootstrap_current != USER_INPUT` guard of `physics.py:549-551` is not
    reproduced: it is answered by *which occupant exists* (the user-input arm has none),
    per `_audit/naming_convention.md` § "switches are not ports".
    """
    if f_c_plasma_bootstrap > f_c_plasma_bootstrap_max:
        return min(f_c_plasma_bootstrap, f_c_plasma_bootstrap_max)
    return f_c_plasma_bootstrap


def _reference_calculate_plasma_current_fractions(
    f_c_plasma_bootstrap,
    f_c_plasma_diamagnetic,
    f_c_plasma_pfirsch_schluter,
    f_c_plasma_non_inductive,
):
    """`process/models/physics/physics.py:558-588`, **transcribed**.

    Same caveat as `_reference_enforce_bootstrap_current_fraction_max`: six statements
    inline in `Physics.run()` with no callable of their own. The `.physics.err243` flag
    of `:569`/`:578` is reporting-only and not produced by either side.
    """
    f_c_plasma_internal = (
        f_c_plasma_bootstrap + f_c_plasma_diamagnetic + f_c_plasma_pfirsch_schluter
    )
    if f_c_plasma_internal > f_c_plasma_non_inductive:
        f_c_plasma_internal = min(f_c_plasma_internal, f_c_plasma_non_inductive)
    f_c_plasma_inductive = max(1.0e-10, 1.0e0 - f_c_plasma_non_inductive)
    f_c_plasma_auxiliary = f_c_plasma_non_inductive - f_c_plasma_internal
    return f_c_plasma_internal, f_c_plasma_auxiliary, f_c_plasma_inductive


# ---------------------------------------------------------------------------
# Contracts -- the Sauter scaling's internals
# ---------------------------------------------------------------------------


class TestTrappedParticleFractionSauter(Tier1Contract):
    """`_trapped_particle_fraction_sauter` -> the same, at `fit = 0`.

    The port's signature is two arguments against PROCESS's four: `fit` is a
    method-choice static kwarg with no `DataStructure` field behind it and no caller that
    passes it, and `triang` is read only by the `fit == 2` branch it therefore cannot
    reach.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_trapped_particle_fraction_sauter
    ported = _trapped_particle_fraction_sauter
    static_argnames = ("radial_elements",)

    samples = [
        legacy_sample(
            "trapped_particle_fraction_sauter-large_tokamak_nof",
            radial_elements=_RADIAL_ELEMENTS,
            sqeps=_SQEPS,
        ),
    ]

    fuzz_bounds = {"sqeps": _band(np.maximum(_SQEPS, 0.02), 0.8, 1.2)}
    fuzz_fixed = {"radial_elements": _RADIAL_ELEMENTS}


class TestElectronCollisionalitySauter(Tier1Contract):
    """`_electron_collisionality_sauter` -> the same, unchanged.

    Reached through `_electron_collisions_sauter` and `_coulomb_logarithm_sauter`, which
    have no contract of their own: they are one-line factors of this one with no
    independent caller, so a disagreement in either surfaces here and names itself.
    """

    audit_record = _AUDIT_RECORD
    reference = _SAUTER._electron_collisionality_sauter
    ported = _electron_collisionality_sauter
    static_argnames = ("radial_elements",)

    samples = [
        legacy_sample(
            "electron_collisionality_sauter-large_tokamak_nof",
            radial_elements=_RADIAL_ELEMENTS,
            rmajor=_LEGACY_PHYSICS["rmajor"],
            zeff=_ZEFF,
            inverse_q=_INVERSE_Q,
            sqeps=_SQEPS,
            tempe=_TEMPE,
            ne=_NE_19,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (6.0, 10.0),
        "zeff": _band(_ZEFF),
        "inverse_q": _band(_INVERSE_Q),
        "sqeps": _band(np.maximum(_SQEPS, 0.02), 0.8, 1.2),
        "tempe": _band(_TEMPE),
        "ne": _band(_NE_19),
    }
    fuzz_fixed = {"radial_elements": _RADIAL_ELEMENTS}


class TestIonCollisionalitySauter(Tier1Contract):
    """`_ion_collisionality_sauter` -> the same, unchanged.

    **`zeff` here is not the effective charge.** The live call site passes `zmain`
    (`bootstrap_current.py:2231-2233`) into a parameter PROCESS's signature calls `zeff`
    (`:1786`), so the sample supplies `1 + f_plasma_fuel_helium3`, which is what the
    scaling actually raises to the fourth power. Audit record **D1**.
    """

    audit_record = _AUDIT_RECORD
    reference = _SAUTER._ion_collisionality_sauter
    ported = _ion_collisionality_sauter
    static_argnames = ("radial_elements",)

    samples = [
        legacy_sample(
            "ion_collisionality_sauter-large_tokamak_nof",
            radial_elements=_RADIAL_ELEMENTS,
            rmajor=_LEGACY_PHYSICS["rmajor"],
            inverse_q=_INVERSE_Q,
            sqeps=_SQEPS,
            tempi=_TEMPI,
            amain=_AMAIN,
            zeff=_ZMAIN,
            ni=_NI_19,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (6.0, 10.0),
        "inverse_q": _band(_INVERSE_Q),
        "sqeps": _band(np.maximum(_SQEPS, 0.02), 0.8, 1.2),
        "tempi": _band(_TEMPI),
        "amain": _band(_AMAIN),
        "zeff": _band(np.maximum(_ZMAIN, 1.0)),
        "ni": _band(_NI_19),
    }
    fuzz_fixed = {"radial_elements": _RADIAL_ELEMENTS}


class TestBetaPoloidalSauter(Tier1Contract):
    """`_beta_poloidal_sauter` -> the same, unchanged.

    `nr` is static: it is the profile length, and the only thing it does here is decide a
    `where` arm that `radial_elements = arange(2, nr)` can never select. Audit record
    **D2**.
    """

    audit_record = _AUDIT_RECORD
    reference = staticmethod(SauterBootstrapCurrent._beta_poloidal_sauter)
    ported = _beta_poloidal_sauter
    static_argnames = ("radial_elements", "nr")

    samples = [
        legacy_sample(
            "beta_poloidal_sauter-large_tokamak_nof",
            radial_elements=_RADIAL_ELEMENTS,
            nr=_N_POINTS,
            rmajor=_LEGACY_PHYSICS["rmajor"],
            b_plasma_toroidal_on_axis=_LEGACY_PHYSICS["b_plasma_toroidal_on_axis"],
            ne=_NE_19,
            tempe=_TEMPE,
            inverse_q=_INVERSE_Q,
            rho=_RHO,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (6.0, 10.0),
        "b_plasma_toroidal_on_axis": (4.0, 7.0),
        "ne": _band(_NE_19),
        "tempe": _band(_TEMPE),
        "inverse_q": _band(_INVERSE_Q),
        "rho": _band(np.maximum(_RHO, 0.05), 0.9, 1.1),
    }
    fuzz_fixed = {"radial_elements": _RADIAL_ELEMENTS, "nr": _N_POINTS}


class TestBetaPoloidalTotalSauter(Tier1Contract):
    """`_beta_poloidal_total_sauter` -> the same, unchanged.

    See `TestBetaPoloidalSauter`.
    """

    audit_record = _AUDIT_RECORD
    reference = staticmethod(SauterBootstrapCurrent._beta_poloidal_total_sauter)
    ported = _beta_poloidal_total_sauter
    static_argnames = ("radial_elements", "nr")

    samples = [
        legacy_sample(
            "beta_poloidal_total_sauter-large_tokamak_nof",
            radial_elements=_RADIAL_ELEMENTS,
            nr=_N_POINTS,
            rmajor=_LEGACY_PHYSICS["rmajor"],
            b_plasma_toroidal_on_axis=_LEGACY_PHYSICS["b_plasma_toroidal_on_axis"],
            ne=_NE_19,
            ni=_NI_19,
            tempe=_TEMPE,
            tempi=_TEMPI,
            inverse_q=_INVERSE_Q,
            rho=_RHO,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (6.0, 10.0),
        "b_plasma_toroidal_on_axis": (4.0, 7.0),
        "ne": _band(_NE_19),
        "ni": _band(_NI_19),
        "tempe": _band(_TEMPE),
        "tempi": _band(_TEMPI),
        "inverse_q": _band(_INVERSE_Q),
        "rho": _band(np.maximum(_RHO, 0.05), 0.9, 1.1),
    }
    fuzz_fixed = {"radial_elements": _RADIAL_ELEMENTS, "nr": _N_POINTS}


_L_COEFFICIENT_SAMPLE = {
    "radial_elements": _RADIAL_ELEMENTS,
    "number_of_elements": _N_POINTS,
    "rmajor": _LEGACY_PHYSICS["rmajor"],
    "b_plasma_toroidal_on_axis": _LEGACY_PHYSICS["b_plasma_toroidal_on_axis"],
    "ne": _NE_19,
    "ni": _NI_19,
    "tempe": _TEMPE,
    "tempi": _TEMPI,
    "inverse_q": _INVERSE_Q,
    "rho": _RHO,
    "zeff": _ZEFF,
    "sqeps": _SQEPS,
}
"""The point `bootstrap_fraction_sauter` hands its two `grad(ln n_e)`/`grad(ln T_e)`
coefficient functions at `_LEGACY_PHYSICS`, rebuilt from `:1494-1527`'s intermediates."""

_L_COEFFICIENT_FUZZ = {
    "rmajor": (6.0, 10.0),
    "b_plasma_toroidal_on_axis": (4.0, 7.0),
    "ne": _band(_NE_19),
    "ni": _band(_NI_19),
    "tempe": _band(_TEMPE),
    "tempi": _band(_TEMPI),
    "inverse_q": _band(_INVERSE_Q),
    "rho": _band(np.maximum(_RHO, 0.05), 0.9, 1.1),
    "zeff": _band(_ZEFF),
    "sqeps": _band(np.maximum(_SQEPS, 0.02), 0.8, 1.2),
}

_L_COEFFICIENT_FIXED = {
    "radial_elements": _RADIAL_ELEMENTS,
    "number_of_elements": _N_POINTS,
}


class TestCalculateL31Coefficient(Tier1Contract):
    """`_calculate_l31_coefficient` -> the same, with `triang` pinned.

    **Registered at the zero boundary on three arguments**: `inverse_q` and `sqeps`,
    which are two factors of `_electron_collisionality_sauter`'s denominator
    (`bootstrap_current.py:1728-1733`), so zeroing either sends the collisionality to
    `+inf` and leaves `f31_teff = f_trapped / (a + b * inf)` a finite `0` with a `nan`
    tangent; and `rmajor`, which is the same site from the numerator side -- it zeroes
    the collisionality while `_beta_poloidal_total_sauter`'s `(rmajor / ...) ** 2` stays
    finite. All three are `_harness/boundary.py`'s unguarded-division class, recorded
    there rather than repaired: deciding what "the electron collisionality at infinite
    safety factor" should be is a modelling question, not a mechanical one.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_calculate_l31_coefficient
    ported = _calculate_l31_coefficient
    static_argnames = ("radial_elements", "number_of_elements")

    samples = [
        legacy_sample(
            "calculate_l31_coefficient-large_tokamak_nof", **_L_COEFFICIENT_SAMPLE
        ),
    ]

    fuzz_bounds = _L_COEFFICIENT_FUZZ
    fuzz_fixed = _L_COEFFICIENT_FIXED


class TestCalculateL3132Coefficient(Tier1Contract):
    """`_calculate_l31_32_coefficient` -> the same, with `triang` pinned.

    Calls `_calculate_l31_coefficient` and both local poloidal betas internally, so a
    disagreement in any of the three surfaces here as well as in its own contract. Same
    `inverse_q`/`sqeps` zero-boundary registration as `TestCalculateL31Coefficient`;
    `rmajor` is *not* registered here, because the extra
    `_beta_poloidal_sauter / _beta_poloidal_total_sauter` quotient cancels the factor
    that made it singular there.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_calculate_l31_32_coefficient
    ported = _calculate_l31_32_coefficient
    static_argnames = ("radial_elements", "number_of_elements")

    samples = [
        legacy_sample(
            "calculate_l31_32_coefficient-large_tokamak_nof", **_L_COEFFICIENT_SAMPLE
        ),
    ]

    fuzz_bounds = _L_COEFFICIENT_FUZZ
    fuzz_fixed = _L_COEFFICIENT_FIXED


class TestCalculateL34Alpha31Coefficient(Tier1Contract):
    """`_calculate_l34_alpha_31_coefficient` -> the same, with `triang` pinned.

    Argument order is PROCESS's own, which differs from the other two coefficient
    functions' and carries both `zmain` and `zeff` (audit record **D1**). `inverse_q`
    alone is registered at the zero boundary here: `sqeps` and `rmajor` are recovered by
    the `alpha` factor, which carries its own `sqeps`/`rmajor` dependence through
    `_ion_collisionality_sauter` and cancels the singular factor.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_calculate_l34_alpha_31_coefficient
    ported = _calculate_l34_alpha_31_coefficient
    static_argnames = ("radial_elements", "number_of_elements")

    samples = [
        legacy_sample(
            "calculate_l34_alpha_31_coefficient-large_tokamak_nof",
            radial_elements=_RADIAL_ELEMENTS,
            number_of_elements=_N_POINTS,
            rmajor=_LEGACY_PHYSICS["rmajor"],
            b_plasma_toroidal_on_axis=_LEGACY_PHYSICS["b_plasma_toroidal_on_axis"],
            inverse_q=_INVERSE_Q,
            sqeps=_SQEPS,
            tempi=_TEMPI,
            tempe=_TEMPE,
            amain=_AMAIN,
            zmain=_ZMAIN,
            ni=_NI_19,
            ne=_NE_19,
            rho=_RHO,
            zeff=_ZEFF,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (6.0, 10.0),
        "b_plasma_toroidal_on_axis": (4.0, 7.0),
        "inverse_q": _band(_INVERSE_Q),
        "sqeps": _band(np.maximum(_SQEPS, 0.02), 0.8, 1.2),
        "tempi": _band(_TEMPI),
        "tempe": _band(_TEMPE),
        "amain": _band(_AMAIN),
        "zmain": _band(np.maximum(_ZMAIN, 1.0)),
        "ni": _band(_NI_19),
        "ne": _band(_NE_19),
        "rho": _band(np.maximum(_RHO, 0.05), 0.9, 1.1),
        "zeff": _band(_ZEFF),
    }
    fuzz_fixed = _L_COEFFICIENT_FIXED


class TestBootstrapFractionSauter(Tier1Contract):
    """`bootstrap_fraction_sauter` -> `SauterBootstrapCurrent.bootstrap_fraction_sauter`,
    called on a real `DataStructure` with a real `PlasmaProfile`.

    **This is the contract the unit rests on.** Both sides return
    `(fraction, current_density_profile)`; `_as_array` flattens the pair the same way on
    both, so the 9-entry `jboot` array is compared component by component alongside the
    scalar. It exercises all three `_calculate_l*` coefficient functions, both local
    poloidal betas, both collisionalities, the trapped-particle fraction, the three
    logarithmic gradients and the annulus sum -- in value and, under `--fp-gradients`, in
    derivative.

    Sample is `tests/unit/models/physics/test_physics.py::
    test_bootstrap_fraction_sauter`'s only parametrisation, verbatim, at
    `n_plasma_profile_elements = 11` rather than its recorded 501 (see `_N_POINTS`);
    `test_bootstrap_fraction_sauter_at_process_default` runs the same code at 501 and
    asserts the recorded answer.

    `radius_plasma_profile_norm` is held fixed under fuzzing. It is
    `linspace(0, 1, n)` for a static `n` -- graph-assembly-time data, per
    `profiles.py`'s `ProfileGrid` -- and drawing it independently per component would
    produce a non-monotone grid on which `_gradient`'s stencil and PROCESS's
    `np.gradient` are both meaningless. It is still differentiated: it appears in the
    declared sample, so `test_gradient_agreement` walks all eleven of its components.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_bootstrap_fraction_sauter
    ported = bootstrap_fraction_sauter
    static_argnames = ("n_plasma_profile_elements",)

    samples = [
        legacy_sample(
            "bootstrap_fraction_sauter-large_tokamak_nof",
            n_plasma_profile_elements=_N_POINTS,
            radius_plasma_profile_norm=_RHO_NORM,
            nd_plasma_electron_profile=_NE_PROFILE,
            temp_plasma_electron_profile_kev=_TE_PROFILE,
            a_plasma_poloidal=_LEGACY_PHYSICS["a_plasma_poloidal"],
            rminor=_LEGACY_PHYSICS["rminor"],
            rmajor=_LEGACY_PHYSICS["rmajor"],
            nd_plasma_ions_total_vol_avg=_LEGACY_PHYSICS["nd_plasma_ions_total_vol_avg"],
            nd_plasma_electrons_vol_avg=_LEGACY_PHYSICS["nd_plasma_electrons_vol_avg"],
            temp_plasma_ion_vol_avg_kev=_LEGACY_PHYSICS["temp_plasma_ion_vol_avg_kev"],
            temp_plasma_electron_vol_avg_kev=_LEGACY_PHYSICS[
                "temp_plasma_electron_vol_avg_kev"
            ],
            n_charge_plasma_effective_vol_avg=_LEGACY_PHYSICS[
                "n_charge_plasma_effective_vol_avg"
            ],
            q0=_LEGACY_PHYSICS["q0"],
            q95=_LEGACY_PHYSICS["q95"],
            m_ions_total_amu=_LEGACY_PHYSICS["m_ions_total_amu"],
            f_plasma_fuel_helium3=_LEGACY_PHYSICS["f_plasma_fuel_helium3"],
            b_plasma_toroidal_on_axis=_LEGACY_PHYSICS["b_plasma_toroidal_on_axis"],
            plasma_current=_LEGACY_PHYSICS["plasma_current"],
        ),
    ]

    fuzz_bounds = {
        "nd_plasma_electron_profile": _band(_NE_PROFILE),
        "temp_plasma_electron_profile_kev": _band(_TE_PROFILE),
        "a_plasma_poloidal": (30.0, 50.0),
        "rminor": (2.0, 3.5),
        "rmajor": (6.0, 10.0),
        "nd_plasma_ions_total_vol_avg": (5.0e19, 9.0e19),
        "nd_plasma_electrons_vol_avg": (6.0e19, 1.0e20),
        "temp_plasma_ion_vol_avg_kev": (8.0, 18.0),
        "temp_plasma_electron_vol_avg_kev": (8.0, 18.0),
        "n_charge_plasma_effective_vol_avg": (1.5, 3.5),
        "q0": (0.8, 1.5),
        "q95": (2.5, 5.0),
        "m_ions_total_amu": (2.0, 3.0),
        "f_plasma_fuel_helium3": (0.0, 0.2),
        "b_plasma_toroidal_on_axis": (4.0, 7.0),
        "plasma_current": (1.0e7, 2.5e7),
    }
    fuzz_fixed = {
        "n_plasma_profile_elements": _N_POINTS,
        "radius_plasma_profile_norm": _RHO_NORM,
    }


# ---------------------------------------------------------------------------
# Contracts -- the `physics.py` current-fraction bookkeeping
# ---------------------------------------------------------------------------


class TestEnforceBootstrapCurrentFractionMax(Tier1Contract):
    """`enforce_bootstrap_current_fraction_max` -> `physics.py:546-556`, transcribed.

    See `_reference_enforce_bootstrap_current_fraction_max` for why the oracle is weaker
    here than anywhere else in this file, and
    `test_reference_arm_matches_recorded_mfile` for the compensating evidence.

    Both the sample and the fuzz bounds keep the two arguments well apart, so no point
    sits on the `min` kink where PROCESS's one-sided difference and `jnp.minimum`'s
    subgradient could legitimately disagree. The legacy point is
    `large_tokamak_eval`'s own: an uncapped Sauter fraction of `0.4056` against a
    `f_c_plasma_bootstrap_max` of `0.95` (`IN.DAT:121`), which is why that run's MFILE
    reports the enforced and the Sauter fractions as the same number.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_enforce_bootstrap_current_fraction_max
    ported = enforce_bootstrap_current_fraction_max

    samples = [
        legacy_sample(
            "enforce_bootstrap_current_fraction_max-large_tokamak_eval",
            f_c_plasma_bootstrap=0.405562281204731256,
            f_c_plasma_bootstrap_max=0.95,
        ),
        legacy_sample(
            "enforce_bootstrap_current_fraction_max-capped",
            f_c_plasma_bootstrap=0.97,
            f_c_plasma_bootstrap_max=0.9,
        ),
    ]

    fuzz_bounds = {
        "f_c_plasma_bootstrap": (0.1, 0.6),
        "f_c_plasma_bootstrap_max": (0.8, 0.99),
    }


class TestDiamagneticFractionScene(Tier1Contract):
    """`diamagnetic_fraction_scene` -> `PlasmaDiamagneticCurrent.
    diamagnetic_fraction_scene`, unchanged (`@nb.njit` dropped).

    A real PROCESS `@staticmethod` with the port's exact signature, so the oracle is the
    strongest kind in this file -- no adapter, no pinned arguments, no second reading of
    the source. That is worth stating next to `TestCalculatePlasmaCurrentFractions`
    below, whose reference is the weakest kind, because the two contracts sit in the same
    chain and are not equally well evidenced.

    Sample is `tests/unit/models/physics/test_physics.py::
    test_diamagnetic_fraction_scene`'s only point, verbatim (`beta = 0.15`, `q95 = 3.0`,
    `q0 = 1.0`, recorded expectation `0.0460`).

    Fuzz bounds keep `q0` away from zero: the `q95 / q0` quotient is PROCESS's own and is
    unguarded, so `q0 == 0` is outside the domain in value as well as in derivative.
    """

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaDiamagneticCurrent.diamagnetic_fraction_scene)
    ported = diamagnetic_fraction_scene

    samples = [
        legacy_sample(
            "diamagnetic_fraction_scene-unit_test",
            beta=0.15,
            q95=3.0,
            q0=1.0,
        ),
    ]

    fuzz_bounds = {
        "beta": (0.02, 0.35),
        "q95": (2.5, 8.0),
        "q0": (0.8, 3.0),
    }


class TestPsFractionScene(Tier1Contract):
    """`ps_fraction_scene` -> the module-level `ps_fraction_scene` in
    `process/models/physics/physics.py`, unchanged (`@nb.jit` dropped).

    Sample is `tests/unit/models/physics/test_physics.py::test_ps_fraction_scene`'s only
    point, verbatim (`beta = 0.15`, recorded expectation `-0.0135`).

    One argument, one multiplication -- the contract is here for completeness of the
    unit's coverage rather than because the expression is hard, and the zero-boundary
    probe passes trivially (a linear function is finite everywhere).
    """

    audit_record = _AUDIT_RECORD
    reference = staticmethod(process_ps_fraction_scene)
    ported = ps_fraction_scene

    samples = [
        legacy_sample("ps_fraction_scene-unit_test", beta=0.15),
    ]

    fuzz_bounds = {"beta": (0.02, 0.35)}


class TestCalculatePlasmaCurrentFractions(Tier1Contract):
    """`calculate_plasma_current_fractions` -> `physics.py:558-588`, transcribed.

    Same oracle caveat as above. The legacy point is `large_tokamak_eval`'s converged
    state: the Sauter bootstrap fraction it reports, zero diamagnetic and zero
    Pfirsch-Schluter fractions (`MFILE:14426-14427`, both switches at `0`), and the
    non-inductive fraction its `IN.DAT:283` carries as iteration variable 44.

    Fuzz keeps `bootstrap + diamagnetic + pfirsch_schluter` strictly below
    `f_c_plasma_non_inductive` and `f_c_plasma_non_inductive` well below `1`, so neither
    clamp sits on its kink.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_calculate_plasma_current_fractions
    ported = calculate_plasma_current_fractions

    samples = [
        legacy_sample(
            "calculate_plasma_current_fractions-large_tokamak_eval",
            f_c_plasma_bootstrap=0.405562281204731256,
            f_c_plasma_diamagnetic=0.0,
            f_c_plasma_pfirsch_schluter=0.0,
            f_c_plasma_non_inductive=0.4242184436680697,
        ),
    ]

    fuzz_bounds = {
        "f_c_plasma_bootstrap": (0.10, 0.30),
        "f_c_plasma_diamagnetic": (0.0, 0.02),
        "f_c_plasma_pfirsch_schluter": (0.0, 0.02),
        "f_c_plasma_non_inductive": (0.50, 0.80),
    }


# ---------------------------------------------------------------------------
# The claims no contract can carry
# ---------------------------------------------------------------------------


def test_reference_is_invariant_to_triangularity():
    """PROCESS's own Sauter body does not read `.physics.triang` on the live path.

    The port declares no `triang` read, which is only correct because
    `_trapped_particle_fraction_sauter` reaches it exclusively from the `fit == 2` branch
    (`bootstrap_current.py:2522`) and `fit` is `0` at every call site. Moving `triang`
    from `+0.5` to `-0.3` -- across zero, where the `fit == 2` correction
    `0.67 * (1 - 1.4 * triang * |triang|) * eps` changes sign of its own correction term
    -- must therefore change nothing at all, to the last bit.
    """
    kwargs = dict(TestBootstrapFractionSauter.samples[0].kwargs)
    pinned = _reference_bootstrap_fraction_sauter(**kwargs, triang=_TRIANG)
    moved = _reference_bootstrap_fraction_sauter(**kwargs, triang=-0.3)

    assert pinned[0] == moved[0]
    np.testing.assert_array_equal(np.asarray(pinned[1]), np.asarray(moved[1]))


def test_bootstrap_fraction_sauter_at_process_default():
    """The port reproduces the recorded expectation at PROCESS's own profile length.

    `_N_POINTS = 11` everywhere else keeps gradient runs affordable; this checks the same
    code at `n_plasma_profile_elements = 501`, the value
    `tests/unit/models/physics/test_physics.py::test_bootstrap_fraction_sauter` records,
    against that test's own literal expectation. Value only -- 501 points would make a
    `jacfwd` component walk pointless as well as slow.
    """
    rho_norm, ne_profile, te_profile = _profile_arrays(501)

    fraction, _ = bootstrap_fraction_sauter(
        n_plasma_profile_elements=501,
        radius_plasma_profile_norm=rho_norm,
        nd_plasma_electron_profile=ne_profile,
        temp_plasma_electron_profile_kev=te_profile,
        a_plasma_poloidal=_LEGACY_PHYSICS["a_plasma_poloidal"],
        rminor=_LEGACY_PHYSICS["rminor"],
        rmajor=_LEGACY_PHYSICS["rmajor"],
        nd_plasma_ions_total_vol_avg=_LEGACY_PHYSICS["nd_plasma_ions_total_vol_avg"],
        nd_plasma_electrons_vol_avg=_LEGACY_PHYSICS["nd_plasma_electrons_vol_avg"],
        temp_plasma_ion_vol_avg_kev=_LEGACY_PHYSICS["temp_plasma_ion_vol_avg_kev"],
        temp_plasma_electron_vol_avg_kev=_LEGACY_PHYSICS[
            "temp_plasma_electron_vol_avg_kev"
        ],
        n_charge_plasma_effective_vol_avg=_LEGACY_PHYSICS[
            "n_charge_plasma_effective_vol_avg"
        ],
        q0=_LEGACY_PHYSICS["q0"],
        q95=_LEGACY_PHYSICS["q95"],
        m_ions_total_amu=_LEGACY_PHYSICS["m_ions_total_amu"],
        f_plasma_fuel_helium3=_LEGACY_PHYSICS["f_plasma_fuel_helium3"],
        b_plasma_toroidal_on_axis=_LEGACY_PHYSICS["b_plasma_toroidal_on_axis"],
        plasma_current=_LEGACY_PHYSICS["plasma_current"],
    )
    assert float(fraction) == pytest.approx(_LEGACY_EXPECTED_BFS, rel=1e-13)


def test_reference_arm_matches_recorded_mfile():
    """The capped-fraction-to-auxiliary-fraction chain, against a real PROCESS run.

    Every number here is read off `tests/regression/input_files/large_tokamak_eval`:
    the Sauter fraction (`MFILE:6691`), the enforced bootstrap fraction (`:6901`), the
    zero diamagnetic and Pfirsch-Schluter fractions (`:6902`, `:6903`), the
    `f_c_plasma_bootstrap_max` input (`IN.DAT:121`), the non-inductive fraction
    (`IN.DAT:283`, iteration variable 44) and the two answers (`:14428`, `:14429`).

    This is the oracle `TestEnforceBootstrapCurrentFractionMax` and
    `TestCalculatePlasmaCurrentFractions` cannot be: their references are transcriptions
    of `Physics.run()`, and this is PROCESS's own converged output.
    """
    f_c_plasma_bootstrap_sauter = 4.05562281204731256e-01
    f_c_plasma_bootstrap_max = 0.95
    f_c_plasma_non_inductive = 0.4242184436680697

    f_c_plasma_bootstrap = enforce_bootstrap_current_fraction_max(
        f_c_plasma_bootstrap_sauter, f_c_plasma_bootstrap_max
    )
    assert float(f_c_plasma_bootstrap) == pytest.approx(
        4.05562281204731256e-01, rel=1e-15
    )

    internal, auxiliary, inductive = calculate_plasma_current_fractions(
        f_c_plasma_bootstrap=f_c_plasma_bootstrap,
        f_c_plasma_diamagnetic=0.0,
        f_c_plasma_pfirsch_schluter=0.0,
        f_c_plasma_non_inductive=f_c_plasma_non_inductive,
    )
    assert float(internal) == pytest.approx(4.05562281204731256e-01, rel=1e-15)
    assert float(auxiliary) == pytest.approx(1.86561624633384548e-02, rel=1e-12)
    assert float(inductive) == pytest.approx(5.75781556331930289e-01, rel=1e-14)


def test_nodes_assemble_and_the_sauter_arm_does_not_read_triangularity():
    """The four nodes assemble together, and `.physics.triang` is not among their reads.

    The structural half of `test_reference_is_invariant_to_triangularity`: that test
    shows PROCESS does not *use* the triangularity here, this one shows the port does not
    *declare* it -- which is what removes the edge from the graph rather than merely
    leaving it inert.
    """
    node = SauterBootstrapCurrentFraction(n_plasma_profile_elements=_N_POINTS)
    graph = to_graph(
        node,
        NoDiamagneticCurrent(),
        NoPfirschSchluterCurrent(),
        PlasmaCurrentFractions(),
    )
    assert len(graph.definitions) == 4

    read = {inp.var for inp in node.inputs}
    assert resolve(physics.triang, VarPath) not in read
    assert resolve(physics.rminor, VarPath) in read


def test_zero_arms_own_their_fields_and_state_them(reads_only_its_own_statement):
    """`NoDiamagneticCurrent`/`NoPfirschSchluterCurrent` are source nodes stating zero.

    They exist because PROCESS leaves both fields at their `current_drive_variables.py`
    defaults on this input (no `else` branch at `plasma_current.py:1081-1094` or
    `physics.py:538-541`, and neither field is settable from `IN.DAT`), so the honest
    shape is a declared zero rather than a boundary input standing for "PROCESS did not
    run this code".

    **The zero is no longer in the node.** It used to read nothing and return a
    `carried()` field; since `_audit/optimise_design.md` §34 it reads exactly one thing
    -- `^stated.<the place it owns>` -- and hands it back, with the value in
    `indat.STATED_VALUES`. The check that the zero is still a zero moved with it, to
    `test_stated.py`; what is asked here is the structure, which is the half this file
    is about.
    """
    for node in (NoDiamagneticCurrent(), NoPfirschSchluterCurrent()):
        reads_only_its_own_statement(node)
        assert float(node(0.0)) == pytest.approx(0.0, abs=0.0)
