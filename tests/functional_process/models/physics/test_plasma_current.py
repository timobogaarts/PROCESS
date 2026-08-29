"""Harness cases for the ported tokamak plasma-current chain.

Six contracts, one per pure function in
`functional_process/models/physics/plasma_current.py`. Audit record:
`functional_process/_audit/units/models/physics/plasma_current.md`.

**Two kinds of oracle here, and the split is the point.**

- `calculate_cyclindrical_plasma_current`, `calculate_current_coefficient_ipdg89`,
  `calculate_cylindrical_safety_factor`, `calculate_current_profile_index_wesson` and
  `calculate_internal_inductance_wesson` need no adapter at all: each is already a pure
  `@staticmethod` (or module-level function) in `process/`, with the port's exact
  signature, so `reference` is the PROCESS callable itself. Three of the five live in
  `process/models/physics/physics.py` rather than `plasma_current.py` -- see the port's
  module docstring for why this unit spans two source files.
- `calculate_plasma_current_ipdg89` and the `i_ind_plasma_internal_norm == 1` selection
  are diffed against **the real stateful PROCESS bodies**, bound to a real
  `DataStructure` at `large_tokamak_eval.IN.DAT`'s own switch values -- the "close the
  `data` back-door" technique. `_reference_plasma_current_ipdg89` calls the whole
  nine-way `PlasmaCurrent.calculate_plasma_current` with `i_plasma_current = 4`, so the
  contract tests that the port reproduces *the arm PROCESS selects*, not a transcription
  of one branch. `_reference_ind_plasma_internal_norm_wesson` calls
  `PlasmaInductance.run()` end to end with `i_ind_plasma_internal_norm = 1`, which
  exercises the `model_map` selection (`physics.py:4759-4764`) as well as the formula.

**`_reference_plasma_current_ipdg89` is also the evidence for a read-set claim.** The
port's IPDG89 occupant declares seven reads; PROCESS's function takes fourteen
arguments. The adapter pins the other seven (`alphaj`, `alphap`,
`pres_plasma_on_axis`, `len_plasma_poloidal`, `kappa`, `triang`, and the switch) to
constants and the value test still agrees at machine precision across every fuzz point
-- which is what "the IPDG89 arm does not read them" means operationally. The union of
all nine arms' reads is the invented-edge defect this port exists to remove.

**Domain guards.** Every sample and fuzz bound keeps `triang95 >= 0` (PROCESS raises
for `triang < 0` unless `i_plasma_current == 8`, `plasma_current.py:305-309`; the pinned
`triang` in the adapter is `+0.5` for the same reason), `eps` well away from `1`
(`(1 - eps^2)^2` in the denominator of the IPDG89 coefficient), and `q0`,
`plasma_current`, `rmajor` away from zero.

**One registered boundary defect.** `TestCalculateCylindricalSafetyFactor`'s
`b_plasma_toroidal_on_axis` is listed in `_harness/boundary.py`'s
`DIVISION_BY_ZERO_AT_BOUNDARY`, so `test_gradient_finite_at_zero` excuses it rather than
failing. It is PROCESS's own division *by a quotient* (`physics.py:93-99`), kept
verbatim: finite in value and `nan` in derivative at zero field. See the audit record's
**D6** for the repair that exists and why the port does not apply it.
"""

from types import MappingProxyType

from functional_process._harness import Sample, Tier1Contract, legacy_sample
from functional_process.models.physics.plasma_current import (
    calculate_current_coefficient_fiesta,
    calculate_current_coefficient_ipdg89,
    calculate_current_profile_index_wesson,
    calculate_cyclindrical_plasma_current,
    calculate_cylindrical_safety_factor,
    calculate_internal_inductance_wesson,
    calculate_plasma_current_fiesta,
    calculate_plasma_current_ipdg89,
)
from process.core.model import DataStructure
from process.models.physics.physics import (
    Physics,
    PlasmaInductance,
)
from process.models.physics.physics import (
    calculate_cylindrical_safety_factor as process_calculate_cylindrical_safety_factor,
)
from process.models.physics.plasma_current import PlasmaCurrent

_AUDIT_RECORD = "models/physics/plasma_current.md"

# `i_plasma_current = 4` (IPDG89), `large_tokamak_eval.IN.DAT:288`.
_I_PLASMA_CURRENT_IPDG89 = 4
# `i_ind_plasma_internal_norm = 1` (Wesson), `large_tokamak_eval.IN.DAT:311`.
_I_IND_PLASMA_INTERNAL_NORM_WESSON = 1
# `i_plasma_current = 9` (FIESTA ST), `spherical_tokamak_eval.IN.DAT:288` and
# `st_regression.IN.DAT`.
_I_PLASMA_CURRENT_FIESTA = 9


def _reference_plasma_current_ipdg89(
    eps, kappa95, triang95, rminor, rmajor, q95, b_plasma_toroidal_on_axis
):
    """`PlasmaCurrent.calculate_plasma_current` at `i_plasma_current = 4`.

    The seven arguments the IPDG89 arm does not touch are pinned to
    `large_tokamak_eval`-plausible constants (`triang = +0.5`, keeping PROCESS's
    negative-triangularity guard quiet). If any of them were in fact read, this
    contract's value test would fail the moment fuzzing moved the seven it does declare.
    """
    return PlasmaCurrent().calculate_plasma_current(
        alphaj=1.0,
        alphap=0.0,
        b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
        eps=eps,
        i_plasma_current=_I_PLASMA_CURRENT_IPDG89,
        kappa=1.85,
        kappa95=kappa95,
        pres_plasma_on_axis=0.0,
        len_plasma_poloidal=24.081367139525412,
        q95=q95,
        rmajor=rmajor,
        rminor=rminor,
        triang=0.5,
        triang95=triang95,
    )


def _reference_plasma_current_fiesta(
    eps, kappa, triang, rminor, rmajor, q95, b_plasma_toroidal_on_axis
):
    """`PlasmaCurrent.calculate_plasma_current` at `i_plasma_current = 9`.

    The same "close the `data` back-door" adapter as the IPDG89 one above, with the
    complementary pin set: this arm reads the **separatrix** `kappa`/`triang`, so it is
    `kappa95`/`triang95` that are pinned to constants here, and if the FIESTA arm in fact
    read either of them the value test would fail the moment fuzzing moved the seven
    arguments it does declare. The two adapters together are the evidence that the
    family's two occupants have genuinely different read sets and not merely different
    constants.

    `triang` is a *live* argument here rather than a pinned `+0.5`, and every sample and
    fuzz bound keeps it strictly positive: `plasma_current.py:305-309` raises for
    `triang < 0` at every `i_plasma_current` except `8`, so a negative point has no
    PROCESS answer to agree with, and `triang == 0` is where `triang ** 0.060` has an
    infinite derivative (the port's `safe_pow` makes it `0` there; PROCESS's own finite
    difference cannot be asked what it should be).
    """
    return PlasmaCurrent().calculate_plasma_current(
        alphaj=1.0,
        alphap=0.0,
        b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
        eps=eps,
        i_plasma_current=_I_PLASMA_CURRENT_FIESTA,
        kappa=kappa,
        kappa95=1.6517857142857142,
        pres_plasma_on_axis=0.0,
        len_plasma_poloidal=24.081367139525412,
        q95=q95,
        rmajor=rmajor,
        rminor=rminor,
        triang=triang,
        triang95=0.3333333333333333,
    )


def _reference_ind_plasma_internal_norm_wesson(alphaj):
    """`PlasmaInductance.run()` at `i_ind_plasma_internal_norm = 1`, on a real `data`.

    Runs the whole method -- including the two scalings the Wesson arm does not select
    and the three reporting fields the port does not carry -- and reads back only
    `.physics.ind_plasma_internal_norm`. `kappa`, `plasma_current`, `vol_plasma`,
    `rmajor` and `b_plasma_surface_poloidal_average` are pinned to plausible non-zero
    values purely so `calculate_normalised_internal_inductance_iter_3`
    (`physics.py:4940-4945`, which divides by `c_plasma**2 * rmajor`) does not divide by
    zero on the way past; none of them reaches the selected value.
    """
    data = DataStructure()
    data.physics.alphaj = alphaj
    data.physics.kappa = 1.85
    data.physics.plasma_current = 1.8398455678867526e7
    data.physics.vol_plasma = 1888.0
    data.physics.rmajor = 8.0
    data.physics.b_plasma_surface_poloidal_average = 0.9
    data.physics.i_ind_plasma_internal_norm = _I_IND_PLASMA_INTERNAL_NORM_WESSON

    inductance = PlasmaInductance()
    inductance.data = data
    inductance.run()
    return data.physics.ind_plasma_internal_norm


class TestCalculateCyclindricalPlasmaCurrent(Tier1Contract):
    """`calculate_cyclindrical_plasma_current` -> the same, unchanged.

    No recorded legacy expectation in `tests/unit` for this function on its own; the
    sample is the operating point of `tests/unit/models/physics/test_physics.py::
    test_calculate_plasma_current` (the `large_tokamak_nof.IN.DAT` one), which is the
    point the whole chain below is anchored on. The harness computes "expected" by
    calling the
    reference, so the `Sample` carries no stored answer.
    """

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaCurrent.calculate_cyclindrical_plasma_current)
    ported = calculate_cyclindrical_plasma_current

    samples = [
        legacy_sample(
            "calculate_cyclindrical_plasma_current-large_tokamak_nof",
            rminor=2.6666666666666665,
            rmajor=8.0,
            q95=3.5,
            b_plasma_toroidal_on_axis=5.7,
        ),
    ]

    fuzz_bounds = {
        "rminor": (2.0, 3.5),
        "rmajor": (6.0, 10.0),
        "q95": (2.5, 5.0),
        "b_plasma_toroidal_on_axis": (4.0, 7.0),
    }


class TestCalculateCurrentCoefficientIpdg89(Tier1Contract):
    """`calculate_current_coefficient_ipdg89` -> the same, unchanged.

    Sample is the shaping half of the same `large_tokamak_nof` point.
    """

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaCurrent.calculate_current_coefficient_ipdg89)
    ported = calculate_current_coefficient_ipdg89

    samples = [
        legacy_sample(
            "calculate_current_coefficient_ipdg89-large_tokamak_nof",
            eps=0.33333333333333331,
            kappa95=1.6517857142857142,
            triang95=0.33333333333333331,
        ),
    ]

    fuzz_bounds = {
        "eps": (0.2, 0.45),
        "kappa95": (1.4, 2.0),
        "triang95": (0.05, 0.5),
    }


class TestCalculatePlasmaCurrentIpdg89(Tier1Contract):
    """`calculate_plasma_current_ipdg89` -> `PlasmaCurrent.calculate_plasma_current`
    with `i_plasma_current = 4`, called on a real instance.

    Sample is the first parametrisation of `tests/unit/models/physics/test_physics.py::
    test_calculate_plasma_current`, verbatim -- genuinely legacy, generated from
    `large_tokamak_nof.IN.DAT`. Its recorded expectation is
    `plasma_current = 18398455.678867526 A`; PROCESS today returns
    `18398455.670608774` for that input and the legacy test passes only on
    `pytest.approx`'s default `rel=1e-6` (audit record **D5**). Harmless here -- the
    harness computes "expected" by calling the reference, so the recorded literal is
    used as a sample point and never as an answer.

    That test file records **two** parametrisations at this switch value, differing only
    in `alphaj`/`alphap`/`pres_plasma_on_axis` and asserting the *same* current. Those
    three are precisely the arguments the IPDG89 arm ignores, so the pair is PROCESS's
    own unit tests already documenting this arm's read set -- the second point is not
    repeated here because for this port's seven declared arguments it is the same point.
    """

    audit_record = _AUDIT_RECORD
    reference = staticmethod(_reference_plasma_current_ipdg89)
    ported = calculate_plasma_current_ipdg89

    samples = [
        legacy_sample(
            "calculate_plasma_current_ipdg89-large_tokamak_nof",
            eps=0.33333333333333331,
            kappa95=1.6517857142857142,
            triang95=0.33333333333333331,
            rminor=2.6666666666666665,
            rmajor=8.0,
            q95=3.5,
            b_plasma_toroidal_on_axis=5.7,
        ),
    ]

    fuzz_bounds = {
        "eps": (0.2, 0.45),
        "kappa95": (1.4, 2.0),
        "triang95": (0.05, 0.5),
        "rminor": (2.0, 3.5),
        "rmajor": (6.0, 10.0),
        "q95": (2.5, 5.0),
        "b_plasma_toroidal_on_axis": (4.0, 7.0),
    }


class TestCalculateCurrentCoefficientFiesta(Tier1Contract):
    """`calculate_current_coefficient_fiesta` -> the same `@staticmethod`, unchanged
    except that `triang ** 0.060` is `safe_pow(triang, 0.060)`.

    **The `safe_pow` substitution is invisible to the value test by construction** --
    `models/safe_math.py` evaluates the identical expression for every `x != 0` -- and
    the domain guard below keeps every point off `triang == 0`, which is the only place
    the two differ at all (in derivative, never in value).

    No legacy point exists: `tests/unit/models/physics/test_physics.py` has no FIESTA
    case, so the sample is the operating point of `spherical_tokamak_eval.IN.DAT`
    itself -- `aspect = 1.8` (line 260) giving `eps = 1/1.8`, `kappa = 2.8` (285),
    `triang = 0.5` (296). Provenance `input-file`, not `legacy`, because it is derived
    from a tracked input and not lifted from a recorded expectation.
    """

    audit_record = _AUDIT_RECORD
    reference = staticmethod(PlasmaCurrent.calculate_current_coefficient_fiesta)
    ported = calculate_current_coefficient_fiesta

    samples = [
        Sample(
            MappingProxyType(
                {"eps": 1.0 / 1.8, "kappa": 2.8, "triang": 0.5},
            ),
            "input-file",
            "calculate_current_coefficient_fiesta-spherical_tokamak_eval",
        ),
    ]

    fuzz_bounds = {
        "eps": (0.35, 0.75),
        "kappa": (1.8, 3.2),
        "triang": (0.05, 0.6),
    }


class TestCalculatePlasmaCurrentFiesta(Tier1Contract):
    """`calculate_plasma_current_fiesta` -> `PlasmaCurrent.calculate_plasma_current`
    with `i_plasma_current = 9`, called on a real instance.

    Sample is `spherical_tokamak_eval.IN.DAT`'s own operating point: `aspect = 1.8`
    (line 260), `kappa = 2.8` (285), `triang = 0.5` (296), `rmajor = 4.5` (291) with
    `rminor = rmajor / aspect`, `q95 = 5.835830999686161` (286) and
    `b_plasma_toroidal_on_axis = 3.0` (262).

    **Spherical-tokamak fuzz bounds, not the tokamak's.** `eps` runs to `0.75` and
    `kappa` to `3.2` here where the IPDG89 contracts stop at `0.45`/`2.0`; a low-aspect
    machine lives in a different part of the domain, and reusing the conventional bounds
    would have tested this arm nowhere near where it is used.
    """

    audit_record = _AUDIT_RECORD
    reference = staticmethod(_reference_plasma_current_fiesta)
    ported = calculate_plasma_current_fiesta

    samples = [
        Sample(
            MappingProxyType(
                {
                    "eps": 1.0 / 1.8,
                    "kappa": 2.8,
                    "triang": 0.5,
                    "rminor": 4.5 / 1.8,
                    "rmajor": 4.5,
                    "q95": 5.835830999686161,
                    "b_plasma_toroidal_on_axis": 3.0,
                },
            ),
            "input-file",
            "calculate_plasma_current_fiesta-spherical_tokamak_eval",
        ),
    ]

    fuzz_bounds = {
        "eps": (0.35, 0.75),
        "kappa": (1.8, 3.2),
        "triang": (0.05, 0.6),
        "rminor": (1.5, 3.0),
        "rmajor": (3.0, 6.0),
        "q95": (3.0, 8.0),
        "b_plasma_toroidal_on_axis": (1.5, 4.5),
    }


class TestCalculateCylindricalSafetyFactor(Tier1Contract):
    """`calculate_cylindrical_safety_factor` -> the same module-level function in
    `process/models/physics/physics.py`, unchanged (`@nb.jit` dropped).

    Sample is `tests/unit/models/physics/test_physics.py::
    test_calculate_cylindrical_safety_factor_parametrized`'s only point, verbatim;
    recorded expectation `qstar = 2.90080289950078`, and its `plasma_current` is the
    same `18398455.678867526` the contract above produces -- the two are consecutive
    links of one chain, taken from the same run.
    """

    audit_record = _AUDIT_RECORD
    reference = staticmethod(process_calculate_cylindrical_safety_factor)
    ported = calculate_cylindrical_safety_factor

    samples = [
        legacy_sample(
            "calculate_cylindrical_safety_factor-large_tokamak_nof",
            rmajor=8.0,
            rminor=2.6666666666666665,
            plasma_current=18398455.678867526,
            b_plasma_toroidal_on_axis=5.7,
            kappa95=1.6517857142857142,
            triang95=0.33333333333333331,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (6.0, 10.0),
        "rminor": (2.0, 3.5),
        "plasma_current": (1.0e7, 2.5e7),
        "b_plasma_toroidal_on_axis": (4.0, 7.0),
        "kappa95": (1.4, 2.0),
        "triang95": (0.05, 0.5),
    }


class TestCalculateCurrentProfileIndexWesson(Tier1Contract):
    """`calculate_current_profile_index_wesson` -> `Physics.
    calculate_current_profile_index_wesson`, unchanged.

    Sample is `tests/unit/models/physics/test_physics.py::
    test_calculate_current_profile_index_wesson`'s point, verbatim (`qstar = 3.5`,
    `q0 = 1.5`, expected `1.33333`).

    The `i_alphaj` selection this feeds is not exercised here and cannot be: PROCESS
    performs it inline in `Physics.run()` (`physics.py:334-348`), which is the whole
    7000-line plasma model and has no callable sub-shell. The selection is a copy
    (`alphaj = alphaj_wesson`) with no arithmetic in it, so what is left untested is the
    wiring, which is `total_process.py`'s to get right, not this contract's.
    """

    audit_record = _AUDIT_RECORD
    reference = staticmethod(Physics.calculate_current_profile_index_wesson)
    ported = calculate_current_profile_index_wesson

    samples = [
        legacy_sample(
            "calculate_current_profile_index_wesson-unit_test",
            qstar=3.5,
            q0=1.5,
        ),
    ]

    fuzz_bounds = {
        "qstar": (2.0, 4.5),
        "q0": (0.8, 1.5),
    }


class TestCalculateInternalInductanceWesson(Tier1Contract):
    """`calculate_internal_inductance_wesson` -> `PlasmaInductance.run()` at
    `i_ind_plasma_internal_norm = 1`, on a real `DataStructure`.

    Diffed against the stateful shell rather than against the bare
    `@staticmethod`, because here the switch *is* testable: `PlasmaInductance.run()`
    takes no arguments and reads only `data`, so the adapter can exercise the
    `model_map` selection (`physics.py:4759-4764`) as well as the formula. That makes
    this the one contract in the file that validates a switch answer and not just an
    expression.

    Sample is `tests/unit/models/physics/test_physics.py::
    test_calculate_internal_inductance_wesson`'s point, verbatim (`alphaj = 0.8`,
    expected `0.8595087177751706`).
    """

    audit_record = _AUDIT_RECORD
    reference = staticmethod(_reference_ind_plasma_internal_norm_wesson)
    ported = calculate_internal_inductance_wesson

    samples = [
        legacy_sample(
            "calculate_internal_inductance_wesson-unit_test",
            alphaj=0.8,
        ),
    ]

    fuzz_bounds = {
        "alphaj": (0.5, 3.0),
    }
