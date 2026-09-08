"""Harness cases for the ported **tokamak** density-limit chain (`PlasmaDensityLimit`).

Not the stellarator unit -- see `functional_process/cottax/physics/density_limit.py`'s
module docstring and `_audit/units/models/physics/density_limit.md` for the distinction
from `functional_process/cottax/stellarator/density_limits.py`.

Legacy sample values for the eight one-liner formulas are lifted verbatim from
`tests/unit/models/physics/test_physics.py::test_calculate_density_limit`
(`CalculateDensityLimitParam`, generated from `large_tokamak_nof.IN.DAT`) -- one input
point, all eight `expected_dlimit` elements recorded, so every formula gets a free,
already-validated oracle from a single case. `p_perp` is computed here exactly as
`PlasmaDensityLimit.calculate_density_limit` computes it
(`process/models/physics/density_limit.py:526`), not hand-rounded.
"""

import numpy as np

from functional_process.cottax._harness import Tier1Contract
from functional_process.cottax._harness.sample_store import FROM_FILE
from functional_process.cottax.physics.density_limit import (
    calculate_asdex_density_limit,
    calculate_asdex_new_density_limit,
    calculate_borrass_iter_i_density_limit,
    calculate_borrass_iter_ii_density_limit,
    calculate_greenwald_density_limit,
    calculate_greenwald_fraction,
    calculate_hugill_murakami_density_limit,
    calculate_jet_edge_radiation_density_limit,
    calculate_jet_simple_density_limit,
    select_enforced_density_limit_greenwald,
)
from process.models.physics.density_limit import DensityLimitModel, PlasmaDensityLimit

# ---------------------------------------------------------------------------
# The `large_tokamak_nof.IN.DAT`-derived legacy point, shared by every formula below.
# ---------------------------------------------------------------------------

_B_PLASMA_TOROIDAL_ON_AXIS = 5.1847188735686647
_P_PLASMA_SEPARATRIX_MW = 162.32943903093374
_P_HCD_INJECTED_TOTAL_MW = 79.928763793309031
_PLASMA_CURRENT = 16702766.338258133
_PRN1 = 0.4614366315228275
_Q95 = 3.5068029786872268
_QCYL = 3.8769445264202052
_RMAJOR = 8.0
_RMINOR = 2.6666666666666665
_A_PLASMA_SURFACE = 1173.8427771245592
_ZEFF = 2.5668755115791471
_P_PERP = _P_PLASMA_SEPARATRIX_MW / _A_PLASMA_SURFACE

# `expected_dlimit`, `CalculateDensityLimitParam(i_density_limit=7, ...)`.
_EXPECTED_ASDEX = 5.2955542598288974e19
_EXPECTED_BORRASS_ITER_I = 1.0934080161360552e20
_EXPECTED_BORRASS_ITER_II = 4.3286395478282289e19
_EXPECTED_JET_EDGE_RADIATION = 1.9109162908046821e21
_EXPECTED_JET_SIMPLE = 4.2410183109151497e20
_EXPECTED_HUGILL_MURAKAMI = 5.0149533075302982e19
_EXPECTED_GREENWALD = 7.4765470107450917e19
_EXPECTED_ASDEX_NEW = 8.7406037163890049e20


class TestAsdexDensityLimit(Tier1Contract):
    """`calculate_asdex_density_limit` -> `PlasmaDensityLimit.
    calculate_asdex_density_limit`, unchanged. Not wired as an occupant (dead work at
    `i_density_limit != 1` on the reference arm) -- ported for the free oracle.
    """

    audit_record = "models/physics/density_limit.md"
    reference = staticmethod(PlasmaDensityLimit.calculate_asdex_density_limit)
    ported = calculate_asdex_density_limit

    samples = FROM_FILE
    fuzz = True


class TestBorrassIterIDensityLimit(Tier1Contract):
    """`calculate_borrass_iter_i_density_limit` -> the same, unchanged. Not wired
    (see module docstring).
    """

    audit_record = "models/physics/density_limit.md"
    reference = staticmethod(PlasmaDensityLimit.calculate_borrass_iter_i_density_limit)
    ported = calculate_borrass_iter_i_density_limit

    samples = FROM_FILE
    fuzz = True


class TestBorrassIterIIDensityLimit(Tier1Contract):
    """`calculate_borrass_iter_ii_density_limit` -> the same, unchanged. Not wired
    (see module docstring).
    """

    audit_record = "models/physics/density_limit.md"
    reference = staticmethod(PlasmaDensityLimit.calculate_borrass_iter_ii_density_limit)
    ported = calculate_borrass_iter_ii_density_limit

    samples = FROM_FILE
    fuzz = True


class TestJetEdgeRadiationDensityLimit(Tier1Contract):
    """`calculate_jet_edge_radiation_density_limit` -> the same, unchanged. Not wired
    (see module docstring).

    `reference_domain_errors` is not used: PROCESS returns a real, finite `0.0` when
    `denom <= 0.0`, rather than raising (see the port's docstring). The
    `denominator-non-positive` sample exercises exactly that branch, both value and
    gradient.
    """

    audit_record = "models/physics/density_limit.md"
    reference = staticmethod(
        PlasmaDensityLimit.calculate_jet_edge_radiation_density_limit
    )
    ported = calculate_jet_edge_radiation_density_limit

    samples = FROM_FILE
    fuzz = True


class TestJetSimpleDensityLimit(Tier1Contract):
    """`calculate_jet_simple_density_limit` -> the same, unchanged. Not wired (see
    module docstring).
    """

    audit_record = "models/physics/density_limit.md"
    reference = staticmethod(PlasmaDensityLimit.calculate_jet_simple_density_limit)
    ported = calculate_jet_simple_density_limit

    samples = FROM_FILE
    fuzz = True


class TestHugillMurakamiDensityLimit(Tier1Contract):
    """`calculate_hugill_murakami_density_limit` -> the same, unchanged. Not wired
    (see module docstring).
    """

    audit_record = "models/physics/density_limit.md"
    reference = staticmethod(PlasmaDensityLimit.calculate_hugill_murakami_density_limit)
    ported = calculate_hugill_murakami_density_limit

    samples = FROM_FILE
    fuzz = True


class TestGreenwaldDensityLimit(Tier1Contract):
    """`calculate_greenwald_density_limit` -> the same, unchanged.

    **The wired occupant's formula** -- `GreenwaldDensityLimit` in the port module.
    """

    audit_record = "models/physics/density_limit.md"
    reference = staticmethod(PlasmaDensityLimit.calculate_greenwald_density_limit)
    ported = calculate_greenwald_density_limit

    samples = FROM_FILE
    fuzz = True


class TestAsdexNewDensityLimit(Tier1Contract):
    """`calculate_asdex_new_density_limit` -> the same, unchanged. Not wired (see
    module docstring) -- also PROCESS's own bare default for `i_density_limit`
    (`8`), which `large_tokamak_eval.IN.DAT` overrides to `7`.
    """

    audit_record = "models/physics/density_limit.md"
    reference = staticmethod(PlasmaDensityLimit.calculate_asdex_new_density_limit)
    ported = calculate_asdex_new_density_limit

    samples = FROM_FILE
    fuzz = True


def test_expected_dlimit_sanity_check():
    """The eight legacy expected values above actually come from PROCESS's own
    `calculate_density_limit`, called whole -- not just copied from the unit test file
    and trusted. Guards against a transcription slip in the eight `_EXPECTED_*`
    constants above.
    """
    array, enforced = PlasmaDensityLimit().calculate_density_limit(
        b_plasma_toroidal_on_axis=_B_PLASMA_TOROIDAL_ON_AXIS,
        i_density_limit=7,
        p_plasma_separatrix_mw=_P_PLASMA_SEPARATRIX_MW,
        p_hcd_injected_total_mw=_P_HCD_INJECTED_TOTAL_MW,
        plasma_current=_PLASMA_CURRENT,
        prn1=_PRN1,
        qcyl=_QCYL,
        q95=_Q95,
        rmajor=_RMAJOR,
        rminor=_RMINOR,
        a_plasma_surface=_A_PLASMA_SURFACE,
        zeff=_ZEFF,
    )
    expected = [
        _EXPECTED_ASDEX,
        _EXPECTED_BORRASS_ITER_I,
        _EXPECTED_BORRASS_ITER_II,
        _EXPECTED_JET_EDGE_RADIATION,
        _EXPECTED_JET_SIMPLE,
        _EXPECTED_HUGILL_MURAKAMI,
        _EXPECTED_GREENWALD,
        _EXPECTED_ASDEX_NEW,
    ]
    assert np.allclose(array, expected, rtol=1e-12)
    assert enforced == array[6]


# ---------------------------------------------------------------------------
# The two small nodes downstream of the array: the GREENWALD selection arm, and the
# unconditional Greenwald-fraction bookkeeping.
# ---------------------------------------------------------------------------


def _reference_enforced_density_limit_greenwald(nd_plasma_electron_max_array_7):
    """Real call into `PlasmaDensityLimit.get_density_limit_value` for the GREENWALD
    arm. Every other array element is `nan` -- confirmed by reading `model_map`
    (`process/models/physics/density_limit.py:131-141`) that GREENWALD's selection
    reads only index 6, not assumed from the docstring.
    """
    array = np.full(8, np.nan)
    array[int(DensityLimitModel.GREENWALD) - 1] = nd_plasma_electron_max_array_7
    return PlasmaDensityLimit.get_density_limit_value(DensityLimitModel.GREENWALD, array)


class TestSelectEnforcedDensityLimitGreenwald(Tier1Contract):
    """`select_enforced_density_limit_greenwald` -> `PlasmaDensityLimit.
    get_density_limit_value(DensityLimitModel.GREENWALD, ...)`, a real call, not a
    transcription -- see the reference adapter's docstring.
    """

    audit_record = "models/physics/density_limit.md"
    reference = _reference_enforced_density_limit_greenwald
    ported = select_enforced_density_limit_greenwald

    samples = FROM_FILE
    fuzz = True


def _reference_greenwald_fraction(
    nd_plasma_electron_line, nd_plasma_electron_max_array_7
):
    """Transcription of `PlasmaDensityLimit.run`,
    `process/models/physics/density_limit.py:106-109` -- the weakest oracle in this
    unit, same shape `bootstrap_current.md`'s current-fraction tail records: `run()`
    has a callable shell, but reaching this one line through it means also running the
    other seven, unrelated formulas and back-deriving a `(plasma_current, rminor)` pair
    for a synthetic `nd_plasma_electron_max_array_7` -- not done for a one-line
    division.
    """
    return nd_plasma_electron_line / nd_plasma_electron_max_array_7


class TestGreenwaldFraction(Tier1Contract):
    """`calculate_greenwald_fraction` -> a transcription of `PlasmaDensityLimit.run`'s
    Greenwald-fraction assignment (see the reference adapter's docstring).
    """

    audit_record = "models/physics/density_limit.md"
    reference = _reference_greenwald_fraction
    ported = calculate_greenwald_fraction

    samples = FROM_FILE
    fuzz = True
