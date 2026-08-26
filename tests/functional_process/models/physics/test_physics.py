"""Harness cases for the ported tokamak arm of `process/models/physics/physics.py`.

Audit record: `functional_process/_audit/units/models/physics/physics.md`.

Three of the seven ported functions have a PROCESS `@staticmethod`/method to diff
against directly (`PlasmaFields.calculate_surface_averaged_poloidal_field`,
`PlasmaExhaust.calculate_separatrix_power`,
`PlasmaBeta.calculate_plasma_energy_from_beta`). The other four are arithmetic PROCESS
writes inline inside `Physics.run`, so their reference is transcribed from the source
here in `numpy`, with the source lines named -- the same convention
`test_plasma_physics.py`'s `_reference_clipped_radiation_powers` established for
`st_phys`'s inline blocks.

**Sample provenance.** Two of the units have real legacy points lifted from
`tests/unit/models/physics/test_physics.py` (marked in each case). The rest have no
PROCESS unit test at all, so their `legacy_sample`s are hand-built at
`large_tokamak_eval`-scale operating values and the real coverage is the fuzz draws --
recorded as this unit's weak point in the audit record's "tier signal", not papered
over.
"""

import numpy as np

from functional_process._harness import Tier1Contract, fuzz_samples, legacy_sample
from functional_process.models.physics.physics import (
    calculate_plasma_energy_from_beta,
    calculate_pulsed_plant_ramp_times,
    calculate_separatrix_power,
    calculate_surface_averaged_poloidal_field_amperes,
    calculate_total_radiation_power,
    calculate_unclipped_radiation_powers,
    force_positive_separatrix_power,
)
from process.models.physics.exhaust import PlasmaExhaust
from process.models.physics.physics import PlasmaBeta
from process.models.physics.plasma_fields import PlasmaFields

_FIELDS = PlasmaFields()

# `PlasmaCurrentModel.IPDG89_SCALING`, `large_tokamak_eval.IN.DAT`'s value. Any value
# other than `2` (`PENG_DIVERTOR_SCALING`) selects the same arm -- see the port's
# `SurfaceAveragedPoloidalFieldAmperes` docstring.
_I_PLASMA_CURRENT_AMPERES = 4


def _reference_surface_averaged_poloidal_field_amperes(cur_plasma, len_plasma_poloidal):
    """`PlasmaFields.calculate_surface_averaged_poloidal_field` on its Ampere arm.

    The five arguments held fixed here are the *evidence* for the split: they are read
    only by the `PENG_DIVERTOR_SCALING` arm (`plasma_fields.py:86-93`), so their values
    are arbitrary and the reference is invariant under them. Set to
    `large_tokamak_eval`-scale numbers anyway, so that a regression in the branch test
    itself shows up as a value mismatch rather than a `nan`.
    """
    return _FIELDS.calculate_surface_averaged_poloidal_field(
        i_plasma_current=_I_PLASMA_CURRENT_AMPERES,
        cur_plasma=cur_plasma,
        q95=3.5,
        aspect=3.0,
        b_plasma_toroidal_on_axis=5.7,
        kappa=1.85,
        triang=0.5,
        len_plasma_poloidal=len_plasma_poloidal,
    )


class TestSurfaceAveragedPoloidalFieldAmperes(Tier1Contract):
    """`<Bp(a)> = mu_0 * Ip / L_pol`, `plasma_fields.py:83-84`.

    Both legacy points are lifted verbatim from
    `tests/unit/models/physics/test_physics.py::test_calculate_surface_averaged_poloidal_field`
    -- its `i_plasma_current = 3` and `i_plasma_current = 4` rows, the two of its four
    parametrisations that take this arm.
    """

    audit_record = "models/physics/physics.md"
    reference = _reference_surface_averaged_poloidal_field_amperes
    ported = calculate_surface_averaged_poloidal_field_amperes

    samples = [
        legacy_sample(
            "large-tokamak-ipdg89",
            cur_plasma=18398455.678867526,
            len_plasma_poloidal=24.081367139525412,
        ),
        legacy_sample(
            "iter-scaling-row",
            cur_plasma=1.6e7,
            len_plasma_poloidal=24.0,
        ),
        *fuzz_samples(
            {
                "cur_plasma": (1.0e6, 3.0e7),
                "len_plasma_poloidal": (5.0, 40.0),
            },
            count=5,
            seed=0,
        ),
    ]


def _reference_unclipped_radiation_powers(
    pden_plasma_core_rad_mw_unclipped,
    pden_plasma_outer_rad_mw_unclipped,
    vol_plasma,
):
    """`physics.py:751-752` and `:758-763`, transcribed from source.

    The point of the transcription is what is *absent*: `stellarator.py:2153-2158`
    wraps both densities in `max(..., 0.0)` before forming the products and this call
    site does not. The `negative-core-density` sample below is the point where the two
    references disagree, and it is the only reason this function is a node of its own.
    """
    core = pden_plasma_core_rad_mw_unclipped
    outer = pden_plasma_outer_rad_mw_unclipped
    return core, outer, core * vol_plasma, outer * vol_plasma


class TestUnclippedRadiationPowers(Tier1Contract):
    """The tokamak's radiation-density assignments: no clip at zero."""

    audit_record = "models/physics/physics.md"
    reference = _reference_unclipped_radiation_powers
    ported = calculate_unclipped_radiation_powers

    samples = [
        legacy_sample(
            "large-tokamak-scale",
            pden_plasma_core_rad_mw_unclipped=0.057544135593658154,
            pden_plasma_outer_rad_mw_unclipped=0.05525606,
            vol_plasma=2077.5,
        ),
        # The arm the stellarator's clip exists to suppress and the tokamak does not.
        # PROCESS carries the negative straight through here; the port must too.
        legacy_sample(
            "negative-core-density",
            pden_plasma_core_rad_mw_unclipped=-0.004,
            pden_plasma_outer_rad_mw_unclipped=-0.001,
            vol_plasma=2077.5,
        ),
        *fuzz_samples(
            {
                "pden_plasma_core_rad_mw_unclipped": (-0.05, 0.5),
                "pden_plasma_outer_rad_mw_unclipped": (-0.05, 0.5),
                "vol_plasma": (500.0, 5000.0),
            },
            count=5,
            seed=0,
        ),
    ]


def _reference_total_radiation_power(pden_plasma_rad_mw, vol_plasma):
    """`physics.py:764-766`, transcribed from source."""
    return pden_plasma_rad_mw * vol_plasma


class TestTotalRadiationPower(Tier1Contract):
    """`P_rad = pden_rad * V`."""

    audit_record = "models/physics/physics.md"
    reference = _reference_total_radiation_power
    ported = calculate_total_radiation_power

    samples = [
        legacy_sample(
            "large-tokamak-scale",
            pden_plasma_rad_mw=0.1128,
            vol_plasma=2077.5,
        ),
        *fuzz_samples(
            {
                "pden_plasma_rad_mw": (0.001, 1.0),
                "vol_plasma": (500.0, 5000.0),
            },
            count=5,
            seed=0,
        ),
    ]


class TestSeparatrixPower(Tier1Contract):
    """`PlasmaExhaust.calculate_separatrix_power`, `exhaust.py:88-127`, unchanged.

    The reference is PROCESS's own `@staticmethod`, called directly -- it takes no
    `self.data` access, so no adapter is needed. `p_hcd_injected_total_mw` is a live
    argument here rather than the `0.0` the ignited arm passes, which is the whole
    content of the `i_plasma_ignited` split this node's occupant declares.
    """

    audit_record = "models/physics/physics.md"
    reference = staticmethod(PlasmaExhaust.calculate_separatrix_power)
    ported = calculate_separatrix_power

    samples = [
        legacy_sample(
            "large-tokamak-non-ignited",
            f_p_alpha_plasma_deposited=0.95,
            p_alpha_total_mw=396.0,
            p_non_alpha_charged_mw=2.1,
            p_hcd_injected_total_mw=50.0,
            p_plasma_ohmic_mw=0.8,
            p_plasma_rad_mw=234.3,
        ),
        # The configuration the KLUDGE at `physics.py:843-845` exists for: radiated
        # power exceeds everything crossing the separatrix, so the raw answer is
        # negative.
        legacy_sample(
            "radiation-dominated-negative",
            f_p_alpha_plasma_deposited=0.95,
            p_alpha_total_mw=100.0,
            p_non_alpha_charged_mw=0.5,
            p_hcd_injected_total_mw=10.0,
            p_plasma_ohmic_mw=0.5,
            p_plasma_rad_mw=250.0,
        ),
        *fuzz_samples(
            {
                "f_p_alpha_plasma_deposited": (0.5, 1.0),
                "p_alpha_total_mw": (50.0, 800.0),
                "p_non_alpha_charged_mw": (0.0, 20.0),
                "p_hcd_injected_total_mw": (0.0, 200.0),
                "p_plasma_ohmic_mw": (0.0, 5.0),
                "p_plasma_rad_mw": (10.0, 600.0),
            },
            count=5,
            seed=0,
        ),
    ]


def _reference_force_positive_separatrix_power(p_plasma_separatrix_mw_raw):
    """`physics.py:839-845`, transcribed from source."""
    return p_plasma_separatrix_mw_raw / (1 - np.exp(-p_plasma_separatrix_mw_raw))


class TestForcePositiveSeparatrixPower(Tier1Contract):
    """PROCESS's own "KLUDGE" positivity transform, `physics.py:839-845`.

    Three samples, one per regime: far above the transform's scale (where it is the
    identity to machine precision), inside it, and below zero -- the case the transform
    exists for, where a negative separatrix power is mapped to a small positive one.
    `x == 0` is *not* sampled: PROCESS evaluates `0.0/0.0` there and returns `nan`, and
    the port reproduces that rather than inventing the limit (which is `1.0`).
    """

    audit_record = "models/physics/physics.md"
    reference = _reference_force_positive_separatrix_power
    ported = force_positive_separatrix_power

    samples = [
        legacy_sample("large-tokamak-scale", p_plasma_separatrix_mw_raw=164.6),
        legacy_sample("inside-the-transform", p_plasma_separatrix_mw_raw=0.5),
        legacy_sample("negative-raw-power", p_plasma_separatrix_mw_raw=-5.0),
        *fuzz_samples(
            {"p_plasma_separatrix_mw_raw": (-20.0, 400.0)},
            count=5,
            seed=0,
        ),
    ]


def _reference_pulsed_plant_ramp_times(plasma_current):
    """`physics.py:476-483`, the `i_pulsed_plant == 1, pulsetimings == 0` arm."""
    t_plant_pulse_plasma_current_ramp_up = plasma_current / 1.0e5
    return (
        t_plant_pulse_plasma_current_ramp_up,
        t_plant_pulse_plasma_current_ramp_up,
    )


class TestPulsedPlantRampTimes(Tier1Contract):
    """The one arm of `pulsetimings` this port supports.

    `pulsetimings` is read at `physics.py:476` and nowhere else in all of
    `process/models/**`, so this two-line function is the whole of that decision's
    computation.
    """

    audit_record = "models/physics/physics.md"
    reference = _reference_pulsed_plant_ramp_times
    ported = calculate_pulsed_plant_ramp_times

    samples = [
        legacy_sample("large-tokamak", plasma_current=18398455.678867526),
        *fuzz_samples(
            {"plasma_current": (1.0e6, 3.0e7)},
            count=5,
            seed=0,
        ),
    ]


class TestPlasmaEnergyFromBeta(Tier1Contract):
    """`PlasmaBeta.calculate_plasma_energy_from_beta`, `physics.py:4153-4176`.

    The legacy point is `tests/unit/models/physics/test_physics.py::
    test_calculate_plasma_energy_from_beta`'s, verbatim.
    """

    audit_record = "models/physics/physics.md"
    reference = staticmethod(PlasmaBeta.calculate_plasma_energy_from_beta)
    ported = calculate_plasma_energy_from_beta

    samples = [
        legacy_sample("unit-test-point", beta=0.02, b_field=5.3, vol_plasma=1000.0),
        legacy_sample(
            "large-tokamak-scale", beta=0.0357, b_field=5.79, vol_plasma=2077.5
        ),
        *fuzz_samples(
            {
                "beta": (0.001, 0.15),
                "b_field": (1.0, 13.0),
                "vol_plasma": (500.0, 5000.0),
            },
            count=5,
            seed=0,
        ),
    ]
