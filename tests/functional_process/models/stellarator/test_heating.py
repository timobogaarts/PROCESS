"""Harness cases for the ported stellarator auxiliary heating (registry unit #5).

Only the self-contained `isthtr` branches are covered -- ECRH (`isthtr == 1`) and
lower-hybrid (`isthtr == 2`) -- plus the common tail shared by all three branches
(`calculate_injected_power_total`, `calculate_beam_current`, `calculate_fusion_gain`).
The NBI branch (`isthtr == 3`) calls `stellarator.current_drive.culnbi()`, out of scope
per `heating.md`.

No PROCESS unit test exists for `st_heat` (checked: no hits for `st_heat`/`isthtr` under
`tests/unit`), so every sample here is a hand-picked, physically plausible point rather
than one lifted from an existing test -- still run through the real PROCESS function via
the `_stellarator()` adapter, so value/gradient agreement is genuine, not assumed.
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.stellarator.heating import (
    calculate_beam_current,
    calculate_ecrh_heating,
    calculate_fusion_gain,
    calculate_injected_power_total,
    calculate_lowhyb_heating,
)
from process.core.model import DataStructure
from process.models.stellarator.heating import st_heat
from process.models.stellarator.stellarator import Stellarator


def _stellarator():
    """A `Stellarator` instance whose sub-models are never called by the ported branches."""
    stellarator = Stellarator(*([None] * 12))
    stellarator.data = DataStructure()
    return stellarator


def _reference_ecrh_heating(p_hcd_primary_extra_heat_mw, eta_ecrh_injector_wall_plug):
    """Call PROCESS's `st_heat` (`isthtr == 1`) through the port's signature."""
    stellarator = _stellarator()
    data = stellarator.data
    data.stellarator.isthtr = 1
    data.current_drive.p_hcd_primary_extra_heat_mw = p_hcd_primary_extra_heat_mw
    data.current_drive.eta_ecrh_injector_wall_plug = eta_ecrh_injector_wall_plug

    st_heat(stellarator, False, data)

    return (
        data.current_drive.p_hcd_ecrh_injected_total_mw,
        data.current_drive.p_hcd_injected_ions_mw,
        data.current_drive.p_hcd_injected_electrons_mw,
        data.current_drive.eta_hcd_primary_injector_wall_plug,
        data.heat_transport.p_hcd_electric_total_mw,
    )


def _reference_lowhyb_heating(
    p_hcd_primary_extra_heat_mw, eta_lowhyb_injector_wall_plug
):
    """Call PROCESS's `st_heat` (`isthtr == 2`) through the port's signature."""
    stellarator = _stellarator()
    data = stellarator.data
    data.stellarator.isthtr = 2
    data.current_drive.p_hcd_primary_extra_heat_mw = p_hcd_primary_extra_heat_mw
    data.current_drive.eta_lowhyb_injector_wall_plug = eta_lowhyb_injector_wall_plug

    st_heat(stellarator, False, data)

    return (
        data.current_drive.p_hcd_lowhyb_injected_total_mw,
        data.current_drive.p_hcd_injected_ions_mw,
        data.current_drive.p_hcd_injected_electrons_mw,
        data.current_drive.eta_hcd_primary_injector_wall_plug,
        data.heat_transport.p_hcd_electric_total_mw,
    )


def _reference_injected_power_total(p_hcd_injected_electrons_mw, p_hcd_injected_ions_mw):
    """Reproduce `st_heat`'s total-injected-power line directly.

    Not independently reachable through `st_heat` without also exercising a full
    `isthtr` branch (it is one line inside that same function, not callable alone), so
    this reference is the source line copied verbatim rather than re-derived --
    consistent with `structure.py`'s `msupstr` precedent.
    """
    return p_hcd_injected_electrons_mw + p_hcd_injected_ions_mw


def _reference_beam_current(p_hcd_beam_injected_total_mw, e_beam_kev):
    """Reproduce `st_heat`'s neutral-beam-current step directly (same reason as above)."""
    if abs(p_hcd_beam_injected_total_mw) > 1e-8:
        return 1e-3 * (p_hcd_beam_injected_total_mw * 1e6) / e_beam_kev
    return 0.0


def _reference_fusion_gain(
    p_fusion_total_mw, p_hcd_injected_total_mw, p_beam_orbit_loss_mw, p_plasma_ohmic_mw
):
    """Reproduce `st_heat`'s fusion-gain step directly (same reason as above)."""
    denominator = p_hcd_injected_total_mw + p_beam_orbit_loss_mw + p_plasma_ohmic_mw
    if abs(denominator) < 1e-6:
        return 1e18
    return p_fusion_total_mw / denominator


class TestEcrhHeating(Tier1Contract):
    """`st_heat` (`isthtr == 1`) -> `calculate_ecrh_heating`."""

    audit_record = "models/stellarator/heating.md"
    reference = _reference_ecrh_heating
    ported = calculate_ecrh_heating

    samples = [
        legacy_sample(
            "ecrh-round-numbers",
            p_hcd_primary_extra_heat_mw=50.0,
            eta_ecrh_injector_wall_plug=0.5,
        ),
    ]

    fuzz_bounds = {
        "p_hcd_primary_extra_heat_mw": (0.1, 200.0),
        "eta_ecrh_injector_wall_plug": (0.1, 0.9),
    }


class TestLowhybHeating(Tier1Contract):
    """`st_heat` (`isthtr == 2`) -> `calculate_lowhyb_heating`."""

    audit_record = "models/stellarator/heating.md"
    reference = _reference_lowhyb_heating
    ported = calculate_lowhyb_heating

    samples = [
        legacy_sample(
            "lowhyb-round-numbers",
            p_hcd_primary_extra_heat_mw=50.0,
            eta_lowhyb_injector_wall_plug=0.6,
        ),
    ]

    fuzz_bounds = {
        "p_hcd_primary_extra_heat_mw": (0.1, 200.0),
        "eta_lowhyb_injector_wall_plug": (0.1, 0.9),
    }


class TestInjectedPowerTotal(Tier1Contract):
    """`st_heat`'s total-injected-power step -> `calculate_injected_power_total`."""

    audit_record = "models/stellarator/heating.md"
    reference = _reference_injected_power_total
    ported = calculate_injected_power_total

    samples = [
        legacy_sample(
            "injected-power-total-round-numbers",
            p_hcd_injected_electrons_mw=40.0,
            p_hcd_injected_ions_mw=10.0,
        ),
    ]

    fuzz_bounds = {
        "p_hcd_injected_electrons_mw": (0.0, 200.0),
        "p_hcd_injected_ions_mw": (0.0, 200.0),
    }


class TestBeamCurrent(Tier1Contract):
    """`st_heat`'s beam-current step -> `calculate_beam_current`."""

    audit_record = "models/stellarator/heating.md"
    reference = _reference_beam_current
    ported = calculate_beam_current

    samples = [
        legacy_sample(
            "beam-current-nonzero",
            p_hcd_beam_injected_total_mw=20.0,
            e_beam_kev=100.0,
        ),
        legacy_sample(
            "beam-current-below-guard",
            p_hcd_beam_injected_total_mw=1e-10,
            e_beam_kev=100.0,
        ),
    ]

    fuzz_bounds = {
        "p_hcd_beam_injected_total_mw": (1.0, 200.0),
        "e_beam_kev": (10.0, 1000.0),
    }


class TestFusionGain(Tier1Contract):
    """`st_heat`'s fusion-gain step -> `calculate_fusion_gain`."""

    audit_record = "models/stellarator/heating.md"
    reference = _reference_fusion_gain
    ported = calculate_fusion_gain

    samples = [
        legacy_sample(
            "fusion-gain-round-numbers",
            p_fusion_total_mw=500.0,
            p_hcd_injected_total_mw=50.0,
            p_beam_orbit_loss_mw=0.0,
            p_plasma_ohmic_mw=1.0,
        ),
        legacy_sample(
            "fusion-gain-degenerate-denominator",
            p_fusion_total_mw=500.0,
            p_hcd_injected_total_mw=0.0,
            p_beam_orbit_loss_mw=0.0,
            p_plasma_ohmic_mw=1e-9,
        ),
    ]

    fuzz_bounds = {
        "p_fusion_total_mw": (1.0, 2000.0),
        "p_hcd_injected_total_mw": (0.0, 200.0),
        "p_beam_orbit_loss_mw": (0.0, 50.0),
        "p_plasma_ohmic_mw": (0.0, 5.0),
    }
