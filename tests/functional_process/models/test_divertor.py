"""Harness cases for the ported subset of `process/models/divertor.py`
(`.tokamak.divertor`) -- **not** `process/models/stellarator/divertor.py` (registry
unit #4, ported separately).

Audit record: `functional_process/_audit/units/models/divertor.md`. Two units:

- `TestCalculateDivertorHeatFluxSplit` -- `Divertor.run()`'s unconditional preamble.
- `TestCalculateDivertorHeatLoadWade` -- `Divertor.divwade`, `n_divertors == 1`
  (single null) baked in.
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.divertor import (
    calculate_divertor_heat_flux_split,
    calculate_divertor_heat_load_wade,
)
from process.core.model import DataStructure
from process.models.divertor import Divertor


def _reference_divertor_heat_flux_split(
    deg_blkt_inboard_poloidal_plasma, p_plasma_neutron_mw, n_divertors
):
    """Call PROCESS's `single_divertor_angle` property plus `incident_neutron_power`
    through the port's signature -- exactly `Divertor.run()`'s own preamble
    (`process/models/divertor.py:41-50`), composed rather than re-derived.
    """
    data = DataStructure()
    data.blanket.deg_blkt_inboard_poloidal_plasma = deg_blkt_inboard_poloidal_plasma
    d = Divertor()
    d.data = data
    deg_div_poloidal_plasma = d.single_divertor_angle
    f_ster_div_single = deg_div_poloidal_plasma / 360.0
    p_div_nuclear_heat_total_mw = Divertor.incident_neutron_power(
        p_plasma_neutron_mw=p_plasma_neutron_mw,
        f_ster_div_single=f_ster_div_single,
        n_divertors=n_divertors,
    )
    return deg_div_poloidal_plasma, f_ster_div_single, p_div_nuclear_heat_total_mw


class TestCalculateDivertorHeatFluxSplit(Tier1Contract):
    """`calculate_divertor_heat_flux_split` -> `Divertor.run()`'s preamble."""

    audit_record = "models/divertor.md"
    reference = _reference_divertor_heat_flux_split
    ported = calculate_divertor_heat_flux_split

    samples = [
        legacy_sample(
            "single-null-plausible",
            deg_blkt_inboard_poloidal_plasma=100.0,
            p_plasma_neutron_mw=1500.0,
            n_divertors=1,
        ),
    ]

    fuzz_bounds = {
        "deg_blkt_inboard_poloidal_plasma": (10.0, 170.0),
        "p_plasma_neutron_mw": (100.0, 3000.0),
        "n_divertors": (1.0, 2.0),
    }


def _reference_divertor_heat_load_wade(
    rmajor,
    rminor,
    aspect,
    b_plasma_toroidal_on_axis,
    b_plasma_poloidal_average,
    p_plasma_separatrix_mw,
    f_div_flux_expansion,
    nd_plasma_separatrix_electron,
    deg_div_field_plate,
    rad_fraction_sol,
):
    """`Divertor.divwade` at `n_divertors == 1` (the `DataStructure` default is `2`;
    forced to `1` here to match this occupant's baked single-null formula).
    """
    data = DataStructure()
    data.divertor.n_divertors = 1
    data.tfcoil.drtop = 0.0
    d = Divertor()
    d.data = data
    return d.divwade(
        rmajor,
        rminor,
        aspect,
        b_plasma_toroidal_on_axis,
        b_plasma_poloidal_average,
        p_plasma_separatrix_mw,
        f_div_flux_expansion,
        nd_plasma_separatrix_electron,
        deg_div_field_plate,
        rad_fraction_sol,
        # unread at n_divertors == 1; PROCESS's signature still needs a value.
        f_p_div_lower=1.0,
        output=False,
    )


class TestCalculateDivertorHeatLoadWade(Tier1Contract):
    """`calculate_divertor_heat_load_wade` -> `Divertor.divwade` at `n_divertors == 1`.

    `theta_div`'s `arcsin` argument can exceed 1 in magnitude for a small flux-expansion
    angle (`divertor.md` § open questions) -- PROCESS raises `ValueError` from
    `math.asin` there, the port returns `nan`. Declared so a fuzz draw landing in that
    regime is asserted non-finite rather than failing as an unexplained mismatch.
    """

    audit_record = "models/divertor.md"
    reference = _reference_divertor_heat_load_wade
    ported = calculate_divertor_heat_load_wade
    reference_domain_errors = (ValueError,)

    # tests/unit/models/test_divertor.py::TestDivertor.test_divwade, verbatim. That
    # test's own `f_p_div_lower = 1.0` makes the source's `n_divertors == 2` branch
    # collapse to `hldiv_base` too (see divertor.md § sample provenance), so the legacy
    # point is reused unchanged for this `n_divertors == 1` occupant.
    samples = [
        legacy_sample(
            "divwade-legacy",
            rmajor=2.0,
            rminor=1.0,
            aspect=2.0,
            b_plasma_toroidal_on_axis=0.5,
            b_plasma_poloidal_average=0.09595,
            p_plasma_separatrix_mw=1.0e2,
            f_div_flux_expansion=2.0,
            nd_plasma_separatrix_electron=1.0e19,
            deg_div_field_plate=5.0,
            rad_fraction_sol=8.0e-1,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "aspect": (1.5, 4.0),
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "b_plasma_poloidal_average": (0.05, 1.5),
        "p_plasma_separatrix_mw": (10.0, 500.0),
        "f_div_flux_expansion": (1.0, 5.0),
        "nd_plasma_separatrix_electron": (1.0e18, 1.0e20),
        "deg_div_field_plate": (1.0, 15.0),
        "rad_fraction_sol": (0.1, 0.9),
    }
