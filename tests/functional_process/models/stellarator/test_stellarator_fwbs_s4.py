"""Harness cases for `st_fwbs`'s S4 component-mass block (`stellarator.py:1045-1274`).

Audit record: `functional_process/_audit/units/models/stellarator/stellarator_fwbs_s4.md`.

**Why these references copy the formulas from source instead of calling the real
`st_fwbs` end to end, unlike the sibling S1/S5 cases.** Both ported functions take
`.fwbs.vol_blkt_total` / `.fwbs.vol_shld_total` as inputs, and both of those are written
by **S1**, *inside the same `st_fwbs` call*, at `stellarator.py:515-605` -- long before
S4 reads them at `:1068` and `:1196`. Seeding them on the `DataStructure` before calling
`st_fwbs` therefore cannot work: S1 overwrites them from plasma/build geometry on the way
past. The only way to drive them to a chosen value through the real method is to invert
S1's own geometry formula for `a_plasma_surface` -- which would make this unit's
reference depend on a *different* unit's arithmetic being right, and would introduce a
round-trip rounding error into a comparison whose whole purpose is to detect exactly that
size of discrepancy.

So this module follows the S3 precedent instead (`test_stellarator_fwbs_s3.py`, itself
following `test_stellarator_D_structure.py`'s `msupstr` adapter): the references below
reproduce PROCESS's statements directly, transcribed from the source range and textually
independent of `stellarator_fwbs_s4.py`. The end-to-end grounding for this unit comes
from `run_mda_harness.py`, which compares these nodes' outputs against a real converged
PROCESS solve of `stellarator_helias.IN.DAT` -- a stronger check than a synthesised
`st_fwbs` call, and the one that motivated porting S4 at all
(`_audit/boundary_inputs_audit.md` § 7 item 3).

No PROCESS unit test exercises either block: `tests/unit/models/stellarator/
test_stellarator.py` never calls `st_fwbs`, and the `whtshld`/`m_blkt_*` hits in
`tests/unit/models/test_ife.py` and `test_costs_1990.py` are the *IFE* model's own
different formula and `Costs`'s consumption of these fields respectively, not this block.
There is therefore no `legacy_sample` to add -- coverage is fuzz-only, at
`fwbs_variables.py`'s own default fractions and plausible stellarator blanket/shield
volumes.
"""

from functional_process._harness import Tier1Contract
from functional_process.models.stellarator.stellarator_fwbs_s4 import (
    calculate_blanket_component_masses,
    calculate_shield_mass,
)


def _reference_blanket_component_masses(
    vol_blkt_total,
    fblli2o,
    fblbe,
    den_steel,
    fblss,
    fblvd,
):
    """`stellarator.py:1068-1091`, copied not re-derived.

    The `blktmodel == 0` guard (`:1056`) and the `blkttype in {1, 2}` guard (`:1057`) are
    not part of this formula -- they are topology switches resolved at graph-assembly
    time (see the port's docstring), so this transcription is the arm they select, taken
    literally including PROCESS's two-statement accumulation of `m_blkt_total`.
    """
    m_blkt_li2o = vol_blkt_total * fblli2o * 2010.0e0
    m_blkt_beryllium = vol_blkt_total * fblbe * 1850.0e0
    m_blkt_total = m_blkt_li2o + m_blkt_beryllium

    m_blkt_steel_total = vol_blkt_total * den_steel * fblss
    m_blkt_vanadium = vol_blkt_total * 5870.0e0 * fblvd

    m_blkt_total = m_blkt_total + m_blkt_steel_total + m_blkt_vanadium

    return (
        m_blkt_li2o,
        m_blkt_beryllium,
        m_blkt_steel_total,
        m_blkt_vanadium,
        m_blkt_total,
    )


def _reference_shield_mass(vol_shld_total, den_steel, vfshld):
    """`stellarator.py:1196-1206`, copied not re-derived. No guard of any kind."""
    whtshld = vol_shld_total * den_steel * (1.0e0 - vfshld)
    wpenshld = whtshld
    return whtshld, wpenshld


class TestBlanketComponentMasses(Tier1Contract):
    """S4's blanket mass block -> `calculate_blanket_component_masses`."""

    audit_record = "models/stellarator/stellarator_fwbs_s4.md"
    reference = _reference_blanket_component_masses
    ported = calculate_blanket_component_masses

    # `fwbs_variables.py`'s own defaults sit inside these ranges (`fblli2o = 0.08`,
    # `fblbe = 0.6`, `fblss = 0.09705`, `fblvd = 0.0`, `den_steel = 7800.0`); `fblvd`'s
    # lower bound is kept at PROCESS's default of exactly zero rather than lifted off it,
    # since a vanadium-free blanket is the configuration this port is validated against
    # and `m_blkt_vanadium`'s gradient must still be checked there. `vol_blkt_total`'s
    # range spans what S1 (`FwBlanketShieldGeometry`) produces for stellarator geometry.
    fuzz_bounds = {
        "vol_blkt_total": (100.0, 3000.0),
        "fblli2o": (0.0, 0.3),
        "fblbe": (0.0, 0.9),
        "den_steel": (7700.0, 7900.0),
        "fblss": (0.0, 0.3),
        "fblvd": (0.0, 0.3),
    }


class TestShieldMass(Tier1Contract):
    """S4's shield mass block -> `calculate_shield_mass`."""

    audit_record = "models/stellarator/stellarator_fwbs_s4.md"
    reference = _reference_shield_mass
    ported = calculate_shield_mass

    # `vfshld = 0.25` and `den_steel = 7800.0` are `fwbs_variables.py`'s own defaults and
    # the reference run's values (`stellarator_helias.IN.DAT` sets neither).
    fuzz_bounds = {
        "vol_shld_total": (100.0, 5000.0),
        "den_steel": (7700.0, 7900.0),
        "vfshld": (0.0, 0.9),
    }
