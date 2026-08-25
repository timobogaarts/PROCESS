"""Harness cases for the ported subset of `blankets/hcpb.py` (registry unit #13).

All three in-scope methods are tier-1 -- see `hcpb.md`. Legacy samples are lifted
verbatim from `tests/unit/models/blankets/test_ccfe_hcpb.py`'s own parametrised cases
(`baseline_2018_IN.DAT`-derived), the same "free oracle" reuse `test_harness.md`
describes for `density_limits.py`'s legacy points.
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.blankets.hcpb import (
    calculate_nuclear_heating_magnets,
    nuclear_heating_blanket,
    nuclear_heating_shield,
)
from process.core.model import DataStructure
from process.models.blankets.hcpb import CCFE_HCPB
from process.models.fw import FirstWall


def _reference_nuclear_heating_blanket(m_blkt_total, p_fusion_total_mw):
    """Call PROCESS's `nuclear_heating_blanket` staticmethod through the port's
    signature -- no `data` back-door to close, it is already a bare `@staticmethod`.
    """
    return CCFE_HCPB.nuclear_heating_blanket(
        m_blkt_total=m_blkt_total, p_fusion_total_mw=p_fusion_total_mw
    )


def _reference_nuclear_heating_shield(
    itart,
    dr_shld_outboard,
    dr_shld_inboard,
    shield_density,
    whtshld,
    x_blanket,
    p_fusion_total_mw,
):
    """Call PROCESS's `nuclear_heating_shield` staticmethod through the port's
    signature -- already a bare `@staticmethod`.
    """
    return CCFE_HCPB.nuclear_heating_shield(
        itart=itart,
        dr_shld_outboard=dr_shld_outboard,
        dr_shld_inboard=dr_shld_inboard,
        shield_density=shield_density,
        whtshld=whtshld,
        x_blanket=x_blanket,
        p_fusion_total_mw=p_fusion_total_mw,
    )


def _reference_nuclear_heating_magnets(
    radius_fw_channel,
    dx_fw_module,
    dr_fw_inboard,
    dr_fw_outboard,
    den_steel,
    m_blkt_total,
    vol_blkt_total,
    whtshld,
    vol_shld_total,
    dr_vv_inboard,
    dr_vv_outboard,
    m_vv,
    vol_vv,
    itart,
    dr_blkt_outboard,
    dr_blkt_inboard,
    dr_shld_outboard,
    dr_shld_inboard,
    fw_armour_thickness,
    whttflgs,
    m_tf_coils_total,
    p_fusion_total_mw,
):
    """Bind a real `CCFE_HCPB` instance's `data` (closing the back-door the source
    method itself does not close) and call `.nuclear_heating_magnets(False)`, matching
    the port's fused signature.
    """
    hcpb = CCFE_HCPB(fw=FirstWall())
    hcpb.data = DataStructure()

    hcpb.data.fwbs.radius_fw_channel = radius_fw_channel
    hcpb.data.fwbs.dx_fw_module = dx_fw_module
    hcpb.data.build.dr_fw_inboard = dr_fw_inboard
    hcpb.data.build.dr_fw_outboard = dr_fw_outboard
    hcpb.data.fwbs.den_steel = den_steel
    hcpb.data.fwbs.m_blkt_total = m_blkt_total
    hcpb.data.fwbs.vol_blkt_total = vol_blkt_total
    hcpb.data.fwbs.whtshld = whtshld
    hcpb.data.fwbs.vol_shld_total = vol_shld_total
    hcpb.data.build.dr_vv_inboard = dr_vv_inboard
    hcpb.data.build.dr_vv_outboard = dr_vv_outboard
    hcpb.data.fwbs.m_vv = m_vv
    hcpb.data.fwbs.vol_vv = vol_vv
    hcpb.data.physics.itart = itart
    hcpb.data.build.dr_blkt_outboard = dr_blkt_outboard
    hcpb.data.build.dr_blkt_inboard = dr_blkt_inboard
    hcpb.data.build.dr_shld_outboard = dr_shld_outboard
    hcpb.data.build.dr_shld_inboard = dr_shld_inboard
    hcpb.data.fwbs.fw_armour_thickness = fw_armour_thickness
    hcpb.data.tfcoil.whttflgs = whttflgs
    hcpb.data.tfcoil.m_tf_coils_total = m_tf_coils_total
    hcpb.data.physics.p_fusion_total_mw = p_fusion_total_mw

    hcpb.nuclear_heating_magnets(False)

    fwbs, ccfe_hcpb = hcpb.data.fwbs, hcpb.data.ccfe_hcpb
    return (
        fwbs.f_a_fw_coolant_inboard,
        fwbs.f_a_fw_coolant_outboard,
        ccfe_hcpb.armour_density,
        ccfe_hcpb.fw_density,
        ccfe_hcpb.blanket_density,
        ccfe_hcpb.shield_density,
        ccfe_hcpb.vv_density,
        ccfe_hcpb.x_blanket,
        ccfe_hcpb.x_shield,
        ccfe_hcpb.tfc_nuc_heating,
        fwbs.p_tf_nuclear_heat_mw,
    )


class TestNuclearHeatingBlanket(Tier1Contract):
    """`nuclear_heating_blanket` -> the same, unchanged.

    Samples are `tests/unit/models/blankets/test_ccfe_hcpb.py::
    test_nuclear_heating_blanket`'s two parametrised cases verbatim.
    """

    audit_record = "models/blankets/hcpb.md"
    reference = _reference_nuclear_heating_blanket
    ported = nuclear_heating_blanket

    samples = [
        legacy_sample(
            "blanket-baseline-2018-a",
            m_blkt_total=3501027.3252278985,
            p_fusion_total_mw=1986.0623241661431,
        ),
        legacy_sample(
            "blanket-baseline-2018-b",
            m_blkt_total=3507503.3737008357,
            p_fusion_total_mw=1985.4423932312809,
        ),
    ]

    fuzz_bounds = {
        "m_blkt_total": (1.0e5, 1.0e7),
        "p_fusion_total_mw": (100.0, 5000.0),
    }


class TestNuclearHeatingShield(Tier1Contract):
    """`nuclear_heating_shield` -> the same, unchanged.

    Samples are `tests/unit/models/blankets/test_ccfe_hcpb.py::
    test_nuclear_heating_shield`'s two parametrised cases verbatim (both `itart=0`; see
    `hcpb.md`'s "switches touched" -- no PROCESS unit test exercises `itart=1` for this
    method either, so the `itart=1` branch is exercised only by fuzzing below).
    """

    audit_record = "models/blankets/hcpb.md"
    reference = _reference_nuclear_heating_shield
    ported = nuclear_heating_shield

    static_argnames = ("itart",)

    samples = [
        legacy_sample(
            "shield-baseline-2018-a",
            itart=0,
            dr_shld_outboard=0.80000000000000004,
            dr_shld_inboard=0.30000000000000004,
            shield_density=3119.9999999999995,
            whtshld=2294873.8131476045,
            x_blanket=2.3374537748527975,
            p_fusion_total_mw=1986.0623241661431,
        ),
        legacy_sample(
            "shield-baseline-2018-b",
            itart=0,
            dr_shld_outboard=0.80000000000000004,
            dr_shld_inboard=0.30000000000000004,
            shield_density=3120,
            whtshld=2297808.3935174854,
            x_blanket=2.3374537748527979,
            p_fusion_total_mw=1985.4423932312809,
        ),
        # itart == 1 (spherical tokamak arm) -- no legacy PROCESS test exercises this
        # branch for `nuclear_heating_shield`; a fixed point, not fuzzed, so the branch
        # is checked deterministically rather than only-sometimes by chance.
        legacy_sample(
            "shield-st-branch",
            itart=1,
            dr_shld_outboard=0.80000000000000004,
            dr_shld_inboard=0.30000000000000004,
            shield_density=3119.9999999999995,
            whtshld=2294873.8131476045,
            x_blanket=2.3374537748527975,
            p_fusion_total_mw=1986.0623241661431,
        ),
    ]

    fuzz_bounds = {
        "dr_shld_outboard": (0.1, 2.0),
        "dr_shld_inboard": (0.1, 2.0),
        "shield_density": (1000.0, 8000.0),
        "whtshld": (1.0e5, 1.0e7),
        "x_blanket": (0.1, 10.0),
        "p_fusion_total_mw": (100.0, 5000.0),
    }
    fuzz_fixed = {"itart": 0}


class TestNuclearHeatingMagnets(Tier1Contract):
    """`calculate_nuclear_heating_magnets` -> `.nuclear_heating_magnets(False)`.

    Samples are `tests/unit/models/blankets/test_ccfe_hcpb.py::
    test_nuclear_heating_magnets`'s two parametrised cases verbatim (both `itart=0`;
    `dr_fw_outboard` is included even though PROCESS's own fixture also carries
    `p_tf_nuclear_heat_mw`/`f_a_fw_coolant_inboard`/`f_a_fw_coolant_outboard`/density
    fields as pre-existing state -- this port's function has no such state since every
    one of those fields is unconditionally overwritten, see `hcpb.md`'s data-footprint
    table, so the reference adapter only needs to seed the *inputs*, not the stale
    pre-call values).
    """

    audit_record = "models/blankets/hcpb.md"
    reference = _reference_nuclear_heating_magnets
    ported = calculate_nuclear_heating_magnets

    static_argnames = ("itart",)

    samples = [
        legacy_sample(
            "magnets-baseline-2018-a",
            radius_fw_channel=0.0060000000000000001,
            dx_fw_module=0.02,
            dr_fw_inboard=0.018000000000000002,
            dr_fw_outboard=0.018000000000000002,
            den_steel=7800,
            m_blkt_total=3501027.3252278985,
            vol_blkt_total=1397.9003011502937,
            whtshld=2294873.8131476045,
            vol_shld_total=735.53647857295027,
            dr_vv_inboard=0.30000000000000004,
            dr_vv_outboard=0.30000000000000004,
            m_vv=9043937.8018644415,
            vol_vv=1159.4792053672361,
            itart=0,
            dr_blkt_outboard=0.98199999999999998,
            dr_blkt_inboard=0.75500000000000012,
            dr_shld_outboard=0.80000000000000004,
            dr_shld_inboard=0.30000000000000004,
            fw_armour_thickness=0.0050000000000000001,
            whttflgs=0,
            m_tf_coils_total=19649856.627845347,
            p_fusion_total_mw=1986.0623241661431,
        ),
        legacy_sample(
            "magnets-baseline-2018-b",
            radius_fw_channel=0.0060000000000000001,
            dx_fw_module=0.02,
            dr_fw_inboard=0.018000000000000002,
            dr_fw_outboard=0.018000000000000002,
            den_steel=7800,
            m_blkt_total=3507503.3737008357,
            vol_blkt_total=1400.4860764869636,
            whtshld=2297808.3935174854,
            vol_shld_total=736.47704920432227,
            dr_vv_inboard=0.30000000000000004,
            dr_vv_outboard=0.30000000000000004,
            m_vv=9056931.558219457,
            vol_vv=1161.1450715665972,
            itart=0,
            dr_blkt_outboard=0.98199999999999998,
            dr_blkt_inboard=0.75500000000000012,
            dr_shld_outboard=0.80000000000000004,
            dr_shld_inboard=0.30000000000000004,
            fw_armour_thickness=0.0050000000000000001,
            whttflgs=0,
            m_tf_coils_total=19662548.210142396,
            p_fusion_total_mw=1985.4423932312809,
        ),
    ]

    fuzz_bounds = {
        "radius_fw_channel": (0.001, 0.02),
        "dx_fw_module": (0.005, 0.05),
        "dr_fw_inboard": (0.005, 0.05),
        "dr_fw_outboard": (0.005, 0.05),
        "den_steel": (6000.0, 9000.0),
        "m_blkt_total": (1.0e5, 1.0e7),
        "vol_blkt_total": (100.0, 3000.0),
        "whtshld": (1.0e5, 1.0e7),
        "vol_shld_total": (100.0, 2000.0),
        "dr_vv_inboard": (0.05, 1.0),
        "dr_vv_outboard": (0.05, 1.0),
        "m_vv": (1.0e5, 1.0e7),
        "vol_vv": (100.0, 3000.0),
        "dr_blkt_outboard": (0.1, 2.0),
        "dr_blkt_inboard": (0.1, 2.0),
        "dr_shld_outboard": (0.1, 2.0),
        "dr_shld_inboard": (0.1, 2.0),
        "fw_armour_thickness": (0.001, 0.02),
        "whttflgs": (1.0e5, 3.0e7),
        "m_tf_coils_total": (1.0e6, 5.0e7),
        "p_fusion_total_mw": (100.0, 5000.0),
    }
    fuzz_fixed = {"itart": 0}
