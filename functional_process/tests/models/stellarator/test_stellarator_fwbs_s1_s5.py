"""Harness cases for `st_fwbs`'s S1/S5 sub-computations (see
`stellarator_E_fwbs_synthesis.md`, ported in `stellarator_fwbs_s1_s5.py`).

Both reference adapters call PROCESS's real `Stellarator.st_fwbs` end to end (not a
hand-reimplemented formula) -- `st_fwbs` is one large method with no standalone
callable for just S1 or S5, so the adapter builds a `Stellarator` instance without going
through `__init__` (which would need all eleven injected sub-models constructed just to
reach two arithmetic blocks that never touch any of them), sets `.data` and the
`first_call_stfwbs` flag `__init__` would otherwise set, and reads back only the fields
each unit actually owns. `blktmodel = 0` throughout, so S2 (`blanket_neutronics`, gated
on `blktmodel == 1`) never runs -- confirmed by reading S1/S3/S4/S6's bodies for any
`self.<submodel>` reference: none, so nothing beyond `self.data` is ever touched at
`blktmodel = 0`.
"""

from functional_process.cottax._harness import Tier1Contract, legacy_sample
from functional_process.cottax.stellarator.stellarator_fwbs_s1_s5 import (
    calculate_cryostat_and_vv_geometry,
    calculate_fw_blanket_shield_geometry,
)
from process.core.model import DataStructure
from process.models.engineering.pumping import CoolantType
from process.models.power import PumpingPowerModelTypes
from process.models.stellarator.stellarator import Stellarator


def _run_st_fwbs(**overrides):
    """Real `Stellarator.st_fwbs(output=False)` at `blktmodel = 0`, fields overridden.

    Returns the `Stellarator` instance so a reference adapter can read back whatever
    fields its own unit owns.
    """
    s = Stellarator.__new__(Stellarator)
    s.data = DataStructure()
    s.first_call_stfwbs = True
    d = s.data

    defaults = {
        "fwbs.blktmodel": 0,
        "heat_transport.ipowerflow": 0,
        "costs.abktflnc": 15.0,
        "physics.pflux_fw_neutron_mw": 2.0,
        "costs.life_plant": 30.0,
        "first_wall.a_fw_total": 1468.3,
        "physics.rminor": 3.2664,
        "build.dr_fw_plasma_gap_inboard": 0.3,
        "build.dr_fw_inboard": 0.018,
        "build.dr_fw_plasma_gap_outboard": 0.3,
        "build.dr_fw_outboard": 0.018,
        "physics.a_plasma_surface": 1468.3,
        "fwbs.fhole": 0.05,
        "fwbs.f_ster_div_single": 0.115,
        "fwbs.f_a_fw_outboard_hcd": 0.0,
        "build.dr_blkt_inboard": 0.4712,
        "build.dr_blkt_outboard": 0.4712,
        "fwbs.fvolsi": 1.0,
        "fwbs.fvolso": 1.0,
        "build.dr_shld_inboard": 0.4,
        "build.dr_shld_outboard": 0.4,
        "physics.p_neutron_total_mw": 1500.0,
        "stellarator_config.stella_config_neutron_peakfactor": 1.5,
        "build.r_tf_outboard_mid": 12.5,
        "build.dr_tf_outboard": 0.6381,
        "fwbs.dr_pf_cryostat": 0.5,
        "physics.rmajor": 8.1386,
        "build.dr_cryostat": 0.15,
        "build.dr_vv_inboard": 0.07,
        "build.dr_vv_outboard": 0.07,
        "fwbs.fvoldw": 1.2,
        "fwbs.den_steel": 7800.0,
        # Only reached at `ipowerflow == 1` (a later, unrelated block of `st_fwbs`
        # neither S1 nor S5 owns) -- populated so the real end-to-end call doesn't
        # raise; values are physically plausible but not independently verified,
        # since nothing S1/S5 own depends on them.
        "fwbs.i_p_coolant_pumping": int(PumpingPowerModelTypes.FRACTION_OF_HEAT),
        "heat_transport.f_p_fw_coolant_pump_total_heat": 0.05,
        "heat_transport.f_p_blkt_coolant_pump_total_heat": 0.05,
        "current_drive.p_beam_orbit_loss_mw": 0.0,
        "fwbs.f_p_blkt_multiplication": 1.269,
        "fwbs.declblkt": 0.075,
        "fwbs.i_blkt_coolant_type": int(CoolantType.HELIUM),
    }
    defaults.update(overrides)
    for path, value in defaults.items():
        area, field = path.split(".")
        setattr(getattr(d, area), field, value)

    s.st_fwbs(False)
    return s


def _reference_fw_blanket_shield_geometry(
    abktflnc,
    pflux_fw_neutron_mw,
    life_plant,
    a_fw_total,
    rminor,
    dr_fw_plasma_gap_inboard,
    dr_fw_inboard,
    dr_fw_plasma_gap_outboard,
    dr_fw_outboard,
    ipowerflow,
    a_plasma_surface,
    fhole,
    f_ster_div_single,
    f_a_fw_outboard_hcd,
    dr_blkt_inboard,
    dr_blkt_outboard,
    fvolsi,
    fvolso,
    dr_shld_inboard,
    dr_shld_outboard,
    p_neutron_total_mw,
    stella_config_neutron_peakfactor,
):
    """Call PROCESS's real `st_fwbs` through S1's port signature."""
    s = _run_st_fwbs(**{
        "costs.abktflnc": abktflnc,
        "physics.pflux_fw_neutron_mw": pflux_fw_neutron_mw,
        "costs.life_plant": life_plant,
        "first_wall.a_fw_total": a_fw_total,
        "physics.rminor": rminor,
        "build.dr_fw_plasma_gap_inboard": dr_fw_plasma_gap_inboard,
        "build.dr_fw_inboard": dr_fw_inboard,
        "build.dr_fw_plasma_gap_outboard": dr_fw_plasma_gap_outboard,
        "build.dr_fw_outboard": dr_fw_outboard,
        "heat_transport.ipowerflow": ipowerflow,
        "physics.a_plasma_surface": a_plasma_surface,
        "fwbs.fhole": fhole,
        "fwbs.f_ster_div_single": f_ster_div_single,
        "fwbs.f_a_fw_outboard_hcd": f_a_fw_outboard_hcd,
        "build.dr_blkt_inboard": dr_blkt_inboard,
        "build.dr_blkt_outboard": dr_blkt_outboard,
        "fwbs.fvolsi": fvolsi,
        "fwbs.fvolso": fvolso,
        "build.dr_shld_inboard": dr_shld_inboard,
        "build.dr_shld_outboard": dr_shld_outboard,
        "physics.p_neutron_total_mw": p_neutron_total_mw,
        "stellarator_config.stella_config_neutron_peakfactor": (
            stella_config_neutron_peakfactor
        ),
    })
    d = s.data
    return (
        d.fwbs.life_fw_fpy,
        d.first_wall.a_fw_inboard,
        d.first_wall.a_fw_outboard,
        d.build.a_blkt_total_surface,
        d.build.a_blkt_inboard_surface,
        d.build.a_blkt_outboard_surface,
        d.fwbs.vol_blkt_inboard,
        d.fwbs.vol_blkt_outboard,
        d.fwbs.vol_blkt_total,
        d.build.a_shld_total_surface,
        d.build.a_shld_inboard_surface,
        d.build.a_shld_outboard_surface,
        d.fwbs.vol_shld_total,
        d.fwbs.pnucloss,
        d.fwbs.wallpf,
    )


class TestFwBlanketShieldGeometry(Tier1Contract):
    """S1 -> `calculate_fw_blanket_shield_geometry`."""

    audit_record = "models/stellarator/stellarator_E_fwbs_synthesis.md"
    reference = _reference_fw_blanket_shield_geometry
    ported = calculate_fw_blanket_shield_geometry
    static_argnames = ("ipowerflow",)

    samples = [
        legacy_sample(
            "helias5b-ipowerflow-1",
            abktflnc=15.0,
            pflux_fw_neutron_mw=1.8,
            life_plant=30.0,
            a_fw_total=1468.3,
            rminor=3.2664,
            dr_fw_plasma_gap_inboard=0.3,
            dr_fw_inboard=0.018,
            dr_fw_plasma_gap_outboard=0.3,
            dr_fw_outboard=0.018,
            ipowerflow=1,
            a_plasma_surface=1468.3,
            fhole=0.05,
            f_ster_div_single=0.115,
            f_a_fw_outboard_hcd=0.0,
            dr_blkt_inboard=0.4712,
            dr_blkt_outboard=0.4712,
            fvolsi=1.0,
            fvolso=1.0,
            dr_shld_inboard=0.4,
            dr_shld_outboard=0.4,
            p_neutron_total_mw=1500.0,
            stella_config_neutron_peakfactor=1.5,
        ),
        legacy_sample(
            "helias5b-ipowerflow-0",
            abktflnc=15.0,
            pflux_fw_neutron_mw=1.8,
            life_plant=30.0,
            a_fw_total=1468.3,
            rminor=3.2664,
            dr_fw_plasma_gap_inboard=0.3,
            dr_fw_inboard=0.018,
            dr_fw_plasma_gap_outboard=0.3,
            dr_fw_outboard=0.018,
            ipowerflow=0,
            a_plasma_surface=1468.3,
            fhole=0.05,
            f_ster_div_single=0.115,
            f_a_fw_outboard_hcd=0.0,
            dr_blkt_inboard=0.4712,
            dr_blkt_outboard=0.4712,
            fvolsi=1.0,
            fvolso=1.0,
            dr_shld_inboard=0.4,
            dr_shld_outboard=0.4,
            p_neutron_total_mw=1500.0,
            stella_config_neutron_peakfactor=1.5,
        ),
    ]

    fuzz_bounds = {
        "abktflnc": (5.0, 25.0),
        "pflux_fw_neutron_mw": (0.5, 4.0),
        "life_plant": (20.0, 40.0),
        "a_fw_total": (500.0, 3000.0),
        "rminor": (1.5, 5.0),
        "dr_fw_plasma_gap_inboard": (0.05, 0.5),
        "dr_fw_inboard": (0.005, 0.05),
        "dr_fw_plasma_gap_outboard": (0.05, 0.5),
        "dr_fw_outboard": (0.005, 0.05),
        "ipowerflow": (0.0, 0.0),
        "a_plasma_surface": (500.0, 3000.0),
        "fhole": (0.01, 0.1),
        "f_ster_div_single": (0.05, 0.2),
        "f_a_fw_outboard_hcd": (0.0, 0.1),
        "dr_blkt_inboard": (0.2, 0.8),
        "dr_blkt_outboard": (0.2, 0.8),
        "fvolsi": (0.8, 1.2),
        "fvolso": (0.8, 1.2),
        "dr_shld_inboard": (0.1, 0.6),
        "dr_shld_outboard": (0.1, 0.6),
        "p_neutron_total_mw": (500.0, 3000.0),
        "stella_config_neutron_peakfactor": (1.0, 2.0),
    }
    fuzz_fixed = {"ipowerflow": 0}


def _reference_cryostat_and_vv_geometry(
    r_tf_outboard_mid,
    dr_tf_outboard,
    dr_pf_cryostat,
    rmajor,
    dr_cryostat,
    dr_fw_plasma_gap_inboard,
    dr_fw_inboard,
    dr_blkt_inboard,
    dr_shld_inboard,
    dr_fw_plasma_gap_outboard,
    dr_fw_outboard,
    dr_blkt_outboard,
    dr_shld_outboard,
    rminor,
    dr_vv_inboard,
    dr_vv_outboard,
    a_plasma_surface,
    fvoldw,
    den_steel,
):
    """Call PROCESS's real `st_fwbs` through S5's port signature."""
    s = _run_st_fwbs(**{
        "build.r_tf_outboard_mid": r_tf_outboard_mid,
        "build.dr_tf_outboard": dr_tf_outboard,
        "fwbs.dr_pf_cryostat": dr_pf_cryostat,
        "physics.rmajor": rmajor,
        "build.dr_cryostat": dr_cryostat,
        "build.dr_fw_plasma_gap_inboard": dr_fw_plasma_gap_inboard,
        "build.dr_fw_inboard": dr_fw_inboard,
        "build.dr_blkt_inboard": dr_blkt_inboard,
        "build.dr_shld_inboard": dr_shld_inboard,
        "build.dr_fw_plasma_gap_outboard": dr_fw_plasma_gap_outboard,
        "build.dr_fw_outboard": dr_fw_outboard,
        "build.dr_blkt_outboard": dr_blkt_outboard,
        "build.dr_shld_outboard": dr_shld_outboard,
        "physics.rminor": rminor,
        "build.dr_vv_inboard": dr_vv_inboard,
        "build.dr_vv_outboard": dr_vv_outboard,
        "physics.a_plasma_surface": a_plasma_surface,
        "fwbs.fvoldw": fvoldw,
        "fwbs.den_steel": den_steel,
    })
    d = s.data
    return (
        d.fwbs.r_cryostat_inboard,
        d.fwbs.vol_cryostat,
        d.fwbs.vol_vv,
        d.fwbs.m_vv,
        d.fwbs.dewmkg,
    )


class TestCryostatAndVvGeometry(Tier1Contract):
    """S5 -> `calculate_cryostat_and_vv_geometry`."""

    audit_record = "models/stellarator/stellarator_E_fwbs_synthesis.md"
    reference = _reference_cryostat_and_vv_geometry
    ported = calculate_cryostat_and_vv_geometry

    samples = [
        legacy_sample(
            "helias5b",
            r_tf_outboard_mid=12.5,
            dr_tf_outboard=0.6381,
            dr_pf_cryostat=0.5,
            rmajor=8.1386,
            dr_cryostat=0.15,
            dr_fw_plasma_gap_inboard=0.3,
            dr_fw_inboard=0.018,
            dr_blkt_inboard=0.4712,
            dr_shld_inboard=0.4,
            dr_fw_plasma_gap_outboard=0.3,
            dr_fw_outboard=0.018,
            dr_blkt_outboard=0.4712,
            dr_shld_outboard=0.4,
            rminor=3.2664,
            dr_vv_inboard=0.07,
            dr_vv_outboard=0.07,
            a_plasma_surface=1468.3,
            fvoldw=1.2,
            den_steel=7800.0,
        ),
    ]

    fuzz_bounds = {
        "r_tf_outboard_mid": (8.0, 20.0),
        "dr_tf_outboard": (0.3, 1.2),
        "dr_pf_cryostat": (0.2, 1.0),
        "rmajor": (6.0, 15.0),
        "dr_cryostat": (0.05, 0.3),
        "dr_fw_plasma_gap_inboard": (0.05, 0.5),
        "dr_fw_inboard": (0.005, 0.05),
        "dr_blkt_inboard": (0.2, 0.8),
        "dr_shld_inboard": (0.1, 0.6),
        "dr_fw_plasma_gap_outboard": (0.05, 0.5),
        "dr_fw_outboard": (0.005, 0.05),
        "dr_blkt_outboard": (0.2, 0.8),
        "dr_shld_outboard": (0.1, 0.6),
        "rminor": (1.5, 5.0),
        "dr_vv_inboard": (0.03, 0.15),
        "dr_vv_outboard": (0.03, 0.15),
        "a_plasma_surface": (500.0, 3000.0),
        "fvoldw": (0.9, 1.5),
        "den_steel": (7700.0, 7900.0),
    }
