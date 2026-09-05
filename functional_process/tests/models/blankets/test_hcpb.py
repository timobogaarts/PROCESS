"""Harness cases for the ported subset of `blankets/hcpb.py` (registry unit #13).

Every ported function is tier-1 -- see `hcpb.md`. Two families of reference adapter:

- **The seven PROCESS callables.** `nuclear_heating_fw`/`_blanket`/`_shield` are bare
  `@staticmethod`s and need only a keyword rename; `nuclear_heating_magnets`,
  `component_masses` and `powerflow_calc` are `self`-bound and get a bound
  `DataStructure`. Writing those adapters is where the audit's "close the `data`
  back-door" claim gets tested rather than asserted.
- **`_RenormalisationOnly`**, for the one ported block that is not a PROCESS callable at
  all. `run()`'s renormalisation (`hcpb.py:195-276`) is inline, so its reference is a
  subclass of `CCFE_HCPB` with every *other* step of `run()` stubbed out and the four
  unnormalised powers injected where the real `nuclear_heating_*` calls would have
  written them. That runs PROCESS's own source lines for the block under test --
  the alternative, transcribing the arithmetic into the test module, would diff the port
  against itself.

Legacy sample points are lifted verbatim from
`tests/unit/models/blankets/test_ccfe_hcpb.py`'s parametrised cases (its own docstrings
name `baseline_2018_IN.DAT` and `large_tokamak_eval.IN.DAT` as their source), plus a
second point per contract read out of the assembled `DataStructure` after four
`Caller._call_models_once` passes on `large_tokamak_eval.IN.DAT` -- the reference run
`_audit/tokamak_call_surface.md` traced. The four unnormalised powers in
`TestNuclearHeatingRenormalisation` are that run's, and they sum to its own
`.ccfe_hcpb.pnuc_tot_blk_sector` (1411.3788234012184) exactly, which is an independent
check that the sample is on the operating point and not near it.
"""

from functional_process.cottax._harness import Tier1Contract, Tolerance, legacy_sample
from functional_process.cottax.blankets.hcpb import (
    calculate_centrepost_angle_fraction,
    calculate_centrepost_fast_neutron_flux_superconducting,
    calculate_centrepost_neutronics_absent,
    calculate_centrepost_neutronics_spherical_tokamak_superconducting,
    calculate_centrepost_nuclear_heating_superconducting,
    calculate_component_masses,
    calculate_divertor_surface_and_plate_mass_double_null,
    calculate_divertor_surface_and_plate_mass_single_null,
    calculate_first_wall_radiation_powers,
    calculate_fw_coolant_void_fractions,
    calculate_nuclear_heating_magnets_conventional,
    calculate_nuclear_heating_magnets_spherical_tokamak,
    calculate_nuclear_heating_renormalisation_double_null_conventional,
    calculate_nuclear_heating_renormalisation_double_null_spherical_tokamak,
    calculate_nuclear_heating_renormalisation_single_null_conventional,
    calculate_nuclear_heating_renormalisation_single_null_spherical_tokamak,
    calculate_pumping_power_mechanical_with_pressure_drop,
    nuclear_heating_blanket,
    nuclear_heating_fw,
    nuclear_heating_shield_conventional,
    nuclear_heating_shield_spherical_tokamak,
)
from process.core.exceptions import ProcessValueError
from process.core.model import DataStructure
from process.models.blankets.hcpb import CCFE_HCPB
from process.models.engineering.pumping import CoolantType
from process.models.fw import FirstWall

_AUDIT_RECORD = "models/blankets/hcpb.md"


def _hcpb():
    """A `CCFE_HCPB` bound to a fresh `DataStructure`."""
    model = CCFE_HCPB(fw=FirstWall())
    model.data = DataStructure()
    return model


# --------------------------------------------------------------------------------------
# reference adapters
# --------------------------------------------------------------------------------------


def _seed_magnets(model, itart, **kwargs):
    """Seed `nuclear_heating_magnets`' twenty-two reads onto a bound `DataStructure`."""
    data = model.data
    data.physics.itart = itart
    data.fwbs.radius_fw_channel = kwargs["radius_fw_channel"]
    data.fwbs.dx_fw_module = kwargs["dx_fw_module"]
    data.build.dr_fw_inboard = kwargs["dr_fw_inboard"]
    data.build.dr_fw_outboard = kwargs["dr_fw_outboard"]
    data.fwbs.den_steel = kwargs["den_steel"]
    data.fwbs.m_blkt_total = kwargs["m_blkt_total"]
    data.fwbs.vol_blkt_total = kwargs["vol_blkt_total"]
    data.fwbs.whtshld = kwargs["whtshld"]
    data.fwbs.vol_shld_total = kwargs["vol_shld_total"]
    data.build.dr_vv_inboard = kwargs["dr_vv_inboard"]
    data.build.dr_vv_outboard = kwargs["dr_vv_outboard"]
    data.fwbs.m_vv = kwargs["m_vv"]
    data.fwbs.vol_vv = kwargs["vol_vv"]
    data.build.dr_blkt_outboard = kwargs["dr_blkt_outboard"]
    data.build.dr_blkt_inboard = kwargs.get("dr_blkt_inboard", 0.0)
    data.build.dr_shld_outboard = kwargs["dr_shld_outboard"]
    data.build.dr_shld_inboard = kwargs.get("dr_shld_inboard", 0.0)
    data.fwbs.fw_armour_thickness = kwargs["fw_armour_thickness"]
    data.tfcoil.whttflgs = kwargs.get("whttflgs", 0.0)
    data.tfcoil.m_tf_coils_total = kwargs.get("m_tf_coils_total", 0.0)
    data.physics.p_fusion_total_mw = kwargs["p_fusion_total_mw"]


def _read_magnets(model):
    """The nine outputs the ported magnets functions return, in their order."""
    data = model.data
    return (
        data.ccfe_hcpb.armour_density,
        data.ccfe_hcpb.fw_density,
        data.ccfe_hcpb.blanket_density,
        data.ccfe_hcpb.shield_density,
        data.ccfe_hcpb.vv_density,
        data.ccfe_hcpb.x_blanket,
        data.ccfe_hcpb.x_shield,
        data.ccfe_hcpb.tfc_nuc_heating,
        data.fwbs.p_tf_nuclear_heat_mw,
    )


def _reference_fw_coolant_void_fractions(radius_fw_channel, dx_fw_module, dr_fw_inboard):
    """The two void fractions `nuclear_heating_magnets` writes at `hcpb.py:483-490`.

    Driven through the whole PROCESS method rather than by re-deriving three lines: the
    port's claim is that these two fields are *exactly* what that method writes, which is
    what makes lifting them into their own node value-preserving.
    """
    model = _hcpb()
    _seed_magnets(
        model,
        itart=0,
        radius_fw_channel=radius_fw_channel,
        dx_fw_module=dx_fw_module,
        dr_fw_inboard=dr_fw_inboard,
        dr_fw_outboard=0.018,
        den_steel=7800.0,
        m_blkt_total=3.0e6,
        vol_blkt_total=1200.0,
        whtshld=2.4e6,
        vol_shld_total=780.0,
        dr_vv_inboard=0.3,
        dr_vv_outboard=0.3,
        m_vv=7.9e6,
        vol_vv=1000.0,
        dr_blkt_outboard=1.0,
        dr_blkt_inboard=0.7,
        dr_shld_outboard=0.8,
        dr_shld_inboard=0.3,
        fw_armour_thickness=0.005,
        m_tf_coils_total=1.4e7,
        p_fusion_total_mw=1630.0,
    )
    model.nuclear_heating_magnets(False)
    return (
        model.data.fwbs.f_a_fw_coolant_inboard,
        model.data.fwbs.f_a_fw_coolant_outboard,
    )


def _reference_nuclear_heating_magnets_conventional(**kwargs):
    """`nuclear_heating_magnets(False)` at `itart == 0`, less the two void fractions."""
    model = _hcpb()
    _seed_magnets(model, itart=0, **kwargs)
    model.nuclear_heating_magnets(False)
    return _read_magnets(model)


def _reference_nuclear_heating_magnets_spherical_tokamak(**kwargs):
    """`nuclear_heating_magnets(False)` at `itart == 1`."""
    model = _hcpb()
    _seed_magnets(model, itart=1, **kwargs)
    model.nuclear_heating_magnets(False)
    return _read_magnets(model)


def _reference_nuclear_heating_fw(
    m_fw_total, fw_armour_u_nuc_heating, p_fusion_total_mw
):
    """`nuclear_heating_fw`, already a bare `@staticmethod`."""
    return CCFE_HCPB.nuclear_heating_fw(
        m_fw_total=m_fw_total,
        fw_armour_u_nuc_heating=fw_armour_u_nuc_heating,
        p_fusion_total_mw=p_fusion_total_mw,
    )


def _reference_nuclear_heating_blanket(m_blkt_total, p_fusion_total_mw):
    """`nuclear_heating_blanket`, already a bare `@staticmethod`."""
    return CCFE_HCPB.nuclear_heating_blanket(
        m_blkt_total=m_blkt_total, p_fusion_total_mw=p_fusion_total_mw
    )


def _reference_nuclear_heating_shield_conventional(
    dr_shld_outboard,
    dr_shld_inboard,
    shield_density,
    whtshld,
    x_blanket,
    p_fusion_total_mw,
):
    """`nuclear_heating_shield` at `itart == 0`.

    The switch is the adapter's, not a port of the function under test.
    """
    return CCFE_HCPB.nuclear_heating_shield(
        itart=0,
        dr_shld_outboard=dr_shld_outboard,
        dr_shld_inboard=dr_shld_inboard,
        shield_density=shield_density,
        whtshld=whtshld,
        x_blanket=x_blanket,
        p_fusion_total_mw=p_fusion_total_mw,
    )


def _reference_nuclear_heating_shield_spherical_tokamak(
    dr_shld_outboard, shield_density, whtshld, x_blanket, p_fusion_total_mw
):
    """`nuclear_heating_shield` at `itart == 1`.

    `dr_shld_inboard` is not a parameter of the port on this arm, so the adapter supplies
    a deliberately absurd value: if the ported function had kept the read, the two would
    disagree by a mile rather than by a rounding error.
    """
    return CCFE_HCPB.nuclear_heating_shield(
        itart=1,
        dr_shld_outboard=dr_shld_outboard,
        dr_shld_inboard=1.0e6,
        shield_density=shield_density,
        whtshld=whtshld,
        x_blanket=x_blanket,
        p_fusion_total_mw=p_fusion_total_mw,
    )


def _seed_component_masses(model, **kwargs):
    """Seed `component_masses`' reads onto a bound `DataStructure`.

    `n_divertors` defaults to `1`; the double-null divertor-mass adapter passes `2`.
    """
    data = model.data
    data.divertor.n_divertors = kwargs.get("n_divertors", 1)
    data.divertor.a_div_surface_total = kwargs["a_div_surface_total"]
    data.divertor.f_vol_div_coolant = kwargs["f_vol_div_coolant"]
    data.divertor.dx_div_plate = kwargs["dx_div_plate"]
    data.divertor.fdiva = kwargs.get("fdiva", 1.11)
    data.divertor.den_div_structure = kwargs.get("den_div_structure", 10000.0)
    data.physics.rmajor = kwargs.get("rmajor", 8.0)
    data.physics.rminor = kwargs.get("rminor", 2.6666666666666665)
    data.fwbs.vol_blkt_total = kwargs["vol_blkt_total"]
    data.fwbs.f_a_blkt_cooling_channels = kwargs["f_a_blkt_cooling_channels"]
    data.fwbs.vol_shld_total = kwargs["vol_shld_total"]
    data.fwbs.vfshld = kwargs["vfshld"]
    data.first_wall.a_fw_inboard = kwargs["a_fw_inboard"]
    data.first_wall.a_fw_outboard = kwargs["a_fw_outboard"]
    data.first_wall.a_fw_total = kwargs["a_fw_total"]
    data.build.dr_fw_inboard = kwargs["dr_fw_inboard"]
    data.build.dr_fw_outboard = kwargs["dr_fw_outboard"]
    data.fwbs.f_a_fw_coolant_inboard = kwargs["f_a_fw_coolant_inboard"]
    data.fwbs.f_a_fw_coolant_outboard = kwargs["f_a_fw_coolant_outboard"]
    data.fwbs.den_steel = kwargs["den_steel"]
    data.physics.a_plasma_surface = kwargs["a_plasma_surface"]
    data.fwbs.fw_armour_thickness = kwargs["fw_armour_thickness"]
    data.fwbs.breeder_f = kwargs["breeder_f"]
    data.fwbs.breeder_multiplier = kwargs["breeder_multiplier"]
    data.fwbs.vfcblkt = kwargs["vfcblkt"]
    data.fwbs.vfpblkt = kwargs["vfpblkt"]


def _reference_divertor_surface_and_plate_mass_single_null(
    fdiva, rmajor, rminor, den_div_structure, f_vol_div_coolant, dx_div_plate
):
    """The divertor pair `component_masses` writes at `hcpb.py:353-367`.

    At `n_divertors == 1`.
    """
    model = _hcpb()
    _seed_component_masses(
        model,
        fdiva=fdiva,
        rmajor=rmajor,
        rminor=rminor,
        den_div_structure=den_div_structure,
        a_div_surface_total=0.0,
        f_vol_div_coolant=f_vol_div_coolant,
        dx_div_plate=dx_div_plate,
        vol_blkt_total=1200.0,
        f_a_blkt_cooling_channels=0.25,
        vol_shld_total=780.0,
        vfshld=0.6,
        a_fw_inboard=600.0,
        a_fw_outboard=1000.0,
        a_fw_total=1600.0,
        dr_fw_inboard=0.018,
        dr_fw_outboard=0.018,
        f_a_fw_coolant_inboard=0.3,
        f_a_fw_coolant_outboard=0.3,
        den_steel=7800.0,
        a_plasma_surface=1170.0,
        fw_armour_thickness=0.005,
        breeder_f=0.5,
        breeder_multiplier=0.75,
        vfcblkt=0.05295,
        vfpblkt=0.1,
    )
    model.component_masses()
    return model.data.divertor.a_div_surface_total, model.data.divertor.m_div_plate


def _reference_divertor_surface_and_plate_mass_double_null(
    fdiva, rmajor, rminor, den_div_structure, f_vol_div_coolant, dx_div_plate
):
    """The divertor pair `component_masses` writes at `hcpb.py:353-367`.

    At `n_divertors == 2` -- the same seed as the single-null adapter with the divertor
    count changed, since the arms read the same six fields and differ only in
    `:360-361`'s factor of two.
    """
    model = _hcpb()
    _seed_component_masses(
        model,
        n_divertors=2,
        fdiva=fdiva,
        rmajor=rmajor,
        rminor=rminor,
        den_div_structure=den_div_structure,
        a_div_surface_total=0.0,
        f_vol_div_coolant=f_vol_div_coolant,
        dx_div_plate=dx_div_plate,
        vol_blkt_total=1200.0,
        f_a_blkt_cooling_channels=0.25,
        vol_shld_total=780.0,
        vfshld=0.6,
        a_fw_inboard=600.0,
        a_fw_outboard=1000.0,
        a_fw_total=1600.0,
        dr_fw_inboard=0.018,
        dr_fw_outboard=0.018,
        f_a_fw_coolant_inboard=0.3,
        f_a_fw_coolant_outboard=0.3,
        den_steel=7800.0,
        a_plasma_surface=1170.0,
        fw_armour_thickness=0.005,
        breeder_f=0.5,
        breeder_multiplier=0.75,
        vfcblkt=0.05295,
        vfpblkt=0.1,
    )
    model.component_masses()
    return model.data.divertor.a_div_surface_total, model.data.divertor.m_div_plate


def _reference_component_masses(**kwargs):
    """`component_masses()` less the divertor pair, read back in the port's order."""
    model = _hcpb()
    _seed_component_masses(model, **kwargs)
    model.component_masses()

    data = model.data
    return (
        data.fwbs.m_fw_blkt_div_coolant_total,
        data.fwbs.fwclfr,
        data.fwbs.whtshld,
        data.fwbs.wpenshld,
        data.fwbs.vol_fw_total,
        data.fwbs.m_fw_total,
        data.fwbs.fw_armour_vol,
        data.fwbs.fw_armour_mass,
        data.fwbs.f_vol_blkt_li4sio4,
        data.fwbs.f_vol_blkt_tibe12,
        data.fwbs.m_blkt_tibe12,
        data.fwbs.m_blkt_li4sio4,
        data.fwbs.m_blkt_beryllium,
        data.fwbs.m_blkt_li2o,
        data.fwbs.f_vol_blkt_steel,
        data.fwbs.m_blkt_steel_total,
        data.fwbs.m_blkt_total,
        data.fwbs.armour_fw_bl_mass,
    )


class _RenormalisationOnly(CCFE_HCPB):
    """`CCFE_HCPB.run()` with everything except `hcpb.py:103-276` stubbed out.

    The renormalisation block is not a method in PROCESS, only lines inside `run()`, so
    the only way to diff the port against *PROCESS's own source* rather than against a
    transcription of it is to run `run()` with its other steps replaced. Each override
    below corresponds to one call `run()` makes; the four `nuclear_heating_*` stubs write
    or return the sample's unnormalised powers exactly where the real routines would.

    What is left executing is `:103-148` (the `itart` branch -- the `else` arm's four
    zeros, which `TestCentrepostNeutronicsAbsent` reads back) and `:195-276` (the
    renormalisation), which is precisely the two blocks this file ports out of `run()`.
    """

    unnormalised = ()

    # `component_volumes` and the module-geometry chain: `blanket_library.py`'s job, and
    # nothing they write is read by the block under test.
    def component_volumes(self):
        """Stub."""

    def blkt_outboard_poloidal_plasma_angle(self, **kwargs):
        """Stub."""
        return 0.0

    f_deg_blkt_outboard_poloidal_plasma = 0.0
    """Shadows `OutboardBlanket`'s `@property` with a plain value."""

    def calculate_blkt_inboard_poloidal_plasma_angle(self, **kwargs):
        """Stub."""
        return 0.0

    def pipe_hydraulic_diameter(self, **kwargs):
        """Stub."""
        return 0.0

    def set_blanket_module_geometry(self):
        """Stub."""

    def calculate_blanket_inboard_module_geometry(self, **kwargs):
        """Stub."""
        return 0.0

    def calculate_blanket_outboard_module_geometry(self, **kwargs):
        """Stub."""
        return 0.0

    def component_masses(self):
        """Stub."""

    def nuclear_heating_magnets(self, output):
        """Inject the unnormalised TF-coil heating where `hcpb.py:572` would write it."""
        self.data.fwbs.p_tf_nuclear_heat_mw = self.unnormalised[3]

    def nuclear_heating_fw(self, **kwargs):
        """Return the unnormalised first-wall heating."""
        return self.unnormalised[0]

    def nuclear_heating_blanket(self, **kwargs):
        """Return the unnormalised blanket heating, plus a dummy `exp_blanket`."""
        return self.unnormalised[1], 0.0

    def nuclear_heating_shield(self, **kwargs):
        """Return the unnormalised shield heating, plus three dummies."""
        return self.unnormalised[2], 0.0, 0.0, 0.0

    def powerflow_calc(self, output):
        """Stub."""


def _run_renormalisation(
    p_fw_nuclear_heat_total_mw_unnormalised=0.0,
    p_blkt_nuclear_heat_total_mw_unnormalised=0.0,
    p_shld_nuclear_heat_mw_unnormalised=0.0,
    p_tf_nuclear_heat_mw_unnormalised=0.0,
    f_ster_div_single=0.0725040362777958,
    f_p_blkt_multiplication=1.269,
    p_neutron_total_mw=1301.2682862201025,
    n_divertors=1,
    itart=0,
    **centrepost,
):
    """Drive `_RenormalisationOnly.run(False)`, returning its bound `DataStructure`.

    `n_divertors` defaults to `1` and `itart` to `0`; the double-null and the
    spherical-tokamak contracts pass the others. At `itart == 1` the eight `centrepost`
    keywords seed `run():103-141`'s reads and the three `st_*` routines run for real --
    `_RenormalisationOnly` never stubbed them, because on the conventional arm there was
    nothing there to stub.
    """
    model = _RenormalisationOnly(fw=FirstWall())
    model.data = DataStructure()
    model.unnormalised = (
        p_fw_nuclear_heat_total_mw_unnormalised,
        p_blkt_nuclear_heat_total_mw_unnormalised,
        p_shld_nuclear_heat_mw_unnormalised,
        p_tf_nuclear_heat_mw_unnormalised,
    )

    data = model.data
    data.physics.itart = itart
    data.divertor.n_divertors = n_divertors
    data.fwbs.f_ster_div_single = f_ster_div_single
    data.fwbs.f_p_blkt_multiplication = f_p_blkt_multiplication
    data.physics.p_neutron_total_mw = p_neutron_total_mw

    if itart == 1:
        data.tfcoil.i_tf_sup = 1  # superconducting: the one spherical cell written
        data.physics.rmajor = centrepost["rmajor"]
        data.physics.rminor = centrepost["rminor"]
        data.physics.triang = centrepost["triang"]
        data.build.dr_fw_plasma_gap_inboard = centrepost["dr_fw_plasma_gap_inboard"]
        data.build.z_plasma_xpoint_upper = centrepost["z_plasma_xpoint_upper"]
        data.build.r_sh_inboard_out = centrepost["r_sh_inboard_out"]
        data.build.dr_shld_inboard = centrepost["dr_shld_inboard"]

    model.run(output=False)
    return data


def _reference_centrepost_angle_fraction(z_cp_top, r_cp_mid, r_cp_top, rmajor):
    """`CCFE_HCPB.st_cp_angle_fraction` (`hcpb.py:1008-1080`), called directly.

    A `@staticmethod` with no `self.data` access at all, so the adapter is a bare
    forward -- the cheapest kind of reference there is, and the one that leaves no room
    for the test to differ from the source.
    """
    return CCFE_HCPB.st_cp_angle_fraction(
        z_cp_top=z_cp_top, r_cp_mid=r_cp_mid, r_cp_top=r_cp_top, rmajor=rmajor
    )


def _reference_centrepost_fast_neutron_flux_superconducting(
    p_neutron_total_mw, sh_width, rmajor
):
    """`st_tf_centrepost_fast_neut_flux` at `i_tf_sup == 1` (`hcpb.py:1082-1134`).

    `self`-bound on one field, `.tfcoil.i_tf_sup`, which the arm under test pins to
    `SUPERCONDUCTING`. Pinning it here rather than passing it is the point of the
    occupant: the other two conductor models return the literal `0` of `:1112` and are a
    different node.
    """
    model = _hcpb()
    model.data.tfcoil.i_tf_sup = 1
    return model.st_tf_centrepost_fast_neut_flux(
        p_neutron_total_mw=p_neutron_total_mw, sh_width=sh_width, rmajor=rmajor
    )


def _reference_centrepost_nuclear_heating_superconducting(pneut, sh_width, rmajor):
    """`st_centrepost_nuclear_heating`'s `else` arm (`hcpb.py:1200-1285`).

    Two `self.data` reads to close: `.tfcoil.i_tf_sup`, pinned away from
    `HELIUM_COOLED_ALUMINIUM` so the `else` arm runs, and `.physics.rmajor`, which the
    source reads off `self` in four places instead of taking it as a parameter and which
    the port makes an ordinary argument.
    """
    model = _hcpb()
    model.data.tfcoil.i_tf_sup = 1
    model.data.physics.rmajor = rmajor
    return model.st_centrepost_nuclear_heating(pneut=pneut, sh_width=sh_width)


def _reference_centrepost_and_renormalisation_spherical(n_divertors, **kwargs):
    """`run()`'s `hcpb.py:103-148` **and** `:195-276` at `itart == 1`, `i_tf_sup == 1`.

    One reference for two nodes, for the same reason `_reference_powerflow_calc` is one
    for two: the seam between them is a field PROCESS writes twice.
    `st_centrepost_nuclear_heating` puts its MCNP shield-heat fit in
    `.fwbs.p_cp_shield_nuclear_heat_mw` at `:137` and `:267` overwrites it, so seeding
    the renormalisation's centrepost reads and calling it alone would diff the port
    against a value PROCESS discards.

    The ten values read back are the ones that survive `run()`. The two that do not --
    `f_geom_cp`, a local, and the overwritten fit -- are covered by
    `TestCentrepostAngleFraction` and `TestCentrepostNuclearHeatingSuperconducting`,
    which reach PROCESS's own routines directly.
    """
    data = _run_renormalisation(itart=1, n_divertors=n_divertors, **kwargs)
    return (
        data.fwbs.pnuc_cp_tf,
        data.fwbs.pnuc_cp,
        data.fwbs.neut_flux_cp,
        data.ccfe_hcpb.pnuc_tot_blk_sector,
        data.fwbs.p_fw_nuclear_heat_total_mw,
        data.fwbs.p_blkt_nuclear_heat_total_mw,
        data.fwbs.p_shld_nuclear_heat_mw,
        data.fwbs.p_tf_nuclear_heat_mw,
        data.fwbs.p_cp_shield_nuclear_heat_mw,
        data.fwbs.p_blkt_multiplication_mw,
    )


def _reference_centrepost_and_renormalisation_single_null(**kwargs):
    """The pair at `n_divertors == 1`."""
    return _reference_centrepost_and_renormalisation_spherical(1, **kwargs)


def _reference_centrepost_and_renormalisation_double_null(**kwargs):
    """The pair at `n_divertors == 2` -- the cell both ST input files select."""
    return _reference_centrepost_and_renormalisation_spherical(2, **kwargs)


def _ported_centrepost_and_renormalisation_spherical(renormalise, **kwargs):
    """`CentrepostNeutronicsSphericalTokamakSuperconducting` then the renormalisation.

    The graph edge between the two nodes written out as a composition -- the two mints
    `.ccfe_hcpb.f_geom_cp` and `.fwbs.pnuc_cp_tf` crossing from the first to the second,
    in the order `Blocking.scc` puts them in.
    """
    (
        f_geom_cp,
        neut_flux_cp,
        pnuc_cp_tf,
        _p_cp_shield_nuclear_heat_mw_fit,
        pnuc_cp,
    ) = calculate_centrepost_neutronics_spherical_tokamak_superconducting(
        kwargs["rmajor"],
        kwargs["rminor"],
        kwargs["triang"],
        kwargs["dr_fw_plasma_gap_inboard"],
        kwargs["z_plasma_xpoint_upper"],
        kwargs["r_sh_inboard_out"],
        kwargs["p_neutron_total_mw"],
        kwargs["dr_shld_inboard"],
    )

    (
        pnuc_tot_blk_sector,
        p_fw_nuclear_heat_total_mw,
        p_blkt_nuclear_heat_total_mw,
        p_shld_nuclear_heat_mw,
        p_tf_nuclear_heat_mw,
        p_cp_shield_nuclear_heat_mw,
        p_blkt_multiplication_mw,
    ) = renormalise(
        kwargs["p_fw_nuclear_heat_total_mw_unnormalised"],
        kwargs["p_blkt_nuclear_heat_total_mw_unnormalised"],
        kwargs["p_shld_nuclear_heat_mw_unnormalised"],
        kwargs["p_tf_nuclear_heat_mw_unnormalised"],
        kwargs["f_ster_div_single"],
        kwargs["f_p_blkt_multiplication"],
        kwargs["p_neutron_total_mw"],
        f_geom_cp,
        pnuc_cp_tf,
    )

    return (
        pnuc_cp_tf,
        pnuc_cp,
        neut_flux_cp,
        pnuc_tot_blk_sector,
        p_fw_nuclear_heat_total_mw,
        p_blkt_nuclear_heat_total_mw,
        p_shld_nuclear_heat_mw,
        p_tf_nuclear_heat_mw,
        p_cp_shield_nuclear_heat_mw,
        p_blkt_multiplication_mw,
    )


def _ported_centrepost_and_renormalisation_single_null(**kwargs):
    """The composition at `n_divertors == 1`."""
    return _ported_centrepost_and_renormalisation_spherical(
        calculate_nuclear_heating_renormalisation_single_null_spherical_tokamak,
        **kwargs,
    )


def _ported_centrepost_and_renormalisation_double_null(**kwargs):
    """The composition at `n_divertors == 2`."""
    return _ported_centrepost_and_renormalisation_spherical(
        calculate_nuclear_heating_renormalisation_double_null_spherical_tokamak,
        **kwargs,
    )


def _reference_nuclear_heating_renormalisation(**kwargs):
    """`run()`'s `hcpb.py:195-276`, at `itart == 0` and `n_divertors == 1`."""
    data = _run_renormalisation(**kwargs)
    return (
        data.ccfe_hcpb.pnuc_tot_blk_sector,
        data.fwbs.p_fw_nuclear_heat_total_mw,
        data.fwbs.p_blkt_nuclear_heat_total_mw,
        data.fwbs.p_shld_nuclear_heat_mw,
        data.fwbs.p_tf_nuclear_heat_mw,
        data.fwbs.p_blkt_multiplication_mw,
    )


def _reference_nuclear_heating_renormalisation_double_null(**kwargs):
    """`run()`'s `hcpb.py:195-276`, at `itart == 0` and `n_divertors == 2`."""
    data = _run_renormalisation(n_divertors=2, **kwargs)
    return (
        data.ccfe_hcpb.pnuc_tot_blk_sector,
        data.fwbs.p_fw_nuclear_heat_total_mw,
        data.fwbs.p_blkt_nuclear_heat_total_mw,
        data.fwbs.p_shld_nuclear_heat_mw,
        data.fwbs.p_tf_nuclear_heat_mw,
        data.fwbs.p_blkt_multiplication_mw,
    )


def _reference_centrepost_neutronics_absent():
    """`run()`'s `else` arm at `hcpb.py:143-148`, plus `:267`'s second write."""
    data = _run_renormalisation(
        p_fw_nuclear_heat_total_mw_unnormalised=164.98107822556855,
        p_blkt_nuclear_heat_total_mw_unnormalised=1245.0032300478786,
        p_shld_nuclear_heat_mw_unnormalised=1.3640954387315272,
        p_tf_nuclear_heat_mw_unnormalised=0.03041968903979381,
    )
    return (
        data.fwbs.pnuc_cp_tf,
        data.fwbs.p_cp_shield_nuclear_heat_mw,
        data.fwbs.pnuc_cp,
        data.fwbs.neut_flux_cp,
    )


def _reference_powerflow_calc(
    p_plasma_rad_mw,
    f_a_fw_outboard_hcd,
    p_div_rad_total_mw,
    a_fw_outboard,
    a_fw_total,
    p_beam_orbit_loss_mw,
    p_fw_alpha_mw,
    p_he,
    dp_he,
    gamma_he,
    t_in_bb,
    t_out_bb,
    etaiso,
    f_p_fw_blkt_pump,
    p_fw_nuclear_heat_total_mw,
    p_blkt_nuclear_heat_total_mw,
    f_p_shld_coolant_pump_total_heat,
    p_shld_nuclear_heat_mw,
    p_cp_shield_nuclear_heat_mw,
    f_p_div_coolant_pump_total_heat,
    p_plasma_separatrix_mw,
    p_div_nuclear_heat_total_mw,
):
    """`powerflow_calc(False)` at `i_p_coolant_pumping == 3`, all seven outputs.

    The prologue and the pumping arm are one adapter because they cannot be separated on
    the PROCESS side: `powerflow_calc` **overwrites** `psurffwi`/`psurffwo`
    (`hcpb.py:805-814`) before the pumping arm reads them, so seeding them and calling
    the method would test nothing. The port's two functions are chained in the same order
    and the whole seven-value result is diffed, which covers both.
    """
    model = _hcpb()
    data = model.data

    data.fwbs.i_blkt_coolant_type = CoolantType.HELIUM
    data.fwbs.i_p_coolant_pumping = 3

    data.physics.p_plasma_rad_mw = p_plasma_rad_mw
    data.fwbs.f_a_fw_outboard_hcd = f_a_fw_outboard_hcd
    data.fwbs.p_div_rad_total_mw = p_div_rad_total_mw
    data.first_wall.a_fw_outboard = a_fw_outboard
    data.first_wall.a_fw_total = a_fw_total
    data.current_drive.p_beam_orbit_loss_mw = p_beam_orbit_loss_mw
    data.physics.p_fw_alpha_mw = p_fw_alpha_mw

    data.primary_pumping.p_he = p_he
    data.primary_pumping.dp_he = dp_he
    data.primary_pumping.gamma_he = gamma_he
    data.primary_pumping.t_in_bb = t_in_bb
    data.primary_pumping.t_out_bb = t_out_bb
    data.primary_pumping.f_p_fw_blkt_pump = f_p_fw_blkt_pump
    data.fwbs.etaiso = etaiso

    data.fwbs.p_fw_nuclear_heat_total_mw = p_fw_nuclear_heat_total_mw
    data.fwbs.p_blkt_nuclear_heat_total_mw = p_blkt_nuclear_heat_total_mw
    data.heat_transport.f_p_shld_coolant_pump_total_heat = (
        f_p_shld_coolant_pump_total_heat
    )
    data.fwbs.p_shld_nuclear_heat_mw = p_shld_nuclear_heat_mw
    data.fwbs.p_cp_shield_nuclear_heat_mw = p_cp_shield_nuclear_heat_mw
    data.heat_transport.f_p_div_coolant_pump_total_heat = f_p_div_coolant_pump_total_heat
    data.physics.p_plasma_separatrix_mw = p_plasma_separatrix_mw
    data.fwbs.p_div_nuclear_heat_total_mw = p_div_nuclear_heat_total_mw

    model.powerflow_calc(False)

    return (
        data.fwbs.p_fw_hcd_rad_total_mw,
        data.fwbs.p_fw_rad_total_mw,
        data.fwbs.psurffwo,
        data.fwbs.psurffwi,
        data.primary_pumping.p_fw_blkt_coolant_pump_mw,
        data.heat_transport.p_shld_coolant_pump_mw,
        data.heat_transport.p_div_coolant_pump_mw,
    )


def _ported_powerflow_calc(
    p_plasma_rad_mw,
    f_a_fw_outboard_hcd,
    p_div_rad_total_mw,
    a_fw_outboard,
    a_fw_total,
    p_beam_orbit_loss_mw,
    p_fw_alpha_mw,
    p_he,
    dp_he,
    gamma_he,
    t_in_bb,
    t_out_bb,
    etaiso,
    f_p_fw_blkt_pump,
    p_fw_nuclear_heat_total_mw,
    p_blkt_nuclear_heat_total_mw,
    f_p_shld_coolant_pump_total_heat,
    p_shld_nuclear_heat_mw,
    p_cp_shield_nuclear_heat_mw,
    f_p_div_coolant_pump_total_heat,
    p_plasma_separatrix_mw,
    p_div_nuclear_heat_total_mw,
):
    """`FirstWallRadiationPowers` then `PumpingPowerMechanicalWithPressureDrop`.

    The graph edge between the two nodes, written out as a composition -- exactly the
    order `Blocking.scc` puts them in.
    """
    (
        p_fw_hcd_rad_total_mw,
        p_fw_rad_total_mw,
        psurffwo,
        psurffwi,
    ) = calculate_first_wall_radiation_powers(
        p_plasma_rad_mw,
        f_a_fw_outboard_hcd,
        p_div_rad_total_mw,
        a_fw_outboard,
        a_fw_total,
        p_beam_orbit_loss_mw,
        p_fw_alpha_mw,
    )

    pumps = calculate_pumping_power_mechanical_with_pressure_drop(
        p_he,
        dp_he,
        gamma_he,
        t_in_bb,
        t_out_bb,
        etaiso,
        f_p_fw_blkt_pump,
        p_fw_nuclear_heat_total_mw,
        psurffwi,
        psurffwo,
        p_blkt_nuclear_heat_total_mw,
        f_p_shld_coolant_pump_total_heat,
        p_shld_nuclear_heat_mw,
        p_cp_shield_nuclear_heat_mw,
        f_p_div_coolant_pump_total_heat,
        p_plasma_separatrix_mw,
        p_div_nuclear_heat_total_mw,
        p_div_rad_total_mw,
    )

    return (p_fw_hcd_rad_total_mw, p_fw_rad_total_mw, psurffwo, psurffwi, *pumps)


# --------------------------------------------------------------------------------------
# contracts
# --------------------------------------------------------------------------------------

_MAGNETS_REFERENCE_RUN = {
    "radius_fw_channel": 0.006,
    "dx_fw_module": 0.02,
    "dr_fw_inboard": 0.018000000000000002,
    "dr_fw_outboard": 0.018000000000000002,
    "den_steel": 7800.0,
    "m_blkt_total": 3110067.3947664234,
    "vol_blkt_total": 1241.7966910494447,
    "whtshld": 2449818.833849217,
    "vol_shld_total": 785.1983441824412,
    "dr_vv_inboard": 0.3,
    "dr_vv_outboard": 0.3,
    "m_vv": 7938816.368934795,
    "vol_vv": 1017.7969703762558,
    "dr_blkt_outboard": 1.0,
    "dr_shld_outboard": 0.8,
    "fw_armour_thickness": 0.005,
    "p_fusion_total_mw": 1630.323245464875,
}

_MAGNETS_FUZZ = {
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
    "dr_shld_outboard": (0.1, 2.0),
    "fw_armour_thickness": (0.001, 0.02),
    "p_fusion_total_mw": (100.0, 5000.0),
}


class TestFirstWallCoolantVoidFractions(Tier1Contract):
    """`calculate_fw_coolant_void_fractions` -> the two fields
    `nuclear_heating_magnets` writes at `hcpb.py:483-490`.

    The node this contract covers is what makes `ComponentMasses` and
    `NuclearHeatingMagnetsConventional` orderable at all -- see the port's docstring.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_fw_coolant_void_fractions
    ported = calculate_fw_coolant_void_fractions

    samples = [
        legacy_sample(
            "void-fractions-reference-run",
            radius_fw_channel=0.006,
            dx_fw_module=0.02,
            dr_fw_inboard=0.018000000000000002,
        ),
    ]

    fuzz_bounds = {
        "radius_fw_channel": (0.001, 0.02),
        "dx_fw_module": (0.005, 0.05),
        "dr_fw_inboard": (0.005, 0.05),
    }


class TestDivertorSurfaceAndPlateMassSingleNull(Tier1Contract):
    """`calculate_divertor_surface_and_plate_mass_single_null` -> `hcpb.py:353-367`.

    Legacy point is `test_ccfe_hcpb.py::test_component_masses`' single parametrised case
    (`n_divertors=1`, generated from `large_tokamak_eval.IN.DAT`), whose
    `expected_a_div_surface_total`/`expected_m_div_plate` are 148.78582807401261 and
    36452.527878133093.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_divertor_surface_and_plate_mass_single_null
    ported = calculate_divertor_surface_and_plate_mass_single_null

    samples = [
        legacy_sample(
            "divertor-large-tokamak-eval",
            fdiva=1.1100000000000001,
            rmajor=8,
            rminor=2.6666666666666665,
            den_div_structure=10000,
            f_vol_div_coolant=0.29999999999999999,
            dx_div_plate=0.035000000000000003,
        ),
    ]

    fuzz_bounds = {
        "fdiva": (0.8, 1.5),
        "rmajor": (5.0, 12.0),
        "rminor": (1.0, 4.0),
        "den_div_structure": (5000.0, 15000.0),
        "f_vol_div_coolant": (0.1, 0.5),
        "dx_div_plate": (0.01, 0.1),
    }


class TestDivertorSurfaceAndPlateMassDoubleNull(Tier1Contract):
    """`calculate_divertor_surface_and_plate_mass_double_null` -> the divertor pair at
    `n_divertors == 2`.

    Same legacy point and same fuzz box as the single-null contract, so the pair reads
    as one comparison: everything but the divertor count is held fixed and the answer
    doubles.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_divertor_surface_and_plate_mass_double_null
    ported = calculate_divertor_surface_and_plate_mass_double_null

    samples = [
        legacy_sample(
            "divertor-double-null-large-tokamak-eval",
            fdiva=1.1100000000000001,
            rmajor=8,
            rminor=2.6666666666666665,
            den_div_structure=10000,
            f_vol_div_coolant=0.29999999999999999,
            dx_div_plate=0.035000000000000003,
        ),
    ]

    fuzz_bounds = {
        "fdiva": (0.8, 1.5),
        "rmajor": (5.0, 12.0),
        "rminor": (1.0, 4.0),
        "den_div_structure": (5000.0, 15000.0),
        "f_vol_div_coolant": (0.1, 0.5),
        "dx_div_plate": (0.01, 0.1),
    }


class TestComponentMasses(Tier1Contract):
    """`calculate_component_masses` -> `component_masses()` less the divertor pair.

    Two legacy points: `test_ccfe_hcpb.py::test_component_masses`' parametrised case, and
    the reference run's own state at the moment `run()` calls `component_masses`
    (`hcpb.py:150`). The two differ in exactly the way the cycle-dissolution predicts:
    PROCESS's own test carries `f_a_fw_coolant_inboard = 0` (the `DataStructure` default,
    because nothing has run `nuclear_heating_magnets` yet) where the reference run
    carries `0.3141592653589793`, the converged value.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_component_masses
    ported = calculate_component_masses

    samples = [
        legacy_sample(
            "masses-large-tokamak-eval",
            a_div_surface_total=0,
            f_vol_div_coolant=0.29999999999999999,
            dx_div_plate=0.035000000000000003,
            vol_blkt_total=1182.5433772195902,
            f_a_blkt_cooling_channels=0.25,
            vol_shld_total=783.69914576548854,
            vfshld=0.60000000000000009,
            a_fw_inboard=505.96109565204046,
            a_fw_outboard=838.00728058362097,
            a_fw_total=1343.9683762356615,
            dr_fw_inboard=0.018000000000000002,
            dr_fw_outboard=0.018000000000000002,
            f_a_fw_coolant_inboard=0,
            f_a_fw_coolant_outboard=0,
            den_steel=7800,
            a_plasma_surface=1173.8427771245592,
            fw_armour_thickness=0.0050000000000000001,
            breeder_f=0.5,
            breeder_multiplier=0.75,
            vfcblkt=0.052949999999999997,
            vfpblkt=0.10000000000000001,
        ),
        legacy_sample(
            "masses-reference-run",
            a_div_surface_total=148.7858280740126,
            f_vol_div_coolant=0.3,
            dx_div_plate=0.035,
            vol_blkt_total=1241.7966910494447,
            f_a_blkt_cooling_channels=0.25,
            vol_shld_total=785.1983441824412,
            vfshld=0.6,
            a_fw_inboard=633.0209600168059,
            a_fw_outboard=1048.452495290143,
            a_fw_total=1681.4734553069488,
            dr_fw_inboard=0.018000000000000002,
            dr_fw_outboard=0.018000000000000002,
            f_a_fw_coolant_inboard=0.3141592653589793,
            f_a_fw_coolant_outboard=0.3141592653589793,
            den_steel=7800.0,
            a_plasma_surface=1173.8427771245592,
            fw_armour_thickness=0.005,
            breeder_f=0.5,
            breeder_multiplier=0.75,
            vfcblkt=0.05295,
            vfpblkt=0.1,
        ),
    ]

    fuzz_bounds = {
        "a_div_surface_total": (50.0, 400.0),
        "f_vol_div_coolant": (0.1, 0.5),
        "dx_div_plate": (0.01, 0.1),
        "vol_blkt_total": (100.0, 3000.0),
        "f_a_blkt_cooling_channels": (0.05, 0.5),
        "vol_shld_total": (100.0, 2000.0),
        "vfshld": (0.1, 0.8),
        "a_fw_inboard": (200.0, 1200.0),
        "a_fw_outboard": (400.0, 2000.0),
        "a_fw_total": (600.0, 3200.0),
        "dr_fw_inboard": (0.005, 0.05),
        "dr_fw_outboard": (0.005, 0.05),
        "f_a_fw_coolant_inboard": (0.05, 0.6),
        "f_a_fw_coolant_outboard": (0.05, 0.6),
        "den_steel": (6000.0, 9000.0),
        "a_plasma_surface": (500.0, 2500.0),
        "fw_armour_thickness": (0.001, 0.02),
        # Inside iteration variable 108's declared bounds (0.060, 1.0), so the clamp at
        # `hcpb.py:404-405` is inert here -- deliberately, since the clamp's only effect
        # is on points the solver is not allowed to visit.
        "breeder_f": (0.06, 1.0),
        "breeder_multiplier": (0.5, 0.9),
        "vfcblkt": (0.01, 0.1),
        "vfpblkt": (0.02, 0.2),
    }


class TestNuclearHeatingMagnetsConventional(Tier1Contract):
    """`calculate_nuclear_heating_magnets_conventional` -> `nuclear_heating_magnets`
    at `itart == 0`.

    Legacy points are `test_ccfe_hcpb.py::test_nuclear_heating_magnets`' two parametrised
    cases (both `itart=0`), plus the reference run's own inputs.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_nuclear_heating_magnets_conventional
    ported = calculate_nuclear_heating_magnets_conventional

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
            dr_blkt_outboard=0.98199999999999998,
            dr_blkt_inboard=0.75500000000000012,
            dr_shld_outboard=0.80000000000000004,
            dr_shld_inboard=0.30000000000000004,
            fw_armour_thickness=0.0050000000000000001,
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
            dr_blkt_outboard=0.98199999999999998,
            dr_blkt_inboard=0.75500000000000012,
            dr_shld_outboard=0.80000000000000004,
            dr_shld_inboard=0.30000000000000004,
            fw_armour_thickness=0.0050000000000000001,
            m_tf_coils_total=19662548.210142396,
            p_fusion_total_mw=1985.4423932312809,
        ),
        legacy_sample(
            "magnets-reference-run",
            **_MAGNETS_REFERENCE_RUN,
            dr_blkt_inboard=0.7,
            dr_shld_inboard=0.3,
            m_tf_coils_total=14339045.099299619,
        ),
    ]

    fuzz_bounds = {
        **_MAGNETS_FUZZ,
        "dr_blkt_inboard": (0.1, 2.0),
        "dr_shld_inboard": (0.1, 2.0),
        "m_tf_coils_total": (1.0e6, 5.0e7),
    }


class TestNuclearHeatingMagnetsSphericalTokamak(Tier1Contract):
    """`calculate_nuclear_heating_magnets_spherical_tokamak` -> the same at `itart == 1`.

    No PROCESS unit test exercises this branch (`test_ccfe_hcpb.py`'s two cases are both
    `itart=0` with `whttflgs=0`, which would make the result identically zero), so the
    point below is the reference run's geometry with a plausible outboard-leg mass
    substituted -- a fixed point, not fuzz-only, so the arm is checked deterministically
    rather than only-sometimes by chance. Same treatment the shield's ST arm already had.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_nuclear_heating_magnets_spherical_tokamak
    ported = calculate_nuclear_heating_magnets_spherical_tokamak

    samples = [
        legacy_sample(
            "magnets-st-branch",
            **_MAGNETS_REFERENCE_RUN,
            whttflgs=5.0e6,
        ),
    ]

    fuzz_bounds = {**_MAGNETS_FUZZ, "whttflgs": (1.0e5, 3.0e7)}


class TestNuclearHeatingFw(Tier1Contract):
    """`nuclear_heating_fw` -> the same, unchanged.

    Samples are `test_ccfe_hcpb.py::test_nuclear_heating_fw`'s two parametrised cases.
    `reference_domain_errors` declares the `ProcessValueError` PROCESS raises on a
    negative result (`hcpb.py:646-650`); the port returns `nan` there instead, and the
    contract asserts that rather than skipping.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_nuclear_heating_fw
    ported = nuclear_heating_fw
    reference_domain_errors = (ProcessValueError,)

    samples = [
        legacy_sample(
            "fw-baseline-2018-a",
            m_fw_total=224802.80270851994,
            fw_armour_u_nuc_heating=6.2500000000000005e-07,
            p_fusion_total_mw=1986.0623241661431,
        ),
        legacy_sample(
            "fw-baseline-2018-b",
            m_fw_total=182115.83467868491,
            fw_armour_u_nuc_heating=6.2500000000000005e-07,
            p_fusion_total_mw=1985.4423932312809,
        ),
        legacy_sample(
            "fw-reference-run",
            m_fw_total=161912.50777733992,
            fw_armour_u_nuc_heating=6.25e-07,
            p_fusion_total_mw=1630.323245464875,
        ),
    ]

    fuzz_bounds = {
        "m_fw_total": (1.0e4, 1.0e6),
        "fw_armour_u_nuc_heating": (1.0e-7, 1.0e-6),
        "p_fusion_total_mw": (100.0, 5000.0),
    }


class TestNuclearHeatingBlanket(Tier1Contract):
    """`nuclear_heating_blanket` -> the same, unchanged.

    Samples are `test_ccfe_hcpb.py::test_nuclear_heating_blanket`'s two parametrised
    cases, plus the reference run's own.
    """

    audit_record = _AUDIT_RECORD
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
        legacy_sample(
            "blanket-reference-run",
            m_blkt_total=3110067.3947664234,
            p_fusion_total_mw=1630.323245464875,
        ),
    ]

    fuzz_bounds = {
        "m_blkt_total": (1.0e5, 1.0e7),
        "p_fusion_total_mw": (100.0, 5000.0),
    }


class TestNuclearHeatingShieldConventional(Tier1Contract):
    """`nuclear_heating_shield_conventional` -> `nuclear_heating_shield` at `itart == 0`.

    Samples are `test_ccfe_hcpb.py::test_nuclear_heating_shield`'s two parametrised cases
    (both `itart=0`), plus the reference run's own -- where `shield_density`/`x_blanket`
    are `NuclearHeatingMagnetsConventional`'s measured outputs at that point.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_nuclear_heating_shield_conventional
    ported = nuclear_heating_shield_conventional

    samples = [
        legacy_sample(
            "shield-baseline-2018-a",
            dr_shld_outboard=0.80000000000000004,
            dr_shld_inboard=0.30000000000000004,
            shield_density=3119.9999999999995,
            whtshld=2294873.8131476045,
            x_blanket=2.3374537748527975,
            p_fusion_total_mw=1986.0623241661431,
        ),
        legacy_sample(
            "shield-baseline-2018-b",
            dr_shld_outboard=0.80000000000000004,
            dr_shld_inboard=0.30000000000000004,
            shield_density=3120,
            whtshld=2297808.3935174854,
            x_blanket=2.3374537748527979,
            p_fusion_total_mw=1985.4423932312809,
        ),
        legacy_sample(
            "shield-reference-run",
            dr_shld_outboard=0.8,
            dr_shld_inboard=0.3,
            shield_density=3120.0,
            whtshld=2449818.833849217,
            x_blanket=2.2911207098527977,
            p_fusion_total_mw=1630.323245464875,
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


class TestNuclearHeatingShieldSphericalTokamak(Tier1Contract):
    """`nuclear_heating_shield_spherical_tokamak` -> the same at `itart == 1`.

    No PROCESS unit test exercises this branch, so the point is
    `shield-baseline-2018-a`'s with the arm switched -- fixed rather than fuzz-only so it
    is checked deterministically. The adapter passes `dr_shld_inboard = 1e6`, a value the
    port has no parameter for: if the arm had kept that read the two would disagree
    grossly rather than subtly.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_nuclear_heating_shield_spherical_tokamak
    ported = nuclear_heating_shield_spherical_tokamak

    samples = [
        legacy_sample(
            "shield-st-branch",
            dr_shld_outboard=0.80000000000000004,
            shield_density=3119.9999999999995,
            whtshld=2294873.8131476045,
            x_blanket=2.3374537748527975,
            p_fusion_total_mw=1986.0623241661431,
        ),
    ]

    fuzz_bounds = {
        "dr_shld_outboard": (0.1, 2.0),
        "shield_density": (1000.0, 8000.0),
        "whtshld": (1.0e5, 1.0e7),
        "x_blanket": (0.1, 10.0),
        "p_fusion_total_mw": (100.0, 5000.0),
    }


class TestCentrepostNeutronicsAbsent(Tier1Contract):
    """`calculate_centrepost_neutronics_absent` -> `run()`'s `else` arm at
    `hcpb.py:143-148`.

    A node with no inputs, so there is nothing to fuzz and nothing to differentiate --
    the contract's remaining content is that PROCESS really does write four zeros here,
    which is what makes the node's four `Out`s legitimate rather than invented.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_centrepost_neutronics_absent
    ported = calculate_centrepost_neutronics_absent

    samples = [legacy_sample("no-centrepost-reference-run")]


class TestNuclearHeatingRenormalisation(Tier1Contract):
    """`calculate_nuclear_heating_renormalisation_single_null_conventional` ->
    `run()`'s `hcpb.py:195-276`, driven through `_RenormalisationOnly`.

    The legacy point's four unnormalised powers are the reference run's own, recomputed
    from its inputs through PROCESS's four `nuclear_heating_*` routines; they sum to
    1411.3788234012184, which is the `.ccfe_hcpb.pnuc_tot_blk_sector` measured on that
    run to the last digit.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_nuclear_heating_renormalisation
    ported = calculate_nuclear_heating_renormalisation_single_null_conventional

    samples = [
        legacy_sample(
            "renormalisation-reference-run",
            p_fw_nuclear_heat_total_mw_unnormalised=164.98107822556855,
            p_blkt_nuclear_heat_total_mw_unnormalised=1245.0032300478786,
            p_shld_nuclear_heat_mw_unnormalised=1.3640954387315272,
            p_tf_nuclear_heat_mw_unnormalised=0.03041968903979381,
            f_ster_div_single=0.0725040362777958,
            f_p_blkt_multiplication=1.269,
            p_neutron_total_mw=1301.2682862201025,
        ),
    ]

    fuzz_bounds = {
        "p_fw_nuclear_heat_total_mw_unnormalised": (50.0, 500.0),
        "p_blkt_nuclear_heat_total_mw_unnormalised": (500.0, 2500.0),
        "p_shld_nuclear_heat_mw_unnormalised": (0.1, 10.0),
        "p_tf_nuclear_heat_mw_unnormalised": (0.001, 1.0),
        "f_ster_div_single": (0.02, 0.2),
        "f_p_blkt_multiplication": (1.0, 1.5),
        "p_neutron_total_mw": (100.0, 4000.0),
    }


class TestNuclearHeatingRenormalisationDoubleNull(Tier1Contract):
    """`calculate_nuclear_heating_renormalisation_double_null_conventional` ->
    `run()`'s `hcpb.py:195-276` at `n_divertors == 2`, `itart == 0`.

    The same reference-run point as the single-null contract. `f_ster_div_single` stays
    below `0.5` in the fuzz box for the same reason as
    `test_blanket_library.py`'s coverage contracts: above it `1 - 2 * f_ster_div_single`
    changes sign and the renormalisation stops meaning anything.

    Not reachable from either spherical-tokamak input file -- both set `itart = 1`, and
    this slot refuses on `('itart_hcpb', 1)` first. Written because `hcpb.py:213-217` is
    one of the double-null wave's named sites, and tested here so the arm is not merely
    asserted to exist.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_nuclear_heating_renormalisation_double_null
    ported = calculate_nuclear_heating_renormalisation_double_null_conventional

    samples = [
        legacy_sample(
            "renormalisation-double-null-reference-run",
            p_fw_nuclear_heat_total_mw_unnormalised=164.98107822556855,
            p_blkt_nuclear_heat_total_mw_unnormalised=1245.0032300478786,
            p_shld_nuclear_heat_mw_unnormalised=1.3640954387315272,
            p_tf_nuclear_heat_mw_unnormalised=0.03041968903979381,
            f_ster_div_single=0.0725040362777958,
            f_p_blkt_multiplication=1.269,
            p_neutron_total_mw=1301.2682862201025,
        ),
    ]

    fuzz_bounds = {
        "p_fw_nuclear_heat_total_mw_unnormalised": (50.0, 500.0),
        "p_blkt_nuclear_heat_total_mw_unnormalised": (500.0, 2500.0),
        "p_shld_nuclear_heat_mw_unnormalised": (0.1, 10.0),
        "p_tf_nuclear_heat_mw_unnormalised": (0.001, 1.0),
        "f_ster_div_single": (0.02, 0.2),
        "f_p_blkt_multiplication": (1.0, 1.5),
        "p_neutron_total_mw": (100.0, 4000.0),
    }


_CP_ANGLE_TOLERANCE = Tolerance(
    rtol=5e-9,
    atol=0.0,
    reason=(
        "`st_cp_angle_fraction` amplifies one ulp of `arcsin` into ~1e-8 in an "
        "intermediate, and it does so *by design*. The trapezoid's last panel sits at "
        "`phy_cp = arcsin(1 / rho_maj)`, where `1 - rho_maj**2 * sin(phy_cp)**2` is "
        "analytically zero and numerically either sign; PROCESS clamps it with "
        "`max(., 0)` and takes its square root. On the FNSF legacy point `np.arcsin` "
        "and `jnp.arcsin` differ by one ulp, which puts that radicand at `+1.11e-16` "
        "for numpy and `-2.22e-16` for jax -- so numpy's square root is `1.05e-8` and "
        "jax's is `0`, and a 1e-8 difference in one of twenty trapezoid terms follows. "
        "Measured over the contract's own fuzz box (4000 points): 686 disagree at all, "
        "worst 4.8e-10 relative, the FNSF point 8.7e-11, and the "
        "`spherical_tokamak_eval.IN.DAT` point exactly 0 because there the two "
        "`arcsin`s agree. The bar is set a factor of ~10 above the measured worst. "
        "Worth being explicit about which side is right: the analytic radicand at that "
        "panel is exactly zero, so it is PROCESS's `1.05e-8` that is spurious and the "
        "port's `0` that is correct -- this tolerance covers PROCESS's noise, not the "
        "port's"
    ),
)
"""Shared by the angle-fraction contract and the two composites that consume it."""

_CP_ANGLE_GRADIENT_SAFETY = 250.0
"""The same clamp instability, seen from the gradient side.

`gradient_safety` multiplies PROCESS's *own* finite-difference error bar, which is a
Richardson extrapolation of the smooth truncation error. It cannot see a
**discontinuity**, and that is exactly what `max(., 0)` puts in PROCESS's function: as an
input is perturbed, the last panel's radicand crosses zero and PROCESS's square root
jumps between `0` and ~`1e-8`. The finite difference then measures a real step in
PROCESS's answer that the analytic derivative has no reason to reproduce, and reports a
tiny error bar while doing it.

Measured, at `--fp-fuzz 40` over the three contracts that share this: the worst case
needs 7.3x the default 25 (`d(f_geom_cp)/d(r_cp_top)`, `|diff| = 2.10e-8` against an
error bar of `1.16e-10`), and the two composites need 1.1x and 1.4x. Set at 250, which
is ~1.4x the measured worst requirement.

The detection power this costs is small for the reason `Tier1Contract.gradient_safety`
already gives: a wrong derivative is wrong by an `O(1)` relative amount, and the worst
of these is `1.9e-7` relative.
"""

_ST_REFERENCE_RUN = {
    "p_fw_nuclear_heat_total_mw_unnormalised": 141.3749135766222,
    "p_blkt_nuclear_heat_total_mw_unnormalised": 1872.8676544479313,
    "p_shld_nuclear_heat_mw_unnormalised": 0.586506103734957,
    "p_tf_nuclear_heat_mw_unnormalised": 0.00436359182345013,
    "f_ster_div_single": 0.048589790347528944,
    "f_p_blkt_multiplication": 1.269,
    "p_neutron_total_mw": 1990.7428899980614,
    "rmajor": 4.5,
    "rminor": 2.5,
    "triang": 0.5,
    "dr_fw_plasma_gap_inboard": 0.1,
    "z_plasma_xpoint_upper": 7.0,
    "r_sh_inboard_out": 1.8720000000000003,
    "dr_shld_inboard": 0.39314459807893426,
}
"""`spherical_tokamak_eval.IN.DAT`'s own converged operating point.

Read off the `DataStructure` PROCESS leaves behind after solving that file --
`aspect = 1.8`, `itart = 1`, `i_tf_sup = 1`, `i_single_null = 0` -- which makes this the
first sample in the port taken from a *spherical tokamak* rather than from a
conventional one. The four unnormalised powers are that run's own, recomputed from its
inputs through PROCESS's four `nuclear_heating_*` routines, and they sum to
2014.833437720112, the `.ccfe_hcpb.pnuc_tot_blk_sector` measured on the same run to the
last digit -- the same independent check the conventional renormalisation sample
carries, and the reason to believe the point sits *on* the operating point rather than
near it.
"""

_ST_FUZZ = {
    "p_fw_nuclear_heat_total_mw_unnormalised": (50.0, 500.0),
    "p_blkt_nuclear_heat_total_mw_unnormalised": (500.0, 2500.0),
    "p_shld_nuclear_heat_mw_unnormalised": (0.1, 10.0),
    "p_tf_nuclear_heat_mw_unnormalised": (0.001, 1.0),
    # Capped well below the point where `1 - 2 * f_ster_div_single - f_geom_cp` changes
    # sign, for the reason the conventional double-null contract caps it: past that the
    # renormalisation stops meaning anything.
    "f_ster_div_single": (0.02, 0.2),
    "f_p_blkt_multiplication": (1.0, 1.5),
    "p_neutron_total_mw": (100.0, 4000.0),
    # The geometry box is deliberately narrow: `rho_maj = 2 * rmajor / (r_cp_mid +
    # r_cp_top)` must stay above 1 for `arcsin(1 / rho_maj)` to be real at all, and a
    # centrepost machine's radii are not free of each other.
    "rmajor": (3.0, 6.0),
    "rminor": (1.5, 3.0),
    "triang": (0.2, 0.8),
    "dr_fw_plasma_gap_inboard": (0.05, 0.3),
    "z_plasma_xpoint_upper": (3.0, 10.0),
    "r_sh_inboard_out": (1.0, 2.5),
    "dr_shld_inboard": (0.15, 0.8),
}


class TestCentrepostAngleFraction(Tier1Contract):
    """`calculate_centrepost_angle_fraction` -> `st_cp_angle_fraction`
    (`hcpb.py:1008-1080`).

    The legacy point is `test_ccfe_hcpb.py::test_st_cp_angle_fraction`'s single
    parametrised case, whose own docstring names the FNSF IN.DAT (no longer in this
    repository); the second is `spherical_tokamak_eval.IN.DAT`'s own geometry, with
    `r_cp_top` formed the way `run():106-110` forms it.

    **This is the port's first `safe_sqrt` whose zero is reached on the operating
    point rather than at a fuzzed edge.** The integral's last panel sits exactly at
    `phy_cp = arcsin(1 / rho_maj)`, where `1 - rho_maj**2 * sin(phy_cp)**2` is
    analytically zero, so the gradient test would see `inf` from `jnp.sqrt` on every
    sample rather than on none.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_centrepost_angle_fraction
    ported = calculate_centrepost_angle_fraction

    value_tolerance = _CP_ANGLE_TOLERANCE
    gradient_safety = _CP_ANGLE_GRADIENT_SAFETY

    samples = [
        legacy_sample(
            "cp-angle-fnsf",
            z_cp_top=2.6714285714285717,
            r_cp_mid=0.20483000000000001,
            r_cp_top=0.92643571428571436,
            rmajor=1.7000000000000002,
        ),
        legacy_sample(
            "cp-angle-spherical-tokamak-eval",
            z_cp_top=7.0,
            r_cp_mid=1.8720000000000003,
            # `rmajor - rminor * triang - 3 * dr_fw_plasma_gap_inboard`.
            r_cp_top=4.5 - 2.5 * 0.5 - 3 * 0.1,
            rmajor=4.5,
        ),
    ]

    fuzz_bounds = {
        "z_cp_top": (0.5, 8.0),
        # `2 * rmajor` stays above `r_cp_mid + r_cp_top` over the whole box, which is
        # what keeps `rho_maj > 1`.
        "r_cp_mid": (0.1, 1.0),
        "r_cp_top": (0.5, 2.0),
        "rmajor": (2.0, 6.0),
    }


class TestCentrepostFastNeutronFluxSuperconducting(Tier1Contract):
    """`calculate_centrepost_fast_neutron_flux_superconducting` ->
    `st_tf_centrepost_fast_neut_flux` at `i_tf_sup == 1` (`hcpb.py:1114-1132`).

    Legacy points are `test_ccfe_hcpb.py::test_st_tf_centrepost_fast_neut_flux`'s two
    parametrised cases (both `i_tf_sup = 1`; the docstring names the Menard_HTS-PP
    IN.DAT), plus `spherical_tokamak_eval.IN.DAT`'s own.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_centrepost_fast_neutron_flux_superconducting
    ported = calculate_centrepost_fast_neutron_flux_superconducting

    samples = [
        legacy_sample(
            "cp-flux-menard-a",
            p_neutron_total_mw=400.65875490746737,
            sh_width=0.60000000000000009,
            rmajor=3,
        ),
        legacy_sample(
            "cp-flux-menard-b",
            p_neutron_total_mw=409.82485143909827,
            sh_width=0.60000000000000009,
            rmajor=3,
        ),
        legacy_sample(
            "cp-flux-spherical-tokamak-eval",
            p_neutron_total_mw=1990.7428899980614,
            sh_width=0.39314459807893426,
            rmajor=4.5,
        ),
    ]

    fuzz_bounds = {
        "p_neutron_total_mw": (100.0, 4000.0),
        "sh_width": (0.1, 1.5),
        "rmajor": (1.5, 6.0),
    }


class TestCentrepostNuclearHeatingSuperconducting(Tier1Contract):
    """`calculate_centrepost_nuclear_heating_superconducting` ->
    `st_centrepost_nuclear_heating`'s `else` arm (`hcpb.py:1200-1285`).

    **The arm is `i_tf_sup in {0, 1}`**, not `{1}`: `hcpb.py:1192` branches on aluminium
    alone. The contract pins `i_tf_sup = 1` because that is the cell registered, and the
    water-cooled-copper cell shares this arithmetic exactly -- it differs only in the
    fast-neutron flux, which is why the two are separate slots' worth of question
    answered by one joint arm.

    Legacy points are `test_ccfe_hcpb.py::test_st_centrepost_nuclear_heating`'s two
    Menard_HTS-PP cases plus the spherical-tokamak reference run's own. The middle of
    the three returned values is the MCNP fit that `run():267` discards on this arm;
    this contract is where it is checked at all.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_centrepost_nuclear_heating_superconducting
    ported = calculate_centrepost_nuclear_heating_superconducting

    samples = [
        legacy_sample(
            "cp-heating-menard-a",
            pneut=400.65875490746737,
            sh_width=0.60000000000000009,
            rmajor=3,
        ),
        legacy_sample(
            "cp-heating-menard-b",
            pneut=409.82485143909827,
            sh_width=0.60000000000000009,
            rmajor=3,
        ),
        legacy_sample(
            "cp-heating-spherical-tokamak-eval",
            pneut=1990.7428899980614,
            sh_width=0.39314459807893426,
            rmajor=4.5,
        ),
    ]

    fuzz_bounds = {
        "pneut": (100.0, 4000.0),
        "sh_width": (0.1, 1.5),
        "rmajor": (1.5, 6.0),
    }


class TestCentrepostAndRenormalisationSingleNullSphericalTokamak(Tier1Contract):
    """`CentrepostNeutronicsSphericalTokamakSuperconducting` then
    `NuclearHeatingRenormalisationSingleNullSphericalTokamak` ->
    `run()`'s `hcpb.py:103-148` and `:195-276` at `itart == 1`, `n_divertors == 1`.

    One contract for two nodes; `_reference_centrepost_and_renormalisation_spherical`
    says why. The sample is `spherical_tokamak_eval.IN.DAT`'s converged point with its
    divertor count flipped -- that file is double-null, so this cell is exercised on its
    own machine's numbers rather than on invented ones, which is the closest a
    single-null spherical contract can get without a single-null spherical input file.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_centrepost_and_renormalisation_single_null
    ported = _ported_centrepost_and_renormalisation_single_null

    value_tolerance = _CP_ANGLE_TOLERANCE
    gradient_safety = _CP_ANGLE_GRADIENT_SAFETY

    samples = [legacy_sample("st-centrepost-single-null", **_ST_REFERENCE_RUN)]

    fuzz_bounds = _ST_FUZZ


class TestCentrepostAndRenormalisationDoubleNullSphericalTokamak(Tier1Contract):
    """The same pair at `n_divertors == 2` -- **the cell both spherical-tokamak input
    files select**, on the operating point one of them converges to.

    Every number in `_ST_REFERENCE_RUN` is `spherical_tokamak_eval.IN.DAT`'s own, so
    this contract is the port's first end-to-end agreement with PROCESS on a spherical
    machine: PROCESS's `.fwbs.pnuc_cp_tf` there is 0.5523229613563773, its
    `.fwbs.p_cp_shield_nuclear_heat_mw` 354.03174669334163, and its
    `.fwbs.neut_flux_cp` 1.2771358544430766e+16, all of which this composition has to
    reproduce.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_centrepost_and_renormalisation_double_null
    ported = _ported_centrepost_and_renormalisation_double_null

    value_tolerance = _CP_ANGLE_TOLERANCE
    gradient_safety = _CP_ANGLE_GRADIENT_SAFETY

    samples = [legacy_sample("st-centrepost-double-null", **_ST_REFERENCE_RUN)]

    fuzz_bounds = _ST_FUZZ


class TestPowerflowCalcMechanicalWithPressureDrop(Tier1Contract):
    """`FirstWallRadiationPowers` then `PumpingPowerMechanicalWithPressureDrop`
    -> `powerflow_calc(False)` at `i_p_coolant_pumping == 3`.

    One contract for two nodes, because PROCESS's own method overwrites the two fields
    that would otherwise be the seam -- see `_reference_powerflow_calc`. Legacy points
    are `test_ccfe_hcpb.py::test_powerflow_calc`'s two parametrised cases (both
    `i_p_coolant_pumping=3`, `i_blkt_coolant_type=HELIUM`) and the reference run's own.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_powerflow_calc
    ported = _ported_powerflow_calc

    samples = [
        legacy_sample(
            "powerflow-baseline-2018-a",
            p_plasma_rad_mw=287.44866938104849,
            f_a_fw_outboard_hcd=0,
            p_div_rad_total_mw=33.056596978820579,
            a_fw_outboard=988.92586580655245,
            a_fw_total=1601.1595634509963,
            p_beam_orbit_loss_mw=0,
            p_fw_alpha_mw=19.835845058655043,
            p_he=8000000,
            dp_he=550000,
            gamma_he=1.667,
            t_in_bb=573.13,
            t_out_bb=773.13,
            etaiso=0.90000000000000002,
            f_p_fw_blkt_pump=1.0,
            p_fw_nuclear_heat_total_mw=276.80690153753221,
            p_blkt_nuclear_heat_total_mw=1504.9215740808861,
            f_p_shld_coolant_pump_total_heat=0.0050000000000000001,
            p_shld_nuclear_heat_mw=1.3611259588044891,
            p_cp_shield_nuclear_heat_mw=0,
            f_p_div_coolant_pump_total_heat=0.0050000000000000001,
            p_plasma_separatrix_mw=143.6315222649435,
            p_div_nuclear_heat_total_mw=182.71773382328519,
        ),
        legacy_sample(
            "powerflow-baseline-2018-b",
            p_plasma_rad_mw=287.44866938104849,
            f_a_fw_outboard_hcd=0,
            p_div_rad_total_mw=33.056596978820579,
            a_fw_outboard=1168.1172772224481,
            a_fw_total=1891.2865102700493,
            p_beam_orbit_loss_mw=0,
            p_fw_alpha_mw=19.829653483586444,
            p_he=8000000,
            dp_he=550000,
            gamma_he=1.667,
            t_in_bb=573.13,
            t_out_bb=773.13,
            etaiso=0.90000000000000002,
            f_p_fw_blkt_pump=1.0,
            p_fw_nuclear_heat_total_mw=230.98304919926957,
            p_blkt_nuclear_heat_total_mw=1550.1447895848396,
            f_p_shld_coolant_pump_total_heat=0.0050000000000000001,
            p_shld_nuclear_heat_mw=1.4038170956592293,
            p_cp_shield_nuclear_heat_mw=0,
            f_p_div_coolant_pump_total_heat=0.0050000000000000001,
            p_plasma_separatrix_mw=143.51338080047339,
            p_div_nuclear_heat_total_mw=182.66070017727785,
        ),
        legacy_sample(
            "powerflow-reference-run",
            p_plasma_rad_mw=218.68809520720959,
            f_a_fw_outboard_hcd=0.0,
            p_div_rad_total_mw=15.855769588425584,
            a_fw_outboard=1048.452495290143,
            a_fw_total=1681.4734553069488,
            p_beam_orbit_loss_mw=0.0,
            p_fw_alpha_mw=16.391644584793067,
            p_he=8000000.0,
            dp_he=550000.0,
            gamma_he=1.667,
            t_in_bb=573.13,
            t_out_bb=773.13,
            etaiso=0.9,
            f_p_fw_blkt_pump=1.0,
            p_fw_nuclear_heat_total_mw=179.03215391121836,
            p_blkt_nuclear_heat_total_mw=1351.03741774038,
            f_p_shld_coolant_pump_total_heat=0.005,
            p_shld_nuclear_heat_mw=1.4802724479874645,
            p_cp_shield_nuclear_heat_mw=0.0,
            f_p_div_coolant_pump_total_heat=0.005,
            p_plasma_separatrix_mw=174.97328472847954,
            p_div_nuclear_heat_total_mw=94.34720303124747,
        ),
    ]

    fuzz_bounds = {
        "p_plasma_rad_mw": (50.0, 600.0),
        # Non-zero, unlike every legacy point: `p_fw_hcd_rad_total_mw` is identically
        # zero at `f_a_fw_outboard_hcd == 0`, so without this the fuzz would never
        # exercise that output at all.
        "f_a_fw_outboard_hcd": (0.001, 0.15),
        "p_div_rad_total_mw": (1.0, 60.0),
        "a_fw_outboard": (400.0, 2000.0),
        "a_fw_total": (600.0, 3200.0),
        "p_beam_orbit_loss_mw": (0.0, 20.0),
        "p_fw_alpha_mw": (1.0, 60.0),
        "p_he": (4.0e6, 1.2e7),
        # Kept well below `p_he`'s lower bound: `pfactor`'s base is
        # `p_he / (p_he - dp_he)` and goes singular as the two approach each other.
        "dp_he": (1.0e5, 1.0e6),
        "gamma_he": (1.3, 1.7),
        "t_in_bb": (450.0, 650.0),
        "t_out_bb": (700.0, 900.0),
        "etaiso": (0.7, 0.99),
        "f_p_fw_blkt_pump": (0.5, 1.5),
        "p_fw_nuclear_heat_total_mw": (50.0, 500.0),
        "p_blkt_nuclear_heat_total_mw": (500.0, 2500.0),
        "f_p_shld_coolant_pump_total_heat": (0.001, 0.05),
        "p_shld_nuclear_heat_mw": (0.1, 10.0),
        "p_cp_shield_nuclear_heat_mw": (0.0, 5.0),
        "f_p_div_coolant_pump_total_heat": (0.001, 0.05),
        "p_plasma_separatrix_mw": (50.0, 400.0),
        "p_div_nuclear_heat_total_mw": (20.0, 300.0),
    }
