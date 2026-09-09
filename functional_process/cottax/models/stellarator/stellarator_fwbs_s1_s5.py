"""Pure-functional port of `st_fwbs`'s S1 and S5 sub-computations (see
`stellarator_E_fwbs_synthesis.md`).
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    build,
    costs,
    first_wall,
    fwbs,
    heat_transport,
    physics,
    stellarator_config,
)
from functional_process.cottax.wraps import WrapsFunction
from functional_process.models.stellarator.stellarator_fwbs_s1_s5 import (
    calculate_cryostat_and_vv_geometry,
    calculate_fw_blanket_shield_geometry,
)


class FwBlanketShieldGeometry(WrapsFunction):
    """cottax node: `calculate_fw_blanket_shield_geometry` (S1), unchanged."""

    fn = calculate_fw_blanket_shield_geometry

    abktflnc = From(costs)
    pflux_fw_neutron_mw = From(physics)
    life_plant = From(costs)
    a_fw_total = From(first_wall)
    rminor = From(physics)
    dr_fw_plasma_gap_inboard = From(build)
    dr_fw_inboard = From(build)
    dr_fw_plasma_gap_outboard = From(build)
    dr_fw_outboard = From(build)
    ipowerflow = From(heat_transport)
    a_plasma_surface = From(physics)
    fhole = From(fwbs)
    f_ster_div_single = From(fwbs)
    f_a_fw_outboard_hcd = From(fwbs)
    dr_blkt_inboard = From(build)
    dr_blkt_outboard = From(build)
    fvolsi = From(fwbs)
    fvolso = From(fwbs)
    dr_shld_inboard = From(build)
    dr_shld_outboard = From(build)
    p_neutron_total_mw = From(physics)
    stella_config_neutron_peakfactor = From(stellarator_config)

    life_fw_fpy = OutputInto(fwbs)
    a_fw_inboard = OutputInto(first_wall)
    a_fw_outboard = OutputInto(first_wall)
    a_blkt_total_surface = OutputInto(build)
    a_blkt_inboard_surface = OutputInto(build)
    a_blkt_outboard_surface = OutputInto(build)
    vol_blkt_inboard = OutputInto(fwbs)
    vol_blkt_outboard = OutputInto(fwbs)
    vol_blkt_total = OutputInto(fwbs)
    a_shld_total_surface = OutputInto(build)
    a_shld_inboard_surface = OutputInto(build)
    a_shld_outboard_surface = OutputInto(build)
    vol_shld_total = OutputInto(fwbs)
    pnucloss = OutputInto(fwbs)
    wallpf = OutputInto(fwbs)


class CryostatAndVvGeometry(WrapsFunction):
    """cottax node: `calculate_cryostat_and_vv_geometry` (S5), unchanged."""

    fn = calculate_cryostat_and_vv_geometry

    r_tf_outboard_mid = From(build)
    dr_tf_outboard = From(build)
    dr_pf_cryostat = From(fwbs)
    rmajor = From(physics)
    dr_cryostat = From(build)
    dr_fw_plasma_gap_inboard = From(build)
    dr_fw_inboard = From(build)
    dr_blkt_inboard = From(build)
    dr_shld_inboard = From(build)
    dr_fw_plasma_gap_outboard = From(build)
    dr_fw_outboard = From(build)
    dr_blkt_outboard = From(build)
    dr_shld_outboard = From(build)
    rminor = From(physics)
    dr_vv_inboard = From(build)
    dr_vv_outboard = From(build)
    a_plasma_surface = From(physics)
    fvoldw = From(fwbs)
    den_steel = From(fwbs)

    r_cryostat_inboard = OutputInto(fwbs)
    vol_cryostat = OutputInto(fwbs)
    vol_vv = OutputInto(fwbs)
    m_vv = OutputInto(fwbs)
    dewmkg = OutputInto(fwbs)
