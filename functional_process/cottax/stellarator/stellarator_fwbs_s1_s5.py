"""Pure-functional port of `st_fwbs`'s S1 and S5 sub-computations (see
`stellarator_E_fwbs_synthesis.md`).

Both are self-contained tier-1 chunks the synthesis record already classified as
"portable now, no audit blocker, just execution" -- ported directly rather than queued,
per `unit_registry.md`'s standing practice for a confirmed self-contained tier-1 chunk.

- **S1** `fw_blanket_shield_geometry_setup` (`process/models/stellarator/stellarator.py`
  515-605): first wall/blanket/shield areas and volumes, `life_fw_fpy`, the neutron
  power lost through first-wall holes (`pnucloss`), and the neutron peaking factor
  (`wallpf`, a straight lookup off `stellarator_config`). One real branch, on
  `.heat_transport.ipowerflow`.
- **S5** `cryostat_and_vv_geometry` (1282-1330): cryostat and vacuum-vessel geometry and
  masses. No branches. `dewmkg` is a real downstream dependency -- chunk 1D's already-
  registered `StructureMasses` node (`structure.py`) reads it.

Neither calls into another model (`self.hcpb`/`self.physics`/etc.) -- both are pure
`self.data` arithmetic, confirmed by reading `st_fwbs`'s full body (422-1682) for any
sub-model reference in these two line ranges: none. `local-intermediate` classification
applies to `vol_shld_inboard`/`vol_shld_outboard` (S1) -- computed, summed into
`vol_shld_total`, never themselves written to `data` -- so they are not ported as
separate return values, same convention `structure.py` uses.
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
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
from functional_process.models.stellarator.stellarator_fwbs_s1_s5 import (
    calculate_cryostat_and_vv_geometry,
    calculate_fw_blanket_shield_geometry,
)


class FwBlanketShieldGeometry(ExplicitFunction):
    """cottax node: `calculate_fw_blanket_shield_geometry` (S1), unchanged."""

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

    def __call__(
        self,
        abktflnc=From(costs),
        pflux_fw_neutron_mw=From(physics),
        life_plant=From(costs),
        a_fw_total=From(first_wall),
        rminor=From(physics),
        dr_fw_plasma_gap_inboard=From(build),
        dr_fw_inboard=From(build),
        dr_fw_plasma_gap_outboard=From(build),
        dr_fw_outboard=From(build),
        ipowerflow=From(heat_transport),
        a_plasma_surface=From(physics),
        fhole=From(fwbs),
        f_ster_div_single=From(fwbs),
        f_a_fw_outboard_hcd=From(fwbs),
        dr_blkt_inboard=From(build),
        dr_blkt_outboard=From(build),
        fvolsi=From(fwbs),
        fvolso=From(fwbs),
        dr_shld_inboard=From(build),
        dr_shld_outboard=From(build),
        p_neutron_total_mw=From(physics),
        stella_config_neutron_peakfactor=From(stellarator_config),
    ):
        return calculate_fw_blanket_shield_geometry(
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
        )


class CryostatAndVvGeometry(ExplicitFunction):
    """cottax node: `calculate_cryostat_and_vv_geometry` (S5), unchanged.

    `dewmkg` is a real downstream dependency: `structure.py`'s
    `StructureMasses` node already declares `dewmkg=From(fwbs)`.
    """

    r_cryostat_inboard = OutputInto(fwbs)
    vol_cryostat = OutputInto(fwbs)
    vol_vv = OutputInto(fwbs)
    m_vv = OutputInto(fwbs)
    dewmkg = OutputInto(fwbs)

    def __call__(
        self,
        r_tf_outboard_mid=From(build),
        dr_tf_outboard=From(build),
        dr_pf_cryostat=From(fwbs),
        rmajor=From(physics),
        dr_cryostat=From(build),
        dr_fw_plasma_gap_inboard=From(build),
        dr_fw_inboard=From(build),
        dr_blkt_inboard=From(build),
        dr_shld_inboard=From(build),
        dr_fw_plasma_gap_outboard=From(build),
        dr_fw_outboard=From(build),
        dr_blkt_outboard=From(build),
        dr_shld_outboard=From(build),
        rminor=From(physics),
        dr_vv_inboard=From(build),
        dr_vv_outboard=From(build),
        a_plasma_surface=From(physics),
        fvoldw=From(fwbs),
        den_steel=From(fwbs),
    ):
        return calculate_cryostat_and_vv_geometry(
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
        )
