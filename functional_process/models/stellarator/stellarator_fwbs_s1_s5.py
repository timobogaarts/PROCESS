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
  registered `StructureMasses` node (`stellarator_D_structure.py`) reads it.

Neither calls into another model (`self.hcpb`/`self.physics`/etc.) -- both are pure
`self.data` arithmetic, confirmed by reading `st_fwbs`'s full body (422-1682) for any
sub-model reference in these two line ranges: none. `local-intermediate` classification
applies to `vol_shld_inboard`/`vol_shld_outboard` (S1) -- computed, summed into
`vol_shld_total`, never themselves written to `data` -- so they are not ported as
separate return values, same convention `stellarator_D_structure.py` uses.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.paths import (
    build,
    costs,
    first_wall,
    fwbs,
    heat_transport,
    physics,
    stellarator_config,
)


def calculate_fw_blanket_shield_geometry(
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
    """S1: first wall/blanket/shield areas and volumes.

    Ports `Stellarator.st_fwbs` lines 515-605 exactly, including PROCESS's own
    `ipowerflow`-gated formula for `a_blkt_total_surface` (two extra loss fractions
    subtracted when power-flow tracking is on) -- as `jnp.where`, `ipowerflow` is a
    plain traced argument here, not a topology switch: both branches share the same
    output shape, just a different subtraction.

    Parameters
    ----------
    abktflnc :
        Allowable blanket neutron fluence (MW-yr/m2). `.costs.abktflnc`.
    pflux_fw_neutron_mw :
        First wall neutron flux (MW/m2). `.physics.pflux_fw_neutron_mw`.
    life_plant :
        Plant lifetime (years). `.costs.life_plant`.
    a_fw_total :
        Total first wall area (m2). `.first_wall.a_fw_total`.
    rminor :
        Plasma minor radius (m). `.physics.rminor`.
    dr_fw_plasma_gap_inboard, dr_fw_plasma_gap_outboard :
        First-wall/plasma gap, inboard/outboard (m). `.build.dr_fw_plasma_gap_inboard`/
        `dr_fw_plasma_gap_outboard`.
    dr_fw_inboard, dr_fw_outboard :
        First wall thickness, inboard/outboard (m). `.build.dr_fw_inboard`/
        `dr_fw_outboard`.
    ipowerflow :
        Power-flow tracking switch (0/1). `.heat_transport.ipowerflow`.
    a_plasma_surface :
        Plasma surface area (m2). `.physics.a_plasma_surface`.
    fhole :
        Area fraction lost through holes (ports etc.). `.fwbs.fhole`.
    f_ster_div_single :
        Divertor area fraction (single-null). `.fwbs.f_ster_div_single`.
    f_a_fw_outboard_hcd :
        HCD outboard first-wall area fraction. `.fwbs.f_a_fw_outboard_hcd`.
    dr_blkt_inboard, dr_blkt_outboard :
        Blanket thickness, inboard/outboard (m). `.build.dr_blkt_inboard`/
        `dr_blkt_outboard`.
    fvolsi, fvolso :
        Shield area coverage factors, inboard/outboard. `.fwbs.fvolsi`/`fvolso`.
    dr_shld_inboard, dr_shld_outboard :
        Shield thickness, inboard/outboard (m). `.build.dr_shld_inboard`/
        `dr_shld_outboard`.
    p_neutron_total_mw :
        Total neutron power (MW). `.physics.p_neutron_total_mw`.
    stella_config_neutron_peakfactor :
        Neutron flux peaking factor, machine preset. `.stellarator_config.
        stella_config_neutron_peakfactor`.

    Returns
    -------
    :
        `(life_fw_fpy, a_fw_inboard, a_fw_outboard, a_blkt_total_surface,
        a_blkt_inboard_surface, a_blkt_outboard_surface, vol_blkt_inboard,
        vol_blkt_outboard, vol_blkt_total, a_shld_total_surface,
        a_shld_inboard_surface, a_shld_outboard_surface, vol_shld_total, pnucloss,
        wallpf)`.
    """
    life_fw_fpy = jnp.minimum(abktflnc / pflux_fw_neutron_mw, life_plant)

    a_fw_inboard = 0.5 * a_fw_total
    a_fw_outboard = 0.5 * a_fw_total

    r1 = rminor + 0.5 * (
        dr_fw_plasma_gap_inboard
        + dr_fw_inboard
        + dr_fw_plasma_gap_outboard
        + dr_fw_outboard
    )
    a_blkt_total_surface = jnp.where(
        ipowerflow == 0,
        a_plasma_surface * r1 / rminor * (1.0 - fhole),
        a_plasma_surface
        * r1
        / rminor
        * (1.0 - fhole - f_ster_div_single - f_a_fw_outboard_hcd),
    )
    a_blkt_inboard_surface = 0.5 * a_blkt_total_surface
    a_blkt_outboard_surface = 0.5 * a_blkt_total_surface

    vol_blkt_inboard = a_blkt_inboard_surface * dr_blkt_inboard
    vol_blkt_outboard = a_blkt_outboard_surface * dr_blkt_outboard
    vol_blkt_total = vol_blkt_inboard + vol_blkt_outboard

    r1 = r1 + 0.5 * (dr_blkt_inboard + dr_blkt_outboard)
    a_shld_total_surface = a_plasma_surface * r1 / rminor
    a_shld_inboard_surface = 0.5 * a_shld_total_surface * fvolsi
    a_shld_outboard_surface = 0.5 * a_shld_total_surface * fvolso

    vol_shld_inboard = a_shld_inboard_surface * dr_shld_inboard
    vol_shld_outboard = a_shld_outboard_surface * dr_shld_outboard
    vol_shld_total = vol_shld_inboard + vol_shld_outboard

    pnucloss = p_neutron_total_mw * fhole
    wallpf = stella_config_neutron_peakfactor

    return (
        life_fw_fpy,
        a_fw_inboard,
        a_fw_outboard,
        a_blkt_total_surface,
        a_blkt_inboard_surface,
        a_blkt_outboard_surface,
        vol_blkt_inboard,
        vol_blkt_outboard,
        vol_blkt_total,
        a_shld_total_surface,
        a_shld_inboard_surface,
        a_shld_outboard_surface,
        vol_shld_total,
        pnucloss,
        wallpf,
    )


def calculate_cryostat_and_vv_geometry(
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
    """S5: cryostat and vacuum-vessel geometry and masses.

    Ports `Stellarator.st_fwbs` lines 1282-1330 exactly. No branches.

    Parameters
    ----------
    r_tf_outboard_mid :
        Mid-plane radius of the outboard TF leg (m). `.build.r_tf_outboard_mid`.
    dr_tf_outboard :
        TF coil outboard thickness (m). `.build.dr_tf_outboard`.
    dr_pf_cryostat :
        PF-coil-to-cryostat clearance (m). `.fwbs.dr_pf_cryostat`.
    rmajor :
        Plasma major radius (m). `.physics.rmajor`.
    dr_cryostat :
        Cryostat wall thickness (m). `.build.dr_cryostat`.
    dr_fw_plasma_gap_inboard, dr_fw_plasma_gap_outboard :
        First-wall/plasma gap, inboard/outboard (m).
    dr_fw_inboard, dr_fw_outboard :
        First wall thickness, inboard/outboard (m).
    dr_blkt_inboard, dr_blkt_outboard :
        Blanket thickness, inboard/outboard (m).
    dr_shld_inboard, dr_shld_outboard :
        Shield thickness, inboard/outboard (m).
    rminor :
        Plasma minor radius (m). `.physics.rminor`.
    dr_vv_inboard, dr_vv_outboard :
        Vacuum vessel thickness, inboard/outboard (m). `.build.dr_vv_inboard`/
        `dr_vv_outboard`.
    a_plasma_surface :
        Plasma surface area (m2). `.physics.a_plasma_surface`.
    fvoldw :
        Vacuum vessel volume multiplier (ports, supports, etc.). `.fwbs.fvoldw`.
    den_steel :
        Steel density (kg/m3). `.fwbs.den_steel`.

    Returns
    -------
    :
        `(r_cryostat_inboard, vol_cryostat, vol_vv, m_vv, dewmkg)`.
    """
    r_cryostat_inboard = r_tf_outboard_mid + 0.5 * dr_tf_outboard + dr_pf_cryostat
    adewex = r_cryostat_inboard - rmajor

    vol_cryostat = 4.0 * (jnp.pi**2) * rmajor * adewex * dr_cryostat

    r1 = rminor + 0.5 * (
        dr_fw_plasma_gap_inboard
        + dr_fw_inboard
        + dr_blkt_inboard
        + dr_shld_inboard
        + dr_fw_plasma_gap_outboard
        + dr_fw_outboard
        + dr_blkt_outboard
        + dr_shld_outboard
    )
    vol_vv = (
        (dr_vv_inboard + dr_vv_outboard) / 2.0 * a_plasma_surface * r1 / rminor * fvoldw
    )

    m_vv = vol_vv * den_steel
    dewmkg = (vol_vv + vol_cryostat) * den_steel

    return r_cryostat_inboard, vol_cryostat, vol_vv, m_vv, dewmkg


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

    `dewmkg` is a real downstream dependency: `stellarator_D_structure.py`'s
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
