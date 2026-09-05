"""Pure physics functions extracted from
`functional_process.cottax.stellarator.stellarator_fwbs_s4`, per `_audit/formulas_split.md`
step 2 phase A. The graph declarations that wrap these stay in that sibling
module.
"""


def calculate_blanket_component_masses(
    vol_blkt_total,
    fblli2o,
    fblbe,
    den_steel,
    fblss,
    fblvd,
):
    """Blanket component masses, excluding coolant, at `blktmodel == 0`, solid breeder.

    Ports `stellarator.py:1068-1091` exactly -- the `else` (solid breeder, HCPB) sub-arm
    of the `blkttype` branch at `:1057` plus the steel/vanadium lines both sub-arms
    share, all inside `if blktmodel == 0` at `:1056`.

    Both of those branches are *topology* switches, resolved at graph-assembly time and
    not inside this function (`_audit/naming_convention.md` § "switches are not ports"):
    the liquid-breeder sub-arm writes `.fwbs.wtbllipb`/`.fwbs.m_blkt_lithium` -- two
    entirely different fields -- rather than a different formula for the same ones, and
    the `blktmodel != 0` arm (`:1093-1181`) computes `m_blkt_steel_total`/
    `m_blkt_beryllium` from six `.build.bl{u,m,p}{i,o}th` sub-assembly thicknesses this
    function never reads, writes `.fwbs.whtblbreed` and `.fwbs.f_a_blkt_cooling_channels`
    which it never writes, and writes neither `m_blkt_li2o` nor `m_blkt_vanadium` at all.
    A `jnp.where` over either would invent graph edges the run does not have. See
    `total_process.py`'s `.fwbs.blktmodel,.fwbs.blkttype` `Switch` for the arm table.

    The four hardcoded densities (`2010.0` Li2O, `1850.0` beryllium, `5870.0` vanadium,
    and steel via `.fwbs.den_steel`) are PROCESS's own literals, reproduced verbatim --
    only steel's is a field.

    `m_blkt_total` is accumulated in PROCESS in two statements (`:1074-1076` then
    `:1087-1091`); written here as one left-associated sum, which is the identical
    floating-point operation order (`((li2o + be) + steel) + vanadium`).

    Parameters
    ----------
    vol_blkt_total :
        Total blanket volume (m3). `.fwbs.vol_blkt_total` -- owned by
        `FwBlanketShieldGeometry` (S1, `stellarator_fwbs_s1_s5.py`), an ordinary
        upstream edge.
    fblli2o :
        Lithium oxide fraction of blanket by volume. `.fwbs.fblli2o`.
    fblbe :
        Beryllium fraction of blanket by volume. `.fwbs.fblbe`.
    den_steel :
        Steel density (kg/m3). `.fwbs.den_steel`.
    fblss :
        Stainless steel fraction of blanket by volume. `.fwbs.fblss`.
    fblvd :
        Vanadium fraction of blanket by volume. `.fwbs.fblvd`.

    Returns
    -------
    :
        `(m_blkt_li2o, m_blkt_beryllium, m_blkt_steel_total, m_blkt_vanadium,
        m_blkt_total)`, all in kg.
    """
    m_blkt_li2o = vol_blkt_total * fblli2o * 2010.0
    m_blkt_beryllium = vol_blkt_total * fblbe * 1850.0

    m_blkt_steel_total = vol_blkt_total * den_steel * fblss
    m_blkt_vanadium = vol_blkt_total * 5870.0 * fblvd

    m_blkt_total = m_blkt_li2o + m_blkt_beryllium + m_blkt_steel_total + m_blkt_vanadium

    return (
        m_blkt_li2o,
        m_blkt_beryllium,
        m_blkt_steel_total,
        m_blkt_vanadium,
        m_blkt_total,
    )


def calculate_shield_mass(vol_shld_total, den_steel, vfshld):
    """Shield mass, and the penetration shield PROCESS sets equal to it.

    Ports `stellarator.py:1195-1206` exactly. Unconditional in `st_fwbs` -- no
    `blktmodel`, `blkttype` or `ipowerflow` branch guards either statement.

    `wpenshld = whtshld` is PROCESS's own assignment (`:1206`, comment "Penetration
    shield (set = internal shield)"), reproduced as a second output rather than dropped:
    it is a real `.fwbs.*` field with its own reader (`ShieldCost`), so collapsing the
    two would leave that reader on a boundary input.

    Parameters
    ----------
    vol_shld_total :
        Total shield volume (m3). `.fwbs.vol_shld_total` -- owned by
        `FwBlanketShieldGeometry` (S1), an ordinary upstream edge.
    den_steel :
        Steel density (kg/m3). `.fwbs.den_steel`.
    vfshld :
        Coolant void fraction in the shield. `.fwbs.vfshld`.

    Returns
    -------
    :
        `(whtshld, wpenshld)` -- shield mass and penetration shield mass (kg), equal by
        construction.
    """
    whtshld = vol_shld_total * den_steel * (1.0 - vfshld)
    wpenshld = whtshld

    return whtshld, wpenshld
