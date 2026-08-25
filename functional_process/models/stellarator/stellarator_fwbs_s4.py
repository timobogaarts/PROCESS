"""Pure-functional port of `st_fwbs`'s S4 component-mass block
(`process/models/stellarator/stellarator.py:1045-1274`, excluding S3's 1030-1043).

Audit record:
`functional_process/_audit/units/models/stellarator/stellarator_fwbs_s4.md`.
`stellarator_E_fwbs_synthesis.md` § 1 names this piece `blanket_shield_fw_coolant_mass`
and § 5 recorded it as "portable once S2 and S3's signatures exist" -- both have since
landed (`stellarator_fwbs_s2.py`, `stellarator_fwbs_s3.py`), so the stated blocker is
discharged.

**Two of S4's four sub-blocks are ported here, and the other two deliberately are not.**
The split is not by convenience; it is by whether the sub-block's operands exist as
`VarPath`s at all:

- **Ported -- blanket component masses** (`:1056-1091`, the `blktmodel == 0` arm, solid
  breeder sub-arm): `m_blkt_li2o`, `m_blkt_beryllium`, `m_blkt_steel_total`,
  `m_blkt_vanadium`, `m_blkt_total`. All operands are `.fwbs.*` fields.
- **Ported -- shield mass** (`:1195-1206`): `whtshld`, `wpenshld`. Unconditional inside
  `st_fwbs`, no branch of any kind.
- **Not ported -- the `coolvol` accumulator** (`:1048-1052`, `:1191-1193`, `:1202`,
  `:1222-1226`/`:1238-1246`) and the total coolant mass it feeds
  (`.fwbs.m_fw_blkt_div_coolant_total`, `:1269-1274`). `coolvol` is a plain Python local
  threaded across S3 and all of S4 -- it has no `VarPath` -- and its S3 seed is the
  cross-call `.divertor.a_div_surface_total` read that `stellarator_fwbs_s3.md` records
  as genuine inter-call state.
- **Not ported -- the first-wall mass block** (`:1208-1262`: `.fwbs.m_fw_total`,
  `.fwbs.fwclfr`). Its `ipowerflow != 0` arm reads `f_a_fw_coolant_inboard`/
  `f_a_fw_coolant_outboard`, two Python *locals* produced inside S2 with two different
  formulas selected by `blktmodel` (`stellarator_E_fwbs_synthesis.md` § 2's
  cross-boundary ledger, row 2). They are not `.fwbs.*` fields, so a node cannot bind
  them, and `stellarator_fwbs_s2.py`'s ported arms do not return them either. Measured,
  not assumed: `.fwbs.m_fw_total`, `.fwbs.fwclfr` and
  `.fwbs.m_fw_blkt_div_coolant_total` currently have **zero** readers in
  `total_process.graph_for()`, so leaving them unowned closes no edge that any consumer
  is waiting on -- see the audit record's "what this port leaves undone".

Neither ported function calls into another model -- confirmed by reading the whole
1045-1274 range for a `self.<submodel>.` reference: none, matching the synthesis
record's "no cross-model calls of its own".
"""

from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.paths import fwbs


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


class BlanketComponentMasses(ExplicitFunction):
    """cottax node: `calculate_blanket_component_masses`, unchanged, ports declared.

    Registered in `total_process.py` behind the synthetic
    `.fwbs.blktmodel,.fwbs.blkttype` `Switch` -- the arm this node implements is
    `blktmodel == 0` *and* `blkttype not in {1, 2}`, which is PROCESS's own default
    (`fwbs_variables.py:479,494`: `blktmodel = 0`, `blkttype = 3`) and the reference
    `stellarator_helias.IN.DAT` run's arm (that file sets neither field).

    Four of the five outputs close boundary inputs `BlanketCost` was reading as given
    (`_audit/boundary_inputs_audit.md` § 4c items (b1)-(b4)). `.fwbs.m_blkt_total` has no
    reader in the graph today; it is declared anyway because PROCESS computes it in the
    same straight-line block and dropping it would make the node's output set a subset
    of what the source writes for no reason other than current demand.
    """

    m_blkt_li2o = OutputInto(fwbs)
    m_blkt_beryllium = OutputInto(fwbs)
    m_blkt_steel_total = OutputInto(fwbs)
    m_blkt_vanadium = OutputInto(fwbs)
    m_blkt_total = OutputInto(fwbs)

    def __call__(
        self,
        vol_blkt_total=From(fwbs),
        fblli2o=From(fwbs),
        fblbe=From(fwbs),
        den_steel=From(fwbs),
        fblss=From(fwbs),
        fblvd=From(fwbs),
    ):
        return calculate_blanket_component_masses(
            vol_blkt_total, fblli2o, fblbe, den_steel, fblss, fblvd
        )


class ShieldMass(ExplicitFunction):
    """cottax node: `calculate_shield_mass`, unchanged, ports declared.

    Registered in `COMMON`, not behind any `Switch`: `stellarator.py:1195-1206` sits
    outside every branch in `st_fwbs`, so both outputs exist in every configuration this
    graph can assemble. Closes `_audit/boundary_inputs_audit.md` § 4c items (b5)/(b6) --
    `Bldgs` and `ShieldCost` read `.fwbs.whtshld`, `ShieldCost` reads `.fwbs.wpenshld`.
    """

    whtshld = OutputInto(fwbs)
    wpenshld = OutputInto(fwbs)

    def __call__(
        self,
        vol_shld_total=From(fwbs),
        den_steel=From(fwbs),
        vfshld=From(fwbs),
    ):
        return calculate_shield_mass(vol_shld_total, den_steel, vfshld)
