"""Pure-functional port of `process/models/structure.py`
(`Structure`, `.tokamak.structure`).

Audit record: `functional_process/_audit/units/models/structure.md`. Entry point is
`Structure.run()`, which forwards straight-line algebra (no internal iteration, no call
into another `Model`'s method) into `Structure.structure()`.

`structure()` is not `@staticmethod` but touches `self` only inside its `if output:`
reporting block (`self.outfile`); called with `output=False` it performs no
`self.data`/`self` access at all, so the port needed only a signature promotion, not an
extraction.

Two switches, `i_tf_sup` and `i_pf_conductor`, gate two of `coldmass`'s four additive
terms (`process/models/structure.py:165-168`). Both take their PROCESS default on
`tests/regression/input_files/large_tokamak_eval.IN.DAT` (neither is set in that file):
`i_tf_sup = 1` (superconducting TF, `tfcoil_variables.py:261`) and `i_pf_conductor = 0`
(`PFConductorModel.SUPERCONDUCTING`, `pfcoil_variables.py:230`) -- both conditions true,
so *both* terms are live simultaneously on this run. Per the wave-1 binding policy ("no
switch is a static kwarg" -- a switch read to branch selects an occupant, not a
parameter), `calculate_structure_masses` below bakes in this one live combination
(`i_tf_sup == 1 and i_pf_conductor == SUPERCONDUCTING`) rather than accepting either
switch as an argument: the function is this combination's occupant, not a general
`coldmass` formula. The other three combinations are UNPORTED -- see `structure.md`
§ switches touched.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.paths import (
    build,
    divertor,
    fwbs,
    pf_coil,
    physics,
    structure,
    tfcoil,
)

_DENS = 7.8e3  # kg/m^3, `structure()`'s local `dens` -- coil/gravity-support density.
_SIGAL = 2.5e7  # Pa, `structure()`'s local `sigal` -- allowable stress.


def calculate_structure_masses(
    ai,
    r0,
    a,
    akappa,
    b0,
    tf_h_width,
    tfhmax,
    shldmass,
    dvrtmass,
    pfmass,
    tfmass,
    m_fw_total,
    blmass,
    m_fw_blkt_div_coolant_total,
    dewmass,
):
    """Support-structure masses for `i_tf_sup == 1` (superconducting TF) and
    `i_pf_conductor == PFConductorModel.SUPERCONDUCTING` -- the combination live on
    `large_tokamak_eval.IN.DAT`. Ports `Structure.structure`,
    `process/models/structure.py:76-231`, arithmetic unchanged; `output`'s reporting
    block dropped (pure reporting, no `data` write); the `i_tf_sup`/`i_pf_conductor`
    branches collapsed to this occupant's one live combination (see module docstring).

    Parameters
    ----------
    ai :
        Plasma current, max design value (A). `.physics.plasma_current`.
    r0 :
        Plasma major radius (m). `.physics.rmajor`.
    a :
        Plasma minor radius (m). `.physics.rminor`.
    akappa :
        Plasma elongation. `.physics.kappa`.
    b0 :
        Axial (toroidal) B-field (T). `.physics.b_plasma_toroidal_on_axis`.
    tf_h_width :
        TF coil horizontal bore width (m). `dr_tf_inner_bore + dr_tf_outboard +
        dr_tf_inboard` (`.build.*`).
    tfhmax :
        TF coil max height (m). `.build.z_tf_inside_half`.
    shldmass :
        Total shield mass (kg). `.fwbs.whtshld`.
    dvrtmass :
        Total divertor (+ associated structure) mass (kg). `.divertor.m_div_plate`.
    pfmass :
        Total PF coil (+ case) mass (kg). `.pf_coil.m_pf_coil_conductor_total +
        .pf_coil.m_pf_coil_structure_total`.
    tfmass :
        Total TF coil (+ case) mass (kg). `.tfcoil.m_tf_coils_total`.
    m_fw_total :
        First wall mass (kg). `.fwbs.m_fw_total`.
    blmass :
        Blanket mass (kg). `.fwbs.m_blkt_total`.
    m_fw_blkt_div_coolant_total :
        Total coolant mass (kg). `.fwbs.m_fw_blkt_div_coolant_total`.
    dewmass :
        Vacuum vessel + cryostat mass (kg). `.fwbs.dewmkg`.

    Returns
    -------
    tuple
        `(fncmass, aintmass, clgsmass, coldmass, gsm)`, all kg -- `Structure.structure`'s
        five-way tuple unpack, owned together since they come from one PROCESS function.
    """
    # Outer PF coil fence (1990 ITER fit).
    fncmass = 2.1e-11 * ai * ai * r0 * akappa * a

    # Intercoil support between TF coils, reacting overturning moment (1990 ITER fit).
    aintmass_raw = 1.4e6 * (ai / 2.2e7) * b0 / 4.85e0 * tf_h_width**2 / 50.0e0
    # PROCESS logs and kludges an overflowing `aintmass` to `1e10`
    # (`structure.py:156-158` -- `logger.error`, dropped here as pure reporting)
    # rather than leaving it `inf`.
    aintmass = jnp.where(jnp.isinf(aintmass_raw), 1e10, aintmass_raw)

    # Total mass of coils plus support plus vacuum vessel + cryostat.
    coilmass = tfmass + pfmass + aintmass + dewmass

    # Total mass of cooled components -- both live-run terms unconditional here
    # (`i_tf_sup == 1` and `i_pf_conductor == SUPERCONDUCTING` both hold; see module
    # docstring for the switch this occupant bakes in).
    coldmass = tfmass + aintmass + dewmass + pfmass

    # Coil gravity support mass.
    clgsmass = coilmass * (r0 / 6.0e0) * 9.1e0 * 9.807e0 * _DENS / _SIGAL

    # Gravity support masses, scaled from Spears algorithms (9/90).
    ws1 = m_fw_blkt_div_coolant_total + m_fw_total + blmass + shldmass + dvrtmass
    gsm1 = 5.0e0 * 9.807e0 * ws1 * _DENS / _SIGAL  # Torus leg support.

    ws2 = ws1 + tfmass + pfmass + aintmass + clgsmass
    # Ring beam.
    gsm2 = 1.0e-3 * 34.77e0 * (r0 + 1.0e0) * jnp.sqrt(0.001e0 * ws2) * _DENS

    gsm3 = 1.0e-6 * 0.3e0 * (tfhmax + 2.0e0) * ws2 * _DENS  # Ring legs.

    gsm = gsm1 + gsm2 + gsm3

    return fncmass, aintmass, clgsmass, coldmass, gsm


class Structure(ExplicitFunction):
    """cottax node: `.tokamak.structure`.

    Answers `i_tf_sup == 1` (superconducting TF) and `i_pf_conductor ==
    PFConductorModel.SUPERCONDUCTING` -- the switch combination live on
    `large_tokamak_eval.IN.DAT` (neither field is set in that file, so both take their
    PROCESS default). See module docstring; other combinations UNPORTED.
    """

    fncmass = OutputInto(structure)
    aintmass = OutputInto(structure)
    clgsmass = OutputInto(structure)
    coldmass = OutputInto(structure)
    gsmass = OutputInto(structure)

    def __call__(
        self,
        plasma_current=From(physics),
        rmajor=From(physics),
        rminor=From(physics),
        kappa=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        dr_tf_inner_bore=From(build),
        dr_tf_outboard=From(build),
        dr_tf_inboard=From(build),
        z_tf_inside_half=From(build),
        whtshld=From(fwbs),
        m_div_plate=From(divertor),
        m_pf_coil_conductor_total=From(pf_coil),
        m_pf_coil_structure_total=From(pf_coil),
        m_tf_coils_total=From(tfcoil),
        m_fw_total=From(fwbs),
        m_blkt_total=From(fwbs),
        m_fw_blkt_div_coolant_total=From(fwbs),
        dewmkg=From(fwbs),
    ):
        total_weight_pf = m_pf_coil_conductor_total + m_pf_coil_structure_total
        tf_h_width = dr_tf_inner_bore + dr_tf_outboard + dr_tf_inboard

        return calculate_structure_masses(
            ai=plasma_current,
            r0=rmajor,
            a=rminor,
            akappa=kappa,
            b0=b_plasma_toroidal_on_axis,
            tf_h_width=tf_h_width,
            tfhmax=z_tf_inside_half,
            shldmass=whtshld,
            dvrtmass=m_div_plate,
            pfmass=total_weight_pf,
            tfmass=m_tf_coils_total,
            m_fw_total=m_fw_total,
            blmass=m_blkt_total,
            m_fw_blkt_div_coolant_total=m_fw_blkt_div_coolant_total,
            dewmass=dewmkg,
        )
