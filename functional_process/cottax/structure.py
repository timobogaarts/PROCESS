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

import jax.numpy as jnp  # noqa: F401
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
from functional_process.models.structure import calculate_structure_masses


def calculate_structure(
    plasma_current,
    rmajor,
    rminor,
    kappa,
    b_plasma_toroidal_on_axis,
    dr_tf_inner_bore,
    dr_tf_outboard,
    dr_tf_inboard,
    z_tf_inside_half,
    whtshld,
    m_div_plate,
    m_pf_coil_conductor_total,
    m_pf_coil_structure_total,
    m_tf_coils_total,
    m_fw_total,
    m_blkt_total,
    m_fw_blkt_div_coolant_total,
    dewmkg,
):
    """The two ratios-of-ports `Structure.__call__` computed inline before delegating
    -- `total_weight_pf` (PF conductor + structure mass) and `tf_h_width` (TF coil
    horizontal bore width) -- now live here, so the declaration is a name and not a
    body (`_audit/formulas_split.md` step 1).
    """
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
        return calculate_structure(
            plasma_current,
            rmajor,
            rminor,
            kappa,
            b_plasma_toroidal_on_axis,
            dr_tf_inner_bore,
            dr_tf_outboard,
            dr_tf_inboard,
            z_tf_inside_half,
            whtshld,
            m_div_plate,
            m_pf_coil_conductor_total,
            m_pf_coil_structure_total,
            m_tf_coils_total,
            m_fw_total,
            m_blkt_total,
            m_fw_blkt_div_coolant_total,
            dewmkg,
        )
