"""Pure-functional port of `process/models/structure.py` (`Structure`,
`.tokamak.structure`).
"""

import jax.numpy as jnp  # noqa: F401

from functional_process.cottax.paths import (
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
    """The two ratios-of-ports `Structure.__call__` computed inline before delegating --
    `total_weight_pf` (PF conductor + structure mass) and `tf_h_width` (TF coil
    horizontal bore width) -- now live here, so the declaration is a name and not a body
    (`_audit/formulas_split.md` step 1).
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


# cottax node: `.tokamak.structure`.
Structure = structure.function(
    calculate_structure,
    "fncmass",
    "aintmass",
    "clgsmass",
    "coldmass",
    "gsmass",
    reads={
        "plasma_current": physics,
        "rmajor": physics,
        "rminor": physics,
        "kappa": physics,
        "b_plasma_toroidal_on_axis": physics,
        "dr_tf_inner_bore": build,
        "dr_tf_outboard": build,
        "dr_tf_inboard": build,
        "z_tf_inside_half": build,
        "whtshld": fwbs,
        "m_div_plate": divertor,
        "m_pf_coil_conductor_total": pf_coil,
        "m_pf_coil_structure_total": pf_coil,
        "m_tf_coils_total": tfcoil,
        "m_fw_total": fwbs,
        "m_blkt_total": fwbs,
        "m_fw_blkt_div_coolant_total": fwbs,
        "dewmkg": fwbs,
    },
)
