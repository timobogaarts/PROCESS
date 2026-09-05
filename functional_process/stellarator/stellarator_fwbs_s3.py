"""Pure physics functions extracted from
`functional_process.models.stellarator.stellarator_fwbs_s3`, per `_audit/formulas_split.md`
step 2 phase A. The graph declarations that wrap these stay in that sibling
module.
"""


def calculate_divertor_plate_mass(
    a_div_surface_total,
    den_div_structure,
    f_vol_div_coolant,
    dx_div_plate,
):
    """Divertor plate mass, excluding coolant.

    Ports `stellarator.py:1038-1043`'s `m_div_plate` formula exactly. `a_div_surface_total`
    is taken as a plain argument: in PROCESS this is either a hardcoded `50.0e0`
    bootstrap value (the true first call to `st_fwbs` in a `Stellarator` instance's
    lifetime) or `Divertor`'s own output from the *previous* solver iteration (every
    call after) -- see the audit record's cross-call finding. Which of those two this
    argument holds at a given call is a driving-time fact this function does not need
    to know.

    Parameters
    ----------
    a_div_surface_total :
        Total divertor plate area (m2). `.divertor.a_div_surface_total` -- see the
        module docstring for why this is a cross-call read, not an ordinary same-call
        value.
    den_div_structure :
        Divertor structure density (kg/m3). `.divertor.den_div_structure`.
    f_vol_div_coolant :
        Divertor coolant volume fraction. `.divertor.f_vol_div_coolant`.
    dx_div_plate :
        Divertor plate thickness (m). `.divertor.dx_div_plate`.

    Returns
    -------
    :
        `m_div_plate` -- divertor plate mass, excluding coolant (kg).
    """
    return (
        a_div_surface_total
        * den_div_structure
        * (1.0 - f_vol_div_coolant)
        * dx_div_plate
    )
