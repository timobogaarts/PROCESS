"""Pure-functional port of `st_fwbs`'s S3 fragment (`stellarator.py:1030-1043`).

Audit record: `functional_process/models/stellarator/stellarator_fwbs_s3.md`.
`stellarator_E_fwbs_synthesis.md` names this fragment
`divertor_mass_and_first_call_seed` and flags it as the one piece of `st_fwbs` that is
genuine cross-call state rather than ordinary same-call dataflow: `.divertor.
a_div_surface_total` is written by `Divertor` (unit #4, `st_div`, `divertor.py`), which
runs *after* `st_fwbs` in `Stellarator.run()`'s call order, so this fragment always reads
either a hardcoded `50.0` bootstrap (the true first call) or the *previous* call's
`Divertor` output (every call after).

**This port does not decide how that cycle is driven.** Per this task's brief, the
`Blocking`/`FixedPoint`/`Cut` machinery for `{this fragment, Divertor}` is deferred to a
later composition pass once the whole graph's shape is known. `calculate_
divertor_plate_mass` below is therefore the fragment's arithmetic core only, taking
`a_div_surface_total` as a plain, ordinary argument -- exactly as any other cross-unit
`Input` would be declared. The source's `first_call_stfwbs` branch and its `50.0` literal
are real and important (see the audit record's "the cross-call read/bootstrap,
precisely" section) but are a fact about *which value a future driver feeds in*, not
something this function's own signature encodes. `self.first_call_stfwbs` is also not a
`DataStructure` field at all -- it is plain Python state on the `Stellarator` model
instance (`stellarator.py:95`), with no `VarPath` -- see the audit record's open
questions for why that is left unresolved rather than papered over.

Deliberately **not registered** in `functional_process/total_process.py` by this pass:
out of this task's boundary, and premature regardless, since `.divertor.
a_div_surface_total` is also `Divertor`'s own `Output` on the very same `VarPath` this
node reads -- wiring both into one graph unconditionally today would pick an evaluation
order rather than reproduce PROCESS's real one-call-lagged semantics.
"""

from cottax.interfaces.pytree_namespace_module import ExplicitFunction, Input, Output


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


class DivertorPlateMass(ExplicitFunction):
    """cottax node: `calculate_divertor_plate_mass`, unchanged, ports declared.

    See the module docstring for why this node is not registered in
    `functional_process/total_process.py` yet.
    """

    m_div_plate = Output(lambda s: s.divertor.m_div_plate)

    def __call__(
        self,
        a_div_surface_total=Input(lambda s: s.divertor.a_div_surface_total),
        den_div_structure=Input(lambda s: s.divertor.den_div_structure),
        f_vol_div_coolant=Input(lambda s: s.divertor.f_vol_div_coolant),
        dx_div_plate=Input(lambda s: s.divertor.dx_div_plate),
    ):
        return calculate_divertor_plate_mass(
            a_div_surface_total, den_div_structure, f_vol_div_coolant, dx_div_plate
        )
