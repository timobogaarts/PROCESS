"""Harness case for the ported subset of `process/models/structure.py`
(`.tokamak.structure`).

Audit record: `functional_process/_audit/units/models/structure.md`. One unit:
`calculate_structure_masses`, tier-1, the `(i_tf_sup=1, i_pf_conductor=SUPERCONDUCTING)`
occupant -- the switch combination live on `large_tokamak_eval.IN.DAT`.

`Structure.structure` is not `@staticmethod` but performs no `self`/`self.data` access
at all when called with `output=False` (the only branch that touches `self` is the
`if output:` reporting block), so the reference below is a bound instance method called
directly -- no `DataStructure` backdoor to close.
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.cottax.structure import calculate_structure_masses
from process.data_structure.pfcoil_variables import PFConductorModel
from process.models.structure import Structure


def _reference_structure_masses(
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
    """`Structure.structure` at this occupant's live switch combination.

    `i_tf_sup=1`, `i_pf_conductor=PFConductorModel.SUPERCONDUCTING` -- baked in, matching
    the one combination `calculate_structure_masses` itself bakes in (see
    `structure.md` § switches touched). `structure()` needs no `self.data`, only
    `self.outfile` inside its `output=True` reporting arm, so `output=False` closes the
    backdoor with no adapter.
    """
    fncmass, aintmass, clgsmass, coldmass, gsm = Structure().structure(
        ai=ai,
        r0=r0,
        a=a,
        akappa=akappa,
        b0=b0,
        i_tf_sup=1,
        i_pf_conductor=PFConductorModel.SUPERCONDUCTING,
        tf_h_width=tf_h_width,
        tfhmax=tfhmax,
        shldmass=shldmass,
        dvrtmass=dvrtmass,
        pfmass=pfmass,
        tfmass=tfmass,
        m_fw_total=m_fw_total,
        blmass=blmass,
        m_fw_blkt_div_coolant_total=m_fw_blkt_div_coolant_total,
        dewmass=dewmass,
        output=False,
    )
    return fncmass, aintmass, clgsmass, coldmass, gsm


class TestCalculateStructureMasses(Tier1Contract):
    """`calculate_structure_masses` -> `Structure.structure` at its live switch cell."""

    audit_record = "models/structure.md"
    reference = _reference_structure_masses
    ported = calculate_structure_masses

    # tests/unit/models/test_structure.py::TestStructure.test_structure, verbatim.
    samples = [
        legacy_sample(
            "large-tokamak-legacy",
            ai=17721306.969367817,
            r0=8.8901,
            a=2.8677741935483869,
            akappa=1.848,
            b0=5.3292,
            tf_h_width=15.337464674334223,
            tfhmax=9.0730900215620327,
            shldmass=2294873.8131476026,
            dvrtmass=43563.275828777645,
            pfmass=5446188.2481440185,
            tfmass=21234909.756419446,
            m_fw_total=224802.80270851994,
            blmass=3501027.3252278985,
            m_fw_blkt_div_coolant_total=1199.6389920083477,
            dewmass=16426726.727684354,
        ),
    ]

    fuzz_bounds = {
        "ai": (1.0e6, 3.0e7),
        "r0": (2.0, 20.0),
        "a": (0.5, 5.0),
        "akappa": (1.0, 2.2),
        "b0": (1.0, 12.0),
        "tf_h_width": (1.0, 30.0),
        "tfhmax": (1.0, 20.0),
        "shldmass": (1.0e4, 5.0e6),
        "dvrtmass": (1.0e3, 1.0e5),
        "pfmass": (1.0e4, 1.0e7),
        "tfmass": (1.0e4, 5.0e7),
        "m_fw_total": (1.0e3, 1.0e6),
        "blmass": (1.0e3, 1.0e7),
        "m_fw_blkt_div_coolant_total": (1.0e2, 1.0e4),
        "dewmass": (1.0e4, 5.0e7),
    }
