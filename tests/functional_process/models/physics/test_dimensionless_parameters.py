"""Harness case for `dimensionless_parameters.py`.

`outplas` is 1095 source lines, almost all of it `process_output` calls and other
models' own `.output()` methods (out of scope -- see the audit record). The reference
adapter runs PROCESS's **real** `outplas()`, not a transcription of its first three
lines: `self.geometry.output()`, the very next call after the three assignments this
unit ports, is stubbed to raise a sentinel exception, caught immediately after. Every
`process_output`/other-model call downstream of that point never runs, so this needs no
`outfile`/`mfile` fixture and no stub for any of `outplas`'s other ~15 sub-model calls.
"""

from functional_process._harness import Tier1Contract, fuzz_samples, legacy_sample
from functional_process.models.physics.dimensionless_parameters import (
    calculate_dimensionless_plasma_parameters,
)
from process.core.model import DataStructure
from process.models.physics.physics import Physics


class _StopAfterDimensionlessParameters(Exception):
    """Sentinel: raised by the stubbed `geometry.output()` right after the three
    assignments this unit ports, so `outplas` never reaches any `process_output` call.
    """


def _reference_dimensionless_plasma_parameters(
    dlamie,
    vol_plasma,
    rmajor,
    b_plasma_toroidal_on_axis,
    eps,
    nd_plasma_electron_line,
    kappa,
    e_plasma_beta,
    plasma_current,
    m_ions_total_amu,
):
    """Call PROCESS's real `Physics.outplas`, stopped right after its only computation."""
    data = DataStructure()
    data.physics.dlamie = dlamie
    data.physics.vol_plasma = vol_plasma
    data.physics.rmajor = rmajor
    data.physics.b_plasma_toroidal_on_axis = b_plasma_toroidal_on_axis
    data.physics.eps = eps
    data.physics.nd_plasma_electron_line = nd_plasma_electron_line
    data.physics.kappa = kappa
    data.physics.e_plasma_beta = e_plasma_beta
    data.physics.plasma_current = plasma_current
    data.physics.m_ions_total_amu = m_ions_total_amu

    def _stop_here():
        raise _StopAfterDimensionlessParameters

    stub_geometry = type("StubGeometry", (), {"output": lambda self: _stop_here()})()

    physics = Physics(
        plasma_profile=None,
        current_drive=None,
        plasma_beta=None,
        plasma_inductance=None,
        plasma_density_limit=None,
        plasma_exhaust=None,
        plasma_bootstrap_current=None,
        plasma_confinement=None,
        plasma_transition=None,
        plasma_current=None,
        plasma_fields=None,
        plasma_dia_current=None,
        plasma_geometry=stub_geometry,
        scrape_off_layer=None,
    )
    physics.data = data
    physics.outfile = None
    physics.mfile = None

    try:
        physics.outplas()
    except _StopAfterDimensionlessParameters:
        pass

    return data.physics.nu_star, data.physics.rho_star, data.physics.beta_mcdonald


class TestDimensionlessPlasmaParameters(Tier1Contract):
    """`Physics.outplas` (first three assignments) ->
    `calculate_dimensionless_plasma_parameters`.
    """

    audit_record = "models/physics/dimensionless_parameters.md"
    reference = _reference_dimensionless_plasma_parameters
    ported = calculate_dimensionless_plasma_parameters

    # Realistic large-tokamak-scale magnitudes (no PROCESS unit test exercises `outplas`
    # directly -- it is a pure reporting shell with no assertions of its own in
    # `tests/unit`), chosen only to keep every intermediate finite and positive
    # (`nu_star`/`rho_star` both take a `sqrt` and divide by `e_plasma_beta`/`vol_plasma`,
    # neither of which is ever zero or negative in a real operating point).
    #
    # `plasma_current` and `dlamie`, however, ARE both exactly zero in a real
    # *stellarator* operating point -- neither is ever written outside
    # `Physics.physics()`, which the stellarator pipeline never calls -- so PROCESS's own
    # converged Helias solve stores `nu_star = nan`. The sampling below is deliberately
    # tokamak-only (`plasma_current` bounded away from 0) because that `nan` is
    # PROCESS's, not the port's: see `dimensionless_parameters.md` section "real
    # PROCESS defect found" for the measurement and for why no guard was added. Do not
    # "fix" it by
    # widening these bounds to include 0 -- that would only assert that the port
    # reproduces an upstream defect it already does.
    samples = [
        legacy_sample(
            "large-tokamak-scale",
            dlamie=17.5,
            vol_plasma=1888.0,
            rmajor=8.0,
            b_plasma_toroidal_on_axis=5.7,
            eps=0.32,
            nd_plasma_electron_line=8.0e19,
            kappa=1.7,
            e_plasma_beta=3.0e8,
            plasma_current=1.8e7,
            m_ions_total_amu=2.5,
        ),
        *fuzz_samples(
            {
                "dlamie": (14.0, 20.0),
                "vol_plasma": (100.0, 3000.0),
                "rmajor": (3.0, 25.0),
                "b_plasma_toroidal_on_axis": (2.0, 12.0),
                "eps": (0.1, 0.9),
                "nd_plasma_electron_line": (2.0e19, 1.0e21),
                "kappa": (1.0, 2.5),
                "e_plasma_beta": (1.0e6, 1.0e9),
                "plasma_current": (1.0e6, 3.0e7),
                "m_ions_total_amu": (1.0, 5.0),
            },
            count=5,
            seed=0,
        ),
    ]
