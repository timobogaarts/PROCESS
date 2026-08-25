"""Pure-functional port of `Stellarator.st_strc` (chunk 1D of unit #1).

Audit record:
`functional_process/_audit/units/models/stellarator/structure.md`. The
record proposes splitting the source method in two: the real structural masses (this
module's `calculate_structure_masses`), and the "previous scaling law, kept for
comparison, not fully trusted" reporting value
(`calculate_intercoil_mass_scaling_reference`), which never feeds
`aintmass`/`clgsmass`/`coldmass` and is only ever printed. Splitting it out drops one
otherwise-unused argument (`e_tf_magnetic_stored_total_gj`) from the real function's
signature.

`fncmass` and `gsmass` are not ported: both are unconditional literal `0.0` in the
source (open question 1 in the record -- whether a constant "producer" should be a graph
node at all is a policy question, not resolved here, so neither is wrapped in a
`CallableNode`).

`StructureMasses` below is the `cottax` node -- a thin `ExplicitFunction` declaration
wrapping `calculate_structure_masses` unchanged. See `_audit/schema.md`'s "cottax node"
section for why the pytree-namespace surface
(`cottax.interfaces.pytree_namespace_module`) rather than a hand-built `CallableNode`:
`Output`/`FromExactly` read like the PROCESS `data.<area>.<field>` path they name, instead of a
`VarPath` built from a string one node at a time.
"""

from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.safe_math import safe_pow
from functional_process.paths import (
    fwbs,
    physics,
    stellarator,
    stellarator_config,
    structure,
    tfcoil,
)

_INTERCOIL_THICKNESS_COEFFICIENT = 0.18
"""Effective thickness (m) scaled by the empirical 1.5-power law below."""

_INTERCOIL_REFERENCE_FIELD_T = 5.6
"""Reference toroidal field (T) -- Helias 5-B design point."""

_GRAVITY_SUPPORT_FRACTION = 0.2
"""Gravity support mass as a fraction of the intercoil support mass (Helias 5-B fit)."""

_SCALING_LAW_COEFFICIENT = 1.3483
_SCALING_LAW_EXPONENT = 0.7821


def calculate_structure_masses(
    stella_config_coilsurface,
    f_st_rmajor,
    r_coil_minor,
    stella_config_coil_rminor,
    dx_tf_inboard_out_toroidal,
    len_tf_coil,
    n_tf_coils,
    b_plasma_toroidal_on_axis,
    den_steel,
    m_tf_coils_total,
    dewmkg,
):
    """Intercoil support, gravity support, and cooled-component structural masses.

    Ports `Stellarator.st_strc`'s real (non-constant, non-comparison-only) outputs.

    Parameters
    ----------
    stella_config_coilsurface :
        Reference coil surface area (m2). `.stellarator_config.stella_config_coilsurface`.
    f_st_rmajor :
        Major-radius scaling factor. `.stellarator.f_st_rmajor`.
    r_coil_minor :
        Coil minor radius (m). `.stellarator.r_coil_minor`.
    stella_config_coil_rminor :
        Reference coil minor radius (m). `.stellarator_config.stella_config_coil_rminor`.
    dx_tf_inboard_out_toroidal :
        TF coil inboard-to-outboard toroidal extent (m). `.tfcoil.dx_tf_inboard_out_toroidal`.
    len_tf_coil :
        TF coil length (m). `.tfcoil.len_tf_coil`.
    n_tf_coils :
        Number of TF coils. `.tfcoil.n_tf_coils`.
    b_plasma_toroidal_on_axis :
        Toroidal field on axis (T). `.physics.b_plasma_toroidal_on_axis`.
    den_steel :
        Steel density (kg/m3). `.fwbs.den_steel`.
    m_tf_coils_total :
        Total TF coil mass (kg). `.tfcoil.m_tf_coils_total`.
    dewmkg :
        Cryostat mass (kg). `.fwbs.dewmkg`.

    Returns
    -------
    :
        `(aintmass, clgsmass, coldmass)` -- intercoil support structure mass,
        gravity support structure mass, and total mass of cooled components (kg).
    """
    intercoil_surface = (
        stella_config_coilsurface
        * f_st_rmajor
        * (r_coil_minor / stella_config_coil_rminor)
        - dx_tf_inboard_out_toroidal * len_tf_coil * n_tf_coils
    )

    aintmass = (
        _INTERCOIL_THICKNESS_COEFFICIENT
        * (b_plasma_toroidal_on_axis / _INTERCOIL_REFERENCE_FIELD_T) ** 2
        * intercoil_surface
        * den_steel
    )

    clgsmass = _GRAVITY_SUPPORT_FRACTION * aintmass

    coldmass = m_tf_coils_total + aintmass + dewmkg

    return aintmass, clgsmass, coldmass


def calculate_intercoil_mass_scaling_reference(e_tf_magnetic_stored_total_gj):
    """Previous (Moon 1982) intercoil structure mass scaling, kept for comparison only.

    Ports the `msupstr` calculation in `Stellarator.st_strc`. Not used by
    `calculate_structure_masses` -- source comment: "the previous scaling law for
    intercoil structure... we do not really trust yet." Printed, never stored to
    `data`, never feeds another calculation.

    Parameters
    ----------
    e_tf_magnetic_stored_total_gj :
        Total TF coil stored magnetic energy (GJ). `.tfcoil.e_tf_magnetic_stored_total_gj`.

    Returns
    -------
    :
        `msupstr`, the scaling-law comparison mass (kg).
    """
    m_struc = _SCALING_LAW_COEFFICIENT * safe_pow(
        1000.0 * e_tf_magnetic_stored_total_gj, _SCALING_LAW_EXPONENT
    )
    return 1000.0 * m_struc


class StructureMasses(ExplicitFunction):
    """cottax node: `calculate_structure_masses`, unchanged, with its ports declared.

    `calculate_intercoil_mass_scaling_reference` has no node: it feeds nothing else in
    the graph (reporting-only, per the module docstring), so there is nowhere for it to
    sit as a node with no reader.
    """

    aintmass = OutputInto(structure)
    clgsmass = OutputInto(structure)
    coldmass = OutputInto(structure)

    def __call__(
        self,
        stella_config_coilsurface=From(stellarator_config),
        f_st_rmajor=From(stellarator),
        r_coil_minor=From(stellarator),
        stella_config_coil_rminor=From(stellarator_config),
        dx_tf_inboard_out_toroidal=From(tfcoil),
        len_tf_coil=From(tfcoil),
        n_tf_coils=From(tfcoil),
        b_plasma_toroidal_on_axis=From(physics),
        den_steel=From(fwbs),
        m_tf_coils_total=From(tfcoil),
        dewmkg=From(fwbs),
    ):
        return calculate_structure_masses(
            stella_config_coilsurface,
            f_st_rmajor,
            r_coil_minor,
            stella_config_coil_rminor,
            dx_tf_inboard_out_toroidal,
            len_tf_coil,
            n_tf_coils,
            b_plasma_toroidal_on_axis,
            den_steel,
            m_tf_coils_total,
            dewmkg,
        )
