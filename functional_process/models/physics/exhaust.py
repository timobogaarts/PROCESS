"""Pure-functional port of `process/models/physics/exhaust.py`.

Registry unit #11. Audit record:
`functional_process/_audit/units/models/physics/exhaust.md`.

In scope: `calculate_radiation_fraction` (the registry's stated method) and, since
2026-08-27, `calculate_eu_demo_re_attachment_metric`. The scope note this module used to
carry -- "the other three `PlasmaExhaust` statics are pure and self-contained but out of
the registry's stated scope" -- is **narrowed rather than withdrawn**:

- `calculate_separatrix_power` is ported, but in `physics.py` beside the mint it feeds
  (`.physics.p_plasma_separatrix_mw_raw`); see that module's `SeparatrixPower`.
- `calculate_eu_demo_re_attachment_metric` is here, because `optimise_design.md` §11.5
  found `.physics.p_div_bt_q_aspect_rmajor_mw` to be a boundary zero that PROCESS's own
  solve moves -- constraint 68 is one of the two inequalities `large_tokamak_eval`
  *violates*, and §11.4 measured its whole port gradient row as identically zero because
  of exactly this gap.
- `calculate_psep_over_r_metric` **is ported as of 2026-09-01**, and the sentence that
  used to stand here -- *"no active constraint and no ported node reads
  `.physics.p_plasma_separatrix_rmajor_mw`, so an occupant for it would be a producer
  with no consumer"* -- was **false when `optimise_design.md` §26 measured it**, not
  merely stale. Constraint 56 is active on both tracked spherical tokamaks
  (`spherical_tokamak_eval.IN.DAT:21`, `st_regression.IN.DAT:689`) and reads exactly
  that path. The reason no earlier pin caught it is §26.1's: every boundary pin in this
  port is measured over the *model* graph, and a path read only by a condition is
  invisible to all of them. The docstring's own prediction -- "a two-line follow-up the
  day one appears" -- held; see `PsepOverRMetric`.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.paths import physics


def calculate_radiation_fraction(p_plasma_rad_mw, p_plasma_heating_mw):
    """Radiation fraction of the plasma. Ports `PlasmaExhaust.calculate_
    radiation_fraction`, `process/models/physics/exhaust.py:194-220`.

    Source returns `0.0` (and logs a warning) when `p_plasma_heating_mw == 0`, rather
    than raising -- a real domain guard, not PROCESS-signals-invalid-input-by-raising.
    Ported as a genuine `jnp.where` branch (not `reference_domain_errors`): the safe
    denominator avoids the classic "NaN through the untaken branch" trap
    (`_audit/test_harness.md`'s `test_gradient_finite`) that a bare
    `p_plasma_rad_mw / p_plasma_heating_mw` inside the `where` would otherwise leak
    into the gradient even on the taken (nonzero) branch.
    """
    zero_heating = p_plasma_heating_mw == 0
    safe_denominator = jnp.where(zero_heating, 1.0, p_plasma_heating_mw)
    return jnp.where(zero_heating, 0.0, p_plasma_rad_mw / safe_denominator)


class RadiationFraction(ExplicitFunction):
    """cottax node: `calculate_radiation_fraction`, ports declared."""

    f_p_plasma_separatrix_rad = OutputInto(physics)

    def __call__(
        self,
        p_plasma_rad_mw=From(physics),
        p_plasma_heating_total_mw=From(physics),
    ):
        return calculate_radiation_fraction(p_plasma_rad_mw, p_plasma_heating_total_mw)


def calculate_eu_demo_re_attachment_metric(
    p_plasma_separatrix_mw,
    b_plasma_toroidal_on_axis,
    q95,
    aspect,
    rmajor,
):
    """The EU-DEMO divertor re-attachment metric, P_sep*B_t / (q95*A*R0) [MW T / m].

    Ports `PlasmaExhaust.calculate_eu_demo_re_attachment_metric`,
    `process/models/physics/exhaust.py:150-192`, unchanged.

    Parameters
    ----------
    p_plasma_separatrix_mw :
        Power crossing the separatrix (MW).
    b_plasma_toroidal_on_axis :
        Toroidal field on the plasma axis (T).
    q95 :
        Safety factor at the 95% flux surface.
    aspect :
        Plasma aspect ratio.
    rmajor :
        Plasma major radius (m).

    Returns
    -------
    :
        The re-attachment metric (MW T / m).
    """
    return (p_plasma_separatrix_mw * b_plasma_toroidal_on_axis) / (q95 * aspect * rmajor)


class EuDemoReAttachmentMetric(ExplicitFunction):
    """cottax node: `calculate_eu_demo_re_attachment_metric`, ports declared.

    **Reads the mint, not the field.** `physics.py:818-826` computes this metric from
    `.physics.p_plasma_separatrix_mw` as it stands *before* the KLUDGE at `:843-845`
    divides it by `1 - exp(-P_sep)`. The port names that first value
    `.physics.p_plasma_separatrix_mw_raw` (see `physics.py::force_positive_separatrix_
    power`), and this node reads it -- one of the three call sites that docstring
    enumerates as seeing the pre-transform number.

    At this machine the two are the *same number*: `P_sep = 176.8 MW`, so
    `1 - exp(-P_sep)` differs from 1 by ~1e-77 and no test could tell the readings
    apart. The mint is read anyway, because the transform is a no-op only while P_sep
    is large -- it is unbounded as P_sep approaches zero, which is the whole reason
    PROCESS applies it and the whole reason the mint exists. Wiring by "the values
    agree here" is how a port acquires a defect that only a different input file can
    see.

    Unswitched: `Physics.run` computes it outside every `if`.
    """

    p_div_bt_q_aspect_rmajor_mw = OutputInto(physics)

    def __call__(
        self,
        p_plasma_separatrix_mw_raw=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        q95=From(physics),
        aspect=From(physics),
        rmajor=From(physics),
    ):
        return calculate_eu_demo_re_attachment_metric(
            p_plasma_separatrix_mw_raw,
            b_plasma_toroidal_on_axis,
            q95,
            aspect,
            rmajor,
        )


def calculate_psep_over_r_metric(p_plasma_separatrix_mw, rmajor):
    """Power crossing the separatrix per unit major radius, P_sep / R0 [MW / m].

    Ports `PlasmaExhaust.calculate_psep_over_r_metric`,
    `process/models/physics/exhaust.py:127-147`, unchanged.

    Parameters
    ----------
    p_plasma_separatrix_mw :
        Power crossing the separatrix (MW).
    rmajor :
        Plasma major radius (m).

    Returns
    -------
    :
        `P_sep / R0` (MW/m). `.physics.p_plasma_separatrix_rmajor_mw`.
    """
    return p_plasma_separatrix_mw / rmajor


class PsepOverRMetric(ExplicitFunction):
    """cottax node: `calculate_psep_over_r_metric`, ports declared.

    **Reads the mint, not the field** -- the identical argument
    `EuDemoReAttachmentMetric`'s docstring makes, and for a stronger reason here:
    `physics.py:811-816` is the *first* of the three call sites that see
    `.physics.p_plasma_separatrix_mw` before the KLUDGE at `:843-845` divides it by
    `1 - exp(-P_sep)`, and it sits three lines above the re-attachment metric in the
    same block. So this node reads `.physics.p_plasma_separatrix_mw_raw`
    (`physics.py::SeparatrixPower`) exactly as its neighbour does.

    At both tracked spherical tokamaks the two readings are the same number to every
    bit -- `P_sep` is 180.0/181.3 MW, so `1 - exp(-P_sep)` differs from 1 by ~1e-79 --
    and no test in this port could tell them apart. The mint is read anyway, for the
    reason recorded next door: the transform is a no-op only while P_sep is large, and
    wiring by "the values agree here" is how a port acquires a defect only a different
    input file can see.

    **Why this node matters and the two beside it do not, yet.** `.physics.
    p_plasma_separatrix_rmajor_mw` is read by constraint 56 (`leq(P_sep/R0, 40)`),
    which is active on `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` and
    on no other tracked file. Frozen at the cold `0.0` the constraint read as satisfied
    with the whole of its margin to spare and contributed an identically zero Jacobian
    row; PROCESS's own converged answers are `39.99999999988` (st_regression -- *on*
    the bound, the single most binding constraint of that problem) and `40.2816`
    (spherical_tokamak_eval -- **violated** at PROCESS's own answer, where the port
    printed it comfortably satisfied). `optimise_design.md` §26.3 ranks it 2nd and 3rd
    of the seven live missing-producer rows; §29 measures what porting it moved.

    Unswitched: `Physics.run` computes it outside every `if`, one statement after
    `.physics.p_plasma_separatrix_mw` itself.
    """

    p_plasma_separatrix_rmajor_mw = OutputInto(physics)

    def __call__(
        self,
        p_plasma_separatrix_mw_raw=From(physics),
        rmajor=From(physics),
    ):
        return calculate_psep_over_r_metric(p_plasma_separatrix_mw_raw, rmajor)
