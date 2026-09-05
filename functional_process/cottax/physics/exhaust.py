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

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.paths import physics
from functional_process.models.physics.exhaust import (
    calculate_eu_demo_re_attachment_metric,
    calculate_psep_over_r_metric,
    calculate_radiation_fraction,
)


class RadiationFraction(ExplicitFunction):
    """cottax node: `calculate_radiation_fraction`, ports declared."""

    f_p_plasma_separatrix_rad = OutputInto(physics)

    def __call__(
        self,
        p_plasma_rad_mw=From(physics),
        p_plasma_heating_total_mw=From(physics),
    ):
        return calculate_radiation_fraction(p_plasma_rad_mw, p_plasma_heating_total_mw)


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
