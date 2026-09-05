"""Mutual and self inductances of the PF coils, the CS and the plasma.

Audit record: `functional_process/_audit/units/models/pfcoil/inductance.md`.

One unit, `calculate_pf_cs_plasma_inductances`, porting `PFCoil.induct`
(`process/models/pfcoil.py:1721-1984`, everything before its `if not output: return`)
and `PFCoil.selfinductance` (`:2837-2867`). One cottax node, `PFCoilInductance`
(`.tokamak.pf_coil.inductance`), owning `.pf_coil.ind_pf_cs_plasma_mutual` whole.

**This closes the cycle rather than joining it.** Before this module,
`.pf_coil.ind_pf_cs_plasma_mutual` was a boundary input of
`currents.py::CSFluxSwing` -- nothing in the graph wrote it, because `induct` was
unported. It now has a producer, and that producer reads `.pf_coil.n_pf_coil_turns` and
the coil geometry, which `masses.py::PFCoilSizes` owns. So the SCC `currents.py`
describes grows from three nodes to four, and PROCESS's `first_call` bootstrap
(`pfcoil.py:605-608`, `ind_pf_cs_plasma_mutual[:, :] = 1.0`) is revealed for what it is:
**the cycle's initial guess**, not an external input. See § "The cycle, one node larger"
in the record.

**`noh` is computed, and the discontinuity is real.** `induct` chooses how many pancake
segments to split the CS into as

    noh = ceil(2 * z_pf_coil_upper[CS] / (r_pf_coil_outer[CS] - r_pf_coil_inner[CS]))

(`pfcoil.py:1758-1765`) -- an integer computed from *solved* geometry, on a run where
the CS radial thickness is an iteration variable. Every inductance this routine returns
depends on it, so `ind_pf_cs_plasma_mutual` is a **piecewise-constant-discontinuous**
function of `dr_cs`: PROCESS's answer steps whenever `dz_cs_full / dr_cs` crosses an
integer, and the derivative its own finite difference reports is the derivative of the
piece it happens to be sitting on.

This port computes it, which was not the first answer. It was pinned at `NOH = 30` --
the value `large_tokamak_eval.IN.DAT` takes, its ratio being `29.027` -- on the argument
that a different `noh` is a different occupant, the way a different `i_pf_location`
pattern is. That argument was wrong about the cost. The pin is right on
`large_tokamak_eval` and wrong on the other two solenoid tokamaks, where the solve walks
across several integers: `32 -> 29 -> 27` on `large_tokamak_nof` and `28 -> 27` on
`low_aspect_ratio_DEMO`. It cost **80 of the 85 disagreeing rows** the cold report used
to carry under `cold_start.NOH_WRONG` (`large_tokamak_nof` 668 agreeing/61 off ->
`721/8`; `low_aspect_ratio_DEMO` `695/40` -> `722/13`), with no new disagreement and no
new error anywhere.

**The pinned arm was the one with the wrong derivative** -- a correct tangent taken on
the wrong piece -- which is why the usual "a `ceil` is not differentiable" objection
does not decide this. It is also not the §31.27 situation: that was a *truncated Neumann
series*, a derivative with unbounded error and invisible to the value test. A `ceil`
gives the correct derivative of a piecewise function; the error is confined to the
value, and only within one step.

What was measured before landing it, because a discontinuity inside an SQP loop deserves
more than an argument: the ratio is **constant across every evaluation inside a given
SQP window** on both configurations, so no Jacobian is ever taken across a step -- the
integer changes between iterates, as a pure value change. Driving `large_tokamak_nof`'s
cold start to `7.1e-15` of an integer and perturbing across it eleven times gives seven
iterations and `max|eq| = 2.748e-06` on every row, with a bit-identical objective; the
**pinned** arm's objective is not bit-identical across the same straddle. A ±1e-13 to
±1e-9 jitter table on `.build.dr_cs`, 44 solves, moves no iteration count, status, or
residual on either arm.
"""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.pfcoil import (
    SPHERICAL_TOKAMAK_TOPOLOGY,
    PFCoilTopology,
)
from functional_process.cottax.paths import build, pf_coil, physics
from functional_process.models.pfcoil.inductance import (
    NOH_PAD,  # noqa: F401 -- re-exported for tests/test_cold_start.py
    _cs_segments,  # noqa: F401 -- re-exported for tests/test_cold_start.py
    calculate_pf_cs_plasma_inductances,  # noqa: F401 -- re-exported for tests
    calculate_pf_cs_plasma_inductances_at_reference_width,
    calculate_pf_plasma_inductances_no_central_solenoid,  # noqa: F401 -- re-exported for tests
    calculate_pf_plasma_inductances_no_central_solenoid_for_topology,
    calculate_solenoid_self_inductance,  # noqa: F401 -- re-exported for tests
)

NPLAS = 1
"""`nplas`, a literal `1` in `induct` (`pfcoil.py:1734`) -- the plasma is one filament,
at `(rmajor, 0)`. Every `for ii in range(nplas)` in the source is therefore a one-trip
loop, and every `/ nplas` a division by one; both are dropped here."""


class PFCoilInductance(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.inductance`.

    Occupant for `iohcl = 1`, `n_pf_coils_in_group = (1, 1, 2, 2)` and `noh = 30`.

    **Owns `.pf_coil.ind_pf_cs_plasma_mutual` whole, not per index**, and the evidence
    for that is on both sides. Producer side: `induct` zeroes the entire matrix
    (`pfcoil.py:1750`) and then fills every entry of the eight-circuit block from one
    shared set of geometry reads -- there is no slice of it this node does not compute,
    unlike `b_pf_coil_peak`, whose index 6 comes from an unported routine and which is
    therefore owned per index. Consumer side: `currents.py::CSFluxSwing` reads column
    `[0:6, 7]`, `process/models/pulse.py:228,235` reads `[n_cs_pf_coils - 1, ...]`, and
    `process/models/power.py:320-539` reads the whole matrix; owning six entries would
    leave the rest of what this node computes unowned. `_audit/naming_convention.md`
    § "Array elements" asks for per-index addressing where read and write ranges
    *differ*, and here they do not.

    `.pf_coil.nef` (`:1944-1947`) is **not** owned: it is `n_cs_pf_coils - 1` on this
    arm, loop bookkeeping of the same kind as `n_cs_pf_coils` itself, which
    `__init__.py` records as graph-assembly data rather than a port.
    """

    ind_pf_cs_plasma_mutual = OutputInto(pf_coil)

    def __call__(
        self,
        rmajor=From(physics),
        ind_plasma=From(physics),
        dr_cs=From(build),
        r_cs_middle=From(pf_coil),
        r_pf_coil_middle=From(pf_coil),
        z_pf_coil_middle=From(pf_coil),
        r_pf_coil_inner=From(pf_coil),
        r_pf_coil_outer=From(pf_coil),
        z_pf_coil_upper=From(pf_coil),
        z_pf_coil_lower=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
    ):
        return calculate_pf_cs_plasma_inductances_at_reference_width(
            rmajor=rmajor,
            ind_plasma=ind_plasma,
            dr_cs=dr_cs,
            r_cs_middle=r_cs_middle,
            r_pf_coil_middle=r_pf_coil_middle,
            z_pf_coil_middle=z_pf_coil_middle,
            r_pf_coil_inner=r_pf_coil_inner,
            r_pf_coil_outer=r_pf_coil_outer,
            z_pf_coil_upper=z_pf_coil_upper,
            z_pf_coil_lower=z_pf_coil_lower,
            n_pf_coil_turns=n_pf_coil_turns,
        )


class PFCoilInductanceNoCentralSolenoid(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.inductance`, the `iohcl = 0` occupant.

    Owns `.pf_coil.ind_pf_cs_plasma_mutual` whole, on the same producer-side argument as
    `PFCoilInductance`: `induct` zeroes the matrix (`pfcoil.py:1750`) and fills every
    entry of the circuit block that exists on this arm.

    Not a subclass of `PFCoilInductance`, because it declares **four reads fewer**
    (`dr_cs`, `r_cs_middle`, `r_pf_coil_inner`, `r_pf_coil_outer`) and a subclass may
    only widen a signature, not narrow it. See
    `calculate_pf_plasma_inductances_no_central_solenoid` for which blocks of `induct`
    those reads belong to and why the `noh` discontinuity is not present here.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)

    ind_pf_cs_plasma_mutual = OutputInto(pf_coil)

    def __call__(
        self,
        rmajor=From(physics),
        ind_plasma=From(physics),
        r_pf_coil_middle=From(pf_coil),
        z_pf_coil_middle=From(pf_coil),
        z_pf_coil_upper=From(pf_coil),
        z_pf_coil_lower=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
    ):
        return calculate_pf_plasma_inductances_no_central_solenoid_for_topology(
            rmajor=rmajor,
            ind_plasma=ind_plasma,
            r_pf_coil_middle=r_pf_coil_middle,
            z_pf_coil_middle=z_pf_coil_middle,
            z_pf_coil_upper=z_pf_coil_upper,
            z_pf_coil_lower=z_pf_coil_lower,
            n_pf_coil_turns=n_pf_coil_turns,
            topology=self.topology,
        )
