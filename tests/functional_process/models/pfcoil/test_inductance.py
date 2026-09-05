"""Harness cases for `functional_process/cottax/pfcoil/inductance.py`.

Audit record: `functional_process/_audit/units/models/pfcoil/inductance.md`.

Two tier-1 contracts:

- `calculate_solenoid_self_inductance` -> `PFCoil.selfinductance`, a `@staticmethod`
  with explicit arguments; called directly.
- `calculate_pf_cs_plasma_inductances` -> `PFCoil.induct(False)`. `induct` takes no
  arguments and works entirely through `self.data`, so the adapter binds a
  `DataStructure` carrying exactly the port's declared inputs plus the coil-count
  topology the port bakes in. Everything else keeps its dataclass default -- which is
  the check: a read the port failed to declare would sit at a default here and the two
  sides would disagree.

**Three arguments are `static_argnames`, and the reason is a property of PROCESS, not
of the port.** `induct` picks the number of CS pancake segments as

    noh = ceil(2 * z_pf_coil_upper[CS] / (r_pf_coil_outer[CS] - r_pf_coil_inner[CS]))

so every inductance it returns is a step function of those three. At the reference point
the ratio is `29.027`, i.e. **0.09 % above the integer 29** -- closer than PROCESS's own
`epsfcn = 1e-3` relative finite-difference step, which moves the ratio by 0.1 %. A
gradient comparison in those three directions would therefore be comparing the port's
within-piece derivative against a difference quotient taken *across* a discontinuity of
the reference. That is not something a tolerance can absorb and not a defect of the
port; it is the reference's own derivative being undefined there. Declared static, with
the finding recorded in `inductance.md` § "noh is a step function of the CS geometry".

The continuous dependence those three also carry is not lost wholesale: `z_pf_coil_lower`
stays differentiated and covers the PF-coil diagonal's `rl = |z_upper - z_lower|`, and
`r_pf_coil_inner`/`r_pf_coil_outer` are read by `induct` at index 6 *only*
(`process/models/pfcoil.py:1762-1764`, `:1897-1900`), so what goes unchecked is one term
of Bunet's formula rather than a whole coil's geometry.
"""

import numpy as np

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.cottax.pfcoil import (
    N_COILS_IN_GROUP,
    N_CS_PF_COILS,
    N_PF_GROUPS,
    NGC2,
    PLASMA_INDEX,
    SPHERICAL_TOKAMAK_TOPOLOGY,
)
from functional_process.cottax.pfcoil.inductance import (
    calculate_pf_cs_plasma_inductances,
    calculate_pf_plasma_inductances_no_central_solenoid,
    calculate_solenoid_self_inductance,
)
from process.core.model import DataStructure
from process.models.pfcoil import PFCoil

# Read off a converged, in-process PROCESS run of
# `tests/regression/input_files/large_tokamak_eval.IN.DAT`.
_R_MID = np.array([
    5.566666666666666,
    5.566666666666666,
    16.868931216641325,
    16.868931216641325,
    15.359714853856374,
    15.359714853856374,
    2.2772514872311596,
])
_Z_MID = np.array([
    9.644333333333332,
    -10.878217164127493,
    2.6666666666666665,
    -2.6666666666666665,
    7.466666666666666,
    -7.466666666666666,
    0.0,
])
_R_IN = np.array([
    4.917268464514993,
    4.871145214086073,
    16.303475590374013,
    16.303475590374013,
    14.967229668259906,
    14.967229668259906,
    2.003843190236783,
])
_R_OUT = np.array([
    6.21606486881834,
    6.26218811924726,
    17.434386842908637,
    17.434386842908637,
    15.752200039452843,
    15.752200039452843,
    2.550659784225536,
])
_Z_UP = np.array([
    10.293731535485007,
    -11.573738616708086,
    3.232122292933979,
    -3.232122292933979,
    7.8591518522631345,
    -7.8591518522631345,
    7.936395447714745,
])
_Z_LO = np.array([
    8.994935131181657,
    -10.1826957115469,
    2.1012110403993542,
    -2.1012110403993542,
    7.074181481070197,
    -7.074181481070197,
    -7.936395447714745,
])
_TURNS = np.array([
    463.88982745360926,
    532.1251000998008,
    191.84403916641483,
    191.84403916641483,
    123.23569673015545,
    123.23569673015545,
    4652.995074701359,
])
_R_CS_MIDDLE = 2.2772514872311596
_DR_CS = 0.546816593988753
_RMAJOR = 8.0
_IND_PLASMA = 1.4328756128565802e-05


def _around(base, fraction):
    """Elementwise `(lower, upper)` at `+-fraction` of `base`, sign-safe."""
    base = np.asarray(base, dtype=float)
    low = base * (1.0 - fraction)
    high = base * (1.0 + fraction)
    return np.minimum(low, high), np.maximum(low, high)


def _reference_solenoid_self_inductance(a, b, c, n):
    """`PFCoil.selfinductance`, a `@staticmethod` with no `self` use."""
    return PFCoil.selfinductance(a, b, c, n)


def _reference_pf_cs_plasma_inductances(
    rmajor,
    ind_plasma,
    dr_cs,
    r_cs_middle,
    r_pf_coil_middle,
    z_pf_coil_middle,
    r_pf_coil_inner,
    r_pf_coil_outer,
    z_pf_coil_upper,
    z_pf_coil_lower,
    n_pf_coil_turns,
):
    """`PFCoil.induct(False)` on a cold `DataStructure` seeded with these inputs.

    `induct` needs `self.data` and `self.outfile`, the latter only inside the reporting
    block that `output=False` skips; it never touches `cs_fatigue` or `cs_coil`, so both
    are `None`.
    """
    data = DataStructure()
    p = data.pf_coil

    data.build.iohcl = 1
    data.build.dr_cs = dr_cs
    data.physics.rmajor = rmajor
    data.physics.ind_plasma = ind_plasma

    p.n_cs_pf_coils = N_CS_PF_COILS
    p.n_pf_cs_plasma_circuits = PLASMA_INDEX + 1
    p.n_pf_coil_groups = N_PF_GROUPS
    p.n_pf_coils_in_group = np.array(
        [*N_COILS_IN_GROUP, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int
    )
    p.r_cs_middle = r_cs_middle
    p.ind_pf_cs_plasma_mutual = np.zeros((NGC2, NGC2))

    for name, value in (
        ("r_pf_coil_middle", r_pf_coil_middle),
        ("z_pf_coil_middle", z_pf_coil_middle),
        ("r_pf_coil_inner", r_pf_coil_inner),
        ("r_pf_coil_outer", r_pf_coil_outer),
        ("z_pf_coil_upper", z_pf_coil_upper),
        ("z_pf_coil_lower", z_pf_coil_lower),
        ("n_pf_coil_turns", n_pf_coil_turns),
    ):
        full = np.zeros(NGC2)
        full[:N_CS_PF_COILS] = np.asarray(value, dtype=float)
        setattr(p, name, full)

    model = PFCoil(cs_fatigue=None, cs_coil=None)
    model.data = data
    model.induct(False)
    return p.ind_pf_cs_plasma_mutual


class TestCalculateSolenoidSelfInductance(Tier1Contract):
    """`calculate_solenoid_self_inductance` -> `PFCoil.selfinductance` (Bunet's fit)."""

    audit_record = "models/pfcoil/inductance.md"
    reference = _reference_solenoid_self_inductance
    ported = calculate_solenoid_self_inductance

    # The CS as `induct` describes it: mean radius, full height, winding thickness,
    # turns (`pfcoil.py:1895-1908`). PROCESS returns 22.551156118389258 H here.
    samples = [
        legacy_sample(
            "large-tokamak-converged",
            a=_R_CS_MIDDLE,
            b=2.0 * _Z_UP[6],
            c=_R_OUT[6] - _R_IN[6],
            n=_TURNS[6],
        ),
    ]

    fuzz_bounds = {
        "a": (0.5, 6.0),
        "b": (2.0, 24.0),
        "c": (0.05, 2.0),
        "n": (100.0, 1.0e4),
    }


class TestCalculatePfCsPlasmaInductances(Tier1Contract):
    """`calculate_pf_cs_plasma_inductances` -> `PFCoil.induct(False)`.

    Produces `.pf_coil.ind_pf_cs_plasma_mutual`, which was a boundary input of
    `models/pfcoil/currents.py::CSFluxSwing` until this module existed.
    """

    audit_record = "models/pfcoil/inductance.md"
    reference = _reference_pf_cs_plasma_inductances
    ported = calculate_pf_cs_plasma_inductances
    static_argnames = ("z_pf_coil_upper", "r_pf_coil_inner", "r_pf_coil_outer")

    samples = [
        legacy_sample(
            "large-tokamak-converged",
            rmajor=_RMAJOR,
            ind_plasma=_IND_PLASMA,
            dr_cs=_DR_CS,
            r_cs_middle=_R_CS_MIDDLE,
            r_pf_coil_middle=_R_MID,
            z_pf_coil_middle=_Z_MID,
            r_pf_coil_inner=_R_IN,
            r_pf_coil_outer=_R_OUT,
            z_pf_coil_upper=_Z_UP,
            z_pf_coil_lower=_Z_LO,
            n_pf_coil_turns=_TURNS,
        ),
    ]

    fuzz_fixed = {
        # Held at the reference values, not merely undifferentiated: together they fix
        # `noh = 30`, which this occupant bakes in. A draw that changed `noh` would be a
        # draw for a different occupant, and the port and the reference would then be
        # modelling different machines.
        "z_pf_coil_upper": _Z_UP,
        "r_pf_coil_inner": _R_IN,
        "r_pf_coil_outer": _R_OUT,
    }
    fuzz_bounds = {
        "rmajor": _around(_RMAJOR, 0.10),
        "ind_plasma": _around(_IND_PLASMA, 0.20),
        # Above `delzoh = 2 * z_cs_half / noh = 0.5291 m`, so the Rosa-Grover split
        # keeps its `sqrt((dr_cs^2 - delzoh^2) / 12)` branch. The other branch is a
        # constant `1e-6` with no derivative to compare; see `inductance.md`.
        "dr_cs": (0.535, 0.62),
        "r_cs_middle": _around(_R_CS_MIDDLE, 0.10),
        "r_pf_coil_middle": _around(_R_MID, 0.08),
        "z_pf_coil_middle": _around(_Z_MID, 0.08),
        "z_pf_coil_lower": _around(_Z_LO, 0.05),
        "n_pf_coil_turns": _around(_TURNS, 0.15),
    }


# ---------------------------------------------------------------------------------
# The no-central-solenoid arm (`iohcl = 0`), 2026-08-31.
# ---------------------------------------------------------------------------------
#
# The eight-coil spherical-tokamak geometry, read off
# `test_masses.py::TestPFCoilChainSphericalTokamak`'s own legacy point by running its
# reference chain (`PFCoil.pfcoil()` at `iohcl = 0`) once. So this contract's inputs are
# PROCESS's own outputs for that machine rather than numbers invented here -- but the
# machine is still `_LEGACY_SPHERICAL_TOKAMAK`'s "plausible, not converged" point, and
# that caveat carries over verbatim: no ST file assembles through this port yet, so
# there is no converged run to read a coil set off.
_ST_R_MID = np.array([
    2.4625000000000004,
    2.4625000000000004,
    9.3,
    9.3,
    9.3,
    9.3,
    9.5,
    9.5,
])
_ST_Z_MID = np.array([4.86, -5.36, 3.0, -3.0, 6.25, -6.25, 13.0, -13.0])
_ST_Z_UP = np.array([
    5.492692096669603,
    -5.992692096669603,
    3.3853576857560688,
    -3.3853576857560688,
    6.5386583645056495,
    -6.5386583645056495,
    13.167712578033077,
    -13.167712578033077,
])
_ST_Z_LO = np.array([
    4.227307903330398,
    -4.727307903330398,
    2.6146423142439312,
    -2.6146423142439312,
    5.9613416354943505,
    -5.9613416354943505,
    12.832287421966923,
    -12.832287421966923,
])
_ST_TURNS = np.array([
    440.3292181069958,
    440.3292181069958,
    163.35060056840018,
    163.35060056840018,
    91.65601653898428,
    91.65601653898428,
    30.940259713551036,
    30.940259713551036,
])
_ST_RMAJOR = 4.5
_ST_IND_PLASMA = 8.0e-6
"""`.physics.ind_plasma` for a 4.5 m machine, chosen the way
`TestPFCoilChainSphericalTokamak`'s build quantities are and for the same reason. It is
read straight through into the matrix's `[plasma, plasma]` entry on both sides
(`pfcoil.py:1858-1862`), so no fit depends on it -- what the contract checks is that
both sides put it in the same slot."""

_ST_R_IN_LAST = 9.332287421966923
_ST_R_OUT_LAST = 9.667712578033077
"""`r_pf_coil_inner[7]`/`r_pf_coil_outer[7]` from the same chain, and **the port declares
neither**, which is a finding about `induct` rather than a gap in the port.

`induct` computes `noh` *before* it looks at `iohcl` (`pfcoil.py:1756-1780`), so on a
machine with no CS it still divides `2 * z_pf_coil_upper[n_cs_pf_coils - 1]` by
`r_pf_coil_outer[...] - r_pf_coil_inner[...]` -- which on arm 2 is not the solenoid but
the **last PF coil**, index 7. The quotient is then unused: `roh`/`zoh` are filled only
under `iohcl != 0` (`:1783-1791`) and all three blocks that read them are guarded, so
every inductance is independent of `noh` (`_audit/next_steps.md` §20.6). Two consequences
for this adapter, both stated rather than worked around:

- the two fields must be seeded to something with a non-zero difference or PROCESS
  raises `ZeroDivisionError` on a quantity it will not use;
- on this geometry `z_pf_coil_upper[7]` is *negative* (a below-midplane coil), so
  `math.ceil` of the quotient is negative and `max(noh, 0)` (`:1778`, PROCESS's own
  FNSF-case guard, with its own `TODO`) clamps it to **zero**. `roh`/`zoh` are then
  zero-length arrays. That is the guard doing real work on a machine PROCESS's comment
  did not have in mind.

Held fixed, not fuzzed, because no output depends on them -- fuzzing them would test
`ceil`."""


def _reference_pf_plasma_inductances_no_central_solenoid(
    rmajor,
    ind_plasma,
    r_pf_coil_middle,
    z_pf_coil_middle,
    z_pf_coil_upper,
    z_pf_coil_lower,
    n_pf_coil_turns,
):
    """`PFCoil.induct(False)` on a cold `DataStructure` with `iohcl = 0`.

    The sibling adapter's shape with the spherical tokamaks' topology
    (`n_pf_coils_in_group = (2, 2, 2, 2)`, `n_cs_pf_coils = 8`,
    `n_pf_cs_plasma_circuits = 9`) and four fewer seeded fields: `dr_cs` and
    `r_cs_middle` are never read on this arm, and `r_pf_coil_inner`/`r_pf_coil_outer`
    reach only `noh` -- see `_ST_R_IN_LAST`.

    **No `static_argnames`**, unlike `TestCalculatePfCsPlasmaInductances`. The three that
    contract declares static are the three `noh` is a step function of, and here `noh`
    reaches no output at all, so `z_pf_coil_upper` is an ordinary differentiated input
    whose only use is the PF/PF diagonal's `rl = |z_upper - z_lower| / sqrt(pi)`.
    """
    data = DataStructure()
    p = data.pf_coil

    data.build.iohcl = 0
    data.physics.rmajor = rmajor
    data.physics.ind_plasma = ind_plasma

    topology = SPHERICAL_TOKAMAK_TOPOLOGY
    n = topology.n_cs_pf_coils
    p.n_cs_pf_coils = n
    p.n_pf_cs_plasma_circuits = topology.plasma_index + 1
    p.n_pf_coil_groups = topology.n_pf_coil_groups
    p.n_pf_coils_in_group = np.array(
        [*topology.n_pf_coils_in_group, 0, 0, 0, 0, 0, 0], dtype=int
    )
    p.ind_pf_cs_plasma_mutual = np.zeros((NGC2, NGC2))

    for name, value in (
        ("r_pf_coil_middle", r_pf_coil_middle),
        ("z_pf_coil_middle", z_pf_coil_middle),
        ("z_pf_coil_upper", z_pf_coil_upper),
        ("z_pf_coil_lower", z_pf_coil_lower),
        ("n_pf_coil_turns", n_pf_coil_turns),
    ):
        full = np.zeros(NGC2)
        full[:n] = np.asarray(value, dtype=float)
        setattr(p, name, full)

    # Only `noh` reads these two, and only through `r_out[7] - r_in[7]`.
    p.r_pf_coil_inner = np.zeros(NGC2)
    p.r_pf_coil_outer = np.zeros(NGC2)
    p.r_pf_coil_inner[n - 1] = _ST_R_IN_LAST
    p.r_pf_coil_outer[n - 1] = _ST_R_OUT_LAST

    model = PFCoil(cs_fatigue=None, cs_coil=None)
    model.data = data
    model.induct(False)
    return p.ind_pf_cs_plasma_mutual


class TestCalculatePfPlasmaInductancesNoCentralSolenoid(Tier1Contract):
    """`calculate_pf_plasma_inductances_no_central_solenoid` -> `PFCoil.induct(False)`
    at `iohcl = 0`, on the spherical tokamaks' eight-coil topology.

    Owed since 2026-08-30 (`_audit/next_steps.md` §20.5 item 1): the function was
    verified bit-exact in a scratch script and nothing in the test tree held it.

    **Three of `induct`'s four blocks survive here and the fourth is absence, not a
    zero.** The CS/plasma block, the CS self-inductance and the CS/PF block are all
    guarded on `iohcl != 0`, so the matrix this returns has the plasma self-inductance,
    the PF/plasma column and the full 8x8 PF/PF block -- and `nef = n_cs_pf_coils`
    rather than `- 1` (`pfcoil.py:1943-1947`), so the last PF coil is *in* that block
    where on the reference arm index 7 is the plasma's neighbour. A port that read the
    absent CS fields as zeros would still fill row/column 8 of the matrix; both sides
    leave them at `0.0` because neither computes them.
    """

    audit_record = "models/pfcoil/inductance.md"
    reference = _reference_pf_plasma_inductances_no_central_solenoid
    ported = calculate_pf_plasma_inductances_no_central_solenoid

    samples = [
        legacy_sample(
            "spherical-tokamak-plausible",
            rmajor=_ST_RMAJOR,
            ind_plasma=_ST_IND_PLASMA,
            r_pf_coil_middle=_ST_R_MID,
            z_pf_coil_middle=_ST_Z_MID,
            z_pf_coil_upper=_ST_Z_UP,
            z_pf_coil_lower=_ST_Z_LO,
            n_pf_coil_turns=_ST_TURNS,
        ),
    ]

    fuzz_bounds = {
        "rmajor": _around(_ST_RMAJOR, 0.10),
        "ind_plasma": _around(_ST_IND_PLASMA, 0.20),
        "r_pf_coil_middle": _around(_ST_R_MID, 0.08),
        "z_pf_coil_middle": _around(_ST_Z_MID, 0.08),
        # Upper and lower edges are drawn independently: their difference is the
        # diagonal's equivalent circular radius, so a draw that moved them together
        # would leave `rl` untested.
        "z_pf_coil_upper": _around(_ST_Z_UP, 0.05),
        "z_pf_coil_lower": _around(_ST_Z_LO, 0.05),
        "n_pf_coil_turns": _around(_ST_TURNS, 0.15),
    }
