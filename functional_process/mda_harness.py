"""Block-by-block comparison: does `functional_process.mda`'s driven graph reproduce
a real, converged PROCESS run's own values?

Two halves, deliberately separable (`converged_data` is specific to one input file;
`compare` is not):

- `converged_data(input_file)` runs PROCESS's own `SingleRun` in-process and returns
  the live, converged `DataStructure` -- not a round trip through `MFILE.DAT` (loses
  precision) or the CLI (re-parses text).
- `compare(graph, data)` seeds `functional_process.mda`'s `Schedule` for `graph` from
  `data`'s own values (every boundary input, and every driven block's starting guess
  -- same converged run, per this session's established principle: we are checking
  whether the graph reproduces an answer PROCESS already found, not solving cold),
  runs it, and diffs every value the schedule produced against `data`'s own value at
  the same field.

**`DuctDiameterRootFind` is excluded.** Confirmed directly in its own docstring
(`vacuum.py:334-344`): every one of its `VarPath`s (`l1`, `l2`, `l3`, `xmult_i`,
`ceff_i`, `d_duct`) is minted, not a real PROCESS `data` field -- in real PROCESS
these are locals inside `_solve_vacuum_pumping_old`'s per-species Newton loop, never
written to `DataStructure`. There is nothing in a converged run to seed or compare
this node against; it is a deliberate island (`VacuumOld`, not this node, is the
PROCESS-faithful, registered vacuum path). `compare` drops it (and its own
`^problem[...]` partner) before running.
"""

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from cottax.blocking import Blocking
from cottax.evaluate import schedule_for
from cottax.plan import Delete
from cottax.spec import NodePath, VarPath

from functional_process.mda import default_drivers, driven_graph

EXCLUDED_NODE_NAMES = (
    "DuctDiameterRootFind",
    "Intersect",
    "WindingPackIntersectInputs",
    "WindingPackTotalSizePost",
)
"""`DuctDiameterRootFind`: see this module's own docstring -- no real `DataStructure`
field backs any of its `VarPath`s.

`Intersect`/`WindingPackIntersectInputs`/`WindingPackTotalSizePost` (one 4-node SCC,
`coils/calculate.py`): same root cause, found empirically rather than pre-flagged.
`.stellarator.wp_width_r_min` (`Intersect`'s own resolved crossing point) is minted,
not a real PROCESS field either -- a `0.0` placeholder for it is a bad enough starting
guess that `NewtonDriver`'s `optx.root_find` fails to converge within `max_steps` and
raises, which aborts the *entire* `schedule()` call, not just this one block's
comparison (`Schedule` runs one JIT-traced program, not node-by-node). Excluded for
the same reason `DuctDiameterRootFind` is: nothing here to compare against, and no
safe placeholder shape/value to invent for a `RootFind`'s own unknown the way
`KNOWN_MINT_VALUES` could for `profile_x`'s array.
"""

KNOWN_MINT_VALUES = {}
"""`{ungrounded VarPath: data -> its exact analytic value}`, for cases where `0.0` is
not just ungrounded but actively wrong-*shaped*.

**Was `{".physics.profile_x": ...}`, now empty -- fixed at the source, not worked
around here.** `.physics.profile_x` (minted, read by `FusionRates`) turned out to be a
duplicate of `ProfileGrid`'s own `.physics.radius_plasma_profile_norm` (same
`linspace(0, 1, n_plasma_profile_elements)` formula, same real source) that this
harness's own comparison run surfaced -- `fusion_reactions.py`/`.md` were corrected to
read `radius_plasma_profile_norm` directly, so `.physics.profile_x` is no longer read
by anything (confirmed: `grep -rn "physics\\.profile_x" functional_process/models/`
finds no remaining `Input`/`Output` binding, only historical docstring mentions). Kept
as an empty dict, not deleted outright, since a future ungrounded-and-wrong-shaped
case is exactly what this mechanism is for.
"""

KNOWN_UNVERIFIABLE_OUTPUTS = frozenset({
    ".fwbs.f_a_fw_coolant_inboard",
    ".fwbs.f_a_fw_coolant_outboard",
})
"""`VarPath`s where a real `DataStructure` field exists (so `_ground_truth` succeeds,
unlike `errors`' "no field at all" case) but PROCESS itself never actually writes a
meaningful value there for the arm this port's node represents -- comparing against
whatever the field's uninitialised default happens to be is not a real check, it is a
guaranteed false positive.

Both entries are `DetailedPowerflowBlanketShieldPower`'s own two "best-effort"
outputs -- its class docstring (`stellarator_fwbs_s2.py:378-382`) already says these
stay Python-locals in the real PROCESS source, never written to `data`, "matching
their PROCESS field names" only for naming convenience. Found via this harness's own
comparison run flagging `f_a_fw_coolant_inboard` as a "disagreement" (`got=0.444`,
`expected=0.0`) that traced back to exactly this documented, pre-existing caveat --
`expected=0.0` was `DataStructure()`'s bare default, not a PROCESS answer. The node's
other 14 outputs are real fields and stay in ordinary comparison scope; only these two
are excluded.
"""


def converged_data(input_file: str):
    """Run PROCESS's own `SingleRun` on `input_file` to convergence, in-process, and
    return the resulting live `DataStructure`. Writes the usual `OUT.DAT`/`MFILE.DAT`
    beside `input_file` as a side effect (same as any real PROCESS run) -- not
    suppressed, since nothing here depends on it not happening.
    """
    from process.main import SingleRun

    run = SingleRun(input_file, "vmcon")
    run.run()
    return run.data


def _without_excluded(graph):
    to_delete = tuple(
        n
        for n in graph.nodes
        if any(name in n.path_str() for name in EXCLUDED_NODE_NAMES)
    )
    if not to_delete:
        return graph
    return Delete(to_delete).apply(graph)


@dataclass
class Disagreement:
    """One `VarPath` where the schedule's answer and `data`'s own converged value
    don't agree within tolerance."""

    var: VarPath
    owner: NodePath
    got: float
    expected: float

    @property
    def rel_diff(self) -> float:
        """`|got - expected| / |expected|` (or `/1` at `expected == 0`)."""
        denom = abs(self.expected) if self.expected != 0 else 1.0
        return abs(self.got - self.expected) / denom


@dataclass
class ComparisonReport:
    """The result of one `compare()` call -- see each field's own docstring."""

    agreements: int = 0
    disagreements: list = field(default_factory=list)
    driven_block_disagreements: list = field(default_factory=list)
    acyclic_disagreements: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    ungrounded_inputs: list = field(default_factory=list)
    """Boundary inputs / driven unknowns with no real `DataStructure` field to seed
    from -- the same shape as `DuctDiameterRootFind`'s exclusion (a minted local
    PROCESS never stores), just discovered empirically rather than pre-flagged.
    Seeded `0.0` (a scalar placeholder -- an array-shaped ungrounded input will still
    break shape-dependent downstream arithmetic, see this module's own known-issues
    note) so the schedule can still run; anything downstream is reported separately,
    in `unverifiable`, not scored as a pass or a fail.
    """
    unverifiable: list = field(default_factory=list)
    """Outputs that read an `ungrounded_inputs` entry, directly or transitively --
    not a disagreement, since there is nothing real to disagree with.
    """

    def summary(self) -> str:
        lines = [
            f"agreements: {self.agreements}",
            f"disagreements: {len(self.disagreements)}",
            f"  in driven (cyclic) blocks: {len(self.driven_block_disagreements)}",
            f"  in ordinary acyclic nodes: {len(self.acyclic_disagreements)}",
            f"unverifiable (depends on an ungrounded input): {len(self.unverifiable)}",
            f"ungrounded inputs (no real DataStructure field): "
            f"{len(self.ungrounded_inputs)}",
            f"errors (could not evaluate at all): {len(self.errors)}",
        ]
        if self.ungrounded_inputs:
            lines.append("\nungrounded inputs:")
            for v in self.ungrounded_inputs:
                lines.append(f"  {v.path_str()}")
        by_owner: dict = {}
        for d in self.disagreements:
            by_owner.setdefault(d.owner.path_str(), []).append(d)
        worst = sorted(by_owner.items(), key=lambda kv: -max(d.rel_diff for d in kv[1]))
        lines.append("\nworst offenders by node:")
        for owner, ds in worst[:20]:
            worst_d = max(ds, key=lambda d: d.rel_diff)
            lines.append(
                f"  {owner}: {len(ds)} var(s) off, worst "
                f"{worst_d.var.path_str()} got={worst_d.got!r} "
                f"expected={worst_d.expected!r} rel_diff={worst_d.rel_diff:.3e}"
            )
        if self.errors:
            lines.append("\nerrors:")
            for e in self.errors[:20]:
                lines.append(f"  {e}")
        return "\n".join(lines)


def _ground_truth(data, var: VarPath):
    """`data`'s own value at `var`, resolved in order:

    1. `KNOWN_MINT_VALUES` -- an exact analytic value for a mint with no real field.
    2. `cottax.tools.minting.unminted` -- if `var` is a `FixedPointCut`'s own minted
       unknown (e.g. `^hat.physics.proton_rate_density`), the real place it names. A
       minted copy has no `DataStructure` field of its own; at the fixed point it
       equals the real variable it was cut from, which *does* --
       `FixedPointFunction`'s structural self-loops are the opposite way round (the
       problem owns the real var, reads the minted copy), so this only matters for
       the two `FixedPointCut`s this module adds, not the 8 pre-existing structural
       ones.
    3. `var` itself.
    """
    from cottax.tools.minting import unminted
    from cottax.tools.pytree import get_at

    known = KNOWN_MINT_VALUES.get(var.path_str())
    if known is not None:
        return known(data)
    return get_at(data, unminted(var).keys)


def compare(graph, data, rtol=1e-6, atol=1e-9) -> ComparisonReport:
    """Drive `graph` (any `total_process`-shaped `Graph`) from `data`'s own converged
    values, and diff every value the schedule produces against `data` itself.
    """
    graph = _without_excluded(graph)
    driven = driven_graph(graph)
    blocking = Blocking.scc(driven)
    schedule = schedule_for(blocking, default_drivers(blocking))

    report = ComparisonReport()

    env = {}
    ungrounded = []
    for var in driven.unowned_inputs:
        try:
            env[var] = jnp.asarray(_ground_truth(data, var))
        except (AttributeError, KeyError):
            ungrounded.append(var)
            env[var] = jnp.asarray(0.0)  # placeholder -- see `ungrounded_inputs`
    # Starting guesses for every driven unknown -- same converged run's own value.
    for problem, problem_type in zip(
        blocking.problems, blocking.problem_types, strict=True
    ):
        if problem_type is None:
            continue
        for var in driven[problem].owns:
            try:
                env[var] = jnp.asarray(_ground_truth(data, var))
            except (AttributeError, KeyError):
                ungrounded.append(var)
                env[var] = jnp.asarray(0.0)

    report.ungrounded_inputs = ungrounded
    # Everything that reads an ungrounded input, directly or transitively, can only
    # ever disagree with `data` because its own input was a placeholder, not because
    # the port is wrong -- excluded from pass/fail, reported separately.
    unverifiable_owners = set()
    for var in ungrounded:
        for reader in driven.readers.get(var, ()):
            unverifiable_owners.add(reader)
            unverifiable_owners |= set(driven.descendants([reader]))

    try:
        out = schedule(env)
    except Exception as e:  # noqa: BLE001 -- report, don't crash the harness
        report.errors.append(f"schedule() raised: {type(e).__name__}: {e}")
        return report

    for var, owner in driven.owners.items():
        if owner in unverifiable_owners or var.path_str() in KNOWN_UNVERIFIABLE_OUTPUTS:
            report.unverifiable.append(var)
            continue
        try:
            expected = _ground_truth(data, var)
        except (AttributeError, KeyError) as e:
            report.errors.append(f"no DataStructure field for {var.path_str()}: {e}")
            continue
        if var not in out:
            report.errors.append(f"schedule did not produce {var.path_str()}")
            continue
        got = out[var]
        try:
            got_f = float(np.asarray(got))
            expected_f = float(np.asarray(expected))
        except (TypeError, ValueError):
            continue  # non-scalar or non-numeric field, skip
        if np.isclose(got_f, expected_f, rtol=rtol, atol=atol, equal_nan=True):
            report.agreements += 1
        else:
            d = Disagreement(var=var, owner=owner, got=got_f, expected=expected_f)
            report.disagreements.append(d)
            block_index = blocking.index[owner]
            if blocking.problem_types[block_index] is not None:
                report.driven_block_disagreements.append(d)
            else:
                report.acyclic_disagreements.append(d)

    return report
