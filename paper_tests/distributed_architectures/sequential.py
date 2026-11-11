"""Two optimisers in sequence on one graph: the magnet, then the plasma.

PROCESS states one problem over all eight of Helias's iteration variables. Here it is
split into two, solved one after the other on the same models:

- **OptMagnet** sizes the coil: the variables the TF constraints reach (`b_t`,
  `rmajor`, `tdmptf`, `fcutfsu`), those constraints, and its own objective,
  figure of merit 1 (`0.2 rmajor`), with `b_t >= 5 T`.
- **OptPlasma** finds the operating point: the remaining variables (`te`, `dene`,
  `hfact`, `f_alpha`), the remaining constraints, and the file's merit, COE.

No variable belongs to both, and nothing the plasma computes is read by the magnet, so
the run order is derived: magnet, then plasma, which reads the magnet's answer as
given. Not equivalent to the file's problem: the magnet stage optimises its own
objective, not COE.

The recipe starts from the AAO base (`partition.base`) and follows the three steps of
jaxgraph's `examples/sellar_plans.py`, one op at a time; inside each stage it does what
AAO's SAND does to the whole problem (the same cuts and closures, the coil solve
absorbed). A split whose two problems lie on one cycle is refused (`refused_split`).

Re-implemented from `functional_process/architecture_examples/two_opt_driver` (removed
in a62a1fac, last run 2026-09-16 at cottax a3e4c56). One change: the old `b_t >= 5 T`
constraint node is the magnet stage's lower bound on `b_t`, the same feasible set.

    python paper_tests/distributed_architectures/sequential.py [--bt-min 5.0] [--no-aao]
"""

from __future__ import annotations

import argparse

import cottax.assemblies.jax as cx
from cottax.core import Function, Mint, Plan
from cottax.core.maths.plan import Combine, Delete, Determine, Undetermine
from cottax.core.realisation import ImplementedFunction

from functional_process.cottax.architectures.drivers import SlsqpDriver
from functional_process.cottax.models.constraints import NUMERICS, CostOfElectricity, area_of
from functional_process.models import objectives as O

import partition as C

MAGNET = C.node("OptMagnet")
PLASMA = C.node("OptPlasma")
MAGNET_MERIT_NODE = C.node("merit_magnet")

MAGNET_MERIT = NUMERICS.place("merit_magnet")
"""Where the magnet stage's objective is computed, beside the file's merit."""

RMAJOR = area_of("rmajor").place("rmajor")


class StageLower(Mint):
    """A stage's own lower bound on a variable, where it is not the file's."""


BT_MIN = 5.0
"""The magnet stage's lower bound on `b_t`. Without it the magnet drives `b_t` to the
file's bound of 4 T and leaves the plasma stage no feasible point (the 2026-09-16
report's first finding)."""


def magnet_merit():
    """Figure of merit 1, `0.2 rmajor`, at `MAGNET_MERIT`. WORKAROUND(FR-10): a plain
    body for the inserted function."""
    return ImplementedFunction(Function((RMAJOR,), (MAGNET_MERIT,)),
                               lambda rmajor: O.objective_metric_1(rmajor))


def _with_lower(relations, x, lower):
    """`relations` with `x`'s lower bound read from `lower` instead."""
    out = []
    for r in relations:
        if r.op is cx.Le and r.rhs == x and r.lhs is not None and r.lhs.spelling.startswith("^lower."):
            out.append(cx.Relation(lower, x, cx.Le))
        else:
            out.append(r)
    return tuple(out)


def recipe(b: C.Base, part: C.Partition, bt_min=BT_MIN) -> tuple:
    """The sequential split as a recipe on the AAO base, in the Sellar plans' three steps.

    1. **Remove.** AAO's cuts of the cycles inside the plasma; `Opt` and the coil solve
       undetermined, and `Opt` deleted (its relations are split between the stages).
    2. **Assemble.** The magnet stage (`Opt`'s bounds of its design, `b_t`'s lower one
       at `bt_min`, and the coil constraints) and its objective function, the plasma
       stage (`Opt`'s bounds of its design, the other constraints, `Opt`'s objective),
       AAO's closures; the closures combined into the plasma, the coil solve into the
       magnet.
    3. **Add.** Each stage determined: its design and what was combined into it. Both
       at the root: the plasma reads the magnet's answer, so it runs after.
    """
    stages = {"magnet": part.magnet_nodes, "plasma": part.plasma_nodes}
    label = {"magnet": MAGNET, "plasma": PLASMA}
    design = {"magnet": part.magnet_design, "plasma": part.plasma_design}
    ids = {"magnet": tuple(C.cid(c) for c in part.magnet_constraints),
           "plasma": tuple(C.cid(c) for c in (*part.plasma_equalities, *part.plasma_inequalities))}
    owner = {c.at: b.owner(stages, c.at) for c in b.closures}
    owner |= {s: b.owner(stages, s) for s in b.solves}

    # 1. Remove
    ops = [cut for c in b.closures for cut in c.cuts]
    ops.append(Undetermine(C.TOP))
    ops += [Undetermine(s) for s in b.solves]
    ops.append(Delete((C.TOP,)))

    # 2. Assemble
    magnet = b.share(design["magnet"], ids["magnet"])
    extra = ()
    if bt_min is not None:
        b_t = design["magnet"][0]
        magnet = _with_lower(magnet, b_t, StageLower()(b_t))
        extra = (StageLower()(b_t),)
    ops.append(C.insert(MAGNET, C.condition(MAGNET_MERIT, magnet,
                                            (*C.held_parameters(b, magnet), *extra))))
    ops.append(C.insert(MAGNET_MERIT_NODE, magnet_merit()))
    plasma = b.share(design["plasma"], ids["plasma"])
    ops.append(C.insert(PLASMA, C.condition(b.objective.place, plasma,
                                            C.held_parameters(b, plasma))))
    ops += [C.insert(c.at, C.requirement_of(c)) for c in b.closures]
    for k in ("magnet", "plasma"):
        if mine := tuple(n for n, o in owner.items() if o == k):
            ops.append(Combine(label[k], (label[k], *mine), cx.RelationalCondition))

    # 3. Add
    for k in ("magnet", "plasma"):
        absorbed = (*(u for s in b.solves if owner[s] == k for u in b.graph[s].unknowns),
                    *(m for c in b.closures if owner[c.at] == k for m in c.copies))
        ops.append(Determine(label[k], (*design[k], *absorbed)))
    return tuple(ops)


def plan(configuration, part, bt_min=BT_MIN) -> tuple:
    """`(base model, plan, given)`: the AAO base, the recipe on it, and the values the
    recipe adds (the magnet stage's `b_t` bound)."""
    b = C.base(configuration)
    assert b.objective.place == C.MERIT and configuration.problem.minimise is CostOfElectricity
    p = Plan(b.graph)
    for op in recipe(b, part, bt_min):
        p = p + op
    C.refuse_shared_cycles(p.graph, (MAGNET, PLASMA))
    b_t = part.magnet_design[0]
    given = {} if bt_min is None else {StageLower()(b_t): float(bt_min)}
    return b.model, p, given


def driven(configuration, part, env, traces, bt_min=BT_MIN, optimiser=SlsqpDriver,
           max_iter=C.study.MAX_ITER) -> tuple:
    """`(model, graph, given)`: the recipe applied, each stage answered by `optimiser`."""
    m, p, given = plan(configuration, part, bt_min)
    graph = C.drive(p, (MAGNET, PLASMA), env, traces, max_iter=max_iter, optimiser=optimiser).graph
    return m, graph, given


def run(configuration, bt_min=BT_MIN, env=None, state=None, optimiser=SlsqpDriver,
        max_iter=C.study.MAX_ITER) -> dict:
    part = C.partition(configuration)
    if env is None:
        state, env = C.mda_env(configuration)
    traces: dict = {}
    m, graph, given = driven(configuration, part, env, traces, bt_min, optimiser, max_iter)
    inputs = C.values(graph, state, {**dict(m.given), **given}, env, set(part.design))
    out, seconds = C.timed(graph, inputs)
    rows = {str(t): C.report(graph, out, t, traces[t]) for t in (MAGNET, PLASMA)}
    return {"graph": graph, "out": out, "seconds": seconds, "rows": rows, "part": part,
            "design": {C.short(d): C.number(out, d) for d in part.design},
            "coe": C.number(out, C.MERIT), "magnet_merit": C.number(out, MAGNET_MERIT)}


def refused_split(configuration):
    """The obvious alternative, machine `{b_t, rmajor, te, dene, hfact, f_alpha}` against
    TF conductor `{tdmptf, fcutfsu}`, as the same recipe: COE reads the conductor and the
    conductor constraints read `b_t` and `rmajor`, so both problems land on one cycle.
    Returns the refusal."""
    part = C.partition(configuration)
    tf = part.magnet_local
    machine = tuple(d for d in part.design if d not in tf)
    split = C.Partition(**{**part.__dict__, "magnet_design": tf, "plasma_design": machine})
    try:
        plan(configuration, split, bt_min=None)
    except (ValueError, TypeError) as refusal:
        return refusal
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bt-min", type=float, default=BT_MIN)
    ap.add_argument("--no-aao", action="store_true")
    args = ap.parse_args(argv)
    configuration = C.load(C.CONFIGURATION)
    print(C.reach_table(configuration), "\n")
    part = C.partition(configuration)
    print("magnet design:", [C.short(d) for d in part.magnet_design],
          " shared:", [C.short(d) for d in part.shared])
    print("plasma design:", [C.short(d) for d in part.plasma_design])
    print("magnet constraints:", [C.cid(c) for c in part.magnet_constraints])
    print("plasma constraints:", [C.cid(c) for c in (*part.plasma_equalities, *part.plasma_inequalities)])
    print(f"{len(part.magnet_nodes)} magnet models, {len(part.plasma_nodes)} plasma models, "
          f"{len(part.crossing)} places cross magnet -> plasma, {len(part.back)} back\n")

    _, p, _ = plan(configuration, part, args.bt_min)
    print("the recipe, on the AAO base:")
    for op in p.ops:
        text = repr(op)
        print("  ", text if len(text) < 140 else text[:137] + "...")
    print()

    result = run(configuration, args.bt_min)
    print(f"solved in {result['seconds']:.1f} s (cold: includes compilation)")
    for name, row in result["rows"].items():
        print(f"  {name:14s} {row['status']:10s} status {row['vmcon_status']} "
              f"iterations {row['iterations']:3d}  max|eq| {row['max_eq']:.1e}  min ie {row['min_ie']:+.1e}")
    print("  design:", {k: round(v, 4) for k, v in result["design"].items()})
    print(f"  COE {result['coe']:.6f}   magnet merit (0.2 rmajor) {result['magnet_merit']:.4f}")

    refusal = refused_split(configuration)
    print("\nthe split {b_t, rmajor, te, dene, hfact, f_alpha} | {tdmptf, fcutfsu}:",
          "scheduled -- unexpected" if refusal is None else f"refused:\n  {str(refusal)[:600]}")

    if not args.no_aao:
        ref = C.aao(configuration, optimiser=SlsqpDriver)
        print(f"\nAAO (SAND, SLSQP): {ref['status']}, {ref['iterations']} iterations, COE {ref['objf']:.6f}")
        print(f"sequential: COE {result['coe']:.6f} ({100 * (result['coe'] / ref['objf'] - 1):+.2f} %)")
    return result


if __name__ == "__main__":
    main()
