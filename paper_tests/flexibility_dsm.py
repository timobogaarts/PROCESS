"""The graph the flexibility analysis solves, and the recipe that builds it.

Every step is cottax ops on the previous graph. Counts are measured
(`--dsm` renders the current graph; the appendix is `scratchpad/bsplit/`).

**2026-09-22: steps 4-6 were replaced.** The belief table was pruned to the beliefs
that actually move the answer (PLASMA + LIMITS, `configurations.kinds.BELIEFS`), and
the helium fraction stopped being a free operator knob: `^cond.constraints.c62`
(`tau_He*/tau_E >= f_t_alpha_energy_confinement_min`, an *inequality*) is now named as
a closure alongside `c2`, so particle balance -- not the operator -- sets
`f_nd_alpha_thermal_electron`. Steps 0-3 are untouched and kept below for the record;
re-measuring them against the current cottax found `boundary_inputs` reporting 318 at
step 3 where the table originally read 305/312 -- cottax's own boundary accounting has
moved in the meantime, so steps 4'-6' below are measured against that current 318, not
against the historical 312, and the two are not directly subtractable.

| # | step | ops | nodes | inputs | problems |
|---|---|---|---|---|---|
| 0 | the declared machine | `indat.graph_for(machine)` | 154 | 305 | 4 |
| 1 | minus unbacked nodes | `Delete(.vacuum.duct_diameter_root_find)` | 152 | 300 | 3 |
| 2 | the MDA | `FixedPointCut` x2 + `Assign(drivers)` | 154 | 300 | 5 |
| 3 | + constraints, objective | `Insert(icc x14)` + `Insert(Optimise)` | 170 | 305 (318 today) | 6 |
| 4' | + `c2` **and** `c62` closed together | `Insert(RootFind)` x2 + `Combine` + `Nest` | 170 | 318 | 141 |
| 5' | + the winding pack lifted | `Undrive`+`Unnest`+`Undetermine`+`Delete`+`Insert` | 171 | 319 | 143 |
| 6' | + the operator's problem (1 unknown) | `Insert(Optimise)` + `nested_inside` | 172 | | |

Step 2 applies only **2** of `mda.CUTS`' nine: the other seven cycles do not exist in
this stellarator's graph. The problems it leaves are the four the models declare
themselves (`proton_rate_density.cycle`, `f_ster_div_single`,
`profiles.ion_vol_avg_temperature`, `power.delta_eta_step`) plus
`stellarator.coils.intersect` -- the winding-pack sizing rule, which step 5' takes apart.

**Step 4' is two `RootFind`s that land on one node.** `closing.close` inserts
`RootFind((c2,), (n_e,))` at `.Close.c2` and `RootFind((c62,), (f_He,))` at `.Close.c62`
exactly as before (one insert per pairing entry); naming c62 -- an inequality -- as a
closure is legitimate because `sand.constraint_nodes` gives every active constraint,
equality or inequality, the same one scalar (the normalised residual), so driving it to
zero is the same point where the original `<=` would sit tight (`closing.close`'s
docstring). Because both conditions read the density and the alpha fraction through the
same power-balance / He-production cycle, their two `RootFind` components overlap --
`close` detects the overlap and `Combine`s them into **one** square problem, so the
graph ends with a single `.Close.c2` node whose four unknowns are the density, two
already-cut copies (`^hat.physics.nd_plasma_fuel_ions_vol_avg`,
`^hat.physics.nd_plasma_ions_total_vol_avg`) and the thermal alpha fraction -- not two
nodes. Node count is unchanged at 170 (an inequality's own trivial node is absorbed
into the combined cycle, not deleted, so components drop from 143 to 141 rather than
the node count moving), and boundary inputs are unchanged too at 318: the fraction
stops being a plain design/boundary input and instead gains a `^guess.Close.c2` start
port, a wash. `report["inequalities"]` drops from 12 to 11 (c62 leaves it, the same way
a closed equality never entered it) -- the pruned belief table also drops the constant
`c62`'s own former belief-limit role into `.constraints.f_t_alpha_energy_confinement_min`,
now a sampled LIMITS row rather than a fixed threshold.

Step 5' *removes* a problem (the coil stops solving for its own width) and adds one node
(the inserted signed residual, `^cond.lift.stellarator.wp_width_r_min`), same as before;
`design` grows from 6 to 7 places (the lifted pack width joins the kept `ixc`).

Step 6' is this file. The `Optimise` now owns exactly **one** unknown -- `T_e` -- and
carries only the **five** conditions the operator can still move (c24, c8, c17, c18,
c67). The other **seven** (c82, c83, c32, c34, c35, c65 and the lifted pack rule) are
constants once the build is fixed, and cottax refuses them outright: *"reads ... as a
condition, but nothing in its cycle produces it, so no unknown of the problem can move
it -- a constant is not a condition"*. That refusal is how the five/seven split was
found rather than assumed -- the same test that found six/seven under the old two-knob
problem, now against one knob and eleven (not twelve) remaining inequalities, since
c62 left the set at step 4'. In an ignited plasma T_e parametrises the ignition curve
(the operator physically sets fuelling; closing c2 by n_e at the chosen T_e traces the
same curve), so this one knob is a label on that curve, not a claim of a second
physical degree of freedom -- the helium fraction never was one, once particle balance
is stated as the closure it always was.

## Appendix: the field split, measured and rejected

`b_plasma_toroidal_on_axis` reaches 122 of 154 nodes -- the plasma *and* the whole magnet
chain -- so lowering it re-sizes the magnet. The magnet-sizing cone is 9 nodes and the
field enters it through exactly one reader, so separating the built field from the
operated one is two ops:

    Insert(.Split.field)   an identity owning B, read from a new input  (B is a boundary
                           input, and `Cut` only splits *computed* places)
    Cut(B, readers=(stellarator_scaling_factors,))   mints the copy the magnet reads

Deterministically inert: `objf 1.3351987988619283` either way, and the lifted pack width
seeds at 0.7169 either way. `stages.leaves` then refuses the graph until each new place
is given a kind, which is the modelling claim made explicit. Plumbing it needs beyond the
two ops: re-point ixc 2, re-key `reference.bounds`, pin the built field into the
configuration.

**Result: worth nothing here.** With `B_operating <= B_design` enforced the operator picks
the maximum field in every world and operability is unchanged (0.4102 -> 0.4102). Without
that cap it appears to gain 47 %, which is a 4.70 T magnet being run at 6.08 T.
"""
import jax
jax.config.update("jax_enable_x64", True)

from common import OUT, deterministic_values
from cottax.pytree.executable import ExecutableGraph
from cottax.pytree.names import PathMap
from cottax.pytree.plan import Insert, Plan
from cottax.pytree.problem import Optimise
from functional_process.cottax.architectures import closing, lift, ouu, session
from functional_process.cottax.architectures.mda import assign_drivers, default_drivers
from functional_process.cottax.queries import nested_inside
from functional_process.configurations import kinds
from functional_process.cottax.visualization.grouping import (
    Drawn,
    dependency_group_sequence,
    graph_of,
    provenance_order,
    render_grouped_dsm_html,
    structure_order,
)
from functional_process.cottax.visualization.render_xdsm import SPELLING


def draw(drawn: Drawn, outdir, name: str, title: str) -> list[str]:
    """Both orderings of one answerable graph as interactive DSM pages; the two file
    names. (Was `paper_tests/dsms.py`'s, the one piece of it the UQ still needs.)
    """
    graph = graph_of(drawn)
    axis = dependency_group_sequence(graph, depth=None)
    common = {"depth": None, "outdir": str(outdir), "write": True, "formatter": SPELLING}
    render_grouped_dsm_html(
        drawn,
        order=provenance_order(graph.nodes, depth=None, owners=graph.graph.owners, groups=axis),
        title=f"{title} -- ordered by provenance",
        file_name=f"{name}_provenance",
        mode="provenance",
        **common,
    )
    render_grouped_dsm_html(
        drawn,
        order=structure_order(drawn),
        title=f"{title} -- ordered by structure (run order, solves nested)",
        file_name=f"{name}_scc",
        mode="structure",
        **common,
    )
    return [f"{name}_provenance.html", f"{name}_scc.html"]

OPERATING = (kinds.TE,)
"""The operator's only remaining knob (2026-09-22): T_e. The helium fraction is now
closed by c62 inside the MDA (`kinds.PAIRINGS["he"]`), alongside c2 by the density --
see the module docstring."""

def render():
    live = session.open_session("stellarator_helias")
    dv, cv = deterministic_values(live, "he")
    model = ouu.two_stage(live, n=4, alpha=0.9, seed=0, lifts=(lift.lift_winding_pack,),
                          pairing="he", closure="flattened", beliefs=kinds.BELIEFS,
                          held=(), design_values=dv, closing_values=cv)
    built = model.closed
    operating = tuple(v for v in built.design if v.spelling in OPERATING)
    fixed = tuple(v for v in built.design if v.spelling not in OPERATING)
    assert len(operating) == 1, [v.spelling for v in built.design]
    print("optimised:", [v.spelling for v in operating])
    # `built.design` is the closed problem's, so it still carries `hfact` (ixc 10),
    # which is a *belief* sampled per draw rather than a build number. The five real
    # build places are the rest. The helium fraction is no longer here at all: `he`
    # closes it by c62 inside the MDA, so it is one of the combined root find's
    # unknowns now, not a design place (`built.unknowns(...)` has it, not this list).
    BELIEF = ".physics.hfact"
    print("held fixed (build):", [v.spelling for v in fixed if v.spelling != BELIEF], flush=True)
    print("sampled per draw   :", [v.spelling for v in fixed if v.spelling == BELIEF], flush=True)

    # Only the conditions the operator can actually move belong to the operator's
    # problem. With the build fixed the other seven are constants, and cottax refuses a
    # constant as a condition -- which is how this split was found rather than assumed.
    g = built.problem.graph
    owner = {v.spelling: n for n, d in g.definitions.items() for v in getattr(d, "owns", ())}
    readers = [n for n, d in g.definitions.items()
               if any(v.spelling in OPERATING for v in getattr(d, "reads", ()))]
    reach = {n.spelling for n in g.graph.descendants(tuple(readers))} | {n.spelling for n in readers}
    movable = tuple(c for c in built.report["inequalities"]
                    if owner.get(c.spelling) is not None and owner[c.spelling].spelling in reach)
    fixed_conds = [c.spelling for c in built.report["inequalities"] if c not in movable]
    print(f"movable by the operator ({len(movable)}):", [c.spelling.rsplit('.',1)[-1] for c in movable])
    print(f"constant given the build ({len(fixed_conds)}):", [c.rsplit('.',1)[-1] for c in fixed_conds], flush=True)

    node = Optimise(
        objective=built.report["objective"],
        unknowns=operating,
        equalities=(),
        inequalities=movable,
    )
    graph = (Plan(built.problem.graph) + Insert(PathMap(((closing.OPTIMISE, node),)))).graph
    graph = nested_inside(graph, closing.OPTIMISE)
    drivers = default_drivers(graph)
    # `he` folds c2's and c62's root finds into one combined, 4-unknown square problem
    # (`flatten=True`, since their cycles overlap) -- `bracketed()` only answers one
    # unknown, so the combined place gets the Newton driver `close` itself defaults to.
    for place in set(built.places.values()):
        drivers[place] = closing.safeguarded()
    drawn = ExecutableGraph(assign_drivers(graph, drivers))

    out = OUT / "dsm" / "stellarator_helias"
    out.mkdir(parents=True, exist_ok=True)
    files = draw(drawn, out, "flexibility",
                      "the operator's problem: T_e optimised over a fixed build, the "
                      "density and the helium fraction closing the power balance and "
                      "the He-exhaust limit (c2, c62) together inside the MDA")
    print("wrote", files, f"({len(graph.nodes)} nodes)")


if __name__ == "__main__":
    render()
