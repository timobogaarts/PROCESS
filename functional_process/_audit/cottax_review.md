# cottax design review, from the PROCESS port

A usability/design review of `cottax` (`~/jaxgraph`) as exercised by its largest real
client. Not a bug hunt — nothing in either repo was modified as part of writing it.

## The three verdicts

**Ease of use: good at the seams, bad at the surface.** The parts touched once per
architecture (`Graph`, `Blocking`, `Plan`/`rewrites`, `Schedule`, `AbstractDriver`) are
excellent. The part touched once per node (the declaration surface,
`interfaces/pytree_namespace_module.py`) was the weakest thing in the package: at review
time, 30% of the port's non-test model code sat inside node-declaration bodies, 87% of
those were a single forwarding call to an already-pure function, and 97.8% of `Input`
declarations just restated a name the parameter already carried. **Since fixed** — this is
exactly what `From`/`OutputInto` sugar (recommendation 1 below) addresses; see
`path_refactor.md`.

**Restrictiveness: right almost everywhere.** Exactly one refusal cost the client real
modelling fidelity — the no-projection rule biting array-element ownership. Every other
traced refusal either caught a genuine error or forced a decision that should be forced
(`Blocking`'s nesting invariant, `Graph.problem_type`'s two-problem refusal, `Drive`'s
`issubclass` check, the `reads ∩ owns` refusal).

**Looseness: narrower than "missing producer" suggests.** cottax correctly can't and
shouldn't try to catch "a variable no node owns" structurally — it has no way to know a
boundary input is unintended. But it was too loose in four specific, checkable places
where it had the information and declined to use it: single-output result binding,
`GraphOp.apply` being public and skipping its own check, the `ConditionMap` seam dropping
problem structure, and no hook to validate declared `VarPath`s against the caller's data
structure at assembly time (despite the check already existing, unused, in
`tools/pytree.py`).

## Recommendations, as given (status not re-verified since)

**Do, high value/low cost:** default-area sugar on `Input`/`Output` (landed — see
`From`/`OutputInto`); give `ConditionMap` the problem definition as a field instead of a
positional contract; make `GraphOp.apply` check its own precondition, or rename it
`_apply`; promote the private `_params` hook to documented (instance-derived ports,
confirmed working); reconcile `~/jaxgraph/CLAUDE.md` with the actual tree.

**Do next, higher cost:** let `Drive.body` be a `Step` and `Schedule.steps` consume
`Blocking.inner` (this is what makes in-graph MDF-shaped root-finding possible — see
`in_graph_rootfind.md`, since built); an assembly-time `check_against(cds)` on `to_graph`
using the existing-but-unused pytree check; opt-in arity strictness at the declaration
surface (seven recorded 1-tuple bugs at review time); implement `Blocking.merge` or
delete it from the docs.

**Decide, don't build:** whether `Optimise` bounds are structure or algorithm; whether
`flat_namespace_module` is frozen (silent divergence between the two declaration surfaces
was the worst of the three options).

**Do not do:** build array-element projection (the one restriction that genuinely costs
the client, but also the largest conceivable change to `graph.py`, and cottax is
pre-Phase-4 — revisit only when a second client hits it); add a `Switch`/`Alternative` to
cottax itself (the client's local layer is the right place); try to catch "missing
producer" structurally (cottax cannot know; that's the client's own audit's job); weaken
`Graph.problem_type`/`Graph.driven`'s refusals (both produced messages that told the
client what its model actually was).
