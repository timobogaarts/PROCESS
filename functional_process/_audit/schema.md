# Audit record schema (draft)

Three record shapes: model-unit, constraint, switch. Every record is one markdown file
at the mirrored path (model-units, `core/solver/constraints.md`, `core/solver/switches.md`
— the latter two are single files holding one section per entry, since constraints/
switches don't have their own source file to mirror).

Every record starts with frontmatter:

```yaml
---
kind: model-unit | constraint | switch
status: pending | draft | reviewed | final
confidence: high | medium | low   # omit for status: pending
---
```

`status: pending` records may be created ahead of time (by the unit registry) with just
the frontmatter and a `source:` pointer — nothing else filled in until a fork picks it up.

## Model-unit record

```markdown
## source
`process/models/<path>.py`, lines `<a>-<b>` if partial (state the scope reason if
partial, e.g. "only methods called by Stellarator.run(), see unit_registry.md").

## data footprint
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.stellarator.powerht_constraint` | write | explicit-arg | ... |
| `.physics.rmajor` | read | implicit-io | read again after `plasma_profile.run()` call, value may differ from entry |

Classification values (six now — added incrementally after real findings, see
`unit_registry.md`'s pilot-batch section and the stellarator.py chunk audits for the
concrete examples that motivated each):
- **`explicit-arg`** — read once, used as a plain parameter.
- **`implicit-io`** — read mid-loop or across branches, depends on state written earlier
  in the same call in a way that could plausibly diverge (order-dependent, conditional,
  or otherwise not statically resolvable to "always the same value already computed").
  Reserve this for the genuinely risky cases — see `local-intermediate` below for the
  common case that isn't.
- **`implicit-io-via-callee`** — written onto a copy/proxy of a stateful object, then an
  opaque callee (another `Model`'s method) reads it off `self` rather than as an argument
  (the `copy.deepcopy(model)` + mutate-then-call pattern). Distinct from plain
  `implicit-io`: the write and the read are in *different* call frames connected only by
  aliasing, not by loop-local sequencing.
- **`redundant-duplicate-write`** — the same `VarPath` is written twice to the same value
  by the unit under audit (once internally, once via the caller re-assigning the return
  value). Not a correctness issue, but the port should keep exactly one write — the
  return value — and drop the internal one; flag it so the redundancy doesn't get carried
  into the pure port as two ports for one value.
- **`local-intermediate`** — written once, unconditionally, with no branching, and read
  back later in the *same straight-line function* — mechanically identical to an ordinary
  Python local variable that happens to be routed through `self.data` because that's this
  codebase's universal idiom for any value, not because it's actually shared or order-
  sensitive. Use this instead of `implicit-io` whenever the write-then-read has no
  possibility of diverging from what was just computed (no intervening branch, no loop,
  no call to anything that could also touch the same field). This label exists
  specifically to keep `implicit-io` reserved for cases that need a careful read — applied
  loosely, almost every Fortran-derived function in this codebase would show several
  "implicit-io" entries that are actually trivial.
- **`conditional-ownership-by-run-config`** — the write only happens when a specific
  `VarPath` (typically the same field, or a related one) is *not* currently an active
  entry in `data.numerics.ixc` (the iteration-variable set). Found in
  `st_new_config()`: `.physics.aspect` is only forward-computed when `aspect` isn't
  itself an optimizer-owned unknown — otherwise the model leaves it alone. This is not an
  implementation detail to normalize away: it's PROCESS's model code enacting the same
  cut-point duality already discussed for constraints (an iteration variable is a
  boundary input with no forward producer) directly inside a physics model, not just at
  the constraint layer. Whether this node "owns" the output is a run-configuration fact,
  not a static property of the function — the pure port's graph-assembly step needs to
  read `ixc` at the same place it resolves other run-config-driven structure (see
  `naming_convention.md`'s "switches are not ports").

## proposed signature(s)
```python
def calculate_<x>(<explicit args, VarPath-derived names>) -> <output shape>:
    ...
```
One per identifiable pure unit within the source range — a 2000-line file may propose
several.

## cottax node
Thin wrap of the signature above — mechanical once the reads/writes in the data-footprint
table are settled, not a second design pass. Written as actual, importable code in the
port's own `.py` file, right after the function it wraps — not just documented here (a
node that only exists in prose can't be imported into `total_process.py`, see below).
Skip this section (and the class in the `.py` file) while open questions about the
signature itself are unresolved — the risk a wrap-before-settled relocates rather than
removes is a silent mismatch between the declared ports and what `fn` actually does.

**Use `cottax.interfaces.pytree_namespace_module`, not a hand-built `CallableNode`.**
PROCESS's `data.<area>.<field>` is already a real nested namespace, and this surface
lets a port's `Output`/`Input` read exactly like that path instead of a `VarPath` built
one `GetAttrKey` at a time:

```python
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, Input, Output

class <Name>(ExplicitFunction):
    <output> = Output(lambda s: s.<area>.<field>)   # one per element of the return tuple
    ...

    def __call__(
        self,
        <arg>=Input(lambda s: s.<area>.<field>),     # one per `read` row, fn's own arg order
        ...
    ):
        return <ported function>(<arg>, ...)
```
The class's `__call__` only ever forwards to the already-proposed, already-tested
function — it is not a second place the arithmetic lives. A switch resolved as a static
kwarg per `naming_convention.md`'s "switches are not ports" becomes a plain (non-`Input`)
field on the class instead of a `__call__` parameter — see `EcrhDensityLimit` in
`models/stellarator/density_limits.py` for `i_plasma_pedestal`.

**Tier-2 (self-contained internal solve):** use `ImplicitFunction` (residual form,
paired with a `RootFind`) or `FixedPointFunction` (next-iterate form, paired with a
`FixedPoint`) from the same module instead of `ExplicitFunction`. Either declares *two*
nodes at once — the body (`residual`/`step`) and the problem that closes it — because the
body reads the very unknown the problem owns: that mutual read/write is the "hole" the
problem drives to close, not a bug to route around. Only reach for these once a chunk's
internal solve is confirmed self-contained (calls no other not-yet-ported unit) — see
`stellarator_B_st_phys.md`'s `power_at_ignition_point` for a tier-2 unit that is *not*
self-contained (it calls into unit #1's `st_phys`) and therefore cannot get one of these
yet.

**Every ported node belongs in `functional_process/total_process.py`** (imported and
passed to `to_graph(...)`) so it shows up in `render_xdsm.py`'s diagram — a node that
exists in a port file but not in `total_process.py` is invisible to inspection.

## tier signal
1 / 2 / 3, with one line of justification (calls `scipy.optimize`? calls another
model's method? neither?).

## switches touched
List of `(switch, values seen, note)` — cross-reference to `core/solver/switches.md`,
don't duplicate the reads-set diff here.

## calls into other models
For tier-3/4 mapping — which other `Model` instances/methods this unit calls, by name.

## JAX-difficulty flags
List, each tagged per `traceability_policy.md` (`non-traceable-external-call`,
`needs-lax-cond-or-where`, etc.) with severity (`blocker` / `workaround-known` / `minor`).

## open questions
Anything the auditing agent is unsure of — this is what a low-confidence review pass
reads first.
```

## Constraint record (one section per constraint, inside `core/solver/constraints.md`)

```markdown
### Constraint <id>: <name/symbol/units>
**calls**: <function(s) it calls>
**data footprint**: (same table as above)
**proposed signature**: (same as above)
**hole-in-MDA**: yes / no / unsure — <reasoning>. If yes, candidate producer: <where,
if known>.
**current closure mechanism**: VMCON-joint / internal-fsolve / idempotence-loop / N/A
**candidate iteration variable(s)**: <best-effort, explicitly non-authoritative>
**confidence**: high / medium / low
**open questions**: ...
```

## Switch record (one section per switch, inside `core/solver/switches.md`)

```markdown
### `data.<area>.<field>` (values: ...)
**sites**: <file:line list>
**per-value reads-set**: <filled in as model-unit audits touching this switch land;
"pending" until then>
**entangled switches**: <any sibling switch whose value changes this switch's own
reads-set, e.g. a branch reads the same field under two different constants depending on
switch A, but which of those two reads happens at all is gated by a second switch B —
A's split decision then isn't decidable without B. Name B here rather than resolving A in
isolation. "None found" is a valid, worth-stating answer.>
**resolved above this file**: <if this switch is never branched on locally because it's
already dispatched to a concrete implementation one layer up (e.g. a `@property` on
`Models` picking a model instance before any model runs — see `i_cost_model` /
`Models.costs` in `process/main.py` for the precedent), say so explicitly. This is the
*target* pattern for topology-changing switches, not a gap — note it as evidence the
pattern already exists in the codebase, not as "pending.">
**split decision**: split / keep-static / unsure — <evidence>
**confidence**: high / medium / low
```
