# Splitting the physics from the graph declarations

Status: **planned, nothing started.** Agreed 2026-09-03; scheduled for the following week.

## The goal

`functional_process/formulas/**` becomes **PROCESS's physics as pure JAX functions**,
importing nothing from cottax, and `models/**` keeps only the graph declarations. The
physics library is then usable without the graph machinery at all.

That is a *product*, not tidiness. The OpenMDAO generator in `~/openmdao_process` already
imports those bodies, so it is the first consumer and a live check that the boundary holds.

## Why this is cheap: the files are already shaped for it

`models/physics/density_limit.py` is representative — module-level pure functions
(`calculate_greenwald_density_limit`, `select_enforced_density_limit_greenwald`, …) at the
top, then thin `ExplicitFunction` declarations whose `__call__` is four lines delegating to
them:

```python
class GreenwaldDensityLimit(ExplicitFunction):
    nd_plasma_electron_greenwald_max = OutputInto(physics)
    def __call__(self, plasma_current=From(physics), rminor=From(physics)):
        return calculate_greenwald_density_limit(plasma_current, rminor)
```

So for files of that shape this is a **file move plus an import**, not a rewrite. What is
unknown is how many declarations are *not* that shape. Step 1 answers it.

## The plan

1. **Census, in reporting mode.** An AST meta-test in the family of
   `tests/functional_process/test_registry_coverage.py`: for every `ExplicitFunction`
   subclass, parse `inspect.getsource(cls.__call__)` and check the body is an optional
   docstring followed by exactly one `return <call>`, whose callee resolves to a
   **module-level function** — not a lambda, a closure or a method.

   Two nuances to decide once rather than discover:
   - **Multi-output nodes.** Some return a tuple, and at least one pattern is *four rows
     each indexing one position of the same call* (`CentrepostNeutronicsAbsent`). The rule
     is "one call, optionally destructured", not "one call".
   - **Trivial adaptation.** A unit conversion or an unpack is arguably fine. Decide
     whether it is allowed and write down the reason.

   **Report first, gate later.** Turning enforcement on before the list is empty means 230
   failures and no information. This is the same pattern the guard audit and the carried
   census used.

2. **Extract the non-thin declarations** — inline physics moves to module-level functions,
   one commit per file or group. Pure refactor, so **bitwise identity is the gate** and the
   cold matrix is the instrument that already proves it.

3. **Turn the check on as a test.** This is what makes the split durable rather than a
   tidy-up that decays.

4. **Move the files.** Suggested layout, because it extends a convention that already
   exists — a unit's port, audit record and test are bound by **shared stem across three
   trees**, so add a fourth:

   | tree | holds |
   |---|---|
   | `functional_process/formulas/physics/density_limit.py` | the pure functions |
   | `functional_process/models/physics/density_limit.py` | the declarations |
   | `_audit/units/models/physics/density_limit.md` | the record |
   | `tests/functional_process/…/test_density_limit.py` | the case |

   `unit_registry.md` names paths explicitly, so it extends naturally and nothing needs
   renaming.

5. **Enforce the boundary**: a test asserting `functional_process/formulas/**` **never
   imports cottax**. Without it, the first person in a hurry imports `From` into a formulas
   file and the separation is over.

## The declaration interface — considered, and deliberately not done now

The thin-delegator shape invites a further change: make the declaration name its
implementation rather than contain it.

```python
class GreenwaldDensityLimit(ExplicitFunction):
    fn = calculate_greenwald_density_limit
    nd_plasma_electron_greenwald_max = OutputInto(physics)
    plasma_current = From(physics)
    rminor         = From(physics, parameter="minor_radius")   # only where names differ
```

called by **keyword**, with `__check_init__` comparing declared field names against
`inspect.signature(fn)` and refusing a mismatch.

**Why it is attractive**: the declaration becomes pure data with no user-authored body,
which is `~/jaxgraph/plans/graph_vs_execution_layer.md` §2 one level down — §2 says a
*graph* should not contain callables, this says a *declaration* should not contain a body.
It also makes step 4 fall out for free.

**Why not now**, in order of weight:

- Steps 1–5 above deliver the reusable-physics product with **no upstream change and
  almost no churn**. This adds an `interfaces/pytree_namespace_module.py` change plus ~230
  declarations rewritten, for a benefit that is architectural rather than immediate.
- It is **close to cottax's own `AbstractImplementation`** direction, so inventing a second
  spelling of it now risks having to unify them later.
- Steps 1–3 make it *smaller* when it is done: the extractions are needed either way.

**Rejected alternatives, so they are not re-proposed:**

- **Positional `fn(*args)`.** Opaque — the parameter names are invisible and the order is
  silent. Consistent with how ports already bind (by position; 15–29 nodes take `*args` and
  4–14 have parameter names differing from the VarPath leaf), but there is no reason to
  accept opacity when a construction-time check costs one method.
- **Signature-as-declaration with the body removed** (`def ports(self, x=From(a)): ...`).
  Keeps today's readability, but leaves *two* signatures to keep in sync — the opacity
  relocated rather than removed, unless validated, at which point the keyword form is
  simpler.
- **Deriving ports from the signature.** Inverts the dependency: renaming a parameter in a
  physics function would silently change the graph. Fragile.

**One implementation caution, learned the hard way elsewhere.** OpenMDAO has exactly this
alias mechanism (`primal_name`) and its implementation carries a silent-failure bug:
`get_function_deps` returns `wrt` in primal names and then filters them against OpenMDAO
names, so nothing matches and the Jacobian comes back **all zero with no warning** — it
surfaced as 15 totals of exactly 0.0 while the values stayed bitwise correct. Reproduced as
a runnable case in `~/openmdao_process`. If the alias is built here, it must be **tested**,
not merely supported.
