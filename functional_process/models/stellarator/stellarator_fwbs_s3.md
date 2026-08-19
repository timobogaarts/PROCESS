---
kind: model-unit
status: draft
confidence: high
---

## source

`process/models/stellarator/stellarator.py`, lines 1030-1043 — the `st_fwbs` fragment
`stellarator_E_fwbs_synthesis.md` names **S3**, `divertor_mass_and_first_call_seed`:

```python
        # Divertor mass
        # N.B. self.data.divertor.a_div_surface_total is calculated in stdiv after this
        # point, so will be zero on first lap, hence the initial approximation

        if self.first_call_stfwbs:
            self.data.divertor.a_div_surface_total = 50.0e0
            self.first_call_stfwbs = False

        self.data.divertor.m_div_plate = (
            self.data.divertor.a_div_surface_total
            * self.data.divertor.den_div_structure
            * (1.0e0 - self.data.divertor.f_vol_div_coolant)
            * self.data.divertor.dx_div_plate
        )
```

Strictly within this range only — `st_fwbs`'s S2 (`blanket_shield_tf_nuclear_power`,
422-480+608-1030, adjacent above) and S4 (`blanket_shield_fw_coolant_mass`, starting
1045 with the `coolvol` seed) are out of scope for this record, per the synthesis
document's boundary and the concurrent-audit note below.

**Boundary discrepancy, not resolved here**: the synthesis document's § 1 boundary table
puts S3 at exactly 1030-1043 and S4 at "1045-1274 excl. 1030-1043", but its § 2
cross-boundary ledger describes `coolvol` as created "S3 (1048-1052, ...)" — 1048-1052 is
textually inside the § 1 S4 range, not S3's. This audit follows § 1's (and this task's)
explicit 1030-1043 bound and does **not** port the `coolvol` seed (1048-1052); flagging
the inconsistency for whoever picks up S4 rather than silently resolving it by expanding
scope here.

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.divertor.a_div_surface_total` | read | implicit-io (cross-call, see § below) | read as a plain float in the `m_div_plate` formula |
| `.divertor.a_div_surface_total` | write (conditional) | conditional-ownership-by-run-config | only written when `self.first_call_stfwbs` is true — see § below; **not** a plain `explicit-arg` write since it doesn't happen every call |
| `.divertor.den_div_structure` | read | explicit-arg | divertor structure density (kg/m3), plain default field, never written anywhere in `stellarator.py` |
| `.divertor.f_vol_div_coolant` | read | explicit-arg | divertor coolant volume fraction, plain default field, never written anywhere in `stellarator.py` |
| `.divertor.dx_div_plate` | read | explicit-arg | divertor plate thickness (m), plain default field, never written anywhere in `stellarator.py` |
| `.divertor.m_div_plate` | write | explicit-arg | divertor plate mass (kg); consumed by `process/models/structure.py:66` (mass sum, out of this port's scope) and printed in S6's output block (1679-1680), out of port scope |

`self.first_call_stfwbs` is **not** a `data.<area>.<field>` at all — it is a plain
Python instance attribute of the `Stellarator` model object (set `True` once in
`Stellarator.__init__`, `stellarator.py:95`), not a field of `DataStructure`. It has no
`VarPath`. Grepped and confirmed: read/written only at these two sites in the entire
codebase (`stellarator.py:95,1034,1036`) — nowhere else touches it. It therefore cannot
be classified with the five `schema.md` VarPath labels at all; it is recorded here as a
distinct finding, not shoehorned into one of them.

### The cross-call read/bootstrap, precisely

Traced `Stellarator.run()`'s non-`output` call order (lines 156-164, confirmed against
source): `st_strc → st_fwbs → st_div`. `st_fwbs` (this fragment lives inside it) runs
**before** `st_div` (`Divertor`, unit #4, `divertor.md`/`divertor.py` — confirmed the
unconditional, sole producer of `.divertor.a_div_surface_total`) in every ordinary solver
iteration. The source's own comment states why: `.divertor.a_div_surface_total` "is
calculated in stdiv after this point, so will be zero on first lap, hence the initial
approximation."

Concretely, across the life of one `Stellarator` instance:

- **The very first call to `st_fwbs`** (`first_call_stfwbs` still `True`, set in
  `__init__`): `.divertor.a_div_surface_total` is force-overwritten to the literal
  `50.0` **before** being read in the `m_div_plate` formula, and `first_call_stfwbs` is
  permanently cleared to `False`. Whatever `.divertor.a_div_surface_total` held at this
  point (its class default, `0.0` — `divertor_variables.py`, confirmed) is irrelevant;
  it is unconditionally replaced.
- **Every call after the first, forever** (`first_call_stfwbs` is `False` from here on —
  it is never set back to `True`): the bootstrap branch is skipped, and
  `.divertor.a_div_surface_total` is read as whatever `st_div` wrote **during the
  previous `run()` invocation** — `st_div` has not run yet *this* cycle when S3 executes,
  since it runs after `st_fwbs` in the same call. This is a genuine one-call-lagged
  read, not resolvable as ordinary same-call dataflow — the value read is never the
  value this call's own `st_div` will go on to produce.

One caveat already on record (`stellarator_E_fwbs_synthesis.md` § 2.1, re-confirmed by
this audit, not re-derived): `run()`'s `output=True` reporting path (125-146) calls
`st_div` **before** `st_fwbs(True)`, the reverse order — but by the time `output()` is
ever invoked, `first_call_stfwbs` has already been cleared by many prior normal-path
calls, so this reordering has no observable effect on this fragment; noted for
completeness only.

**This audit reproduces, not resolves, the synthesis document's classification**: S3 and
`Divertor` form a genuine two-node cycle across calls (`first_call_stfwbs` is the
driver's own "is this the first iterate" test, `50.0` its initial guess), which is a
graph-composition/driving question this audit is explicitly not deciding — see "Framing"
below.

## Framing this audit follows (not a design decision made here)

Per this task's brief: the `Blocking`/`FixedPoint`/`Cut` question of *how* this cycle
gets driven is deferred to a later composition pass once the whole graph shape is known.
This record and its port therefore treat `.divertor.a_div_surface_total` as an **ordinary
`Input`** off its real `VarPath` — exactly like any other cross-unit read this project has
already ported (e.g. `Divertor`'s own `In`s onto other units' outputs). The pure function
below takes `a_div_surface_total` as a plain float argument and does **not** encode
`first_call_stfwbs` or the `50.0` bootstrap in its signature at all: which concrete value
(the hardcoded `50.0` on a true first call, or `Divertor`'s previous-iteration output
otherwise) gets fed into that argument is a fact about how a future driver wires this
node, not about this node's own arithmetic. The conditional overwrite of
`.divertor.a_div_surface_total` itself (the write half of the cross-call finding above)
is consequently **not ported as an output of this node** — it is bootstrap/driving-time
state-seeding, not a computation this fragment's `m_div_plate` formula depends on
producing.

## proposed signature(s)

```python
def calculate_divertor_plate_mass(
    a_div_surface_total: float,
    den_div_structure: float,
    f_vol_div_coolant: float,
    dx_div_plate: float,
) -> float:
    # returns m_div_plate
    ...
```

One function, one output — the `first_call_stfwbs` branch and its `50.0` literal are
deliberately not part of this signature (see "Framing" above).

## cottax node

```python
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, Input, Output


class DivertorPlateMass(ExplicitFunction):
    m_div_plate = Output(lambda s: s.divertor.m_div_plate)

    def __call__(
        self,
        a_div_surface_total=Input(lambda s: s.divertor.a_div_surface_total),
        den_div_structure=Input(lambda s: s.divertor.den_div_structure),
        f_vol_div_coolant=Input(lambda s: s.divertor.f_vol_div_coolant),
        dx_div_plate=Input(lambda s: s.divertor.dx_div_plate),
    ):
        return calculate_divertor_plate_mass(
            a_div_surface_total, den_div_structure, f_vol_div_coolant, dx_div_plate
        )
```

Written in `stellarator_fwbs_s3.py`. **Not registered in `functional_process/total_process.py`**
by this pass — out of this task's strict boundary, and premature regardless: `DivertorPlateMass`
reads `.divertor.a_div_surface_total`, which `Divertor` (unit #4) also owns as an
`Output`, on the same `VarPath` `Divertor` writes from the *same* call's own computation —
wiring both into one `COMMON` graph unconditionally today would silently pick whichever
evaluation order cottax happens to use, not reproduce PROCESS's genuine one-call-lagged
semantics. That composition decision (present in the same "third/fourth confirmed
instance" family as `unit_registry.md` row 17's `Avail`/`.costs.cplife` self-loop, and
`plasma_composition`'s `first_call`) is exactly what this task defers to a later
consolidation pass.

## tier signal

**Tier 1.** Once `.divertor.a_div_surface_total` is treated as an ordinary `Input` (per
"Framing" above), this is four reads, one multiplication, one write — no internal
iteration, no calls into other models, no data-dependent branching left inside the pure
function itself (the branch that *does* exist in the source, on `first_call_stfwbs`, is
excluded from the node's signature by design, not hidden inside a `jnp.where`).

## switches touched

None that are ports. `self.first_call_stfwbs` behaves like a switch in spirit (it
selects between two producers of one value: the `50.0` literal vs. `Divertor`'s previous
output) but is not a `data.<area>.<field>` and is not read anywhere `naming_convention.md`'s
"switches are not ports" framework currently covers (it is neither a topology-changing
input-parse-time switch nor a formula-changing static kwarg on a node — it is per-instance
mutable model state that changes exactly once per `Stellarator` object's lifetime,
independent of any run configuration). Recorded as an open question, not force-fit into
the switches table.

## calls into other models

None directly. `.divertor.a_div_surface_total` is produced by `Divertor` (unit #4,
`st_div`) but S3 does not call it — the two are connected only via `self.data`/`.data`
state and call order, which is exactly the cross-call finding above, not a same-call
model-to-model call.

## JAX-difficulty flags

None. Plain arithmetic (one conditional-in-source-but-excluded-from-signature, one
multiplication), no external calls, no dynamic shapes.

## open questions

1. **What `self.first_call_stfwbs` "is" in cottax vocabulary is genuinely unresolved,
   not just deferred.** It is per-`Model`-instance Python state, not part of
   `DataStructure`, so it has no natural `VarPath` at all — unlike every other
   `conditional-ownership-by-run-config` case on record (`.physics.aspect` vs.
   `numerics.ixc`, `.costs.cplife` vs. `.physics.itart`), which are gated by an ordinary
   `data.<area>.<field>` value. Whoever designs the eventual `Blocking`/`FixedPoint`/`Cut`
   over `{DivertorPlateMass-adjacent-write, Divertor}` will need to decide whether
   `first_call_stfwbs` becomes a driver-level "is this the first solve" flag (most
   natural, since it is exactly that) or gets promoted into `DataStructure` first for
   uniformity. Not decided here.
2. This record does not port the conditional `.divertor.a_div_surface_total` write
   (the `50.0` bootstrap) as any node's `Output` — see "Framing". A future consolidation
   pass may want a tiny `Bootstrap`/`Seed`-shaped node for it, or may fold it directly
   into whichever `FixedPoint`/`Square` problem drives the `{S3, Divertor}` cycle as its
   initial guess. Left open.
3. The § "Boundary discrepancy" above (this document's own S3=1030-1043 vs. the
   synthesis's § 2 mention of `coolvol` at 1048-1052) should be reconciled by whoever
   audits S4 — not resolved here, per the strict line-range boundary this task set.
