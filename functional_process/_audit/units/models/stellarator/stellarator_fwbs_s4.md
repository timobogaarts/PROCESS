---
kind: model-unit
status: draft
confidence: high
---

# `st_fwbs` S4 — component masses — audit record

Companion to `stellarator_E_fwbs_synthesis.md`, whose § 1 names this piece **S4**
`blanket_shield_fw_coolant_mass` and whose § 5 recorded it as *"portable once S2 and S3's
signatures exist (needs their outputs as `In` arguments) — purely blocked on its two
producers' signatures settling, not on any further audit work."* Both have since landed
(`stellarator_fwbs_s2.md`/`.py`, `stellarator_fwbs_s3.md`/`.py`), and
`unit_registry.md`'s S4 row recorded the same blocker; it is discharged.

The immediate reason this unit was picked up is `_audit/boundary_inputs_audit.md` § 4c
items **(b1)–(b6)** and § 7 item **3**: six fields that PROCESS computes on the reference
run's path and that the port was taking as *boundary inputs* — `.fwbs.m_blkt_li2o`,
`m_blkt_beryllium`, `m_blkt_steel_total`, `m_blkt_vanadium` (read by `BlanketCost`),
`.fwbs.whtshld` (read by `Bldgs` and `ShieldCost`) and `.fwbs.wpenshld` (read by
`ShieldCost`). That audit measured zero cycle risk for all six; § "cycle risk,
re-measured" below re-measures it independently rather than taking it on trust.

## source

`process/models/stellarator/stellarator.py`, lines **1045–1274**, excluding S3's
1030–1043 — inside `Stellarator.st_fwbs`. Four sub-blocks, of which **two are ported**
and two are not:

| sub-block | lines | ported? |
|---|---|---|
| `coolvol` seed + accumulation | 1048-1052, 1191-1193, 1202, 1222-1226/1238-1246 | no — see § "what this port leaves undone" |
| blanket component masses (`blktmodel` × `blkttype` dispatch) | 1056-1181 | **yes**, the `blktmodel == 0` + solid-breeder arm (1068-1091) |
| shield mass + penetration shield | 1195-1206 | **yes**, unconditional |
| first-wall mass (`ipowerflow` dispatch) + total coolant mass | 1208-1274 | no — see § "what this port leaves undone" |

The two ported sub-blocks are verbatim:

```python
if self.data.fwbs.blktmodel == 0:
    if self.data.fwbs.blkttype in {1, 2}:  # liquid breeder (WCLL or HCLL)
        ...  #   -> wtbllipb, m_blkt_lithium
    else:  # solid breeder (HCPB); always for ipowerflow=0
        self.data.fwbs.m_blkt_li2o = (
            self.data.fwbs.vol_blkt_total * self.data.fwbs.fblli2o * 2010.0e0
        )
        self.data.fwbs.m_blkt_beryllium = (
            self.data.fwbs.vol_blkt_total * self.data.fwbs.fblbe * 1850.0e0
        )
        self.data.fwbs.m_blkt_total = (
            self.data.fwbs.m_blkt_li2o + self.data.fwbs.m_blkt_beryllium
        )

    self.data.fwbs.m_blkt_steel_total = (
        self.data.fwbs.vol_blkt_total * self.data.fwbs.den_steel * self.data.fwbs.fblss
    )
    self.data.fwbs.m_blkt_vanadium = (
        self.data.fwbs.vol_blkt_total * 5870.0e0 * self.data.fwbs.fblvd
    )

    self.data.fwbs.m_blkt_total = (
        self.data.fwbs.m_blkt_total
        + self.data.fwbs.m_blkt_steel_total
        + self.data.fwbs.m_blkt_vanadium
    )
```

```python
        # Shield mass
        self.data.fwbs.whtshld = (
            self.data.fwbs.vol_shld_total
            * self.data.fwbs.den_steel
            * (1.0e0 - self.data.fwbs.vfshld)
        )
        ...
        # Penetration shield (set = internal shield)

        self.data.fwbs.wpenshld = self.data.fwbs.whtshld
```

## data footprint

### `calculate_blanket_component_masses` (1068-1091)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.fwbs.vol_blkt_total` | read | explicit-arg | S1's output (`FwBlanketShieldGeometry`, `stellarator_fwbs_s1_s5.py`), written at `:567` in the same `st_fwbs` call — an ordinary same-call upstream edge, and the *only* operand of this block with a producer in the graph |
| `.fwbs.fblli2o` | read | explicit-arg | Li2O volume fraction; default `0.08` (`fwbs_variables.py:455`), never written anywhere in `process/` |
| `.fwbs.fblbe` | read | explicit-arg | beryllium volume fraction; default `0.6` (`:198`) |
| `.fwbs.den_steel` | read | explicit-arg | steel density (kg/m3); default `7800.0` (`:22`) |
| `.fwbs.fblss` | read | explicit-arg | stainless steel volume fraction; default `0.09705` (`:37`) |
| `.fwbs.fblvd` | read | explicit-arg | vanadium volume fraction; default `0.0` (`:461`) |
| `.fwbs.m_blkt_li2o` | write | explicit-arg | kg; read by `Costs.acc222` → `BlanketCost` |
| `.fwbs.m_blkt_beryllium` | write | explicit-arg | kg; read by `BlanketCost` |
| `.fwbs.m_blkt_steel_total` | write | explicit-arg | kg; read by `BlanketCost` |
| `.fwbs.m_blkt_vanadium` | write | explicit-arg | kg; read by `BlanketCost` |
| `.fwbs.m_blkt_total` | write (twice, accumulating) | local-intermediate → explicit-arg | PROCESS writes it at `:1074` (breeder part) and again at `:1087` (+ steel + vanadium). The first write is read only by the second, in the same straight-line block with no intervening branch — `local-intermediate` by `schema.md`'s definition. The **final** value is a genuine `.fwbs.*` output and is declared as one; the intermediate is not a separate port. Currently has **no reader** in `total_process.graph_for()` — declared anyway, see § "why `m_blkt_total` is declared" |

Nothing here is `implicit-io`: every read is a plain field read, once, with no branch or
callee between it and its use.

### `calculate_shield_mass` (1196-1206)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.fwbs.vol_shld_total` | read | explicit-arg | S1's output (`FwBlanketShieldGeometry`), written at `:593` in the same call |
| `.fwbs.den_steel` | read | explicit-arg | as above |
| `.fwbs.vfshld` | read | explicit-arg | shield coolant void fraction; default `0.25` (`fwbs_variables.py:413`) |
| `.fwbs.whtshld` | write | explicit-arg | kg; read by `Bldgs` and `ShieldCost` |
| `.fwbs.wpenshld` | write | explicit-arg | kg; read by `ShieldCost`. PROCESS assigns it `= whtshld` outright (`:1206`) — reproduced as a second declared output, not collapsed, because it is a real field with its own reader |

One statement (`coolvol += vol_shld_total * vfshld`, `:1202`) sits textually *between*
these two writes. It touches no `.fwbs.*` field — it accumulates a Python local — so it
is skipped here and recorded in § "what this port leaves undone" instead of being
silently absorbed.

## proposed signature(s)

```python
def calculate_blanket_component_masses(
    vol_blkt_total: float,
    fblli2o: float,
    fblbe: float,
    den_steel: float,
    fblss: float,
    fblvd: float,
) -> tuple[float, float, float, float, float]:
    # (m_blkt_li2o, m_blkt_beryllium, m_blkt_steel_total, m_blkt_vanadium, m_blkt_total)
    ...


def calculate_shield_mass(
    vol_shld_total: float,
    den_steel: float,
    vfshld: float,
) -> tuple[float, float]:
    # (whtshld, wpenshld)
    ...
```

Written in `stellarator_fwbs_s4.py`; see each function's own docstring for the full
parameter descriptions.

## cottax node

`BlanketComponentMasses` and `ShieldMass`, both `ExplicitFunction`, in
`stellarator_fwbs_s4.py`. **Both registered** in `functional_process/total_process.py` —
see § "registration" for which goes where and why.

## registration

Two different answers for the two nodes, because the two sub-blocks genuinely differ in
whether a PROCESS switch guards them.

**`ShieldMass` → `COMMON`.** `stellarator.py:1195-1206` is outside every branch in
`st_fwbs`: no `blktmodel`, no `blkttype`, no `ipowerflow`, no `i_tf_sup` guard. Read the
enclosing block directly to confirm rather than inferring it from indentation alone.
Both outputs therefore exist in every configuration this graph can assemble, which is
exactly `COMMON`'s meaning.

**`BlanketComponentMasses` → a new synthetic `Switch(path=".fwbs.blktmodel,.fwbs.blkttype")`.**
The reasoning, since the task explicitly asked for it:

1. *Is it configuration-dependent at all?* **Yes, and on two axes.** Under
   `blktmodel != 0` (`:1093-1181`) PROCESS computes `m_blkt_steel_total` and
   `m_blkt_beryllium` from six `.build.bl{u,m,p}{i,o}th` sub-assembly thicknesses this
   node never reads, additionally writes `.fwbs.whtblbreed` and
   `.fwbs.f_a_blkt_cooling_channels`, and **never writes `m_blkt_li2o` or
   `m_blkt_vanadium` at all**. Under `blktmodel == 0` *and* `blkttype in {1, 2}` (liquid
   breeder, `:1058-1066`) it writes `.fwbs.wtbllipb`/`.fwbs.m_blkt_lithium` instead of
   `m_blkt_li2o`/`m_blkt_beryllium`. Registering this node unconditionally in `COMMON`
   would be the `EcrhDensityLimit`/`ScTfCoilNuclearHeating` bug class `configuration.py`
   exists to make impossible — a node computing fields the selected configuration does
   not compute, by a formula it does not use.
2. *Can it join the existing joint `.fwbs.blktmodel,.heat_transport.ipowerflow` switch?*
   **No.** That switch is S2's dispatch. Its three arms are `(blktmodel == 1)`,
   `(blktmodel != 1, ipowerflow == 0)`, `(blktmodel != 1, ipowerflow == 1)`. S4's mass
   block does not read `ipowerflow` at all, so this node would have to be declared in
   *both* of the latter two arms — duplicating a node across arms to express "this axis
   is irrelevant here", which conflates two independent dispatches and would make each
   arm's node list stop meaning "what this arm computes". It also could not express the
   `blkttype` axis, which that switch has no notion of.
3. *Why one synthetic joint switch rather than two real ones (`.fwbs.blktmodel` and
   `.fwbs.blkttype` separately)?* Two real switches would be individually checkable by
   `test_configuration.py::test_switch_defaults_match_process`, which is a genuine
   advantage — but they cannot express this block. `m_blkt_total` is accumulated **across**
   both axes (breeder part chosen by `blkttype`, then steel + vanadium added, both inside
   `blktmodel == 0`), so the block does not factor into a `blktmodel`-only node plus a
   `blkttype`-only node without inventing a third node to own a sum PROCESS writes as two
   statements in one straight line. The dispatch really is joint, exactly as S2's is, and
   the codebase already has that precedent and the tests that skip comma-joined paths for
   it (`test_switch_defaults_match_process`, `test_reference_configuration_matches_the_
   input_file`, both of which `continue` on `"," in switch.path`).
4. *Which arm is the reference run's?* `stellarator_helias.IN.DAT` sets **neither**
   `blktmodel` nor `blkttype`, so both fall through to PROCESS's own defaults:
   `blktmodel = 0` (`fwbs_variables.py:479`) and `blkttype = 3` (`:494`). `3 ∉ {1, 2}`, so
   the solid-breeder (HCPB) sub-arm is live — the arm this node implements, and the
   `Switch`'s `default`.

The other two arms are `unported`, not `unproduced`: assembling either while leaving
`m_blkt_*` unowned would hand `BlanketCost` a boundary value silently, which is precisely
what `Alternative`'s docstring says `unported` is for. It also matches the existing joint
S2 switch, whose `blktmodel == 1` arm already raises.

**A wart worth stating rather than hiding**: `.fwbs.blktmodel` now appears as an axis of
*two* synthetic switch keys (S2's and S4's), and nothing mechanically enforces that a
caller choosing arms for both keeps them consistent about `blktmodel`. That is a real gap
in `configuration.py`'s model — arm indices on synthetic keys are opaque to it. It is not
introduced by this unit (S2's key already had the property), and it does not bite the
reference run or PROCESS's default run, both of which sit at `blktmodel = 0` in both
switches' defaults. Recorded as an open question below rather than papered over.

## cycle risk, re-measured

`boundary_inputs_audit.md` § 4c states zero cycle risk for all six fields. Re-measured
here independently against `total_process.graph_for()` before writing the nodes, by
building the node-level producer→consumer DAG and taking each reader's transitive
descendants:

| reader | descendants | `FwBlanketShieldGeometry` (S1) among them? |
|---|---|---|
| `BlanketCost` | 6 | no |
| `ShieldCost` | 6 | no |
| `Bldgs` | 20 | no |

`.fwbs.vol_blkt_total` and `.fwbs.vol_shld_total` are the only operands with a producer
(both `FwBlanketShieldGeometry`); everything else (`fblli2o`, `fblbe`, `fblss`, `fblvd`,
`den_steel`, `vfshld`) is a boundary input and stays one. So closing these edges cannot
create a cycle. Confirmed after the fact too: the assembled graph's SCC list is unchanged
(same 12 multi-node/self-loop cycles, same members).

Incidentally measured on the way, and worth recording because it is the mirror image of a
boundary input: **before this unit, `.fwbs.vol_blkt_total` and `.fwbs.vol_shld_total` had
zero readers** — S1 computed both and the graph threw them away. This unit is their first
consumer.

## why `m_blkt_total` is declared

It is a sixth output beyond the six fields the boundary audit asked for, and it has **no
reader** in the graph today (measured). It is declared anyway because PROCESS writes it in
the same straight-line block from the same four masses, and a node whose output set is a
strict subset of what its source range writes — chosen by current demand rather than by
what the code does — is the shape that lets a field silently keep a stale boundary value
later. The cost is one extra compared output in the MDA harness, not an extra edge.

## tier signal

**Tier 1**, both functions. Six reads / five writes and four multiplications for the
first; three reads / two writes and two multiplications for the second. No internal
iteration, no calls into other models, no data-dependent branching inside either pure
function (the two branches that exist in the source are topology switches resolved at
graph-assembly time, not `jnp.where`s).

## switches touched

| switch | values seen | note |
|---|---|---|
| `.fwbs.blktmodel` | `0` (ported arm), `1` | `choices=[0, 1]` (`core/input.py:978`). **Topology switch, split.** The two arms write partially disjoint field sets, not two formulas for one set. Note PROCESS spells this switch `== 1` at `:608` (S2) and `== 0` at `:1056` (S4) — `stellarator_E_fwbs_synthesis.md` § 4 already confirmed the domain is exactly `{0, 1}`, so the two spellings partition it identically; inconsistent style, not a gap |
| `.fwbs.blkttype` | `{1, 2}` (liquid, unported), `3` (solid, ported) | `choices=[1, 2, 3]`, default `3`. **Topology switch, split.** This is the one computational site `core/solver/switches.md` records for `blkttype`, and the synthesis record's § 4 "three-values-two-arms" `Alternative`-shape problem: values 1 and 2 select the *same* liquid-breeder formula. Handled the same way the `.tfcoil.i_tf_sup` switch already handles `{0, 2}` — one arm carries the reason, the other points at it — except that here neither liquid value is ported, so both are simply `unported` with the same text |
| `.heat_transport.ipowerflow` | — | read at `:1208`, inside the first-wall mass block this unit does **not** port. Not a switch this unit touches |

## calls into other models

None. Confirmed by reading the whole 1045-1274 range for a `self.<submodel>.` reference:
zero hits, matching `stellarator_E_fwbs_synthesis.md` § 5's "no cross-model calls of its
own".

## JAX-difficulty flags

None. Both functions are products and sums of scalars — no `jnp` call is needed at all,
and neither imports it.

## what this port leaves undone, and why

Two of S4's four sub-blocks are deliberately not ported. Both reasons are structural, not
effort:

1. **The `coolvol` accumulator and `.fwbs.m_fw_blkt_div_coolant_total`** (`:1048-1052`,
   `:1191-1193`, `:1202`, `:1222-1246`, `:1269-1274`). `coolvol` is a Python local with no
   `VarPath`, threaded from S3 into S4 (`stellarator_E_fwbs_synthesis.md` § 2, row 3), and
   its seed is `.divertor.a_div_surface_total` — the cross-call read
   `stellarator_fwbs_s3.md` documents as genuine inter-call state. Porting the accumulator
   means deciding S3's `{DivertorPlateMass, Divertor}` composition question, which that
   record explicitly defers.
2. **The first-wall mass block** (`:1208-1262`: `.fwbs.m_fw_total`, `.fwbs.fwclfr`). Its
   `ipowerflow != 0` arm — the reference run's arm — reads `f_a_fw_coolant_inboard` and
   `f_a_fw_coolant_outboard`, two Python **locals** produced inside S2 by one of two
   different formulas selected by `blktmodel` (`stellarator_E_fwbs_synthesis.md` § 2,
   row 2). They are not fields, so no node can bind them, and
   `stellarator_fwbs_s2.py`'s ported arms do not return them either. Porting this block
   requires widening S2's node signature first — a change to an already-registered unit,
   out of this unit's boundary.

Measured, so the omission is a stated cost rather than an assumed-harmless one:
`.fwbs.m_fw_total`, `.fwbs.fwclfr` and `.fwbs.m_fw_blkt_div_coolant_total` all currently
have **zero readers** in `total_process.graph_for()`. Leaving them unowned therefore
closes no edge any consumer is waiting on, and adds no boundary input that was not already
one.

## PROCESS bugs found

**None.** Both ported sub-blocks were checked statement by statement against the source
and are internally consistent: no stale read, no discarded return value, no unassigned
name, no arm that writes a field the sibling arm leaves untouched *and* that a downstream
reader depends on unconditionally. Two things that *look* like findings and are not, both
checked rather than waved past:

- `m_blkt_total`'s two-statement accumulation is not a `redundant-duplicate-write`: the
  second statement reads the first's value, so both are load-bearing. It is the ordinary
  `local-intermediate` pattern.
- `wpenshld = whtshld` is not a modelling stub gone wrong; PROCESS's own comment
  ("Penetration shield (set = internal shield)") states the intent, and `ShieldCost`
  consumes them as two separate accounts.

The `blktmodel == 1` arm this unit refuses is not bug-free — but its bugs live in
`blanket_neutronics()` (S2's territory) and are already on record in
`stellarator_fwbs_s2.md` § 2.2 and `hcpb.md`, not here.

## open questions

1. **Nothing enforces `blktmodel` consistency across the two synthetic switch keys.**
   `.fwbs.blktmodel` is now an axis of both `".fwbs.blktmodel,.heat_transport.ipowerflow"`
   (S2) and `".fwbs.blktmodel,.fwbs.blkttype"` (S4), and `configuration.py` treats each
   comma-joined path as an opaque key with integer arm indices, so a `Configuration`
   asking for S2's `blktmodel == 1` arm and S4's `blktmodel == 0` arm is describable and
   would not be rejected. It is currently harmless (S2's `blktmodel == 1` arm is
   `unported` and raises first, and both defaults agree at `blktmodel = 0`), but the
   general shape — synthetic keys hiding a shared real field — is a gap in
   `configuration.py`'s model, not in this unit. A real fix is probably predicate-keyed
   arms over real field values rather than arm indices, which is the same mechanism
   `stellarator_E_fwbs_synthesis.md` § 4 wanted for `blkttype`'s "three values, two arms".
2. `boundary_inputs_audit.md` § 8's caveat applies to this unit as it does to the rest:
   the cycle-risk measurement above is over *currently registered* nodes, so it cannot see
   coupling that would appear once the blanket/first-wall producers this port has not
   reached are added. The claim is "no cycle among the producers this configuration
   executes", not "no cycle exists".
3. `stellarator_fwbs_s3.md` § "Boundary discrepancy" asked whoever audits S4 to reconcile
   the synthesis record's § 1 boundary (S3 = 1030-1043, S4 = 1045-1274) against its § 2
   ledger, which describes the `coolvol` seed at 1048-1052 as belonging to "S3". **Closed
   here in favour of § 1**: 1048-1052 is textually inside S4's range and this record
   claims it (as unported, item 1 of the previous section); § 2's "S3" is a slip of
   attribution in a ledger about *locals*, not a competing boundary. No line is claimed by
   two records, and no line is claimed by none.
