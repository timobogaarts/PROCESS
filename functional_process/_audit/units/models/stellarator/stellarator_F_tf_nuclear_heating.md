---
kind: model-unit
status: reviewed
confidence: high
---

**Ported.** `stellarator_F_tf_nuclear_heating.py` / `test_stellarator_F_tf_nuclear_heating.py`,
tier-1 contract passing (legacy + fuzz). Only the SUPERCONDUCTING branch is ported, per
the elimination recommendation below — the resistive branch is not represented as a
function at all, consistent with `i_tf_sup` being resolved at graph-build time.

## source
`process/models/stellarator/stellarator.py`, lines 1683-1885:
`sc_tf_coil_nuclear_heating_iter90()`. Chunk 1F of unit #1 (see
`../../_audit/unit_registry.md`). Single method, no sub-calls, no internal loop — one of
the cleanest units found so far.

Called from two sites, both inside chunk 1E1's scope (422-880), not split across the
1E1/1E2 boundary: line 476 (`blanket_neutronics`, discards 9 of 10 return values, keeps
only `flu_tf_neutron_fast_peak`) and line 728 (`st_fwbs`, keeps
`flu_tf_neutron_fast_peak` and `p_tf_nuclear_heat_mw`, discards the other 8 into unused
locals). Flagging for whoever audits 1E1: **the same two output fields
(`self.data.fwbs.flu_tf_neutron_fast_peak`, `self.data.fwbs.p_tf_nuclear_heat_mw`) are
written by two different call sites of this function within one `run()`** — worth
checking there whether the two call sites are mutually exclusive branches (only one
executes per run) or whether the second call's write is a real overwrite of the first's;
if the latter, that's a `redundant-duplicate-write` or worse at the 1E1 level, not this
one, since this function itself performs no writes.

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.i_tf_sup` | read | explicit-arg (switch) | selects the whole branch; see "switches touched" |
| `.build.dr_shld_inboard` | read | explicit-arg | SUPERCONDUCTING branch only |
| `.build.dr_fw_inboard` | read | explicit-arg | ″ |
| `.build.dr_blkt_inboard` | read | explicit-arg | ″ |
| `.build.dr_shld_outboard` | read | explicit-arg | ″ |
| `.build.dr_fw_outboard` | read | explicit-arg | ″ |
| `.build.dr_blkt_outboard` | read | explicit-arg | ″ |
| `.tfcoil.dr_tf_wp_with_insulation` | read | explicit-arg | ″ |
| `.tfcoil.dx_tf_wp_insulation` | read | explicit-arg | ″ |
| `.physics.pflux_fw_neutron_mw` | read | explicit-arg | read 7x across the function, same immutable value each time |
| `.tfcoil.tfsai` | read | explicit-arg | ″ |
| `.tfcoil.tfsao` | read | explicit-arg | ″ |
| `.tfcoil.dr_tf_plasma_case` | read | explicit-arg | read 5x, same value each time |
| `.costs.f_t_plant_available` | read | explicit-arg | ″ |
| `.costs.life_plant` | read | explicit-arg | ″ |

No writes to `self.data` anywhere in this function — every output leaves via the return
tuple only. This is the cleanest unit audited so far: 15 scalar reads, all `explicit-arg`,
zero `implicit-io`, zero `implicit-io-via-callee`, zero `redundant-duplicate-write`. The
only impurity relative to a pure function is the `self.data.*` access pattern itself,
which is mechanical to remove.

## proposed signature(s)

Per `traceability_policy.md`'s split-default: `i_tf_sup`'s two branches have starkly
different reads-sets (resistive branch reads nothing and returns 10 zeros; superconducting
branch reads all 15 fields above) — this is independent confirmation of `switches.md`'s
existing `i_tf_sup` → **split** decision (already high-confidence there; adding this as a
second data point).

```python
def calculate_sc_tf_coil_nuclear_heating(
    dr_shld_inboard: float,
    dr_fw_inboard: float,
    dr_blkt_inboard: float,
    dr_shld_outboard: float,
    dr_fw_outboard: float,
    dr_blkt_outboard: float,
    dr_tf_wp_with_insulation: float,
    dx_tf_wp_insulation: float,
    pflux_fw_neutron_mw: float,
    tfsai: float,
    tfsao: float,
    dr_tf_plasma_case: float,
    f_t_plant_available: float,
    life_plant: float,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    # (coilhtmx, dpacop, htheci, flu_tf_neutron_fast_peak, pheci, pheco, ptfiwp,
    #  ptfowp, raddose, p_tf_nuclear_heat_mw)
    ...
```

The resistive (`i_tf_sup != SUPERCONDUCTING`) branch is a degenerate case worth flagging
as a candidate for **elimination rather than porting**: it takes no inputs and always
returns the same 10 zeros. In the graph, this isn't really a node at all — it's the
absence of one (the outputs simply don't exist / are definitionally zero when
`i_tf_sup` selects a non-superconducting coil). Worth deciding at the graph-assembly
level (per `naming_convention.md`'s "switches are not ports") rather than writing a
trivial all-zero function to port faithfully.

`ishmat` is hardcoded to `0` unconditionally (docstring: "stainless steel coil casing is
assumed"; `j = 2` / tungsten column of `coef`/`decay` is dead — never selected). Not a
switch, not a bug, just note: the ported function can drop the unused second column of
`coef`/`decay` entirely rather than carrying a always-`[..., 0]` index.

## cottax node

**Actually written**, in `stellarator_F_tf_nuclear_heating.py`
(`ScTfCoilNuclearHeating`, an `ExplicitFunction`), registered in
`functional_process/total_process.py`:

```python
class ScTfCoilNuclearHeating(ExplicitFunction):
    coilhtmx = OutputInto(fwbs)
    ...
    p_tf_nuclear_heat_mw = OutputInto(fwbs)

    def __call__(self, dr_shld_inboard=From(build), ..., life_plant=From(costs)):
        return calculate_sc_tf_coil_nuclear_heating(dr_shld_inboard, ..., life_plant)
```
Output `VarPath` areas are best-effort (`coilhtmx`/`dpacop`/`htheci`/`pheci`/`pheco`/
`ptfiwp`/`ptfowp`/`raddose` are never written to `self.data` by the source — they leave
only via the return tuple, per the data-footprint table above — so there is no existing
`data.<area>.<field>` to port; `.fwbs` is inferred from where the two fields that *are*
stored (`flu_tf_neutron_fast_peak`, `p_tf_nuclear_heat_mw`, both confirmed
`.fwbs.*` by chunk 1E1) land, on the assumption the rest of this function's outputs share
that area. Flagging for whoever audits 1E1/1E2 to confirm or correct once those chunks'
own `self.data.fwbs.*` assignments are cross-checked against this function's full return
tuple.

## tier signal
**Tier 1.** No internal solve, no calls to other models/methods, no loop. Straightforward
port once split by `i_tf_sup` as above.

## switches touched
- `i_tf_sup` (`.tfcoil.i_tf_sup`) — see `core/solver/switches.md`'s existing entry
  (already **split**, high confidence, based on evidence from `stellarator.py` lines 1022
  and 1724). This unit adds a third, particularly clean data point: reads-set here is
  "everything" vs. "nothing", not just "differs" — about as strong a split signal as this
  switch will produce anywhere.

## calls into other models
None. Self-contained.

## JAX-difficulty flags
None found. Plain scalar arithmetic (`np.exp`, static array indexing with constant index
`ishmat=0`) — all directly `jnp`-portable, no control flow on a traced value beyond the
already-classified `i_tf_sup` switch.

## open questions
- Whether the two call sites noted above (line 476 vs. 728) are mutually exclusive or a
  genuine double-write — belongs to chunk 1E1's audit, flagged there and here for
  cross-reference.
- Whether eliminating the resistive branch entirely (rather than porting an all-zero
  function) is the right call structurally, or whether downstream nodes expect these
  fields to always exist with a zero default when TF coil isn't superconducting — needs a
  read of what `.fwbs.p_tf_nuclear_heat_mw` etc. are used for elsewhere (out of this
  unit's scope).
