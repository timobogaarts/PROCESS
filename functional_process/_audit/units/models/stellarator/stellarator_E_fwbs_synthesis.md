---
kind: model-unit
status: draft
confidence: medium-high
---

## source

`process/models/stellarator/stellarator.py`, lines 422-1682: `blanket_neutronics()`
(422-480) + the whole of `st_fwbs()` (481-1682). This is **not a fourth chunk** — it is
the synthesis pass `_audit/next_steps.md` § 3 asked for, reading chunks 1E1
(`stellarator_E1_fwbs_setup.md`), 1E2 (`stellarator_E2_fwbs_neutronics.md`) and 1E3
(`stellarator_E3_fwbs_shield_divertor.md`) together against the real source rather than
against their own even-thirds line cuts. Those three records are **not superseded in their
data-footprint findings** (all confirmed correct on direct re-read) but their "one function
per chunk" framing is superseded by the re-chunking below. Each of the three now carries a
short pointer to this file; their `status` is left at `draft` for the human to re-triage.

## 1. Real function boundaries (vs. the arbitrary 1E1/1E2/1E3 thirds)

`st_fwbs` has six semantically distinct pieces, none of which land on the 880/1280 cuts:

| # | name | lines | shape |
|---|---|---|---|
| S1 | `fw_blanket_shield_geometry_setup` | 515-605 | unconditional per-call geometry/flux setup (areas, volumes, `wallpf`, `life_fw_fpy`, `pnucloss`). One `ipowerflow` branch (`a_blkt_total_surface` formula only). Tier 1. |
| S2 | `blanket_shield_tf_nuclear_power` | 422-480 (`blanket_neutronics`) + 608-1030 | the `blktmodel`×`ipowerflow` dispatch — three live arms (see § 4). Tier 3 in two of three arms (calls `hcpb.py`/chunk 1F). Produces the `sc_tf_coil_nuclear_heating_iter90` locals (`coilhtmx`, `dpacop`, `htheci`, `pheci`, `pheco`, `ptfiwp`, `ptfowp`, `raddose`) that survive into S6's output block. Contains the confirmed `p_div_rad_total_mw` bug (§ 6). |
| S3 | `divertor_mass_and_first_call_seed` | 1030-1043 | `first_call_stfwbs` check + `.divertor.a_div_surface_total`/`m_div_plate`. The one piece that is genuine cross-call state, not ordinary dataflow (§ 2). |
| S4 | `blanket_shield_fw_coolant_mass` | 1045-1274 (excl. 1030-1043) | blanket mass (`blktmodel`×`blkttype` dispatch), shield mass, FW mass (third independent `ipowerflow` site), total coolant mass (`blktmodel`/`i_blkt_coolant_type` dispatch). Tier 1, no cross-model calls, but consumes locals threaded from S2 (`f_a_fw_coolant_inboard`/`outboard`) and S3 (`coolvol`, seeded at 1048-1052 before S4 adds to it). |
| S5 | `cryostat_and_vv_geometry` | 1282-1330 | external cryostat + internal VV volumes/masses. **Fully self-contained** — reads only `.build.*`/`.physics.rmajor`/`.physics.rminor`/`.fwbs.fvoldw`/`.fwbs.den_steel`, none of which are written by S1-S4. No dependency on `blktmodel`/`ipowerflow`/any S2-S4 local. Matches 1E3's `calculate_cryostat_and_vv` proposal essentially as written. |
| S6 | `st_fwbs_output` | 1331-1682 | `if output:` reporting shell (88% of the method's tail, per 1E3). References S2's `coilhtmx`-family locals (guarded by the exact same `ipowerflow==0 and blktmodel==0` condition that produced them, see § 6.5 — safe, not a bug) and S4's mass outputs. Two small non-inert inline computations already flagged by 1E3 (`r_cryostat_inboard - 2*adewex`, `dewmkg - m_vv`). |

The 880 cut falls inside S2's third arm (`blktmodel!=1`/`ipowerflow==1`, 730-1030), exactly
as 1E1/1E2 both independently suspected. The 1280 cut, by contrast, **is** a real seam —
it falls in the two-line gap between S4 ending (1274) and S5 starting (1282), which is why
1E2/1E3 found "nothing dangling" there. One accidental correct cut, one real one.

## 2. Cross-boundary state — full ledger

| local/state | created | consumed | verdict |
|---|---|---|---|
| `sc_tf_coil_nuclear_heating_iter90()` outputs (`coilhtmx`, `dpacop`, `htheci`, `pheci`, `pheco`, `ptfiwp`, `ptfowp`, `raddose`) | S2, only in the `blktmodel!=1 & ipowerflow==0` arm (716-728) | S6's output block (1438-1474), gated by the identical `ipowerflow==0 and blktmodel==0` condition (1434-1436) | **Ordinary `In`/`Out` port, not hidden state.** Verified by direct read: the output block's guard exactly reproduces the arm's own condition, so there is no path where these names are referenced unassigned — it just looks alarming because ~700 lines of unrelated code (S3/S4) sit textually between assignment and use. A resynthesized S2 node simply returns these 8 values; S6 (or whatever composes S2+S6) takes them as ordinary arguments. `flu_tf_neutron_fast_peak` and `p_tf_nuclear_heat_mw` (positions 4 and 10) *are* written to `.fwbs.*` in this arm and read elsewhere too — only the other 8 stay Python-local. |
| `f_a_fw_coolant_inboard`, `f_a_fw_coolant_outboard` | S2, in **either** the `blktmodel==1&ipowerflow==1` arm (676-678, formula A: `1 - fblbe - fblbreed - fblss`) **or** the `blktmodel!=1&ipowerflow==1` arm (828-837, formula B: `radius_fw_channel²/bfw²`) — mutually exclusive, both under `ipowerflow==1` | S4's FW-mass `else` branch (1229-1246), gated by `ipowerflow != 0` at line 1208 | **Ordinary `In` port**, but with a wrinkle: two different producers, same names, disjoint formulas, selected by the same `blktmodel` switch that already splits S2. A pure port for S4 must accept these as plain float arguments; *which* formula feeds them is decided one level up in S2, not inside S4. Confirmed safe (no unassigned-reference path) because S4's consuming branch (`ipowerflow==1`) is only reachable when one of the two producing sub-arms (both also `ipowerflow==1`) has already run in the same call. |
| `coolvol` | S3 (1048-1052, seeded from `.divertor.a_div_surface_total`) | accumulated across S4 (1191-1192, 1202, 1222-1226 or 1238-1246) | **Ordinary local accumulator threaded S3→S4** — a ladder of `+=`, no branching risk. `local-intermediate` in schema terms, just spanning further than any one chunk. |
| `pnucfwbsi`/`pnucfwbso`, `p_fw_inboard/outboard_nuclear_heat_mw`, `psurffwi/o`, `decayfwi/o` (1E1's flagged list) | S2's `ipowerflow==1` arm, 747-856 | consumed later in the **same arm** (868-923), i.e. entirely inside S2, not actually crossing into S3/S4 | 1E1's original worry ("1E2's audit needs to treat these as inherited inputs") is **resolved as a non-issue** now that the real boundary (S2, ending at 1030) is known — these never left S2's territory in the first place; they only looked cross-chunk because the arbitrary 880 cut sliced through the middle of S2. |
| `.divertor.a_div_surface_total` / `self.first_call_stfwbs` | S3 (1034-1036), and `stdiv`/`Divertor` (unit #4, called via `st_div(self, ...)` in `run()`) | S3 itself (`m_div_plate`, 1038-1043) and S4 (`coolvol` seed, 1048-1052) | **Genuine cross-call state — see § 2.1, not resolvable as ordinary intra-call dataflow.** |

### 2.1 `first_call_stfwbs`: real call-order state, and what it actually means

Traced `self.first_call_stfwbs` to its only two touch points: set `True` once in
`Stellarator.__init__` (line 95), read-and-cleared at 1034-1036. Traced `run()`'s call
order (lines 156-164, the non-`output` path — the one that runs every solver iteration):

```
st_strc → st_fwbs → st_div
```

`st_fwbs` runs **before** `st_div` in every ordinary call. The source comment at 1030-1032
confirms why: "`self.data.divertor.a_div_surface_total` is calculated in stdiv after this
point, so will be zero on first lap, hence the initial approximation." So:

- **First call ever** (object construction to first `run()`): `first_call_stfwbs` is
  `True`, `a_div_surface_total` doesn't exist yet meaningfully, and S3 substitutes a
  hardcoded `50.0` bootstrap value, then permanently clears the flag.
- **Every subsequent call, forever**: `first_call_stfwbs` is `False`, so S3 reads whatever
  `.divertor.a_div_surface_total` currently holds — which is the value `st_div` wrote
  **during the previous `run()` invocation**, not this one (`st_div` hasn't run yet this
  cycle when S3 executes).

This is a genuine **one-call-lagged fixed point** between `st_fwbs` and `st_div`
(`Divertor`, unit #4 — already confirmed by `unit_registry.md` row 4 to unconditionally
produce `a_div_surface_total`), not intra-call implicit-io and not resolvable by treating
`first_call_stfwbs` as an ordinary boolean `In` port of one node. It maps cleanly onto
cottax vocabulary, though: `st_fwbs`'s S3 piece and `Divertor` form a two-node SCC exactly
like the `ipowerflow` SCC `next_steps.md` § 1 already found between `AFwTotalWithPowerflow`
and `Divertor`. `first_call_stfwbs` **is** the driver's "is this the first iterate" test,
and `50.0` **is** the fixed point's initial guess — both belong to a `Blocking` + driven
`FixedPoint`/`Square` over `{S3, Divertor}`, not to either node's own signature. Whether
that fixed point needs more than one round in practice (PROCESS's own idempotence loop
allows up to 10) is unmeasured — flagging for whoever drives the port, not resolved here.

One caveat: `run()`'s `output=True` path (125-146) calls `st_div` **before**
`self.st_fwbs(True)` (142 vs. 146) — the opposite order from the normal path. By the time
`output()` is ever called, `first_call_stfwbs` has already been permanently cleared by
many prior normal-path calls, so this reordering has no observable effect on `st_fwbs`
itself; it does mean the *reporting* run always sees a same-cycle-fresh
`a_div_surface_total` rather than a lagged one. Noted for completeness, not a discrepancy
requiring action.

## 3. Reconciling the two switch-discovery discrepancies

**Verdict: 1E2's `first_call_stfwbs` finding is real and is the only genuine cross-call
mechanism in this method. 1E1's `f_p_blkt_multiplication` worry is a false alarm, resolved
directly from source — it is an ordinary `InputVariable`, not stale state, and it has
nothing to do with `first_call_stfwbs`.**

Traced `.fwbs.f_p_blkt_multiplication` fully:
- `process/core/input.py:391`: `InputVariable("fwbs", float, range=(1.0, 2.0))` — this is
  a real IN.DAT-settable input, not a pure output.
- `process/data_structure/fwbs_variables.py:31`: default `1.269`.
- `blanket_neutronics()` (line 461, only called under `blktmodel==1`) **overwrites** it
  unconditionally with the literal `1.269` — i.e., the exact same value as the class
  default — before using it downstream in the same arm (S2, 921-949 continuation... no,
  correction: that usage is in the *other* arm, see below).
- The `blktmodel!=1` arm (line 690) reads it **without** `blanket_neutronics()` ever having
  run this run (since `blktmodel` is a switch, constant for the entire run — confirmed
  `choices=[0, 1]` at `input.py:978` — so if `blktmodel==0`, `blanket_neutronics()` is
  never called at all, not "not called yet"). This read gets whatever IN.DAT (or the
  `1.269` default) supplied — an ordinary `explicit-arg` read of an input field, no
  different from any other material-fraction constant in this file.

So the two chunks found two *different*, unrelated phenomena that happened to look
similar under the same "read without a same-run write" symptom: `f_p_blkt_multiplication`
is fully explained by ordinary input-variable semantics (`blktmodel` gates whether a
*hardcoded constant matching the input default* is force-written, not whether the field
has a value at all); `a_div_surface_total`/`first_call_stfwbs` is the one place in
`st_fwbs` with a real inter-call dependency. **1E1's open question 1 should be closed as
resolved-not-a-bug**; 1E2's open question 1 stands as the correct and only finding of its
kind.

## 4. `blktmodel` and `blkttype` arms, now that the boundaries are known

`blktmodel` (`choices=[0, 1]`, default `0`) is checked with **asymmetric spellings** at two
sites — `== 1` at line 608 (S2) and `== 0` at line 1056 (S4) — which looked like it might
leave a silent third case for some `blktmodel` value that is neither 1 nor 0. Confirmed
from `input.py:978` that the domain is exactly `{0, 1}`, so `==1`/`else` and `==0`/`else`
partition the same two-element set identically — no gap, just inconsistent spelling within
one file. Not a bug; worth a one-line note in the port so the two spellings aren't
mistaken for different conditions.

- **`blktmodel == 1` arm** (S2: 608-678, plus `blanket_neutronics()` 422-480): reads —
  `.fwbs.breedmat`, `.fwbs.vol_blkt_total`, TF coil geometry (`.tfcoil.len_tf_coil`,
  `.tfcoil.a_tf_inboard_total`, `.tfcoil.a_tf_leg_outboard`, `.tfcoil.n_tf_coils`), plus
  everything `self.hcpb.nuclear_heating_{blanket,magnets,shield}()` reads (unit #13, not
  yet audited — this arm's reads-set is not fully known until then). Writes —
  `.fwbs.breeder`, `.densbreed`, `.m_blkt_total`, `.ptfnucpm3`,
  `.f_p_blkt_multiplication` (unconditional overwrite, see § 3),
  `.flu_tf_neutron_fast_peak`, and (only under the nested `ipowerflow==1`, no `else`)
  `.p_div_nuclear_heat_total_mw`, `.p_fw_hcd_nuclear_heat_mw`,
  `.p_fw_nuclear_heat_total_mw`, `.pradloss`, `.p_div_rad_total_mw` (**correctly**
  computed here, unlike the other arm, see § 6), `.p_fw_hcd_rad_total_mw`,
  `.p_fw_rad_total_mw`, four `heat_transport.p_*_coolant_pump_mw` fields. `ipowerflow==0`
  under `blktmodel==1` writes almost nothing beyond `blanket_neutronics()`'s own outputs —
  a real "quiet" combination, confirmed present, not resolved further here (would need
  unit #13's audit to know if `hcpb`'s calls have side effects that matter for this
  combination). Masses (S4, `else` branch 1093-1146): reads six `bl{u,m,p}i/oth` thickness
  fields plus `fblss`/`fblhebmi/mo/pi/po`/`fblbe`/`densbreed`/`fblbreed`, writes
  `.m_blkt_steel_total`, `.m_blkt_beryllium`, `.whtblbreed`, `.m_blkt_total`,
  `.f_a_blkt_cooling_channels` — **independent of `blkttype`**, unlike the `==0` arm.
- **`blktmodel == 0` arm**: self-contained "old model" for powers (S2, 684-1030) — no
  `hcpb` calls, exponential-attenuation formulas, reads `.fwbs.f_p_blkt_multiplication`
  as a plain input (§ 3) plus material fractions; under `ipowerflow==0` calls
  `sc_tf_coil_nuclear_heating_iter90()` (chunk 1F) with all 10 return values used; under
  `ipowerflow==1` is entirely local arithmetic plus the CoolProp/`irefprop` branch (already
  in `switches.md`) and contains the confirmed bug (§ 6). Masses (S4, `if` branch
  1056-1092): reads `.fwbs.blkttype` — **this is where `blkttype` actually matters**,
  `blktmodel==1` masses never look at it.

`blkttype` (`choices=[1, 2, 3]`, default `3`) has exactly the one computational site
`switches.md` already found (`stellarator.py:1057`), now located precisely inside S4 and
confirmed nested under `blktmodel == 0`:
- `blkttype in {1, 2}` (liquid breeder, WCLL/HCLL): writes `.fwbs.wtbllipb`,
  `.fwbs.m_blkt_lithium`, `.fwbs.m_blkt_total` (partial, +steel/vanadium added after,
  1078-1091) from `.fwbs.vol_blkt_total`, `.fwbs.fbllipb`, `.fwbs.fblli`.
- `blkttype == 3` (else, solid breeder HCPB): writes `.fwbs.m_blkt_li2o`,
  `.fwbs.m_blkt_beryllium`, `.fwbs.m_blkt_total` from `.fwbs.fblli2o`, `.fwbs.fblbe`.
  Both arms then share the steel/vanadium addition (1078-1091).

This confirms `switches.md`'s "three-values-two-arms" characterization exactly; no new
information changes that entry's `Alternative`-shape problem. What *is* new: the arm now
has a known, small, tier-1, self-contained (no cross-model calls) home — S4 — so once S2's
signature is settled (blocked on unit #13/hcpb.py) and S3's fixed-point question is
decided, `blkttype`'s two arms are otherwise ready to implement as soon as the
`Alternative`/predicate-keyed-arm mechanism exists.

## 5. Recommended re-chunking

Retire the even-thirds 1E1/1E2/1E3 split as the port unit boundary (their audit content
stays valid and is cross-referenced, not thrown away). Re-chunk `st_fwbs` +
`blanket_neutronics` into the six pieces of § 1:

- **Portable now, tier 1, no blockers**: **S5** (`cryostat_and_vv_geometry`,
  1282-1330). Fully self-contained; recommend porting this first, exactly as 1E3 already
  proposed (`calculate_cryostat_and_vv`).
- **Portable once unit #13 (`hcpb.py`) is audited**: **S2** (`blanket_shield_tf_nuclear_power`,
  422-480 + 608-1030). Tier 3 in two of its three arms; the `blktmodel!=1 & ipowerflow==1`
  arm is otherwise tier 1 but should be fixed (or at least explicitly ported *with* the bug
  reproduced and flagged, per this project's "reproduce PROCESS's existing behavior first"
  stance) for the `p_div_rad_total_mw` issue (§ 6), and is gated by the `irefprop`
  CoolProp blocker already on record.
- **Not portable as an ordinary node — needs a `Blocking`+`FixedPoint` over
  `{S3, Divertor}`**: **S3** (`divertor_mass_and_first_call_seed`, 1030-1043). Small (14
  lines) but structurally the hardest piece: real inter-call state, a real two-node SCC
  with unit #4. Recommend this be the concrete worked example for how `next_steps.md`'s
  "most of PROCESS is probably not actually cyclic once made explicit" claim looks when a
  cycle *is* found — it's tiny, and now precisely bounded.
- **Portable once S2 and S3's signatures exist** (needs their outputs as `In` arguments):
  **S4** (`blanket_shield_fw_coolant_mass`, 1045-1274 excl. 1030-1043). Tier 1 internally,
  no cross-model calls of its own — purely blocked on its two producers' signatures
  settling, not on any further audit work. This is also where `blktmodel`'s and
  `blkttype`'s mass-side arms live (§ 4) — both become straightforward `Alternative`
  arms once S4 itself is ported.
- **Portable now, tier 1, no blockers**: **S1** (`fw_blanket_shield_geometry_setup`,
  515-605). Nothing here depends on S2-S5; only produces inputs consumed by S2/S4.
  Recommend porting alongside S5 as the second immediately-actionable piece.
- **Out of port scope**: **S6** (`st_fwbs_output`, 1331-1682), same treatment as
  `stellarator_G_output.md` and `coils/output.md` — reporting shell, not ported, two small
  non-inert inline computations already flagged by 1E3 kept on record.

Net: of six pieces, **two (S1, S5) are portable today**, **one (S4) is blocked only on
S2/S3's signatures**, **one (S2) is blocked on unit #13's audit**, and **one (S3) needs a
genuine SCC/`FixedPoint` treatment**, not a blocker in the "more audit needed" sense but a
different node shape than anything else in `st_fwbs`.

## 6. Latent bugs — confirmed against full source

**`.fwbs.p_div_rad_total_mw` read-but-never-assigned: confirmed real, in the
`blktmodel!=1 & ipowerflow==1` arm only (S2, 730-1030).** Direct read of lines 768-780:

```python
# Radiation power incident on divertor (MW)
self.data.fwbs.p_fw_hcd_rad_total_mw = (
    self.data.physics.p_plasma_rad_mw * self.data.fwbs.f_a_fw_outboard_hcd
)
# Radiation power incident on HCD apparatus (MW)
self.data.fwbs.p_fw_hcd_rad_total_mw = (
    self.data.physics.p_plasma_rad_mw * self.data.fwbs.f_a_fw_outboard_hcd
)
```

Both statements write the identical expression to the identical field
(`p_fw_hcd_rad_total_mw`) — a `redundant-duplicate-write` — under two different comments,
the first of which (correctly, per the pattern the `blktmodel==1` arm actually uses at
line 630-632: `p_plasma_rad_mw * f_ster_div_single`) should have computed
`p_div_rad_total_mw` instead. `p_div_rad_total_mw` is then read three lines later (792)
in `p_fw_rad_total_mw`'s formula, and is **never written anywhere in this arm**.
`fwbs_variables.py:373` confirms its class default is `0.0` and it is not an
`InputVariable` (not in `core/input.py`), and since `blktmodel` is a run-constant switch,
a run with `blktmodel != 1` never calls `blanket_neutronics()` (the only other writer) at
all — so the value is **deterministically `0.0`** for the lifetime of any such run, not
merely "stale from a previous call." `p_fw_rad_total_mw` in this arm is therefore missing
its `p_div_rad_total_mw` subtraction and is wrong by exactly that (nonzero, in the
`blktmodel==1` arm's parallel formula) amount whenever `blktmodel != 1` and
`ipowerflow == 1`. Flagged, not fixed, per audit scope.
