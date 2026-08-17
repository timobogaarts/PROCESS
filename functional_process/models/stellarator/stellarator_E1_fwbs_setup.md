---
kind: model-unit
status: draft
confidence: medium
---

## source
`process/models/stellarator/stellarator.py`, lines 422-880: `blanket_neutronics()`
(422-480) + first third of `st_fwbs()` (481-880, an even-thirds line cut, not a semantic
boundary). Chunk 1E1 of unit #1.

**The 880 cut falls mid-computation, not cleanly.** Confirmed by reading past the
boundary: `pnucbsi`/`pnucbso` (computed 877-878, just before the cut) and
`p_fw_inboard_nuclear_heat_mw`/`p_fw_outboard_nuclear_heat_mw`/`psurffwi`/`psurffwo`
(computed ~847-878) are all local Python variables consumed at 887-924, past line 880.
The entire `else: # blktmodel != 1` → `else: # ipowerflow == 1` branch (roughly 730-1150+,
not yet fully bounded) is one continuous local-variable dataflow chain that chunk 1E2
picks up mid-stream. 1E2's audit needs to treat lines 730-880's local variables
(`pnucfwbsi`, `pnucfwbso`, `p_fw_inboard_nuclear_heat_mw`, `p_fw_outboard_nuclear_heat_mw`,
`psurffwi`, `psurffwo`, `f_a_fw_coolant_inboard`, `f_a_fw_coolant_outboard`, `decayfwi`,
`decayfwo`) as inherited inputs, not derive them fresh.

## data footprint

### `blanket_neutronics()` (422-480)
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.fwbs.breedmat` | read | explicit-arg | material selector (1/2/else), plain input |
| `.fwbs.vol_blkt_total` | read | implicit-io | written by `st_fwbs` itself, lines 567-569, earlier in the same call, before `blanket_neutronics()` is invoked at line 609 |
| `.fwbs.breeder`, `.fwbs.densbreed` | write | — | set from the `breedmat` branch |
| `.fwbs.m_blkt_total` | write | explicit-arg (derived) | `vol_blkt_total * densbreed` |
| `.tfcoil.len_tf_coil`, `.tfcoil.a_tf_inboard_total`, `.tfcoil.a_tf_leg_outboard`, `.tfcoil.n_tf_coils` | read | explicit-arg | TF coil geometry, produced by an earlier-run model, not by this call |
| `.fwbs.p_tf_nuclear_heat_mw` | read | implicit-io | written by `self.hcpb.nuclear_heating_magnets(False)` (line 443) immediately before this read (line 455) — crosses into `hcpb.py` (unit #13), same call frame, real shared `data` (not a copy) so plain `implicit-io` per the policy's "by this unit or one it calls" clause, not `implicit-io-via-callee` (that label is for the deepcopy/proxy pattern specifically, see `density_limits.md`) |
| `.fwbs.ptfnucpm3` | write | explicit-arg (derived) | `p_tf_nuclear_heat_mw / tf_volume` |
| `.fwbs.f_p_blkt_multiplication` | write | — | hardcoded constant `1.269`, no read dependency — **see open question 1, this write only happens when `blktmodel==1`** |
| `.fwbs.flu_tf_neutron_fast_peak` | write | — | 4th of 10 values unpacked from `self.sc_tf_coil_nuclear_heating_iter90()` (chunk 1F), rest discarded (`_`) |

**Calls**: `self.hcpb.nuclear_heating_blanket()` (440), `self.hcpb.nuclear_heating_magnets(False)`
(443), `self.hcpb.nuclear_heating_shield()` (458) — all real cross-model calls (unit #13,
`hcpb.py`, in-scope, not yet audited); `self.sc_tf_coil_nuclear_heating_iter90()` (476) —
own-class method, chunk 1F.

### `st_fwbs()`, unconditional geometry setup (515-605)
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.costs.abktflnc`, `.physics.pflux_fw_neutron_mw`, `.costs.life_plant` | read | explicit-arg | → `.fwbs.life_fw_fpy` |
| `.first_wall.a_fw_total` | read | explicit-arg | → `.first_wall.a_fw_inboard`/`a_fw_outboard` (each half) |
| `.physics.rminor`, `.build.dr_fw_plasma_gap_inboard`, `.build.dr_fw_inboard`, `.build.dr_fw_plasma_gap_outboard`, `.build.dr_fw_outboard` | read | explicit-arg | local `r1` |
| `.heat_transport.ipowerflow` | read | **switch, not one of the original 10** | branches the `a_blkt_total_surface` formula — `==0` reads `.physics.a_plasma_surface`, `.fwbs.fhole`; else additionally reads `.fwbs.f_ster_div_single`, `.fwbs.f_a_fw_outboard_hcd`. **Differing reads-set — split candidate.** Same switch already flagged as entangled with `i_pflux_fw_neutron` in `switches.md`; recommend promoting to its own audit row rather than leaving it as an incidental find twice. |
| `.build.a_blkt_total_surface` | write, then read | implicit-io | written above, read 2 lines later for `a_blkt_inboard_surface`/`a_blkt_outboard_surface` |
| `.build.dr_blkt_inboard`, `.build.dr_blkt_outboard` | read | explicit-arg | → `.fwbs.vol_blkt_inboard`/`vol_blkt_outboard`/`vol_blkt_total` (sum) |
| `.fwbs.fvolsi`, `.fwbs.fvolso` | read | explicit-arg | → `.build.a_shld_inboard_surface`/`a_shld_outboard_surface` |
| `.build.dr_shld_inboard`, `.build.dr_shld_outboard` | read | explicit-arg | → local `vol_shld_inboard`/`vol_shld_outboard` → `.fwbs.vol_shld_total` |
| `.physics.p_neutron_total_mw`, `.fwbs.fhole` | read | explicit-arg | → `.fwbs.pnucloss` |
| `.stellarator_config.stella_config_neutron_peakfactor` | read | explicit-arg | → `.fwbs.wallpf`; ties to unit #8 (`preset_config.py`) — this is the first concrete read from the `stellarator_config` namespace seen in this audit |

### `st_fwbs()`, `blktmodel == 1` branch (608-678, wholly inside this chunk)
`.fwbs.blktmodel` is **a second switch not in the original 10**, and the single biggest
topology fork found in this chunk — see open question 2. Branch calls `blanket_neutronics()`
(above), then if `.heat_transport.ipowerflow == 1` (no `else`, so `ipowerflow == 0` under
`blktmodel == 1` computes almost nothing further — worth noting as a real "quiet" combination):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.p_neutron_total_mw`, `.fwbs.f_ster_div_single`, `.fwbs.f_a_fw_outboard_hcd` | read | explicit-arg | → `.fwbs.p_div_nuclear_heat_total_mw`, `.fwbs.p_fw_hcd_nuclear_heat_mw` |
| `.fwbs.pnucloss` | read | implicit-io | written above, same call |
| `.fwbs.p_div_nuclear_heat_total_mw`, `.fwbs.p_fw_hcd_nuclear_heat_mw` | write, then read | implicit-io | written 2 lines above, read immediately for `.fwbs.p_fw_nuclear_heat_total_mw` |
| `.physics.p_plasma_rad_mw`, `.fwbs.fhole` | read | explicit-arg | → `.fwbs.pradloss`, `.fwbs.p_div_rad_total_mw`, `.fwbs.p_fw_hcd_rad_total_mw`, `.fwbs.p_fw_rad_total_mw` (all implicit-io chained off each other within this block) |
| `.fwbs.p_blkt_nuclear_heat_total_mw` | read | implicit-io | **not written anywhere in 609-678** — must come from `self.hcpb.nuclear_heating_blanket()` inside `blanket_neutronics()`, i.e. crosses into unit #13, same as `p_tf_nuclear_heat_mw` above |
| `.fwbs.p_shld_nuclear_heat_mw` | read | implicit-io | same — from `self.hcpb.nuclear_heating_shield()` |
| `.heat_transport.f_p_fw_coolant_pump_total_heat`, `f_p_blkt_coolant_pump_total_heat`, `f_p_shld_coolant_pump_total_heat`, `f_p_div_coolant_pump_total_heat` | read | explicit-arg | four independent scaling fractions |
| `.current_drive.p_beam_orbit_loss_mw` | read | explicit-arg | |
| `.physics.p_plasma_separatrix_mw` | read | explicit-arg | |
| `.heat_transport.p_fw_coolant_pump_mw`, `p_blkt_coolant_pump_mw`, `p_shld_coolant_pump_mw`, `p_div_coolant_pump_mw` | write | — | four pumping-power outputs |
| `.fwbs.fblbe`, `.fwbs.fblbreed`, `.fwbs.fblss` | read | explicit-arg | → local `f_a_fw_coolant_inboard`/`outboard` (Python locals, not written to `data`) |

### `st_fwbs()`, `blktmodel != 1` branch (680-880, continues past the cut)
`.fwbs.pnuc_cp = 0.0` (write, constant). Branches again on `.heat_transport.ipowerflow`
(the same switch as above, confirming it's read/branched independently in at least two
places in this file).

**`ipowerflow == 0` sub-branch (684-728), wholly inside this chunk**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.p_neutron_total_mw` | read | explicit-arg | |
| `.fwbs.pnucloss`, `.fwbs.pnuc_cp` | read | implicit-io | both written earlier, same call |
| `.fwbs.f_p_blkt_multiplication` | read | **implicit-io — but see open question 1** | this field is only ever *written* inside `blanket_neutronics()` (line 461), which only runs when `blktmodel == 1`. Here `blktmodel != 1`, so `blanket_neutronics()` was never called this run — this read gets whatever value the field held *before this call*, not something this call produced. That's not ordinary implicit-io (same-call dependency), it's a cross-call/stale-state dependency the current four classifications don't cleanly name — flagging as its own open question rather than picking a label that doesn't fit. |
| `.fwbs.f_a_blkt_cooling_channels`, `.fwbs.fblli2o`, `.fwbs.fblbe` | read | explicit-arg | local `decaybl` |
| `.build.dr_blkt_outboard` | read | explicit-arg | → `.fwbs.p_blkt_nuclear_heat_total_mw` |
| `.fwbs.p_blkt_nuclear_heat_total_mw` | write, then read | implicit-io | → `.fwbs.p_shld_nuclear_heat_mw` two lines later |
| `.fwbs.flu_tf_neutron_fast_peak`, `.fwbs.p_tf_nuclear_heat_mw` | write | — | this time genuinely from `self.sc_tf_coil_nuclear_heating_iter90()`'s own return tuple (positions 4 and 10 of 10, **all 10 values used here**, unlike `blanket_neutronics()`'s call to the same method which discards 9 of 10 — see open question 3) |

**`ipowerflow == 1` sub-branch (730-880+), continues past the cut into chunk 1E2** — see
the boundary note above and open question 4 below for a likely source bug found in this
range.

## proposed signature(s)

Tier-1, complete, no cross-model calls:
```python
def calculate_fw_blanket_shield_geometry(
    abktflnc: float,
    pflux_fw_neutron_mw: float,
    life_plant: float,
    a_fw_total: float,
    rminor: float,
    dr_fw_plasma_gap_inboard: float,
    dr_fw_inboard: float,
    dr_fw_plasma_gap_outboard: float,
    dr_fw_outboard: float,
    ipowerflow: int,  # static — see open question re: promoting to its own switch row
    a_plasma_surface: float,
    fhole: float,
    f_ster_div_single: float,
    f_a_fw_outboard_hcd: float,  # only used if ipowerflow != 0
    dr_blkt_inboard: float,
    dr_blkt_outboard: float,
    fvolsi: float,
    fvolso: float,
    dr_shld_inboard: float,
    dr_shld_outboard: float,
    p_neutron_total_mw: float,
    stella_config_neutron_peakfactor: float,
) -> tuple[float, ...]:  # life_fw_fpy, a_fw_inboard, a_fw_outboard, vol_blkt_total,
    # vol_shld_total, pnucloss, wallpf, ...
    ...
```

Tier-3 (composes cross-model calls, cannot be finalised independent of `hcpb.py` unit #13
and chunk 1F):
```python
def calculate_blanket_neutronics(
    breedmat: int,
    vol_blkt_total: float,
    len_tf_coil: float,
    a_tf_inboard_total: float,
    a_tf_leg_outboard: float,
    n_tf_coils: float,
    # + whatever nuclear_heating_blanket/magnets/shield need — TBD, unit #13
    # + whatever sc_tf_coil_nuclear_heating_iter90 needs — TBD, chunk 1F
) -> tuple[float, ...]:  # breeder, densbreed, m_blkt_total, p_tf_nuclear_heat_mw,
    # ptfnucpm3, f_p_blkt_multiplication, flu_tf_neutron_fast_peak
    ...
```

`blktmodel == 1` / `ipowerflow == 1` and `blktmodel != 1` / `ipowerflow == 0` branches:
provisional signatures possible (both complete within this chunk) but not written out here
in full — same shape as above, reads-set differs sharply between the two (confirmed
different reads-sets → both are split candidates per policy default). Deferred to the
consolidation pass once all of 1E1-1E3 are in, so the two switches' full split decision is
made once, not per-chunk.

`blktmodel != 1` / `ipowerflow == 1`: **blocked**, continues into chunk 1E2.

## tier signal
- Geometry setup (515-605): **tier 1**.
- `blanket_neutronics()`: **tier 3** — calls `self.hcpb.*` (unit #13) and
  `self.sc_tf_coil_nuclear_heating_iter90()` (chunk 1F).
- `blktmodel == 1` branch: **tier 3** (via `blanket_neutronics()`).
- `blktmodel != 1` / `ipowerflow == 0` branch: **tier 3** (via
  `sc_tf_coil_nuclear_heating_iter90()`).
- `blktmodel != 1` / `ipowerflow == 1` branch: tier 1 so far within this chunk (no
  cross-model calls seen yet in 730-880), but incomplete — see boundary note.

## switches touched
- `.heat_transport.ipowerflow` — **not one of the original 10**, found here. Already
  showed up as an entangled switch in `switches.md`'s `i_pflux_fw_neutron` entry; this
  chunk gives it two more independent branch sites with clearly differing reads-sets.
  Recommend promoting it to its own row in `switches.md` rather than leaving it scattered
  across other switches' entangled-switch notes.
- `.fwbs.blktmodel` — **not one of the original 10**, found here. The largest topology
  fork in this chunk: `blktmodel == 1` routes through `hcpb.py`'s nuclear-heating methods
  entirely; `blktmodel != 1` uses a self-contained "old model" with no `hcpb` calls at
  all. Reads-sets are almost disjoint. Strong split candidate, also recommend its own row.
- `.fwbs.i_blkt_coolant_type` and `.fwbs.irefprop` — read at line 803-812, inside the
  region this chunk covers but past 730 (the part continuing into 1E2's territory
  numerically, though the branch itself starts and is read within 481-880's nominal range
  at line 803). Already has a full entry in `switches.md` (medium confidence, "unsure,
  leaning keep-static") from the pilot batch — not re-audited here, just confirming the
  site is real and matches that entry's description (CoolProp-vs-polynomial for
  `temp_blkt_coolant_out`).

## calls into other models
`self.hcpb.nuclear_heating_blanket()`, `.nuclear_heating_magnets(False)`,
`.nuclear_heating_shield()` (all via `blanket_neutronics()`) — unit #13, `hcpb.py`, in
scope, not yet audited. `self.sc_tf_coil_nuclear_heating_iter90()` — chunk 1F, not yet
audited, called from two different places with different unpacking (see open question 3).

## JAX-difficulty flags
- **CoolProp call** (`FluidProperties.of(...)`, line ~806, gated by `irefprop`) —
  `non-traceable-external-call`, `blocker`. Already known from `switches.md`; confirmed
  present in this chunk's territory.
- **Enum-construction validation** (`PumpingPowerModelTypes(self.data.fwbs.i_p_coolant_pumping)`
  at line 901, just past this chunk's nominal 880 boundary but visible while checking
  continuity) raises on an invalid value via `IntEnum` construction rather than an
  explicit `if`. Same category as the ProcessValueError pattern found in
  `density_limits.md` — `minor` severity, belongs to switch/input validation (checked
  once at graph-build/input-parse time), not something to compile into a traced function.

## open questions

1. **Possible stale-state read, not ordinary implicit-io**: `.fwbs.f_p_blkt_multiplication`
   is read at line 690 (`blktmodel != 1` branch) but is *only ever written* inside
   `blanket_neutronics()` (line 461, hardcoded `1.269`), which only runs when
   `blktmodel == 1`. When `blktmodel != 1`, this read's value is whatever the field held
   from a *previous* call — not derivable from this call's inputs at all. This is a purity
   violation the pure port cannot silently carry forward: either (a) `1.269` is meant to be
   a genuine constant regardless of `blktmodel` and should be inlined/passed explicitly in
   both branches, or (b) the `blktmodel != 1` path is relying on leftover state from a
   prior `blktmodel == 1` run in the same process lifetime, which would be a real
   behavioural dependency on call history. Needs your read — I can't tell which from the
   code alone, and it changes the port's signature either way.

2. **`.fwbs.blktmodel` is a second undiscovered major switch**, on top of `ipowerflow`.
   Together they produce (at least) four combinations in this file alone
   (`blktmodel==1`×`ipowerflow∈{0,1}`, `blktmodel!=1`×`ipowerflow∈{0,1}`), three of which
   are audited here (the fourth, `blktmodel!=1`/`ipowerflow==1`, continues into 1E2).
   Recommend both get their own rows in the switch registry rather than treating them as
   local-only findings — they very likely fan out into `hcpb.py`, `blanket_library.py` and
   other files not yet in scope, similar to how `i_blkt_coolant_type` did.

3. **`sc_tf_coil_nuclear_heating_iter90()` is called from two different branches with
   asymmetric unpacking** — `blanket_neutronics()` (under `blktmodel==1`) keeps only the
   4th of 10 return values; the `blktmodel!=1`/`ipowerflow==0` branch keeps all 10. Since
   *both* branches call it regardless of which value `blktmodel` takes, this suggests the
   call itself may not actually depend on `blktmodel` at all — only what's done with its
   outputs differs. Worth considering, once chunk 1F is audited, whether this should be
   hoisted out as one unconditional node rather than duplicated inside both branches of the
   `blktmodel` switch.

4. **Likely latent bug**: lines 768-773 (comment: "Radiation power incident on divertor
   (MW)") compute `.fwbs.p_fw_hcd_rad_total_mw = physics.p_plasma_rad_mw *
   fwbs.f_a_fw_outboard_hcd` — the **HCD** formula, under a comment claiming **divertor**.
   Six lines later (777-780, comment: "Radiation power incident on HCD apparatus (MW)"),
   the *identical* expression is written to the *same* field again — a
   `redundant-duplicate-write` in the schema's sense, numerically harmless since both
   writes are identical. But `.fwbs.p_div_rad_total_mw` — read three lines later at 792
   (`p_fw_rad_total_mw = p_plasma_rad_mw - p_div_rad_total_mw - pradloss -
   p_fw_hcd_rad_total_mw`) — is **never assigned anywhere in this branch** (480-880). The
   neutron-power equivalent, `p_div_nuclear_heat_total_mw`, *is* correctly computed a few
   lines earlier (733-736, using `f_ster_div_single`) — the radiation-power analogue looks
   like it was meant to be computed the same way at 768-773 and a copy-paste error
   substituted the HCD formula/field instead. If so, `p_div_rad_total_mw` in this branch
   is silently reading stale/leftover state (same class of issue as open question 1), and
   `p_fw_rad_total_mw`'s value is wrong by whatever `p_div_rad_total_mw` should have been.
   Flagged, not fixed, per scope — this is a `process/` source read finding, worth your
   judgment on whether it's reachable/reproducible in practice.
