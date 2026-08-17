---
kind: model-unit
status: draft
confidence: high
---

## source
`process/models/stellarator/stellarator.py`, lines 1281-1682: final third of `st_fwbs()`
(481-1682, arbitrary even-thirds cut, not a semantic boundary). Chunk 1E3 of unit #1.

**Correction to this chunk's stub note**: the coolant-mass switch (`i_blkt_coolant_type`,
lines 1269-1274) that the earlier note anticipated at this boundary is actually entirely
inside chunk 1E2 (881-1280) — it ends at line 1274, seven lines before my range starts.
Nothing meaningful crosses the 1280/1281 seam itself; my chunk opens cleanly with the
external cryostat radius calculation (line 1282).

**The real cross-chunk finding is different and larger in span**: `coilhtmx`, `ptfiwp`,
`ptfowp`, `htheci`, `pheci`, `pheco`, `raddose`, `dpacop` are used in this chunk's output
block (lines 1438-1474) but never assigned anywhere in my range. Verified by direct grep
across the whole file restricted to lines < 1683: they're unpacked from
`sc_tf_coil_nuclear_heating_iter90()`'s return tuple around line 709-728 — **inside
chunk 1E1's territory (422-880), skipping over 1E2 entirely**. This is independently
confirmed by chunk 1F's own record
(`functional_process/models/stellarator/stellarator_F_tf_nuclear_heating.md`), which
already flagged the same two call sites (lines 476, 728) from the producer side. Not a
bug — `st_fwbs` is one continuous 1200-line function; these are ordinary Python locals
that happen to live across ~730 lines of it. Reinforces 1D's proposed fifth
classification (`local-intermediate`, see `stellarator_D_structure.md`'s open questions)
from a different angle: this case isn't even `self.data`-mediated, it's a plain local
surviving across three audit-chunk boundaries that only exist because of how this audit
was sliced, not because of anything in the source. **The pure port cannot treat 1E1/1E2/1E3
as three independent functions with a `VarPath`-only interface** — at minimum the
`sc_tf_coil_nuclear_heating` outputs must thread through as ordinary return values/
arguments across whatever the real function boundaries turn out to be once 1E1/1E2 are
read together.

**Correction to the dispatching directive**: both `i_tf_sup` sites `switches.md`
currently cites (stellarator.py lines 1022 and 1724) fall **outside** this chunk —
1022 is in 1E2's range (881-1280), 1724 is in 1F's range (1683-1885). Grepped `i_tf_sup`
restricted to 1281-1682: zero matches. `switches.md`'s existing entry is factually
accurate, just not evidence from this chunk — the attribution in my directive was wrong,
not the underlying finding. (1F's own record makes the same correction independently for
line 1724 — its "switches touched" section describes exactly that site.)

## data footprint

Computational section (1282-1330 only — see tier signal for why 1331-1682 is excluded):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.build.r_tf_outboard_mid` | read | explicit-arg | |
| `.build.dr_tf_outboard` | read | explicit-arg | |
| `.fwbs.dr_pf_cryostat` | read | explicit-arg | |
| `.fwbs.r_cryostat_inboard` | write, then read again (1287, and later in the output section) | implicit-io | same-chunk, unconditional, un-branched produce-then-consume — the "too strong a label" pattern 1D already flagged; noting the same nuance rather than re-arguing it |
| `.physics.rmajor` | read (2x) | explicit-arg | |
| `.build.dr_cryostat` | read | explicit-arg | |
| `.fwbs.vol_cryostat` | write, then read again (1328, and in output) | implicit-io | same nuance as above |
| `.build.dr_vv_inboard` | read | explicit-arg | |
| `.build.dr_vv_outboard` | read | explicit-arg | |
| `.physics.a_plasma_surface` | read | explicit-arg | |
| `.physics.rminor` | read | explicit-arg | |
| `.fwbs.fvoldw` | read | explicit-arg | |
| `.build.dr_fw_plasma_gap_inboard` / `dr_fw_inboard` / `dr_blkt_inboard` / `dr_shld_inboard` / `dr_fw_plasma_gap_outboard` / `dr_fw_outboard` / `dr_blkt_outboard` / `dr_shld_outboard` | read (8 fields) | explicit-arg | feed `r1` only |
| `.fwbs.vol_vv` | write, then read again (1323, 1328, and in output) | implicit-io | same nuance |
| `.fwbs.den_steel` | read | explicit-arg | |
| `.fwbs.m_vv` | write, then read again (1328, and in output) | implicit-io | same nuance |
| `.fwbs.dewmkg` | write | explicit-arg | final aggregate, terminal write of this chunk's computation |

No `implicit-io-via-callee`, no `redundant-duplicate-write` in the computational section.

Output section (1331-1682, `if output:`) reads ~35 further `data.*` fields purely for
`po.ovarre`/`po.ocmmnt` display (`pflux_fw_neutron_mw`, `wallpf`, `life_fw_fpy`, several
`build.*` thicknesses, `p_blkt_nuclear_heat_total_mw`, `p_shld_nuclear_heat_mw`,
`p_tf_nuclear_heat_mw`, `f_p_blkt_multiplication`, `npdiv`, `nphcdin`, `nphcdout`,
`f_blkt_li6_enrichment`, `tritprate`, `life_blkt_fpy`, `life_blkt`, `a_fw_total`,
`m_fw_total`, `divertor.a_div_surface_total`, `divertor.m_div_plate`, plus two switches
used only for display text — see below). None are written; none feed back into the
computation above. Not tabulated individually — consistent with this audit's established
"reporting is not in scope" stance (see `stellarator_G_output.md`) — but see the
JAX-difficulty note below for why this section isn't *quite* as inert as
`stellarator_G_output.md`'s.

## proposed signature(s)

Computational core only (1282-1330):
```python
def calculate_cryostat_and_vv(
    r_tf_outboard_mid: float,
    dr_tf_outboard: float,
    dr_pf_cryostat: float,
    rmajor: float,
    dr_cryostat: float,
    dr_vv_inboard: float,
    dr_vv_outboard: float,
    a_plasma_surface: float,
    rminor: float,
    fvoldw: float,
    dr_fw_plasma_gap_inboard: float,
    dr_fw_inboard: float,
    dr_blkt_inboard: float,
    dr_shld_inboard: float,
    dr_fw_plasma_gap_outboard: float,
    dr_fw_outboard: float,
    dr_blkt_outboard: float,
    dr_shld_outboard: float,
    den_steel: float,
) -> tuple[float, float, float, float, float]:
    # returns (r_cryostat_inboard, vol_cryostat, vol_vv, m_vv, dewmkg)
    ...
```
**Provisional — this is only 49 lines out of a 1200-line method sliced by line count, not
by function boundary.** Cannot be finalized as a standalone port target independent of
1E1/1E2; flagging as a real signature proposal for *this arithmetic*, not as "chunk 1E3 =
one function." The `sc_tf_coil_nuclear_heating`-derived locals used only in the output
section are explicitly excluded from this signature since they're not part of the
computation, only the report.

## tier signal
**Tier 1** for the computational section (1282-1330) — no internal solve, no calls to
other models, no data-dependent control flow beyond ordinary `if/else` on already-static
switches. The rest of the chunk (1331-1682, 88% of its line count) is a reporting shell,
out of pure-port scope by this audit's established convention.

## switches touched
- `.fwbs.blktmodel` — **not in the original 10, found incidentally.** Branches both the
  computation (indirectly, via 1E2's territory) and this chunk's reporting extensively
  (gates 6+ separate `if self.data.fwbs.blktmodel...` blocks in the output section alone,
  lines 1341, 1375, 1401, 1434-1436, 1476, 1496, 1591-1593). Given that density, this is
  very likely a major topology-changing switch — recommend adding it to `switches.md` as
  its own entry rather than treating it as this chunk's finding alone; out of scope to
  fully characterize from this chunk since most of its computational effect (not just
  reporting) likely lives in 1E1/1E2.
- `.heat_transport.ipowerflow` — **also not in the original 10.** Gates output-section
  branches at 1208 (also crosses into 1E2, since 1208 < 1281 — wait, verified: 1208 is
  *not* in my range at all, this was misread from earlier context; the actual site in my
  range is 1434 and 1591) and 1591. Reporting-only within my range; not evidence either
  way for its computational reads-set. Add to `switches.md` as its own entry.
- `.fwbs.i_thermal_electric_conversion` — **confirms `switches.md`'s existing finding.**
  My independent read of lines 1591-1601 matches that entry exactly (output-only site,
  same line numbers) — cross-validated, not just re-asserted.
- `.fwbs.hcdportsize`, `.fwbs.breedmat` — reporting-only switches selecting comment text
  (port-size wording, breeder-material name string). Trivial; not computational. Not
  worth a `switches.md` entry unless a computational site turns up elsewhere.

## calls into other models
None in this chunk. Pure arithmetic on already-available `data` fields plus formatted
output.

## JAX-difficulty flags
- The output section is **not purely inert** the way `stellarator_G_output.md`'s was:
  two of its display values (line 1623: `r_cryostat_inboard - 2.0*adewex`; line 1650:
  `dewmkg - m_vv`) are computed inline for the printout only, never stored. Minor
  severity — trivial arithmetic, not a blocker — but worth noting since it's a small
  instance of the same "reporting invoking real (if trivial) computation" pattern flagged
  as a general risk after `density_limits.py`'s `output()`.
- No CoolProp calls, no dynamic shapes, no control flow on a non-switch traced value in
  either section.

## open questions
1. Whether the true function boundary for a pure port should follow 1E1/1E2/1E3 at all,
   given the `sc_tf_coil_nuclear_heating`-derived locals span all three — recommend
   whoever synthesizes unit #1's final signature set treats 1E1-1E3 as one combined
   reading pass rather than three independently portable pieces.
2. `.fwbs.blktmodel` and `.heat_transport.ipowerflow` should each get a `switches.md`
   entry — I don't have enough of this switch's computational footprint from this chunk
   alone to propose a split/keep-static decision, only evidence that they're
   high-fan-out within the reporting section.
