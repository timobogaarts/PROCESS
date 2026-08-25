---
kind: model-unit
status: draft
confidence: medium-high
---

**Ported (2/3 arms).** S2 is `stellarator_E_fwbs_synthesis.md`'s piece #2
(`blanket_shield_tf_nuclear_power`): the `blktmodel`x`ipowerflow` dispatch inside
`st_fwbs`, three live arms. Arm 2 (`blktmodel != 1 & ipowerflow == 0`) and arm 3
(`blktmodel != 1 & ipowerflow == 1`) are self-contained tier-1 and are ported in
`stellarator_fwbs_s2.py`/`test_stellarator_fwbs_s2.py`. Arm 1 (`blktmodel == 1`, i.e.
`blanket_neutronics()` + its `ipowerflow`-nested tail) is audit-only: it is the arm the
synthesis doc flagged as hitting a live, two-part PROCESS bug at `blanket_neutronics()`
the moment it runs (see § 2 below) -- exactly the "stays mostly audit-only for the arm
that actually calls into `blanket_neutronics()`" outcome the dispatching task anticipated
as a fine result.

## source

`process/models/stellarator/stellarator.py`: `blanket_neutronics()` (422-480) is arm 1's
body; the `blktmodel`x`ipowerflow` dispatch itself is 608-1030, inside `st_fwbs()`.
Boundary is exactly `stellarator_E_fwbs_synthesis.md`'s S2 (§ 1 of that record): starts
right after S1 (`fw_blanket_shield_geometry_setup`, 515-605) ends, stops right before S3
(`divertor_mass_and_first_call_seed`, 1030-1043) begins. This audit does not touch
515-605 or 1030-1043 -- those are other agents' ranges (S1/S5 and S3 respectively, per
the dispatching task's boundary note) and this record only treats their outputs as
ordinary upstream `In` values.

## 1. The three arms, restated precisely

```
if blktmodel == 1:                          # ARM 1 -- audit-only, see § 2
    blanket_neutronics()                     # 422-480
    if ipowerflow == 1:                      # 611-678, no `else`
        ... (11 more .fwbs./.heat_transport. writes, tier 1, no bug, not ported here
             only because it is downstream of arm 1's own bug -- see § 2.3)
else:                                        # blktmodel != 1
    pnuc_cp = 0.0                            # 681, shared preamble, both sub-arms
    if ipowerflow == 0:                      # ARM 2 -- ported, § 3
        ...                                  # 684-728
    else:                                    # ARM 3 -- ported, § 4
        ...                                  # 730-1029
```

Both spellings PROCESS uses for the `blktmodel` switch (`== 1` at 608, `== 0` at
`st_fwbs`'s S4, line 1056, outside this unit's range) partition the same `{0, 1}` domain
identically -- `stellarator_E_fwbs_synthesis.md` § 4 already confirmed this from
`core/input.py:978`'s `choices=[0, 1]`, restated here since it's directly relevant to
this unit's own arm split. Not a bug, just inconsistent spelling.

## 2. Arm 1 (`blktmodel == 1`) -- audit-only

### 2.1 What the arm does when correct

`blanket_neutronics()` (422-480): sets `.fwbs.breeder`/`densbreed` from a 3-way
`breedmat` material switch (not a numerics switch, an `InputVariable` choosing a string
label -- out of scope for a `Switch`/`Alternative` decision), computes
`.fwbs.m_blkt_total = vol_blkt_total * densbreed` (`local-intermediate`, immediately
consumed), then should call, in order:

1. `nuclear_heating_blanket(m_blkt_total=.fwbs.m_blkt_total,
   p_fusion_total_mw=.physics.p_fusion_total_mw)` -> assign the 2-tuple to
   `.fwbs.p_blkt_nuclear_heat_total_mw`, `.ccfe_hcpb.exp_blanket`.
2. `nuclear_heating_magnets(output=False)` -- correctly called in the source (line 443,
   matches `nuclear_heating_magnets(self, output: bool)`'s real signature; it is
   `self`-bound and writes `self.data.*` directly, no return-value capture needed). This
   is the one call in `blanket_neutronics()` that is not part of either bug below.
3. (`tf_volume`/`ptfnucpm3` local arithmetic, 447-455, correct and unaffected.)
4. `nuclear_heating_shield(itart=.physics.itart, dr_shld_outboard=.build.dr_shld_outboard,
   dr_shld_inboard=.build.dr_shld_inboard, shield_density=.ccfe_hcpb.shield_density,
   whtshld=.fwbs.whtshld, x_blanket=.ccfe_hcpb.x_blanket,
   p_fusion_total_mw=.physics.p_fusion_total_mw)` -> assign the 4-tuple to
   `.fwbs.p_shld_nuclear_heat_mw`, `.ccfe_hcpb.exp_shield1`, `.ccfe_hcpb.exp_shield2`,
   `.ccfe_hcpb.shld_u_nuc_heating`. `shield_density`/`x_blanket` are step 2's own
   outputs -- an ordinary graph edge, matching `hcpb.md`'s own documented call order
   (magnets before shield) and `CCFE_HCPB.run()`'s real call sites (`hcpb.py:663-681`).

Then: `.fwbs.f_p_blkt_multiplication = 1.269` (unconditional overwrite -- confirmed
not-a-bug by `stellarator_E_fwbs_synthesis.md` § 3, since it happens to match the field's
own class default), and a call to `self.sc_tf_coil_nuclear_heating_iter90()` (chunk 1F,
already ported as `calculate_sc_tf_coil_nuclear_heating` in
`tf_nuclear_heating.py`) keeping only the 4th return value
(`flu_tf_neutron_fast_peak`) and discarding the rest, including its own
`p_tf_nuclear_heat_mw` -- correct as written, source comment confirms intent ("Use older
model to calculate neutron fluence since it is not calculated in the CCFE blanket
model").

### 2.2 What the arm actually does -- two live bugs, not one

`hcpb.md`'s open question #1 already flagged that lines 440/458 call
`self.hcpb.nuclear_heating_blanket()`/`nuclear_heating_shield()` with **zero
arguments**, against `@staticmethod` signatures requiring 2 and 7 keyword arguments --
would `TypeError` immediately. Confirmed by direct re-read.

**Second, independent bug found here, not previously flagged**: even setting the
argument-list bug aside, **neither call's return value is captured at all**:

```python
self.hcpb.nuclear_heating_blanket()  # line 440 -- no assignment
...
self.hcpb.nuclear_heating_shield()  # line 458 -- no assignment
```

Both `nuclear_heating_blanket` and `nuclear_heating_shield` are pure `@staticmethod`s
(confirmed by direct read of `process/models/blankets/hcpb.py:653-769`) -- unlike
`nuclear_heating_magnets`, they have no `self` to write through, so their entire effect
is their return value. A hypothetical minimal fix that only supplied the correct keyword
arguments (per § 2.1's reconstruction) but still discarded the return value would leave
`.fwbs.p_blkt_nuclear_heat_total_mw`, `.ccfe_hcpb.exp_blanket`,
`.fwbs.p_shld_nuclear_heat_mw`, `.ccfe_hcpb.exp_shield1`, `.ccfe_hcpb.exp_shield2`,
`.ccfe_hcpb.shld_u_nuc_heating` **all unwritten** by this arm -- a second, independent
omission from the missing-arguments bug, not a consequence of it. A correct fix needs
both: the right keyword arguments *and* capturing the tuple, exactly as
`CCFE_HCPB.run()`'s own call sites already do (`hcpb.py:663-681`, quoted in full in
`hcpb.md`'s "source" section).

Not fixed here, per this project's standing policy of reproducing PROCESS's documented
behaviour rather than silently correcting it. Both bugs are upstream of anything S2 can
resolve on its own (they are inside `blanket_neutronics()` itself, which *is* arm 1's
body) -- flagged for whoever does the consolidation pass that decides arm 1's node shape.

### 2.3 Why arm 1 is audit-only, not partially ported

Arm 1's `ipowerflow == 1` tail (611-678) is, in isolation, ordinary tier-1 arithmetic
with no bug of its own (same shape as arm 3's, smaller) -- but every one of its inputs
(`.fwbs.p_blkt_nuclear_heat_total_mw`, `.fwbs.p_shld_nuclear_heat_mw`,
`.current_drive.p_beam_orbit_loss_mw` aside, which is a genuine independent read) is
either produced by `blanket_neutronics()` itself or has no meaning without it having
correctly run first. Porting the tail alone, with its two inputs from
`blanket_neutronics()` treated as ordinary `In` arguments (the same treatment arm 2 gives
`sc_tf_coil_nuclear_heating_iter90()`'s outputs), is possible in principle -- but doing
so was judged not worth a third port in this pass: the tail's own arithmetic
(`p_div_nuclear_heat_total_mw`, `p_fw_hcd_nuclear_heat_mw`, `p_fw_nuclear_heat_total_mw`,
`pradloss`, `p_div_rad_total_mw` -- **correctly computed here**, contrast arm 3's bug,
§ 4 below -- `p_fw_hcd_rad_total_mw`, `p_fw_rad_total_mw`, four
`heat_transport.p_*_coolant_pump_mw` fields) is structurally near-identical to arm 3's
already-ported arithmetic (same formula shapes, `p_div_nuclear_heat_total_mw =
p_neutron_total_mw * f_ster_div_single` etc.), so the marginal audit value of a third,
very similar function was low relative to documenting arm 1's real blocker precisely
(§ 2.2). Left for a follow-up pass if the consolidation step wants it; the reads/writes
above are enough to write its signature without re-reading the source.

Under `ipowerflow == 0`, arm 1 writes almost nothing beyond `blanket_neutronics()`'s own
outputs -- confirmed by direct read (no `else` at 611), matching
`stellarator_E_fwbs_synthesis.md` § 4's "real quiet combination" finding.

## data footprint -- arm 1 (audit only, not ported)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.fwbs.breedmat` | read | explicit-arg | 3-way material switch, string-valued output, not a numerics `Switch` |
| `.fwbs.vol_blkt_total` | read | explicit-arg | S1 output |
| `.fwbs.densbreed` | write then read | local-intermediate | written by the `breedmat` branch, read one line later for `m_blkt_total` |
| `.fwbs.m_blkt_total` | write then read | local-intermediate | feeds `nuclear_heating_blanket`'s first arg in the same call |
| `.physics.p_fusion_total_mw` | read | explicit-arg | feeds both `nuclear_heating_blanket` and `nuclear_heating_shield` |
| `.fwbs.p_blkt_nuclear_heat_total_mw`, `.ccfe_hcpb.exp_blanket` | **not written** (bug) | -- | should be `nuclear_heating_blanket`'s captured return, see § 2.2 |
| (21 reads for `nuclear_heating_magnets`) | read | explicit-arg | identical footprint to `hcpb.md`'s own `nuclear_heating_magnets` table -- not re-listed here, cross-reference only |
| `.fwbs.f_a_fw_coolant_inboard/outboard`, `.ccfe_hcpb.armour_density/fw_density/blanket_density/shield_density/vv_density/x_blanket/x_shield/tfc_nuc_heating`, `.fwbs.p_tf_nuclear_heat_mw` | write | own-write (via `nuclear_heating_magnets`, correct call) | see `hcpb.md` |
| `.tfcoil.len_tf_coil`, `.tfcoil.a_tf_inboard_total`, `.tfcoil.a_tf_leg_outboard`, `.tfcoil.n_tf_coils` | read | explicit-arg | `tf_volume` local, 447-452 |
| `.fwbs.ptfnucpm3` | write | own-write | `= p_tf_nuclear_heat_mw / tf_volume`, local-intermediate read of the field `nuclear_heating_magnets` just wrote |
| `.physics.itart`, `.build.dr_shld_outboard/inboard`, `.fwbs.whtshld` | read | explicit-arg | `nuclear_heating_shield`'s remaining args (`shield_density`/`x_blanket` come from `nuclear_heating_magnets` above) |
| `.fwbs.p_shld_nuclear_heat_mw`, `.ccfe_hcpb.exp_shield1/exp_shield2/shld_u_nuc_heating` | **not written** (bug) | -- | should be `nuclear_heating_shield`'s captured return, see § 2.2 |
| `.fwbs.f_p_blkt_multiplication` | write | own-write | unconditional `1.269` literal, confirmed not-a-bug (matches class default) |
| (`sc_tf_coil_nuclear_heating_iter90`'s full read-set, chunk 1F) | read | explicit-arg (tier-3 edge) | see `tf_nuclear_heating.md` |
| `.fwbs.flu_tf_neutron_fast_peak` | write | own-write | only the 4th of `sc_tf_coil_nuclear_heating_iter90`'s 10 outputs is kept here; correct as written |

## proposed signature -- arm 1 (not ported; recorded for whoever does)

```python
def calculate_blanket_neutronics(m_blkt_total, p_fusion_total_mw, ...) -> tuple:
    """Would compose NuclearHeatingBlanket -> NuclearHeatingMagnets ->
    NuclearHeatingShield (hcpb.py, unit #13, already ported) using the CORRECT keyword
    arguments per § 2.1 above -- not the broken zero-arg call site. Tier 3 (calls
    already-validated nodes), not tier 1."""
```

## tier signal -- arm 1

**Tier 3** (composes three already-ported `hcpb.py` nodes plus one already-ported
`tf_nuclear_heating.py` node) -- not ported per this project's tier-3
policy (structural composition, no new solver, `test_harness.md`'s "Not built" section).
Blocked additionally by the two live bugs in § 2.2, which mean *any* port of this arm
must first decide how to route around PROCESS's own broken call site -- a design
decision reserved for the consolidation pass per the dispatching task's framing, not
resolved here.

## 3. Arm 2 (`blktmodel != 1 & ipowerflow == 0`) -- ported

`stellarator.py:684-728`, the "old model": one exponential-attenuation formula for
blanket heating, shield heating as the remainder, then a call to
`self.sc_tf_coil_nuclear_heating_iter90()` (chunk 1F) keeping all 10 outputs (2 to
`.fwbs.*`, 8 staying Python-local for S6's output block per
`stellarator_E_fwbs_synthesis.md` § 2's ledger, row 1).

Ported: `calculate_exponential_attenuation_blanket_shield_power` in
`stellarator_fwbs_s2.py` (the arm's own 5-line arithmetic only, 686-714). The
`sc_tf_coil_nuclear_heating_iter90()` tail (716-728) is **not** reproduced here -- it is
an ordinary tier-3 composition edge onto `tf_nuclear_heating.py`'s
already-ported, already-tested `calculate_sc_tf_coil_nuclear_heating`, per this
project's standing policy against re-porting an already-validated node.

## data footprint -- arm 2

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.p_neutron_total_mw` | read | explicit-arg | |
| `.fwbs.pnucloss` | read | explicit-arg | S1 output (`= p_neutron_total_mw * fhole`, `stellarator.py:598-600`) -- an ordinary upstream read from this arm's perspective, kept as its own parameter even though a real state constrains its value relative to `p_neutron_total_mw`/`fhole` (arm 2 does not read `fhole` itself, so the constraint is invisible to it) |
| `.fwbs.pnuc_cp` | read | **local-intermediate**, not explicit-arg | `stellarator.py:681` sets it to the literal `0.0` unconditionally, in the same straight-line scope, immediately before this arm runs -- not a real upstream edge. **Correction to how this field would naively be classified** (a first read suggests `explicit-arg` since it's genuinely `self.data`-read inside this arm, but tracing its producer shows it's dead as an independent input on the stellarator path) -- inlined as `0.0` in the port, not exposed as a parameter |
| `.fwbs.f_p_blkt_multiplication` | read | explicit-arg | same field arm 1 also reads (§ 2), no conflict -- both are plain reads |
| `.fwbs.f_a_blkt_cooling_channels`, `.fwbs.fblli2o`, `.fwbs.fblbe` | read | explicit-arg | `decaybl`'s denominator |
| `.build.dr_blkt_outboard` | read | explicit-arg | |
| `.fwbs.p_blkt_multiplication_mw` | write | own-write (returned) | |
| `.fwbs.p_blkt_nuclear_heat_total_mw` | write | own-write (returned) | |
| `.fwbs.p_shld_nuclear_heat_mw` | write | own-write (returned) | |
| (10-output `sc_tf_coil_nuclear_heating_iter90` read-set) | read | explicit-arg (tier-3 edge) | see `tf_nuclear_heating.md`; not this unit's own footprint |
| `.fwbs.flu_tf_neutron_fast_peak`, `.fwbs.p_tf_nuclear_heat_mw` | write (via that call) | own-write (returned, tier-3 edge) | 2 of the 10 outputs kept; other 8 stay Python-local (consumed by S6, out of this unit's scope) |

## proposed signature -- arm 2

```python
def calculate_exponential_attenuation_blanket_shield_power(
    p_neutron_total_mw,
    pnucloss,
    f_p_blkt_multiplication,
    f_a_blkt_cooling_channels,
    fblli2o,
    fblbe,
    dr_blkt_outboard,
) -> tuple[float, float, float]:
    """(p_blkt_multiplication_mw, p_blkt_nuclear_heat_total_mw, p_shld_nuclear_heat_mw)"""
```

Implemented verbatim in `stellarator_fwbs_s2.py`.

## cottax node -- arm 2

`ExponentialAttenuationBlanketShieldPower` (`ExplicitFunction`) in
`stellarator_fwbs_s2.py`. **Not registered in `total_process.py`** -- reserved for the
consolidation pass, same reasoning `hcpb.py`'s unit #13 record already gave for its own
three nodes: this arm is one of three `blktmodel`x`ipowerflow` alternatives that write
overlapping fields (`.fwbs.p_blkt_nuclear_heat_total_mw`,
`.fwbs.p_shld_nuclear_heat_mw`, `.fwbs.p_tf_nuclear_heat_mw`) with the other two arms --
an `Alternative`/`Switch` design question for whoever does that pass, not a registration
decision this audit should make unilaterally (see § 5, and the dispatching task's framing
note that this is explicitly out of scope here).

## tier signal -- arm 2

**Tier 1** for the arm's own arithmetic (no internal iteration, no `self.data` access
once extracted). The arm as a whole (including the `sc_tf_coil_nuclear_heating_iter90()`
tail) would be tier 3 if composed as one unit; ported as tier 1 + a separate,
already-existing tier-3 edge instead, per this project's standing practice of not
re-porting already-validated nodes.

## 4. Arm 3 (`blktmodel != 1 & ipowerflow == 1`) -- ported

`stellarator.py:730-1029`, the "new model": inboard/outboard power-flow accounting
through first wall, blanket, shield and divertor, with per-component coolant pumping
power. Self-contained (no cross-model calls anywhere in this arm), the largest of the
three arms.

Ported: `calculate_detailed_powerflow_blanket_shield_power` in
`stellarator_fwbs_s2.py`, the full arm **except**:

- the CoolProp/`irefprop`-gated `.fwbs.temp_blkt_coolant_out` block (803-823) -- see § 4.1.
- the confirmed `.fwbs.p_div_rad_total_mw` bug -- reproduced, not fixed, see § 4.2 (a
  **second, previously unflagged consequence** of the bug
  `stellarator_E_fwbs_synthesis.md` § 6 already found, see below).
- the redundant duplicate write of `.fwbs.p_fw_hcd_rad_total_mw` (770-780, identical
  expression computed twice under two different comments) -- collapsed to one
  computation, `redundant-duplicate-write` per `_audit/schema.md`.
- the trivial branches of two switches (`i_p_coolant_pumping != FRACTION_OF_HEAT`,
  `i_tf_sup != SUPERCONDUCTING`) -- see § 4.3.

### 4.1 CoolProp block, excluded

`stellarator.py:803-823`, gated by `.fwbs.i_blkt_coolant_type == WATER` then
`.fwbs.irefprop`: computes `.fwbs.temp_blkt_coolant_out` either via a real CoolProp call
(`FluidProperties.of(...)`, non-traceable external call) or a polynomial in `.fwbs.coolp`
(the `irefprop == False` branch). Confirmed by direct read of 824-1029 that
`temp_blkt_coolant_out` is **never read again anywhere else in this arm** -- its only
readers are `power.py` and `blankets/blanket_library.py`, both out of S2's scope.
Excluded from the port entirely (not computed, not returned) rather than porting just the
polynomial branch, since it has no downstream consumer inside this unit to justify the
extra surface. `non-traceable-external-call`, `blocker` for the CoolProp half,
`workaround-known` for the polynomial half (same treatment `stellarator_E1_fwbs_setup.md`
already gave this exact block, cross-referenced there under 1E1's own audit of S1).

### 4.2 `.fwbs.p_div_rad_total_mw` bug -- confirmed, and a second read site found

`stellarator_E_fwbs_synthesis.md` § 6 already confirmed `p_div_rad_total_mw` is read at
line 792 (feeding `p_fw_rad_total_mw`) but never written anywhere in this arm (the
duplicate-write bug at 770-780 computes `p_fw_hcd_rad_total_mw` twice, not
`p_div_rad_total_mw` once), and is therefore deterministically the dataclass default
`0.0` for the lifetime of any `blktmodel != 1` run (the only other writer,
`blanket_neutronics()`, never runs when `blktmodel != 1`, since `blktmodel` is a
run-constant switch).

**New finding: there is a second read site of the same never-written field**, at
`stellarator.py:1013`, inside `p_div_coolant_pump_mw`'s formula:

```python
self.data.heat_transport.p_div_coolant_pump_mw = (
    self.data.heat_transport.f_p_div_coolant_pump_total_heat
    * (
        self.data.physics.p_plasma_separatrix_mw
        + self.data.fwbs.p_div_nuclear_heat_total_mw
        + self.data.fwbs.p_div_rad_total_mw  # <- same bug, second site
    )
)
```

Not mentioned in the synthesis doc's § 6 (which only traced the 792 consequence through
to `p_fw_rad_total_mw`). Both sites are deterministically `0.0` for the same reason;
both are reproduced in the port as the literal `0.0`, not as a function parameter (see
the port's docstring) -- this exactly matches PROCESS's own actual runtime value, not an
approximation of it, so it introduces no value or gradient disagreement against the
reference (verified: `test_value_agreement`/`test_gradient_agreement` both pass, see
"harness verification" below).

### 4.3 Switches dropped: `i_p_coolant_pumping`, `i_tf_sup`

- **`.fwbs.i_p_coolant_pumping`** (`PumpingPowerModelTypes`, `process/core/input.py:1089`
  declares range `(0, 3)` globally, but this arm's own `else` branch (925-928) raises
  `ProcessValueError("i_p_coolant_pumping = 0 or 1 only for stellarator")` for any value
  outside `{0, 1}` -- a model-enforced domain restriction narrower than the input
  validator's). Two live values for a stellarator: `USER_INPUT` (0, `pass` -- this arm
  writes nothing to `.heat_transport.p_fw_coolant_pump_mw`/`p_blkt_coolant_pump_mw` at
  all, leaving them at whatever IN.DAT/a previous call supplied) and `FRACTION_OF_HEAT`
  (1, computes both from local nuclear-heating/radiation intermediates). Reads-sets
  genuinely differ (nominally calls for a split per `traceability_policy.md`'s default),
  but the `USER_INPUT` branch is not a computation to port at all (literally `pass`) --
  same shape `tf_nuclear_heating.py` already used for its own trivial
  switch branch (see next bullet), not a fresh judgment call. The port always computes
  as if `FRACTION_OF_HEAT`; `USER_INPUT` and the two raising values are out of port
  scope, flagged here rather than silently defaulted.
- **`.tfcoil.i_tf_sup`** (`TFConductorModel`): gates `p_tf_nuclear_heat_mw` between
  `pnucsi + pnucso - pnucshldi - pnucshldo` (`SUPERCONDUCTING`) and the literal `0.0`
  (resistive). **Direct precedent already on record**: `tf_nuclear_heating.py`'s own
  module docstring drops the identical switch's resistive branch from
  `sc_tf_coil_nuclear_heating_iter90`'s port entirely ("the resistive branch takes no
  inputs and always returns ten zeros, so it is not a computation to port, it is the
  absence of this node in the graph"). Same treatment applied here for the same switch on
  a different field. The port always computes as if `SUPERCONDUCTING`.

Neither switch is added to `core/solver/switches.md` (out of this unit's file boundary,
reserved for the consolidation pass per the dispatching task's boundary note) --
recorded here as the evidence for whoever does add them.

## data footprint -- arm 3

Full 26-input, 16-output ledger (see `stellarator_fwbs_s2.py`'s docstrings for the
per-parameter units/source `VarPath`s -- not duplicated field-by-field here to avoid two
copies drifting; this table gives the classification pass over the same list).

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.p_neutron_total_mw` | read | explicit-arg | |
| `.fwbs.f_ster_div_single` | read | explicit-arg | S1/`Divertor` output |
| `.fwbs.f_a_fw_outboard_hcd` | read | explicit-arg | |
| `.fwbs.pnucloss` | read | explicit-arg | S1 output, see arm 2's note -- same field |
| `.fwbs.pnuc_cp` | read | **local-intermediate**, not explicit-arg | same as arm 2, `stellarator.py:681`; inlined as `0.0` |
| `.first_wall.a_fw_inboard`, `a_fw_outboard`, `a_fw_total` | read | explicit-arg | S1 always sets the first two to exactly half the third (`stellarator.py:521-522`), but this arm reads all three as independent values -- kept as 3 separate parameters, matching the source, not S1's coincidental symmetry (see harness verification note below) |
| `.physics.p_plasma_rad_mw` | read | explicit-arg | |
| `.fwbs.fhole` | read | explicit-arg | also an S1 input (`pnucloss`'s own producer), read here a second, independent time for `pradloss` |
| `.build.dr_fw_inboard`, `dr_fw_outboard` | read | explicit-arg | |
| `.fwbs.radius_fw_channel` | read | explicit-arg | |
| `.fwbs.declfw` | read | explicit-arg | |
| `.build.dr_blkt_inboard`, `dr_blkt_outboard` | read | explicit-arg | |
| `.fwbs.declblkt` | read | explicit-arg | |
| `.heat_transport.f_p_fw_coolant_pump_total_heat` | read | explicit-arg | |
| `.current_drive.p_beam_orbit_loss_mw` | read | explicit-arg | |
| `.heat_transport.f_p_blkt_coolant_pump_total_heat` | read | explicit-arg | |
| `.fwbs.f_p_blkt_multiplication` | read | explicit-arg | same field arms 1/2 also read |
| `.fwbs.declshld` | read | explicit-arg | |
| `.build.dr_shld_inboard`, `dr_shld_outboard` | read | explicit-arg | |
| `.heat_transport.f_p_shld_coolant_pump_total_heat` | read | explicit-arg | |
| `.physics.p_plasma_separatrix_mw` | read | explicit-arg | |
| `.heat_transport.f_p_div_coolant_pump_total_heat` | read | explicit-arg | |
| `.fwbs.i_p_coolant_pumping` | read (dropped) | switch, see § 4.3 | not a port parameter |
| `.tfcoil.i_tf_sup` | read (dropped) | switch, see § 4.3 | not a port parameter |
| `.fwbs.i_blkt_coolant_type`, `.fwbs.irefprop`, `.fwbs.coolp` | read (excluded) | see § 4.1 | not a port parameter, CoolProp block dropped whole |
| `.fwbs.p_div_nuclear_heat_total_mw` | write | own-write (returned) | |
| `.fwbs.p_fw_hcd_nuclear_heat_mw` | write | own-write (returned) | |
| `.fwbs.p_fw_hcd_rad_total_mw` | write (x2 in source) | **redundant-duplicate-write** | collapsed to one write, see § 4 |
| `.fwbs.pradloss` | write | own-write (returned) | |
| `.fwbs.p_div_rad_total_mw` | **never written** (bug) | -- | reproduced as literal `0.0`, see § 4.2 |
| `.fwbs.p_fw_rad_total_mw` | write | own-write (returned) | wrong by the omitted `p_div_rad_total_mw` term, reproduced exactly, see § 4.2 |
| `f_a_fw_coolant_inboard`, `f_a_fw_coolant_outboard` (Python locals, **not** `.fwbs.*` in this arm) | write | own-write (returned) | `stellarator_E_fwbs_synthesis.md` § 2's "same names, disjoint formulas" wrinkle -- consumed by S4, out of this unit's scope; given best-effort `.fwbs.*` `VarPath`s on the cottax node anyway, per `tf_nuclear_heating.py`'s own precedent for best-effort output paths |
| `.fwbs.p_fw_nuclear_heat_total_mw` | write | own-write (returned) | |
| `.fwbs.p_blkt_multiplication_mw` | write (x2, `=` then `+=`, source 930/948) | own-write (returned) | combined into one closed-form expression in the port, not a bug -- both source writes are unconditional, no branch between them |
| `.fwbs.p_blkt_nuclear_heat_total_mw` | write | own-write (returned) | |
| `.heat_transport.p_fw_coolant_pump_mw` | write (conditional, § 4.3) | own-write (returned) | FRACTION_OF_HEAT only |
| `.heat_transport.p_blkt_coolant_pump_mw` | write (conditional, § 4.3) | own-write (returned) | FRACTION_OF_HEAT only |
| `.fwbs.p_shld_nuclear_heat_mw` | write | own-write (returned) | |
| `.heat_transport.p_shld_coolant_pump_mw` | write (conditional, § 4.3) | own-write (returned) | FRACTION_OF_HEAT only |
| `.heat_transport.p_div_coolant_pump_mw` | write (conditional, § 4.3) | own-write (returned) | FRACTION_OF_HEAT only; second `p_div_rad_total_mw` bug site, § 4.2 |
| `.fwbs.p_tf_nuclear_heat_mw` | write (conditional, § 4.3) | own-write (returned) | SUPERCONDUCTING only, else `0.0` |

## proposed signature -- arm 3

```python
def calculate_detailed_powerflow_blanket_shield_power(
    p_neutron_total_mw, f_ster_div_single, f_a_fw_outboard_hcd, pnucloss,
    a_fw_inboard, a_fw_outboard, a_fw_total, p_plasma_rad_mw, fhole,
    dr_fw_inboard, dr_fw_outboard, radius_fw_channel, declfw,
    dr_blkt_inboard, dr_blkt_outboard, declblkt,
    f_p_fw_coolant_pump_total_heat, p_beam_orbit_loss_mw,
    f_p_blkt_coolant_pump_total_heat, f_p_blkt_multiplication,
    declshld, dr_shld_inboard, dr_shld_outboard,
    f_p_shld_coolant_pump_total_heat, p_plasma_separatrix_mw,
    f_p_div_coolant_pump_total_heat,
) -> tuple:  # 16 scalars, see data footprint's write rows for order
```

Implemented in `stellarator_fwbs_s2.py`, with the CoolProp block and the two dropped
switch branches omitted per § 4.1/§ 4.3, and the `p_div_rad_total_mw` bug reproduced as a
literal per § 4.2.

## cottax node -- arm 3

`DetailedPowerflowBlanketShieldPower` (`ExplicitFunction`) in `stellarator_fwbs_s2.py`.
Not registered in `total_process.py`, same reservation as arm 2 (§ 5).

## tier signal -- both ported arms

**Tier 1**: no internal iteration, no cross-model calls (arm 2's
`sc_tf_coil_nuclear_heating_iter90()` tail is excluded from the ported function, treated
as a separate tier-3 edge onto an already-validated node, not part of this tier
assessment).

## switches touched

See § 4.3 for `i_p_coolant_pumping`/`i_tf_sup` (arm 3 only, both dropped to their
"live"/nontrivial branch, not added to `switches.md`, out of file boundary). `blktmodel`
and `ipowerflow` themselves are the topology-changing switches selecting among the three
arms -- see § 5 for the shape `next_steps.md` § 1 already asked this unit to characterize
precisely.

## calls into other models

Arm 1 only (audit-only, not ported): `self.hcpb.nuclear_heating_blanket`/
`nuclear_heating_magnets`/`nuclear_heating_shield` (unit #13, `hcpb.py`, already
ported), `self.sc_tf_coil_nuclear_heating_iter90` (chunk 1F, already ported). Arms 2/3:
arm 2 alone calls `self.sc_tf_coil_nuclear_heating_iter90` (chunk 1F); its tail is not
reproduced in the port, see § 3. Arm 3: none.

## JAX-difficulty flags

- **CoolProp call** (`FluidProperties.of(...)`, arm 3 only, § 4.1) --
  `non-traceable-external-call`, `blocker`. Whole block excluded from the port, not
  worked around, since it has no downstream consumer inside this unit.
- **`i_p_coolant_pumping`'s `ProcessValueError`** (arm 3, values outside `{0, 1}`) --
  the port does not reproduce the raise (a traced function cannot raise on a
  data-dependent condition, per `test_harness.md`'s domain-guard convention) and does
  not reproduce the `USER_INPUT` (0) branch either, since it is not a computation (see
  § 4.3) -- `minor`, `workaround-known`, both branches simply out of port scope rather
  than `jnp.where`-guarded.
- **`i_tf_sup`'s resistive branch** (arm 3) -- `minor`, `workaround-known`, dropped
  entirely per direct precedent (§ 4.3), not `jnp.where`-guarded.
- **Gradient testing entanglement, arm 3 only** (not a JAX-tracing difficulty, a
  reference-construction one -- see "harness verification" below): six of arm 3's 26
  arguments (`p_neutron_total_mw`, `a_fw_total`, `a_fw_inboard`, `a_fw_outboard`,
  `fhole`, `pnucloss`) cannot be independently gradient-checked against a full
  `st_fwbs()` reference, because S1 imposes real relationships between them that the
  reference call cannot help but enforce, while the pure function (correctly, matching
  the source) treats them as independent reads. `minor` for the port itself (the
  function's own derivatives are not in question, only this one reference's ability to
  validate them) -- `workaround-known`, `static_argnames` on the test contract, see the
  test module for the full reasoning.
- No `scipy`, no `copy.deepcopy`, no data-dependent early exit/loop anywhere in the
  ported arithmetic of arms 2 or 3.

## harness verification

`~/miniconda3/envs/process_port/bin/python -m pytest
functional_process/models/stellarator/test_stellarator_fwbs_s2.py -q --fp-gradients`:
all cases pass (value agreement, output finiteness, gradient finiteness and gradient
agreement for both ported arms, including all of arm 3's 20 non-entangled arguments).
Reference adapters construct a real `Stellarator`+`DataStructure` and call
`st_fwbs(output=False)` in full (S1 through S5 run, not just this unit's own arm --
`st_fwbs` has no early return) -- see the test module's docstring for the derived-field
handling this requires (`a_fw_inboard`/`a_fw_outboard` from `a_fw_total`, `pnucloss` from
`p_neutron_total_mw`x`fhole`) and § 4.3 above for why six arguments are excluded from
arm 3's gradient check specifically. No PROCESS unit test exercises `st_fwbs` at all
(`tests/unit/models/stellarator/test_stellarator.py` has no `test_st_fwbs`), so both
contracts are fuzz-only, consistent with `divertor.md`/`heating.md`'s own precedent for
units in the same position.

## 5. `blktmodel`x`ipowerflow` switch shape -- restated precisely, not resolved

This is exactly the question `next_steps.md` § 1 flagged as open ("`blkttype` is three
values over two arms... decide the multi-value-per-arm question when S2 is actually
audited") and `stellarator_E_fwbs_synthesis.md` § 4 already partially answered for
`blkttype` specifically (S4's territory, not S2's -- confirmed not this unit's problem).
For S2 itself, the shape is:

- `blktmodel` (2 values) x `ipowerflow` (2 values) would naively suggest 4 combinations,
  but only 3 are live: `blktmodel==1 & ipowerflow==0` writes almost nothing (§ 2.3's
  "quiet combination"), collapsing the `blktmodel==1` side to effectively one arm with an
  optional tail, not two independent ones.
- The three live arms share **overlapping output fields**, not disjoint ones:
  `.fwbs.p_blkt_nuclear_heat_total_mw`, `.fwbs.p_shld_nuclear_heat_mw`, and
  `.fwbs.p_tf_nuclear_heat_mw` are each written by more than one arm (arm 1 via
  `nuclear_heating_magnets`, when its two bugs are fixed; arm 2 via
  `sc_tf_coil_nuclear_heating_iter90`; arm 3 via its own local formula) -- genuinely
  alternative producers of the *same* `VarPath`, gated by the switch pair, exactly the
  shape `unit_registry.md`'s row 13 already flagged for `p_tf_nuclear_heat_mw` alone
  (`NuclearHeatingMagnets` vs. `ScTfCoilNuclearHeating`) and now confirmed to extend to
  two more fields once arm 3's own formula is counted.
- This is **not** an `Alternative.value`-per-arm shape (one integer selects one
  producer) so much as a genuine 3-way `Switch` over `(blktmodel, ipowerflow)` jointly,
  where the joint pair -- not either switch alone -- selects the producer, and the
  `ipowerflow==0` combination under `blktmodel==1` is a degenerate/quiet fourth case
  rather than a fourth producer. A clean design likely wants a single derived
  enum-like `Switch` with 3 (not 4) values, one per live arm, rather than nesting two
  binary `Switch`es -- but this is the consolidation pass's design decision, per the
  dispatching task's framing, not resolved here.

## open questions

1. Arm 1's two bugs (§ 2.2) need a routing decision before it can be ported at all --
   reproduce both (a `TypeError`-raising node is not meaningfully testable) or treat
   `blanket_neutronics()` as calling the *correct* signatures (per § 2.1) and flag the
   divergence from PROCESS's actual current behaviour explicitly. Not resolved here, per
   the dispatching task's explicit instruction not to route around this bug silently.
2. § 5's `Switch` shape is stated, not built -- no `Alternative`/`Switch` code exists for
   `blktmodel`x`ipowerflow` anywhere in this unit's files, by design (reserved for
   consolidation).
3. Whether arm 1's `ipowerflow==1` tail (§ 2.3) is worth a third port once arm 1's body
   itself is resolved -- flagged, not decided; its arithmetic is now fully recorded in
   § 2.3's prose even though not implemented.
4. S3 (`divertor_mass_and_first_call_seed`, 1030-1043) and S4
   (`blanket_shield_fw_coolant_mass`, 1045-1274) both consume this unit's outputs
   (`f_a_fw_coolant_inboard`/`outboard` from arms 1/3, `coolvol`'s seed from S3) --
   confirmed present in this unit's data footprint above, not re-audited here (other
   agents'/passes' scope per the dispatching task's boundary note).
