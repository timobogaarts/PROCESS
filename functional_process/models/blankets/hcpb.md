---
kind: model-unit
status: draft
confidence: high
---

**Ported (3/3).** All three in-scope methods are self-contained tier-1 and are ported in
`hcpb.py`/`test_hcpb.py`: `nuclear_heating_blanket` and `nuclear_heating_shield` were
already `@staticmethod`s in the source (no `self.data` access at all — the pure port is
almost a verbatim copy); `nuclear_heating_magnets` is a `self`-bound instance method with
a genuine `self.data` footprint and gets the usual "close the data back-door" extraction,
same shape as `PlasmaDensityLimit.calculate_density_limit`/`st_sudo_density_limit`.

**A live call-site bug was found while tracing the stellarator caller — see "open
questions" #1. It does not block this port** (the port targets the staticmethods'
declared signatures, not the broken call site), but it directly affects `st_fwbs`'s S2
sub-computation, which is what this unit was the sole blocker for, so it is flagged
prominently rather than buried.

## source

`process/models/blankets/hcpb.py` (1663 lines total). Registry unit #13, scope: the
three named methods only — `nuclear_heating_blanket` (654-698), `nuclear_heating_shield`
(700-769), `nuclear_heating_magnets` (463-609) — per `unit_registry.md`'s row 13 and
`next_steps.md` §4b's dispatch table.

**Transitive-closure check** (same discipline as unit #10/#22, unit #20/#23): none of the
three methods calls any other `Model` method, any other method within `hcpb.py` itself
(including the fourth sibling `nuclear_heating_fw`, out of scope — the three in-scope
methods neither call it nor are called by it), or any module-level function outside
`numpy`. All three are dead-ends — no further scope expansion needed. Confirmed by direct
read of all three bodies (`nuclear_heating_magnets`: 463-609; `nuclear_heating_blanket`:
653-698; `nuclear_heating_shield`: 700-769) and by grep (`grep -n "self\.\(nuclear_heating\|hcpb\)"` inside the file returns
only the three's own `def` lines and their call sites in `run()`).

**Call sites** (for context, not part of this unit's own footprint):
- `CCFE_HCPB.run()` (tokamak path), lines 150-182: calls all three with correctly
  supplied keyword arguments matching their declared signatures — `component_masses()`
  first (150, out of scope, not required by any of the three), then
  `nuclear_heating_magnets(output=output)` (155), then `nuclear_heating_fw` (157, out of
  scope), then `nuclear_heating_blanket(m_blkt_total=..., p_fusion_total_mw=...)` (164),
  then `nuclear_heating_shield(itart=..., ..., x_blanket=self.data.ccfe_hcpb.x_blanket,
  ...)` (174) — note `x_blanket` here is `nuclear_heating_magnets`'s own output,
  confirming the real call order `magnets → shield` (shield needs magnets' `x_blanket`).
- `stellarator.py`'s `blanket_neutronics()` (`stellarator.py:422-461`, part of unit #1's
  `st_fwbs` S2 per the synthesis doc), calls `self.hcpb.nuclear_heating_blanket()` (440,
  **zero arguments**), `self.hcpb.nuclear_heating_magnets(False)` (443, correct — matches
  `nuclear_heating_magnets(self, output: bool)`), `self.hcpb.nuclear_heating_shield()`
  (458, **zero arguments**). See open question #1 — the zero-argument calls do not match
  either staticmethod's declared signature.

## data footprint

### `nuclear_heating_blanket(m_blkt_total, p_fusion_total_mw)`

Already pure in the source (`@staticmethod`, no `self` access at all).

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.fwbs.m_blkt_total` | read (via caller's kwarg) | explicit-arg | plain parameter |
| `.physics.p_fusion_total_mw` | read (via caller's kwarg) | explicit-arg | plain parameter |
| `.fwbs.p_blkt_nuclear_heat_total_mw` | write (via caller's return-value assignment) | own-write (returned) | first return value |
| `.ccfe_hcpb.exp_blanket` | write (via caller's return-value assignment) | own-write (returned) | second return value |

`logger.error(...)` on `p_blkt_nuclear_heat_total_mw < 1` is a diagnostic side effect,
not a data write — see JAX-difficulty flags.

### `nuclear_heating_shield(itart, dr_shld_outboard, dr_shld_inboard, shield_density, whtshld, x_blanket, p_fusion_total_mw)`

Already pure in the source (`@staticmethod`, no `self` access at all).

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.itart` | read (via caller's kwarg) | explicit-arg | spherical-tokamak indicator, gates a 2-way formula branch (see JAX-difficulty flags and switches touched) |
| `.build.dr_shld_outboard` | read | explicit-arg | |
| `.build.dr_shld_inboard` | read | explicit-arg | unused when `itart==1` (branch doesn't read it), but always supplied — ordinary continuous branch, not a topology split |
| `.ccfe_hcpb.shield_density` | read | explicit-arg | produced by `nuclear_heating_magnets` at the real call site (both `run()` and `blanket_neutronics()` call magnets before shield) — an ordinary graph edge, not a footprint concern for this function itself |
| `.fwbs.whtshld` | read | explicit-arg | shield mass, from `component_masses`/blanket-library territory, out of this unit's scope |
| `.ccfe_hcpb.x_blanket` | read | explicit-arg | also produced by `nuclear_heating_magnets` — second edge from that node into this one |
| `.physics.p_fusion_total_mw` | read | explicit-arg | |
| `.fwbs.p_shld_nuclear_heat_mw` | write (return) | own-write (returned) | 1st return value |
| `.ccfe_hcpb.exp_shield1` | write (return) | own-write (returned) | 2nd |
| `.ccfe_hcpb.exp_shield2` | write (return) | own-write (returned) | 3rd |
| `.ccfe_hcpb.shld_u_nuc_heating` | write (return) | own-write (returned) | 4th |

### `nuclear_heating_magnets(self, output)`

Not a `@staticmethod` — genuine `self.data` extraction needed. 21 reads (verified against
`tests/unit/models/blankets/test_ccfe_hcpb.py::test_nuclear_heating_magnets`'s fixture
field list as a cross-check, which independently enumerates the same 21 — including
`dr_fw_outboard`, easy to miss on a first read since it only appears inside the
`x_blanket` formula's `(dr_fw_inboard + dr_fw_outboard) / 2.0` term, not in the
inboard-only void-fraction calculation two lines above it).

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.fwbs.radius_fw_channel` | read | explicit-arg | |
| `.fwbs.dx_fw_module` | read | explicit-arg | |
| `.build.dr_fw_inboard` | read | explicit-arg | used twice: void-fraction denominator and `x_blanket`'s FW term |
| `.build.dr_fw_outboard` | read | explicit-arg | **easy to miss** — only appears in `x_blanket`'s FW term, not in the void-fraction calc |
| `.fwbs.den_steel` | read | explicit-arg | |
| `.fwbs.m_blkt_total` | read | explicit-arg | same field `nuclear_heating_blanket` also reads — no conflict, both are plain reads of one upstream value |
| `.fwbs.vol_blkt_total` | read | explicit-arg | |
| `.fwbs.whtshld` | read | explicit-arg | same field `nuclear_heating_shield` also reads directly (not through this node) |
| `.fwbs.vol_shld_total` | read | explicit-arg | |
| `.build.dr_vv_inboard` | read | explicit-arg | |
| `.build.dr_vv_outboard` | read | explicit-arg | |
| `.fwbs.m_vv` | read | explicit-arg | |
| `.fwbs.vol_vv` | read | explicit-arg | denominator of `vv_density`; guarded in the port (see JAX-difficulty flags) since it is only read under a `> 1e-6` guard in the source, and PROCESS's own Python `if` short-circuits the division in a way a traced `jnp.where` would not |
| `.physics.itart` | read | explicit-arg | same 2-way formula branch as `nuclear_heating_shield`, applied twice more here (`th_blanket_av`/`th_shield_av` **and** the TF-coil-mass source for `tfc_nuc_heating`) |
| `.build.dr_blkt_outboard` | read | explicit-arg | |
| `.build.dr_blkt_inboard` | read | explicit-arg | unused when `itart==1` |
| `.build.dr_shld_outboard` | read | explicit-arg | |
| `.build.dr_shld_inboard` | read | explicit-arg | unused when `itart==1` |
| `.fwbs.fw_armour_thickness` | read | explicit-arg | |
| `.tfcoil.whttflgs` | read | explicit-arg | only used when `itart==1` (ST outboard-leg mass) |
| `.tfcoil.m_tf_coils_total` | read | explicit-arg | only used when `itart==0` (full TF coil mass) |
| `.physics.p_fusion_total_mw` | read | explicit-arg | |
| `.fwbs.f_a_fw_coolant_inboard` | write (return) | own-write (returned) | |
| `.fwbs.f_a_fw_coolant_outboard` | write (return) | own-write (returned) | **numerically identical** to `f_a_fw_coolant_inboard` (source: `f_a_fw_coolant_outboard = f_a_fw_coolant_inboard`, no separate formula) — not `redundant-duplicate-write` per schema's definition (that label is for one `VarPath` written twice; this is two distinct `VarPath`s sharing one value), but worth flagging so the port's two outputs aren't mistaken for independently-derived quantities |
| `.ccfe_hcpb.armour_density` | write (return) | own-write (returned) | |
| `.ccfe_hcpb.fw_density` | write (return) | own-write (returned) | |
| `.ccfe_hcpb.blanket_density` | write (return) | own-write (returned) | |
| `.ccfe_hcpb.shield_density` | write (return) | own-write (returned) | consumed by `nuclear_heating_shield` |
| `.ccfe_hcpb.vv_density` | write (return) | own-write (returned) | |
| `.ccfe_hcpb.x_blanket` | write (return) | own-write (returned) | consumed by `nuclear_heating_shield` |
| `.ccfe_hcpb.x_shield` | write (return) | own-write (returned) | **not** read by `nuclear_heating_shield` (that function computes its own, differently-scaled shield exponent from `shield_density`/`dr_shld_*` directly) — a reporting/other-consumer output, not a graph edge to the sibling in this unit |
| `.ccfe_hcpb.tfc_nuc_heating` | write (return) | own-write (returned) | |
| `.fwbs.p_tf_nuclear_heat_mw` | write (return) | own-write (returned) | **re-normalised later**, outside this unit's scope, by `CCFE_HCPB.run()`'s lines 253-264 (divides by `pnuc_tot_blk_sector`, multiplies by `f_p_blkt_multiplication`/`f_geom_blanket`/`p_neutron_total_mw`, adds `pnuc_cp_tf`) — this unit only owns the *first* write, not the final value used downstream. Noted so a later S2 port doesn't mistake this node's output for the final `p_tf_nuclear_heat_mw`. |

Local intermediates, all `local-intermediate` per schema (written once, unconditionally,
read back in the same straight-line body, no branch or loop in between): `vffwm`
(= `f_a_fw_coolant_inboard`, reused as a plain float in three density formulas),
`d_vv_all` (`max(dr_vv_inboard, dr_vv_outboard)`), `th_blanket_av`/`th_shield_av` (the one
real 2-way branch, see JAX-difficulty flags).

## proposed signature(s)

```python
def nuclear_heating_blanket(m_blkt_total, p_fusion_total_mw) -> tuple[float, float]:
    """Ports the staticmethod of the same name verbatim (no `self` to close)."""
    ...


def nuclear_heating_shield(
    itart,
    dr_shld_outboard,
    dr_shld_inboard,
    shield_density,
    whtshld,
    x_blanket,
    p_fusion_total_mw,
) -> tuple[float, float, float, float]:
    """Ports the staticmethod of the same name verbatim (no `self` to close)."""
    ...


def calculate_nuclear_heating_magnets(
    radius_fw_channel,
    dx_fw_module,
    dr_fw_inboard,
    dr_fw_outboard,
    den_steel,
    m_blkt_total,
    vol_blkt_total,
    whtshld,
    vol_shld_total,
    dr_vv_inboard,
    dr_vv_outboard,
    m_vv,
    vol_vv,
    itart,
    dr_blkt_outboard,
    dr_blkt_inboard,
    dr_shld_outboard,
    dr_shld_inboard,
    fw_armour_thickness,
    whttflgs,
    m_tf_coils_total,
    p_fusion_total_mw,
) -> tuple:  # 11 scalars, see data footprint's write rows for order
    """Closes the `self.data` back-door on `nuclear_heating_magnets` — `calculate_`
    prefix per naming_convention.md, same as `calculate_density_limit` for an instance
    method with no pre-existing pure core to lift verbatim."""
    ...
```

## cottax node

Three `ExplicitFunction` nodes, one per function, written in `hcpb.py` right after each
function. Straightforward wraps — every input/output is an ordinary `VarPath`, no
switches-as-ports, no minted names needed (every field involved already has PROCESS
storage).

```python
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, Input, Output


class NuclearHeatingBlanket(ExplicitFunction):
    p_blkt_nuclear_heat_total_mw = Output(lambda s: s.fwbs.p_blkt_nuclear_heat_total_mw)
    exp_blanket = Output(lambda s: s.ccfe_hcpb.exp_blanket)

    def __call__(
        self,
        m_blkt_total=Input(lambda s: s.fwbs.m_blkt_total),
        p_fusion_total_mw=Input(lambda s: s.physics.p_fusion_total_mw),
    ):
        return nuclear_heating_blanket(m_blkt_total, p_fusion_total_mw)


class NuclearHeatingShield(ExplicitFunction):
    p_shld_nuclear_heat_mw = Output(lambda s: s.fwbs.p_shld_nuclear_heat_mw)
    exp_shield1 = Output(lambda s: s.ccfe_hcpb.exp_shield1)
    exp_shield2 = Output(lambda s: s.ccfe_hcpb.exp_shield2)
    shld_u_nuc_heating = Output(lambda s: s.ccfe_hcpb.shld_u_nuc_heating)

    def __call__(
        self,
        itart=Input(lambda s: s.physics.itart),
        dr_shld_outboard=Input(lambda s: s.build.dr_shld_outboard),
        dr_shld_inboard=Input(lambda s: s.build.dr_shld_inboard),
        shield_density=Input(lambda s: s.ccfe_hcpb.shield_density),
        whtshld=Input(lambda s: s.fwbs.whtshld),
        x_blanket=Input(lambda s: s.ccfe_hcpb.x_blanket),
        p_fusion_total_mw=Input(lambda s: s.physics.p_fusion_total_mw),
    ):
        return nuclear_heating_shield(
            itart, dr_shld_outboard, dr_shld_inboard, shield_density, whtshld,
            x_blanket, p_fusion_total_mw,
        )


class NuclearHeatingMagnets(ExplicitFunction):
    f_a_fw_coolant_inboard = Output(lambda s: s.fwbs.f_a_fw_coolant_inboard)
    f_a_fw_coolant_outboard = Output(lambda s: s.fwbs.f_a_fw_coolant_outboard)
    armour_density = Output(lambda s: s.ccfe_hcpb.armour_density)
    fw_density = Output(lambda s: s.ccfe_hcpb.fw_density)
    blanket_density = Output(lambda s: s.ccfe_hcpb.blanket_density)
    shield_density = Output(lambda s: s.ccfe_hcpb.shield_density)
    vv_density = Output(lambda s: s.ccfe_hcpb.vv_density)
    x_blanket = Output(lambda s: s.ccfe_hcpb.x_blanket)
    x_shield = Output(lambda s: s.ccfe_hcpb.x_shield)
    tfc_nuc_heating = Output(lambda s: s.ccfe_hcpb.tfc_nuc_heating)
    p_tf_nuclear_heat_mw = Output(lambda s: s.fwbs.p_tf_nuclear_heat_mw)

    def __call__(self, ... 21 Inputs, one per data-footprint read row ...):
        return calculate_nuclear_heating_magnets(...)
```

(Full 21-argument `__call__` written out in `hcpb.py` itself, not repeated here — see the
data footprint table above for the exact field list and `hcpb.py` for the exact
`Input(lambda s: ...)` mapping, one per row.)

**Graph edges within this unit**: `NuclearHeatingMagnets` produces `shield_density` and
`x_blanket`, both consumed by `NuclearHeatingShield` — matching the real call order both
`run()` and `blanket_neutronics()` use (magnets before shield). `NuclearHeatingBlanket`
is independent of both (shares `m_blkt_total`/`p_fusion_total_mw` as external reads, no
internal edge).

## tier signal

All three: **tier 1**. No internal iteration anywhere, no calls into other models (see
"calls into other models" below), and — per this project's standing practice
(`unit_registry.md`, "Standing practice going forward") — self-contained tier-1 chunks are
ported as part of finishing the audit, not queued. `nuclear_heating_blanket`/
`nuclear_heating_shield` were already effectively pure in the source (`@staticmethod`, no
`self` access); `nuclear_heating_magnets` needed the ordinary `self.data` extraction, no
different in kind from `PlasmaDensityLimit.calculate_density_limit`.

## switches touched

- `.physics.itart` — **new, not in `switches.md`'s original 10 or the two added since
  (`i_tf_sc_mat`, `i_plant_availability`)**. Read by both `nuclear_heating_shield` (one
  2-way branch, `dr_shld_average`) and `nuclear_heating_magnets` (two more 2-way
  branches: `th_blanket_av`/`th_shield_av`, and the TF-coil-mass source for
  `tfc_nuc_heating`). Per `traceability_policy.md`'s split-by-default rule, the two
  branches' reads-sets genuinely differ (`itart==1` skips `dr_blkt_inboard`/
  `dr_shld_inboard`/`m_tf_coils_total`, `itart==0` skips `whttflgs`), which would
  nominally call for a split into two functions per the default. **Not split here**: the
  split decision was already made — one way, unified — by whoever wrote
  `nuclear_heating_shield`'s existing PROCESS staticmethod signature (it already takes
  `itart` as one plain `int` argument covering both branches, not two functions), and
  this port's job is to reproduce that function, not redesign it. Treated as an ordinary
  data-dependent branch (`needs-lax-cond-or-where`) inside each of the three ported
  functions, consistently. Recorded here as a genuine open switches-table gap (no
  existing row), with a recommendation below rather than a resolution.
  - **Is it topology-changing in the graph-assembly sense?** No evidence either way was
    found in this unit — `itart` is never read by `stellarator.py` itself (confirmed by
    grep: zero hits for `.itart` in `process/models/stellarator/*.py`), so nothing in the
    stellarator call path forces it to any particular value. It is an ordinary
    `InputVariable` (`itart`, `physics`, `choices=[0, 1]`, default `0`,
    `process/core/input.py:1064`) that a stellarator IN.DAT *could* set to `1` even
    though `itart` nominally means "spherical tokamak" — nothing in this file or
    `stellarator.py` validates the combination is physically sensible. Not provably dead
    on the stellarator path the way `avail_st`'s `itart==1` requirement is provably dead
    (`unit_registry.md`'s `i_plant_availability` row) — this is the opposite shape, a
    switch this unit's own PROCESS code does *not* guard against being set.
  - **Recommendation**: keep-static (plain traced argument, `jnp.where` branch), matching
    what the existing staticmethod signature already committed to. Split is available
    later if a future unit finds a reason `itart` needs topology-level treatment.

## calls into other models

None. All three methods are dead-ends — confirmed by direct read and by grep (see
"source" above). No `scipy`, no CoolProp, no other `Model` instance's method.

## JAX-difficulty flags

- **`logger.error(...)` inside `nuclear_heating_blanket`**, gated on
  `p_blkt_nuclear_heat_total_mw < 1` — a Python-level diagnostic side effect conditioned
  on a traced value. `minor`, `workaround-known`: dropped in the port (same treatment
  `coils.md` gives `intersect`'s `logger.error` bail-out — diagnostic only, does not
  affect the return value, and a traced function cannot branch on a data-dependent
  condition to decide whether to log anyway).
- **`itart`'s three 2-way branches** (one in `nuclear_heating_shield`, two more in
  `nuclear_heating_magnets`) — `minor`, `workaround-known`, `jnp.where` throughout (see
  "switches touched" above for why these aren't split).
- **`vv_density`'s guarded division** (`m_vv / vol_vv` only when `d_vv_all > 1e-6`, else
  `0.0`) — `minor`, `workaround-known`. The source's Python `if` short-circuits the
  division whenever it isn't needed (so `vol_vv == 0` alongside `d_vv_all <= 1e-6` never
  actually divides); a traced `jnp.where` evaluates both branches, so the port guards the
  denominator (`jnp.where(vol_vv == 0.0, 1.0, vol_vv)`) to keep the untaken branch finite
  — same `test_outputs_finite`/`test_gradient_finite` concern `fusion_reactions.md`'s
  `beam_slowing_down_state` already documents for the identical pattern.
- No CoolProp, no `scipy`, no `copy.deepcopy`, no data-dependent early exit/loop anywhere
  in the three in-scope methods.

## open questions

1. **A live call-site bug**: `stellarator.py`'s `blanket_neutronics()` (lines 440, 458)
   calls `self.hcpb.nuclear_heating_blanket()` and `self.hcpb.nuclear_heating_shield()`
   with **zero arguments**, but both are `@staticmethod`s requiring 2 and 7 explicit
   keyword arguments respectively (confirmed: neither has a default value for any
   parameter). This would raise `TypeError: nuclear_heating_blanket() missing 2 required
   positional arguments` (and similarly for `nuclear_heating_shield`) the moment a
   stellarator run with `blktmodel == 1` actually executes `blanket_neutronics()` (the
   arm `st_fwbs`'s S2 sub-computation lives in, per
   `stellarator_E_fwbs_synthesis.md` §4). **Not exercised by any existing test**:
   `tests/unit/models/stellarator/test_stellarator.py` only ever sets `blktmodel=0`
   (confirmed by grep — three hits, all `blktmodel=0`), so this branch appears to be
   dead in practice, not merely untested in principle. This is directly relevant to S2
   (`unit_registry.md`'s `st_fwbs` synthesis row, and `next_steps.md` §3) — whoever
   ports S2's `blktmodel==1` arm will hit this immediately, since `blanket_neutronics()`
   is the first thing that arm calls. Two candidate fixes exist (pass the same explicit
   keyword arguments `run()`'s own call sites already use at lines 164-182, since every
   value `blanket_neutronics()` needs is already on `self.data` at that point; or treat
   it as a genuine PROCESS bug to report upstream) — **not resolved here**, flagged for
   whoever picks up S2 or the consolidation pass. Not a blocker for *this* port, since the
   port targets the staticmethods' own declared (and internally self-consistent)
   signatures, which is what `run()`'s own call sites already use correctly.
2. **`itart`'s switches-table gap** (see "switches touched") — recorded here as a new
   finding, not added to `switches.md` itself (out of this unit's file-boundary; that
   table lives outside `models/blankets/`, reserved for the consolidation pass per this
   dispatch's boundary note).
3. **`.ccfe_hcpb.x_shield`'s consumer**, if any, beyond reporting — not traced in this
   audit (out of scope: would require auditing `write_output()`/`powerflow_calc`, neither
   in scope for this unit). Noted only so a future S2/S4 audit doesn't assume it feeds
   `nuclear_heating_shield` (it doesn't — see data footprint table).
