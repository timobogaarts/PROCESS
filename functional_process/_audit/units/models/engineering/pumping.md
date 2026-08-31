---
kind: model-unit
status: draft
confidence: high
---

**Not a numbered-node unit — it registers nothing.** Like `ivc_functions.md`, this
record documents *shared leaf helpers*: three pure functions called from inside other
units' bodies, none of which owns a `VarPath`. It has a registry row because it has a
record and every record needs one.

Ported 2026-08-31. **Read this section before estimating what it buys.**

## source

`process/models/engineering/pumping.py` (partial — 3 of the file's 3 functions plus one
enum left behind).

- `calculate_reynolds_number` (lines 175-201)
- `darcy_friction_haaland` (lines 46-78)
- `gnielinski_heat_transfer_coefficient` (lines 81-172)

Not ported: `CoolantType`, an `IntEnum` whose `full_name` is a CoolProp fluid name. It
is a lookup key for the external library, not arithmetic, and nothing in the port
consumes it.

## reachability — the honest framing

Both call sites are behind `.fwbs.i_p_coolant_pumping == 2` (`MECHANICAL`):

| caller | functions used |
|---|---|
| `process/models/blankets/blanket_library.py:3220,3233` (`coolant_friction_pressure_drop`) | `calculate_reynolds_number`, `darcy_friction_haaland` |
| `process/models/fw.py:492` (`FirstWall.fw_temp`) | `gnielinski_heat_transfer_coefficient` |

**No tracked regression input selects that arm** — `large_tokamak_eval.IN.DAT:172` sets
`i_p_coolant_pumping = 3`. So this port unblocks no configuration today and fixes no
machine. It is for arbitrary-`IN.DAT` support.

**It does not lift `indat.py`'s refusal of arm 2 either.** That refusal
(`indat.py:1115-1121`) is keyed on CoolProp, `_audit/next_steps.md` §5's unresolved
wrapping policy — and correctly so: every one of the density, viscosity, heat-capacity
and thermal-conductivity arguments these three consume comes from
`FluidProperties.of(...)`, in `fw_temp` and in `thermo_hydraulic_model`. What changes is
that when that policy is settled, the arithmetic sitting behind it is already validated
rather than pending. The refusal text should be amended to say so; the wording is in the
porting agent's report and was **not** applied here, because `indat.py` was owned by
another agent at the time.

## write trace — every write against its next use

The standing check, run before porting. Three findings, none of which changed what was
ported, and one of which is a real dead return.

1. **`fw_temp` returns four values; two are dead at every call site.** `fw.py:670-675`
   returns `(tpeakfw, heatcap_fw_coolant_average, den_fw_coolant_average,
   mflow_fw_coolant)`. The function has exactly two callers, both in
   `blanket_library.py` (`:2308`, `:2322`), and both discard positions 1 and 2 — the
   inboard call with bare `_`, the outboard with `_cf`/`_rhof`. So
   `heatcap_fw_coolant_average` and `den_fw_coolant_average` are computed, returned and
   thrown away on every evaluation, unconditionally. Neither is downstream of any
   function in this record, so nothing here shrank; it is recorded because whoever ports
   `fw_temp` itself should port a two-value return, not a four-value one.

2. **`deltat_solid_1D` is output-only** (`fw.py:534`, read only at `:668` inside the
   `if output:` block). It is one of `tkfw`'s two consumers — the other, `deltat_solid`
   at `:556`, is the live one — and the source says why: *"Model B is given for
   comparison, Model C is used"*. So `eurofer97_thermal_conductivity`'s result feeds one
   live path and one report line. This does **not** make the function dead; it makes
   Model B dead, and a port of `fw_temp` can drop it.

3. **`.fwbs.temp_fw_peak` is a genuine one-variable cycle, not a dead write.** It is
   written at `blanket_library.py:2333` as the max of the two peak temperatures
   `fw_temp` returns, and read at `fw.py:476` — *inside `fw_temp` itself*, on the next
   pass, to form `temp_k = (temp_fw_coolant_out + temp_fw_peak) / 2`, which is
   `eurofer97_thermal_conductivity`'s `temp` argument. The source comment is explicit
   that this is last iteration's estimate. It is closed only by `Caller.call_models`'s
   up-to-10-pass Gauss-Seidel, and it is a live read elsewhere too (constraint 39,
   `core/solver/constraints.py:1063-1070`), so it is not removable. **Whoever ports
   `fw_temp` inherits an SCC**, not a straight-line chain — worth knowing before
   budgeting that work.

Everything in `coolant_friction_pressure_drop`'s chain is live: `darcy_friction_factor`
feeds `f_straight` and both `elbow_coeff` calls, all three feed `dpres_total`, which is
returned and consumed by `total_pressure_drop` (`blanket_library.py:2950`) and thence by
six pressure-drop call sites at `:1426-1528`.

## data footprint

All three are already pure in `process/` — module-level `def`s, explicit float
arguments, no `self.data` access anywhere. Nothing to classify.

| VarPath | read/write | classification | note |
|---|---|---|---|
| *(none)* | — | — | pure math in all three |

## proposed signature(s)

```python
def calculate_reynolds_number(*, den_coolant, vel_coolant, radius_channel, visc_coolant)
def darcy_friction_haaland(*, reynolds, roughness_channel, radius_channel)
def gnielinski_heat_transfer_coefficient(
    *, mflux_coolant, den_coolant, radius_channel, heatcap_coolant,
    visc_coolant, thermcond_coolant, roughness_channel,
)
```

Unchanged from source except that the port makes them keyword-only, per this package's
convention. PROCESS's own call sites already pass every argument by keyword, so the
references bind against the same kwargs with no adapter.

## cottax node

**None.** All three are leaf pure functions called from inside `fw_temp` /
`coolant_friction_pressure_drop`, neither of which is ported; neither owns a `VarPath`,
so there is nothing for `From`/`OutputInto` to wrap and no slot to occupy. Same shape as
`ivc_functions.md`, and for the same reason. When arm 2 is eventually written, the nodes
will be `fw_temp`'s and `coolant_friction_pressure_drop`'s, with these three as their
interior arithmetic.

## tier signal

**Tier 1.** No iteration, no `self.data`, no CoolProp *inside these three* — the
CoolProp calls are in the callers, which is exactly why the leaves could be ported ahead
of the policy.

**Sample provenance.** `tests/unit/models/engineering/test_pumping.py` has one
`pytest.approx` point for each of the three; all three are lifted verbatim as legacy
samples. Fuzz boxes are hand-drawn (none of these arguments is a PROCESS iteration
variable, so `bounds_from_iteration_variables` has nothing to offer) and their reasoning
is in each contract's docstring.

## switches touched

None inside the three. The *arm* that reaches them is `.fwbs.i_p_coolant_pumping == 2`,
which is the callers' switch, not theirs.

## calls into other models

`gnielinski_heat_transfer_coefficient` calls the other two, in-file. Nothing else.

## JAX-difficulty flags

- **`safe_pow`/`safe_sqrt` applied at two sites, both in `gnielinski_*`**:
  `safe_sqrt(f / 8)` and `safe_pow(pr, 2/3)`. Both are `safe_math`'s documented
  `0 < p < 1`-at-zero case; both are value-identical away from zero and only replace an
  `inf` derivative with `0` at it.
- **Deliberately *not* applied in `darcy_friction_haaland`**, and the reason is in the
  port's docstring rather than left to inference: the exponents are `1.11` and `-2`,
  neither in the `0 < p < 1` window. `x ** 1.11` has derivative `1.11 * x ** 0.11`,
  finite (`0`) at `x == 0` — which matters, because `roughness_channel == 0` is a
  physically reachable smooth-pipe input. The `** (-2)` is singular only where
  `1.8 * log10(bracket) == 0`, an ordinary domain restriction outside any turbulent
  regime.
- **Three `logger.error` range checks dropped** (`pumping.py:159-172`, Reynolds,
  Prandtl, and negative friction factor). All three sit *after* the return value is
  computed and neither clamp nor raise, so dropping them changes no value. The bounds
  they check are restated in the port's docstring and used to draw the fuzz box.

## open questions

- **Should the `logger.error` diagnostics survive the port in some form?** They are
  genuinely useful information — a Gnielinski evaluation outside `3e3 < Re < 5e6` is a
  modelling error, not a numerical one — and a traced body cannot emit them. No policy
  exists for this in the port yet; flagged rather than improvised.
- **The `deg_*`/`elbow_coeff` half of `coolant_friction_pressure_drop` is unported**, so
  `darcy_friction_haaland`'s consumer chain is only half present. Whoever ports arm 2
  needs `elbow_coeff` and `pipe_hydraulic_diameter` from the same file.
