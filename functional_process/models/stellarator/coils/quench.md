---
kind: model-unit
status: reviewed
confidence: high
---

**Ported.** `quench.py` / `test_quench.py`. One composed tier-1 function
(`calculate_quench_protection`) plus its three already-pure helpers, tests passing.

## source
`process/models/stellarator/coils/quench.py` (228 lines, full file in scope). Four
functions: `max_dump_voltage` and `calculate_quench_protection_current_density` are
already pure (no `data` argument in the source); `calculate_vv_max_force_density_from_W7X_scaling`
takes `(rad_vv, data)`; `calculate_quench_protection` is the orchestrator, called once
from `coils/calculate.py`'s `st_coil` (registry unit #9, line 118).

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.rmajor` | read (2x) | explicit-arg | feeds `rad_vv_in`/`rad_vv_out` directly, and again as `rad_vv` (the source's own approximation for the VV major radius — see proposed-signature note) |
| `.physics.rminor` | read | explicit-arg | |
| `.build.dr_fw_plasma_gap_inboard`, `.dr_fw_inboard`, `.dr_blkt_inboard`, `.dr_shld_blkt_gap`, `.dr_shld_inboard` | read | explicit-arg | inboard VV radius |
| `.build.dr_fw_plasma_gap_outboard`, `.dr_fw_outboard`, `.dr_blkt_outboard`, `.dr_shld_outboard` | read | explicit-arg | outboard VV radius (`dr_shld_blkt_gap` shared with inboard) |
| `.physics.b_plasma_toroidal_on_axis` | read | explicit-arg | |
| `.tfcoil.c_tf_total` | read (2x) | explicit-arg | fed to the W7-X scaling call, and (new, see below) used to derive `coilcurrent` internally |
| `.tfcoil.t_tf_superconductor_quench` | read (2x) | explicit-arg | W7-X scaling call and `tau_quench` |
| `.build.dr_vv_inboard`, `.dr_vv_outboard` | read | explicit-arg | |
| `.tfcoil.t_tf_quench_detection` | read | explicit-arg | `t_detect` |
| `.tfcoil.f_a_tf_turn_cable_copper` | read (2x) | explicit-arg | `f_cu`, and the copper-current-density denominator |
| `.tfcoil.f_a_tf_turn_cable_space_extra_void` | read | explicit-arg | `1 - this` = `f_cond` |
| `.tfcoil.tftmp` | read | explicit-arg | `temp` |
| `.tfcoil.a_tf_turn_cable_space_no_void` | read | explicit-arg | `a_cable` |
| `.tfcoil.dx_tf_turn_general` | read | explicit-arg | squared, `a_turn` |
| `.tfcoil.a_tf_wp_conductor` | read | explicit-arg | copper-current-density denominator |
| `.tfcoil.e_tf_magnetic_stored_total_gj` | read | explicit-arg | feeds `tf_energy_stored` |
| `.tfcoil.n_tf_coils` | read (2x) | explicit-arg | `tf_energy_stored` denominator, and (new) `coilcurrent` derivation |
| `.tfcoil.c_tf_turn` | read | explicit-arg | `max_dump_voltage`'s `current` |
| `.superconducting_tfcoil.vv_stress_quench` | write | explicit-arg | |
| `.tfcoil.j_tf_wp_quench_heat_max` | write | explicit-arg | |
| `.rebco.coppera_m2` | write | explicit-arg | |
| `.tfcoil.v_tf_coil_dump_quench_kv` | write | explicit-arg | |
| `f_vv_actual` (return value, not stored — see below) | — | invented output | see cottax-node section |

No `implicit-io`, no `implicit-io-via-callee`, no `redundant-duplicate-write`.

**`coilcurrent` eliminated as a parameter — verified, not just flagged.** The source
signature is `calculate_quench_protection(coilcurrent, data)`. Grepped
`coils/calculate.py`: `winding_pack_total_size` (called at `st_coil` line 49, before
`calculate_quench_protection` at line 118) sets
`data.tfcoil.c_tf_total = data.tfcoil.n_tf_coils * coilcurrent * 1.0e6` (line 276) and
that is the *only* write to `c_tf_total` in the file. `coilcurrent` is therefore fully
determined by `c_tf_total`/`n_tf_coils`, both already parameters of
`calculate_quench_protection` for other reasons — carrying it as a third, separately-
sourced argument was redundant, not a genuine independent input. The port derives
`coilcurrent = c_tf_total / (n_tf_coils * 1.0e6)` internally instead of taking it as a
parameter, which removes the only real-`data`-field gap this unit had. Confirmed
numerically: the ported function and the real PROCESS `calculate_quench_protection`
(called with `coilcurrent` computed the same way) agree exactly on all 5 outputs at a
realistic point.

## proposed signature(s)

```python
def calculate_quench_protection(
    rmajor: float,
    rminor: float,
    dr_fw_plasma_gap_inboard: float,
    dr_fw_inboard: float,
    dr_blkt_inboard: float,
    dr_shld_blkt_gap: float,
    dr_shld_inboard: float,
    dr_fw_plasma_gap_outboard: float,
    dr_fw_outboard: float,
    dr_blkt_outboard: float,
    dr_shld_outboard: float,
    b_plasma_toroidal_on_axis: float,
    c_tf_total: float,
    t_tf_superconductor_quench: float,
    dr_vv_inboard: float,
    dr_vv_outboard: float,
    t_tf_quench_detection: float,
    f_a_tf_turn_cable_copper: float,
    f_a_tf_turn_cable_space_extra_void: float,
    tftmp: float,
    a_tf_turn_cable_space_no_void: float,
    dx_tf_turn_general: float,
    a_tf_wp_conductor: float,
    e_tf_magnetic_stored_total_gj: float,
    n_tf_coils: float,
    c_tf_turn: float,
) -> tuple[float, float, float, float, float]:
    # (f_vv_actual, vv_stress_quench, j_tf_wp_quench_heat_max, coppera_m2,
    #  v_tf_coil_dump_quench_kv)
    ...
```
`max_dump_voltage(tf_energy_stored, t_dump, current)` and
`calculate_quench_protection_current_density(tau_quench, t_detect, f_cu, f_cond, temp,
a_cable, a_turn)` are kept as already-pure helper functions, called internally — not
separately noded (see cottax-node section). `calculate_vv_max_force_density_from_w7x_scaling`
likewise: real `data` fields feed it, but only ever via this one call site.

## cottax node

**Actually written**, in `quench.py` (`QuenchProtection`, an `ExplicitFunction`),
registered in `functional_process/total_process.py`:

```python
class QuenchProtection(ExplicitFunction):
    f_vv_actual = OutputInto(superconducting_tfcoil)
    vv_stress_quench = OutputInto(superconducting_tfcoil)
    j_tf_wp_quench_heat_max = OutputInto(tfcoil)
    coppera_m2 = OutputInto(rebco)
    v_tf_coil_dump_quench_kv = OutputInto(tfcoil)

    def __call__(self, rmajor=From(physics), ..., c_tf_turn=From(tfcoil)):
        return calculate_quench_protection(rmajor, ..., c_tf_turn)
```
`.superconducting_tfcoil.f_vv_actual` is an **invented** `VarPath`: the source itself
never stores `f_vv_actual` to `data` either — `calculate_quench_protection` only
`return`s it, and `coils/calculate.py` forwards it straight into `coils/output.py`'s
`write(...)` for the printout (confirmed by grep). Grouped under `.superconducting_tfcoil`
alongside `vv_stress_quench` since the source computes both from the same intermediate
and assigns the latter there.

`max_dump_voltage`/`calculate_quench_protection_current_density`/
`calculate_vv_max_force_density_from_w7x_scaling` get no separate node: in their one real
call site several of their arguments (`tf_energy_stored`, `f_cond`, `rad_vv`) are derived
intermediates, not raw `data` fields — a standalone node for any of them would need
invented `VarPath`s for values that only exist inside this composed chain, same treatment
`stellarator_D_structure.md` gave `calculate_structure_masses`'s internal
`intercoil_surface`.

## tier signal
**Tier 1.** No internal solve, no calls into other not-yet-ported units, no
data-dependent Python control flow on a traced quantity (the source approximation
`rad_vv = rmajor` and the `coilcurrent` derivation are both unconditional).

## switches touched
None.

## calls into other models
None, after the `coilcurrent` elimination above — the last remaining cross-unit
dependency (on unit #9's `calculate_current`) is gone.

## JAX-difficulty flags
None found. Plain scalar arithmetic (`jnp.log`), no CoolProp, no dynamic shapes.

## open questions
1. Whether `.superconducting_tfcoil.f_vv_actual`'s invented name/area is the one a
   later synthesis pass will want — it's a genuinely reporting-only value in PROCESS
   today (see above), so there's no existing field to check it against.

## Derivative-safe power laws (`safe_pow` / `safe_sqrt`)

1 square root in this file has been rewritten from `x ** p` / `jnp.sqrt(x)` to
`models/safe_math.py`'s `safe_pow(x, p)` / `safe_sqrt(x)`.

**Why.** For `0 < p < 1` the function is continuous at `x == 0` and its derivative is
not: `d/dx x**p = p * x**(p-1) -> +inf`. JAX's JVP then returns `inf` along the
direction that perturbs `x` and `nan` (`inf * 0`) along every other, so the *value* is
right everywhere and the *Jacobian row* is poisoned. That is the defect class
`_audit/next_steps.md` §9 records; the most recent instance produced 46 non-finite
Jacobian cells and stalled a cold optimiser start at zero SQP steps, reported by the
solver as "the problem seems to be non-convex".

**Value identity, checked not asserted.** `safe_pow`/`safe_sqrt` dispatch on `x == 0`
and evaluate the identical expression otherwise, so every `x != 0` result is bit-for-bit
what it was, and the `x == 0` result is `0.0 ** p` / `sqrt(0.0)` -- again exactly what
the bare expression returns. Verified two ways: a hex-exact diff of every Tier-1
contract's output over every declared sample plus eight fresh fuzz draws (3655 points,
zero differing bits), and `run_mda_harness.py` unchanged at 492 agreements / 34
disagreements. PROCESS itself does not raise at `x == 0` here -- it is plain Python
`float.__pow__` / `numpy.sqrt`, both of which return `0.0` -- and the reference was
re-evaluated at each boundary point to confirm it returns the port's number.

**What changed is only the derivative at exactly `x == 0`**, which becomes `0` instead
of `inf`/`nan` -- the same convention JAX already uses at `jnp.maximum`'s kink.

`Tier1Contract.test_gradient_finite_at_zero` (`--fp-gradients`) now checks the whole
class automatically: it zeroes each differentiable argument in turn and requires a
finite Jacobian wherever the value is finite.
