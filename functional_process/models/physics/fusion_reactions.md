---
kind: model-unit
status: draft
confidence: medium
---

**Cottax nodes added** (update, not part of the original placeholder): `fusion_reactions.py`
now declares `FusionRates`/`SetFusionPowers` (`ExplicitFunction`s), plus the plain ported
functions they wrap. `beam_fusion()` and its `beam_reaction_rate_coefficient()` helper are
**not** ported — see "tier signal" and "JAX-difficulty flags" below; the rest of
`beam_fusion`'s dependency chain (everything the `scipy.integrate.quad` call does not
touch) is ported as standalone functions with no cottax node, ready for whenever that
blocker resolves.

## source

`process/models/physics/fusion_reactions.py` (1597 lines). Registry unit #19: scope is
whatever `stellarator.py`'s `st_phys()` (chunk 1B, `stellarator_B_st_phys.md`) actually
calls, found via a bare `import ... as reactions` module alias.

**Scope check against the actual call site** (`process/models/stellarator/stellarator.py`
lines 1984-2077): confirmed to match the registry's stated method list exactly, no
correction needed —

```python
fusion_reactions = reactions.FusionReactionRate(self.plasma_profile, self.data)
fusion_reactions.deuterium_branching(self.data.physics.temp_plasma_ion_vol_avg_kev)
fusion_reactions.calculate_fusion_rates()
fusion_reactions.set_physics_variables()
...
reactions.beam_fusion(...)               # only under `p_hcd_beam_injected_total_mw != 0`
                                          # and non-ignited plasma
...
reactions.set_fusion_powers(...)
```

`calculate_fusion_rates()` is not itself the whole computation, though — it calls four
`FusionReactionRate` methods not named in the registry row (`dt_reaction`,
`dhe3_reaction`, `dd_helion_reaction`, `dd_triton_reaction`), which are therefore in scope
**transitively** (same "the registry names the entry point, not everything it reaches"
situation as chunk 1B's own `st_phys` audit). Also reached, transitively, from those four:
the module-level `bosch_hale_reactivity()` and `fusion_rate_integral()`. All four are
audited and ported here; none required expanding scope beyond registry unit #19's own
file.

## data footprint

**`FusionReactionRate.__init__(plasma_profile, data)`** — takes the whole `PlasmaProfile`
object (registry unit #12, `physics/plasma_profiles.py`, already audited and partially
ported) and the whole `DataStructure`. Only two arrays are ever read off `plasma_profile`:
`.neprofile.profile_x`/`.profile_y` and `.teprofile.profile_x`/`.profile_y` — the same
`data`-backdoor-closure situation unit #12's own audit already established and ported
around (`ProfileFactors` in `plasma_profiles.py` takes `ne_profile_y`/`te_profile_y` as
plain array arguments, not an injected profile object).

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.temp_plasma_ion_vol_avg_kev` | read | explicit-arg | passed into `deuterium_branching()` explicitly by `st_phys`; read again *implicitly* off `self.data` inside all four reaction methods for the ion-temperature-profile scaling — same immutable value both times within one `st_phys` call, so `local-intermediate` in effect even though the class routes it through two different access paths |
| `.physics.temp_plasma_electron_vol_avg_kev` | read | explicit-arg | denominator of the same scaling, all four reactions |
| `.physics.f_plasma_fuel_deuterium` | read | explicit-arg | `dt_reaction`, `dhe3_reaction`, `dd_helion_reaction` (squared), `dd_triton_reaction` (squared) |
| `.physics.f_plasma_fuel_tritium` | read | explicit-arg | `dt_reaction` only |
| `.physics.f_plasma_fuel_helium3` | read | explicit-arg | `dhe3_reaction` only |
| `.physics.nd_plasma_fuel_ions_vol_avg` | read | explicit-arg | all four reactions |
| `.physics.nd_plasma_electrons_vol_avg` | read | explicit-arg | all four reactions, as the profile-normalisation denominator |
| `.physics.temp_plasma_electron_profile_kev` | read | explicit-arg | **reused minted `VarPath`** — this *is* `teprofile.profile_y`, the same array `plasma_profiles.md`/`ProfileFactors` already minted this name for (same `PlasmaProfile` instance, same object). Not a new mint. |
| `.physics.nd_plasma_electron_profile` | read | explicit-arg | **reused minted `VarPath`** — `neprofile.profile_y`, same situation |
| `.physics.radius_plasma_profile_norm` | read | explicit-arg | **reused minted `VarPath` (corrected)** — `teprofile.profile_x` and `neprofile.profile_x` are two distinct array objects but numerically identical — both are built by `profiles.py`'s `Profile.normalise_profile_x()` acting on `np.arange(n_plasma_profile_elements)`, with no per-instance randomness or divergence, so this port takes **one** shared grid argument rather than two. Verified, not assumed: `plasma_profiles.py`'s own test harness stub (`test_plasma_profiles.py`'s `_plasma_profile`) already sets `neprofile.profile_x` and `teprofile.profile_x` to the identical array for the same reason. **This row previously said "no existing `VarPath` covers `profile_x`" and minted a fresh `.physics.profile_x`** — wrong by the time this was written: `profiles.py`'s `ProfileGrid` (a source node) already mints this exact grid as `.physics.radius_plasma_profile_norm`, and `radiation_power.py`'s own node already reads it under that name. The block-by-block MDA-vs-PROCESS comparison harness caught the duplicate (`.physics.profile_x` showed up as an ungrounded boundary input shadowing an already-real one); fixed by reading `radius_plasma_profile_norm` here too. |
| `plasma_profile.neprofile.profile_dx` | read (by PROCESS, not this port) | dropped | passed to `scipy.integrate.simpson(..., x=..., dx=...)` alongside `x`; `scipy` ignores `dx` whenever `x` is given (confirmed empirically — see `plasma_profiles.md`'s identical finding for `pedestal_parameterisation`). Not an argument of the ported function. |
| `.physics.f_dd_branching_trit` | write, on the **instance** (`self.f_dd_branching_trit`), *not* on `data` | implicit-io | set by `.deuterium_branching()`, read by `dd_helion_reaction`/`dd_triton_reaction` *within the same `FusionReactionRate` instance* — an always-paired call sequence at the `st_phys` call site (`deuterium_branching()` then `calculate_fusion_rates()`, unconditionally, nothing else touches the instance in between), so this is `local-intermediate` in spirit despite crossing a method boundary. Resolved in the port by having the `calculate_fusion_rates` node call `calculate_deuterium_branching_trit` itself rather than threading a separate port — see "cottax node" below. |
| `.physics.fusrat_plasma_dt_profile` / `_dhe3_profile` / `_dd_helion_profile` / `_dd_triton_profile` | write | — | one array write per reaction method, unconditional, **not** part of `set_physics_variables()`'s copy list — a second write boundary inside `calculate_fusion_rates()` itself. Read nowhere else in scope (confirmed by grep: only `core/io/plot/summary.py`, a reporting/plotting module, and `physics.py`'s mfile writer, both out of scope) |
| `.physics.pden_plasma_alpha_mw`, `.pden_non_alpha_charged_mw`, `.pden_plasma_neutron_mw`, `.fusden_plasma`, `.fusden_plasma_alpha`, `.proton_rate_density`, `.sigmav_dt_average`, `.dt_power_density_plasma`, `.dhe3_power_density`, `.dd_power_density`, `.f_dd_branching_trit` | write | — | `set_physics_variables()`'s copy list, from the instance's accumulated `sum_fusion_rates()` totals (or, for the DT/D-3He/D-D power densities and the DT `sigmav`, straight instance-attribute passthrough — only D-D's is a real accumulation, `dd_helion_reaction`'s `+=` then `dd_triton_reaction`'s `+=` onto the same field) |
| `bosch_hale_reactivity(ion_temperature_profile, reaction_constants)` computed **twice** per reaction (once inside `fusion_rate_integral` for the `sigmav` integration, once again directly for the `fusrat_plasma_*_profile` write) | — | redundant computation, not a redundant write | Same pure inputs both times (`ion_temperature_profile` is identical in both call sites within one reaction method), so the two calls return identical arrays. Not a data-footprint issue (nothing is written twice), but the pure port computes it once per reaction and reuses the result for both outputs — noted so the simplification isn't mistaken for a missed term. |

**`beam_fusion()`'s own footprint** (all explicit-arg, straight from `st_phys`'s call —
see chunk 1B's source excerpt): `beamfus0`, `betbm0`, `b_plasma_total`, `c_beam_total`,
`nd_plasma_electrons_vol_avg`, `nd_plasma_fuel_ions_vol_avg`, `dlamie`, `e_beam_kev`,
`f_plasma_fuel_deuterium`, `f_plasma_fuel_tritium`, `f_beam_tritium`,
`temp_plasma_electron_density_weighted_kev`, `vol_plasma`,
`n_charge_plasma_effective_mass_weighted_vol_avg`. No `implicit-io` here — every argument
is a plain explicit read, and `beam_fusion` itself has no internal iteration
(`scipy.integrate.quad` lives one level deeper, in `beam_reaction_rate_coefficient`, not
here). The blocker is JAX-traceability/value-agreement, not a data-footprint problem — see
"tier signal" and "JAX-difficulty flags".

**`set_fusion_powers()`'s footprint** — all explicit-arg, straight from `st_phys`'s call
(lines 2068-2077): `f_alpha_electron`, `f_alpha_ion`, `p_beam_alpha_mw`,
`pden_non_alpha_charged_mw`, `pden_plasma_neutron_mw`, `vol_plasma`,
`pden_plasma_alpha_mw`, `f_p_alpha_plasma_deposited`. One entry worth flagging:
`.physics.p_beam_alpha_mw` is produced by `beam_fusion()` when beam heating is active, or
left at whatever `DataStructure`'s default/prior value is otherwise (the `if`/`else` at
`stellarator.py` L2006-2053 only resets `fusden_total`/`fusden_alpha_total`/
`p_dt_total_mw` in the `else` arm, not `p_beam_alpha_mw` itself) — since `beam_fusion` is
blocked (below), this input currently has **no producer node** in the pure graph, the same
"iteration variable is a boundary input with no forward producer" situation `schema.md`
already names for `conditional-ownership-by-run-config`, except here the missing producer
is a blocked unit rather than an iteration variable. Not a reason to withhold
`SetFusionPowers`'s own node — its job is just to consume whatever value arrives at that
`VarPath`.

## proposed signature(s)

```python
def calculate_deuterium_branching_trit(ion_temperature: float) -> float:
    """Ports `.deuterium_branching()`."""
    ...

def bosch_hale_reactivity(
    ion_temperature_profile: jax.Array, reaction_constants: BoschHaleConstants
) -> jax.Array:
    """Ports the module-level function of the same name, unchanged in shape."""
    ...

def calculate_fusion_rates(
    profile_x: jax.Array,
    te_profile_y: jax.Array,
    ne_profile_y: jax.Array,
    temp_plasma_ion_vol_avg_kev: float,
    temp_plasma_electron_vol_avg_kev: float,
    f_plasma_fuel_deuterium: float,
    f_plasma_fuel_tritium: float,
    f_plasma_fuel_helium3: float,
    nd_plasma_fuel_ions_vol_avg: float,
    nd_plasma_electrons_vol_avg: float,
    f_dd_branching_trit: float,
) -> tuple:  # 11 scalars (set_physics_variables' copy list) + 4 profile arrays
    """Composes `.calculate_fusion_rates()` **and** `.set_physics_variables()` — see
    'cottax node' below for why these two collapse into one pure function rather than
    staying separate."""
    ...

def set_fusion_powers(
    f_alpha_electron: float,
    f_alpha_ion: float,
    p_beam_alpha_mw: float,
    pden_non_alpha_charged_mw: float,
    pden_plasma_neutron_mw: float,
    vol_plasma: float,
    pden_plasma_alpha_mw: float,
    f_p_alpha_plasma_deposited: float,
) -> tuple:  # 11 scalars, already an exact 1:1 port of the source
    ...
```

**Blocked, not attempted as a node** (see tier signal / JAX-difficulty flags):
```python
def beam_fusion(...) -> tuple[float, float, float]:
    ...  # unchanged signature from source; ported value-for-value except for the one
         # sub-call that cannot be
```

**Ported as standalone pure functions, no node** (building blocks for whenever
`beam_reaction_rate_coefficient`'s blocker resolves — see below):
`beam_slowing_down_state`, `fast_ion_pressure_integral`, `beam_target_reaction_rate`,
`alpha_power_beam`, `beam_fusion_cross_section` (renamed from the source's
`_beam_fusion_cross_section`, since it is now independently useful rather than a private
implementation detail of the one blocked function), `hot_beam_fusion_reaction_rate_integrand`
(similarly renamed from `_hot_beam_fusion_reaction_rate_integrand`).

## tier signal

- `deuterium_branching`, `bosch_hale_reactivity`, `dt_reaction`/`dhe3_reaction`/
  `dd_helion_reaction`/`dd_triton_reaction`, `calculate_fusion_rates`,
  `set_physics_variables`, `set_fusion_powers` — **tier 1**. No internal iteration
  anywhere in this group; `scipy.integrate.simpson` is a fixed, non-adaptive quadrature on
  a grid PROCESS already supplies (not a solve), same classification `plasma_profiles.md`
  already gave the same construct. Verified end-to-end against a real (stubbed-profile)
  `FusionReactionRate` instance before writing the final port — every value matched
  PROCESS bit-for-bit at float64 precision (not just to `rtol=1e-12`), including the four
  `fusrat_plasma_*_profile` arrays and the two composed methods' full field lists.

- `beam_slowing_down_state`, `fast_ion_pressure_integral`, `beam_target_reaction_rate`,
  `alpha_power_beam`, `beam_fusion_cross_section`,
  `hot_beam_fusion_reaction_rate_integrand` — **tier 1** individually (no iteration, pure,
  self-contained), and ported/tested as such. `beam_fusion_cross_section` has a genuine
  data-dependent three-way clamp (`e_beam_kev < 10`, `> 1e4`, else) — ordinary
  `needs-lax-cond-or-where`, not a tier concern.

- `beam_reaction_rate_coefficient` (and therefore `beam_fusion`, which calls it twice) —
  **not cleanly tier 1 or tier 2; a third case this schema doesn't yet name.** It has no
  internal *solve* (nothing converges to a fixed point, so it is not tier 2's "internal
  iterative loop over model state" either), but its `scipy.integrate.quad` call is an
  adaptive, tolerance-stopped numerical integration whose own accuracy is bounded, not
  exact — and empirically, that bound is much looser than tier 1's machine-precision value
  bar. Measured directly (not assumed): replacing `quad` with fixed Gauss-Legendre
  quadrature at 16, 32, 64, 128, and 256 nodes over the same interval, relative
  disagreement with `scipy.integrate.quad`'s answer **plateaus at ~1e-6 to 3e-6 and does
  not improve with more nodes** — the signature of the integrand's own smoothness limit
  (`beam_fusion_cross_section`'s hard clamps at 10 keV and 10 MeV kink the integrand),
  not of quadrature error. So even a *better* quadrature than PROCESS's own would disagree
  with PROCESS's answer by ~1e-6 relative, four orders outside `MACHINE_PRECISION`
  (`rtol=1e-12`) — a tier-1 value test could not pass here even with a numerically
  superior port, for the same structural reason tier 2 exists ("PROCESS's own answer is
  not ground truth"), but via a different mechanism (bounded-but-inexact quadrature, not
  an under-converged fixed-point loop). Left **audit-only**, per this directive's
  instruction to leave genuinely-entangled units undone rather than force a tier onto
  them — flagging for the schema/test-harness review that a residual-based tier-2-style
  pass criterion (compare the integral's own residual, not PROCESS's value) might be the
  right fit once someone wants this ported, rather than inventing a fourth harness tier.

## switches touched

None. No `data.<area>.i_*` field is read anywhere in this file's in-scope methods —
`beam_fusion`'s gating (`p_hcd_beam_injected_total_mw != 0` and
`i_plasma_ignited != IGNITED`) happens at the `st_phys` call site, not inside this unit.

## calls into other models

- `FusionReactionRate.__init__` takes a `PlasmaProfile` instance (registry unit #12,
  audited and partially ported, `functional_process/models/physics/plasma_profiles.md`)
  but only ever reads its two profile objects' `.profile_x`/`.profile_y` arrays — no
  method call, so this unit does not depend on unit #12's own methods being ported, only
  on the array values existing (which the pure port takes as plain arguments, same
  `data`-backdoor closure unit #12 already performed for its own port).
- No other model calls anywhere in the in-scope methods.

## JAX-difficulty flags

- **`scipy.integrate.quad` inside `beam_reaction_rate_coefficient`** —
  `non-traceable-external-call`, `blocker`. Same class of flag as CoolProp
  (`traceability_policy.md`), with one addition specific to this case: even a
  hypothetical `pure_callback` wrapper (preserving value agreement, sacrificing
  differentiability) would not clear tier 1's `test_outputs_finite`, which runs
  unconditionally on every test invocation and differentiates every argument via
  `jacfwd` — a `pure_callback`-wrapped scalar has no JVP rule by default, so the check
  would error outright, not merely fail a tolerance. Precise boundary for whoever resolves
  this: inputs crossing into the opaque call are `relative_mass_ion`, `critical_velocity`,
  `beam_energy_kev` (all continuous, all differentiable in the surrounding physics);
  output is one scalar rate coefficient (m^3/s).
- **`beam_fusion_cross_section`'s three-way clamp** (`e_beam_kev < 10.0` / `> 1.0e4` /
  else) — `workaround-known` (`needs-lax-cond-or-where`), already implemented in the port
  with nested `jnp.where`. Ordinary, not a blocker.
- **`sigmav[t_mask] = 0.0` in-place mask assignment** in `bosch_hale_reactivity` —
  `workaround-known`, `jnp.where(ion_temperature_profile == 0.0, 0.0, sigmav)` in the
  port.
- **`scipy.integrate.simpson(y, x=..., dx=...)`** in all four reaction methods — same
  non-uniform-formula trap `plasma_profiles.md` already found and fixed (`_simpson`,
  imported from that port rather than re-derived — see "cottax node" below). Flagging
  again here because it recurs, not because it is new: worth the test harness noting this
  is now the *second* unit this exact bug class hit.
- No CoolProp calls, no `copy.deepcopy`, no `scipy.optimize`/`fsolve` anywhere in this
  file's in-scope methods.

## cottax node

Two nodes, matching the two-composed-function proposed signature above.

**`FusionRates`** fuses all three in-scope `FusionReactionRate` methods
(`.deuterium_branching()`, `.calculate_fusion_rates()`, `.set_physics_variables()`) into
one node, rather than exposing `calculate_deuterium_branching_trit` as its own node. Not a
simplification taken lightly: `deuterium_branching()`'s only effect PROCESS ever reads
back is an *instance* attribute (`self.f_dd_branching_trit`), which has no `VarPath` of
its own — it only becomes `.physics.f_dd_branching_trit` once `set_physics_variables()`
runs. Two nodes both trying to own that one `VarPath` (one from a hypothetical
`DeuteriumBranchingTrit` node, one from `FusionRates`) is exactly what `to_graph` rejects.
`st_phys` also never calls `deuterium_branching()` anywhere except immediately before
`calculate_fusion_rates()`, unconditionally, on the same instance — a `local-intermediate`
in every respect except the method boundary. So `FusionRates.__call__` computes
`f_dd_branching_trit = calculate_deuterium_branching_trit(temp_plasma_ion_vol_avg_kev)`
internally and forwards it into `calculate_fusion_rates`; `calculate_deuterium_branching_trit`
remains separately defined and separately tested (its own `Tier1Contract` case, directly
against PROCESS's `.deuterium_branching()`) but gets no `ExplicitFunction` wrap of its
own.

`_simpson` is **imported from `functional_process.models.physics.plasma_profiles`**
rather than re-derived — same exact algorithm needed for the same reason (`scipy`'s
general non-uniform composite rule, gradient-verified there against the same trap this
file's audit re-confirms). A cross-unit import rather than a duplicate, since a second,
independently-typed copy of a correctness-critical quadrature formula is a liability, not
caution.

```python
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, Input, Output

class FusionRates(ExplicitFunction):
    pden_plasma_alpha_mw = Output(lambda s: s.physics.pden_plasma_alpha_mw)
    pden_non_alpha_charged_mw = Output(lambda s: s.physics.pden_non_alpha_charged_mw)
    pden_plasma_neutron_mw = Output(lambda s: s.physics.pden_plasma_neutron_mw)
    fusden_plasma = Output(lambda s: s.physics.fusden_plasma)
    fusden_plasma_alpha = Output(lambda s: s.physics.fusden_plasma_alpha)
    proton_rate_density = Output(lambda s: s.physics.proton_rate_density)
    sigmav_dt_average = Output(lambda s: s.physics.sigmav_dt_average)
    dt_power_density_plasma = Output(lambda s: s.physics.dt_power_density_plasma)
    dhe3_power_density = Output(lambda s: s.physics.dhe3_power_density)
    dd_power_density = Output(lambda s: s.physics.dd_power_density)
    f_dd_branching_trit = Output(lambda s: s.physics.f_dd_branching_trit)
    fusrat_plasma_dt_profile = Output(lambda s: s.physics.fusrat_plasma_dt_profile)
    fusrat_plasma_dhe3_profile = Output(lambda s: s.physics.fusrat_plasma_dhe3_profile)
    fusrat_plasma_dd_helion_profile = Output(
        lambda s: s.physics.fusrat_plasma_dd_helion_profile
    )
    fusrat_plasma_dd_triton_profile = Output(
        lambda s: s.physics.fusrat_plasma_dd_triton_profile
    )

    def __call__(
        self,
        profile_x=Input(lambda s: s.physics.radius_plasma_profile_norm),
        te_profile_y=Input(lambda s: s.physics.temp_plasma_electron_profile_kev),
        ne_profile_y=Input(lambda s: s.physics.nd_plasma_electron_profile),
        temp_plasma_ion_vol_avg_kev=Input(lambda s: s.physics.temp_plasma_ion_vol_avg_kev),
        temp_plasma_electron_vol_avg_kev=Input(
            lambda s: s.physics.temp_plasma_electron_vol_avg_kev
        ),
        f_plasma_fuel_deuterium=Input(lambda s: s.physics.f_plasma_fuel_deuterium),
        f_plasma_fuel_tritium=Input(lambda s: s.physics.f_plasma_fuel_tritium),
        f_plasma_fuel_helium3=Input(lambda s: s.physics.f_plasma_fuel_helium3),
        nd_plasma_fuel_ions_vol_avg=Input(lambda s: s.physics.nd_plasma_fuel_ions_vol_avg),
        nd_plasma_electrons_vol_avg=Input(lambda s: s.physics.nd_plasma_electrons_vol_avg),
    ):
        f_dd_branching_trit = calculate_deuterium_branching_trit(
            temp_plasma_ion_vol_avg_kev
        )
        return calculate_fusion_rates(
            profile_x, te_profile_y, ne_profile_y,
            temp_plasma_ion_vol_avg_kev, temp_plasma_electron_vol_avg_kev,
            f_plasma_fuel_deuterium, f_plasma_fuel_tritium, f_plasma_fuel_helium3,
            nd_plasma_fuel_ions_vol_avg, nd_plasma_electrons_vol_avg,
            f_dd_branching_trit,
        )
```

`SetFusionPowers` is a direct 1:1 wrap of `set_fusion_powers`, Inputs/Outputs read
straight off the `st_phys` call site's argument/assignment lists (see "data footprint"
above for the one input with no current producer, `.physics.p_beam_alpha_mw`).

## open questions

1. **`.physics.p_beam_alpha_mw`'s missing producer** (see data footprint) is a real gap
   in the eventual graph, not just a documentation note — `SetFusionPowers` will not be
   assemblable end-to-end into `total_process.py` until either `beam_fusion` is resolved
   or a cut-point default is chosen for the non-beam-heating configuration. Not decided
   here; flagging for whoever wires this node in.
2. **`beam_reaction_rate_coefficient`'s blocker is a new case for the schema/harness**,
   not quite tier 1 or tier 2 (see tier signal). Recommend the schema explicitly name this
   pattern — "PROCESS's own numerical integration has a bounded, non-machine-precision
   error" — the same way it already names CoolProp for opaque external calls, so the next
   unit that hits `scipy.integrate.quad`/`fixed_quad`/similar doesn't have to re-derive the
   Gauss-Legendre-plateau experiment from scratch.
3. **Not independently re-verified**: whether `plasma_profile.neprofile.profile_x` and
   `.teprofile.profile_x` are *guaranteed* identical across every code path that
   constructs a `PlasmaProfile` (only `profiles.py`'s construction was checked here, per
   the minted-`VarPath` note above) — low-confidence on this point specifically, since it
   rests on reading `profiles.py`'s constructors rather than unit #21's own (still
   pending) audit.
