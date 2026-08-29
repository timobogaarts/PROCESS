---
kind: model-unit
status: draft
confidence: medium
---

**Partially ported (2026-08-26), see "## ported" below.** The subset live on
`tests/regression/input_files/large_tokamak_eval.IN.DAT`: the `i_bootstrap_current == 4`
(Sauter) bootstrap fraction with its whole collisionality/transport-coefficient chain,
the cap at `f_c_plasma_bootstrap_max`, the `i_diamagnetic_current == 0` and
`i_pfirsch_schluter_current == 0` empty arms, and the current-fraction bookkeeping that
produces `.physics.f_c_plasma_auxiliary` and `.physics.f_c_plasma_inductive`. No
`unit_registry.md` row and no `next_steps.md` edit — registration is the consolidation
pass's job (`next_steps.md` §4b); see "## ported"'s registration instructions.

## source

Nominally `process/models/physics/bootstrap_current.py` (2529 lines, two classes).
**In practice the unit spans two files**, and that is the first finding of this audit.

### the chain is not one file

`PlasmaBootstrapCurrent.run()` (`bootstrap_current.py:81-262`) produces
`.current_drive.f_c_plasma_bootstrap` and stops. Everything that *uses* it — the cap, the
plasma-driven total, the auxiliary and inductive fractions — is written inline in
`Physics.run()`, `process/models/physics/physics.py:543-588`, in the six statements
immediately after the `self.plasma_bootstrap_current.run()` call at `:543`:

| step | site | writes |
|---|---|---|
| 1 | `bootstrap_current.py:137-143` | `.current_drive.f_c_plasma_bootstrap_sauter`, `.physics.j_plasma_bootstrap_sauter_profile` |
| 2 | `bootstrap_current.py:250-262` + `:264-298` | `.current_drive.f_c_plasma_bootstrap` (the switch's answer) |
| 3 | `physics.py:545-556` | `.current_drive.f_c_plasma_bootstrap` (capped), `.physics.err242` |
| 4 | `physics.py:558-562`, `:569-578` | `.current_drive.f_c_plasma_internal`, `.physics.err243` |
| 5 | `physics.py:581-583` | `.physics.f_c_plasma_inductive` |
| 6 | `physics.py:585-588` | `.physics.f_c_plasma_auxiliary` |

**Filing decision: ported as one unit into
`functional_process/models/physics/bootstrap_current.py`**, with `file:line` attribution
into `physics.py` on both functions that came from there. The decisive fact is step 6:
`physics.py:585-588` is the **only** producer of `.physics.f_c_plasma_auxiliary` anywhere
in `process/` (measured: `grep -rn f_c_plasma_auxiliary process/ --include=*.py` returns
one write and three reads, all reporting or in `current_drive.py`), and that path is a
declared boundary read of `functional_process/models/physics/current_drive.py`'s
`HcdPrimaryInjectedPower`. Porting the bootstrap fraction while leaving its only consumer
chain on the boundary would have left an invented edge in place.

Recorded as a **deviation from the source layout**, not as a claim PROCESS's filing is
wrong; the same call `plasma_current.md` made, for the same file, four hours earlier.
`physics.py` is 6931 lines and holds several units' worth of material. If a later pass
gives the current-fraction bookkeeping its own unit, `PlasmaCurrentFractions`,
`calculate_plasma_current_fractions` and `enforce_bootstrap_current_fraction_max` move
there wholesale. Flagged in "## open questions", not decided here.

### functions in `bootstrap_current.py`

**26 `def`s.** Four are enum/class structure (`__new__`, `full_name`, `__init__`,
`SauterBootstrapCurrent.run`/`output`, both empty at `:1450-1454`). In audit scope:

| # | function | lines | shape |
|---|---|---|---|
| 1 | `PlasmaBootstrapCurrent.run` | 81-262 | evaluates **all fourteen** scalings, then indexes; the unit's head |
| 2 | `get_bootstrap_current_fraction_value` | 264-298 | `dict` lookup on `BootstrapCurrentFractionModel` |
| 3 | `bootstrap_fraction_iter89` | 300-366 | `@nb.njit` `@staticmethod`, pure — value 1 |
| 4 | `bootstrap_fraction_wilson` | 368-496 | `@nb.njit`, pure — value 3 |
| 5 | `_nevins_integral` | 498-585 | `@nb.njit`, pure — value 2's integrand |
| 6 | `PlasmaBootstrapCurrent.bootstrap_fraction_sauter` | 587-604 | one-line delegate to #14 |
| 7 | `bootstrap_fraction_nevins` | 606-710 | `scipy.integrate.quad` over #5 — value 2 |
| 8-13 | `bootstrap_fraction_{sakai,aries,andrade,hoang,wong,gi_I,gi_II}` | 712-1133 | `@nb.njit`, pure — values 5-11 |
| — | `bootstrap_fraction_sugiyama_{l,h}_mode` | 1135-1269 | `@nb.njit`, pure — values 12, 13 |
| — | `PlasmaBootstrapCurrent.output` | 1271-1445 | reporting only |
| 14 | `SauterBootstrapCurrent.bootstrap_fraction_sauter` | 1456-1608 | **value 4**; reads a `PlasmaProfile` object and 16 `data` fields |
| 15 | `_coulomb_logarithm_sauter` | 1610-1651 | `@nb.njit`, pure |
| 16 | `_electron_collisions_sauter` | 1653-1682 | instance method, no `self` state |
| 17 | `_electron_collisionality_sauter` | 1684-1734 | instance method, no `self` state |
| 18 | `_ion_collisions_sauter` | 1736-1777 | `@nb.njit`, pure |
| 19 | `_ion_collisionality_sauter` | 1779-1826 | instance method, no `self` state |
| 20 | `_calculate_l31_coefficient` | 1828-1931 | instance method, no `self` state |
| 21 | `_calculate_l31_32_coefficient` | 1933-2120 | instance method; calls #20 |
| 22 | `_calculate_l34_alpha_31_coefficient` | 2122-2305 | instance method; calls #20 |
| 23 | `_beta_poloidal_sauter` | 2307-2364 | `@nb.njit`, pure |
| 24 | `_beta_poloidal_total_sauter` | 2366-2442 | `@nb.njit`, pure |
| 25 | `_trapped_particle_fraction_sauter` | 2444-2529 | `@nb.njit`, pure; 3-way `fit` branch |

## the extraction seam

**Clean for the Sauter internals, and a real rewrite for the head.** #15-#25 are already
pure (`@nb.njit` staticmethods, or instance methods that touch no `self` state — proved
in the test module by binding them on a `SauterBootstrapCurrent()` with **no**
`DataStructure` at all, so an accidental `self.data` read would raise). Porting them was
transcription plus `np.` → `jnp.`.

#14 is the seam. It takes a `PlasmaProfile` object and reads sixteen
`self.data.physics.*` fields off the back door (`:1494-1532`); the port's signature is
those sixteen minus `.physics.triang` (see below), plus the three profile arrays that
came off the object. Nothing else changed.

#1 is not ported at all in the shape PROCESS wrote it — see "## the family PROCESS
computes and this port does not".

## data footprint

### `bootstrap_fraction_sauter` (#14), the live arm

Sixteen `.physics` reads at `:1494-1532`, of which the port declares **fifteen**:
`a_plasma_poloidal`, `rminor`, `rmajor`, `nd_plasma_ions_total_vol_avg`,
`nd_plasma_electrons_vol_avg`, `temp_plasma_ion_vol_avg_kev`,
`temp_plasma_electron_vol_avg_kev`, `n_charge_plasma_effective_vol_avg`, `q0`, `q95`,
`m_ions_total_amu`, `f_plasma_fuel_helium3`, `b_plasma_toroidal_on_axis`,
`plasma_current`, and `n_plasma_profile_elements` (static — a shape, not a value).
Plus three arrays off the `PlasmaProfile`: `neprofile.profile_x`, `neprofile.profile_y`,
`teprofile.profile_y`, which the port reads as
`.physics.radius_plasma_profile_norm` / `.nd_plasma_electron_profile` /
`.temp_plasma_electron_profile_kev` — the minted paths `profiles.py` (registry unit #21)
already produces. **Nothing about the profiles is re-derived here.**

The sixteenth is `.physics.triang`, and it is not a read.

### the invented `triang` edge

`bootstrap_fraction_sauter` passes `self.data.physics.triang` (`:1553`, `:1569`, `:1585`)
into all three `_calculate_l*` coefficient functions, each of which forwards it to
`_trapped_particle_fraction_sauter(radial_elements, triang, sqeps)`. That function's
signature is `(radial_elements, triang, sqeps, fit=0)` and **`triang` is read only inside
the `fit == 2` branch** (`:2518-2527`). `fit` has no `DataStructure` field behind it, it
defaults to `0`, and no call site anywhere in `process/` passes it — so branches `1` and
`2` are unreachable and `triang` reaches nothing.

Four functions carry an argument to a dead branch. Measured, not inferred:
`test_reference_is_invariant_to_triangularity` moves `triang` from `+0.5` to `-0.3` —
across zero, where the `fit == 2` correction `0.67 * (1 - 1.4 * triang * |triang|) * eps`
changes the sign of its own correction term — and asserts PROCESS's own answer is
**bit-identical**. The port declares no `triang` read and
`test_nodes_assemble_and_the_sauter_arm_does_not_read_triangularity` asserts the
`VarPath` is absent from the node's inputs, so the edge is removed from the graph rather
than left inert.

This is the invented-edge class `_audit/tokamak_scope.md` exists to find, in its purest
form: the dependency is real in the *source text* and absent from the *computation*.

### the family PROCESS computes and this port does not

`PlasmaBootstrapCurrent.run` (#1) evaluates **all fourteen** scalings unconditionally
into fourteen `.current_drive.f_c_plasma_bootstrap_*` fields (`:90-248`) and only then
indexes the family by `i_bootstrap_current` (`:250-262` via #2). Its read set is
therefore the union of fourteen arms:

`aspect`, `beta_total_vol_avg`, `b_plasma_total`, `plasma_current`, `q95`, `q0`,
`rmajor`, `vol_plasma`, `alphan`, `alphat`, `beta_toroidal_vol_avg`,
`b_plasma_toroidal_on_axis`, `nd_plasma_electrons_vol_avg`, `rminor`,
`temp_plasma_electron_vol_avg_kev`, `n_charge_plasma_effective_vol_avg`, `alphaj`,
`alphap`, `beta_thermal_poloidal_vol_avg`, `beta_poloidal_vol_avg`, `eps`,
`ind_plasma_internal_norm`, `nd_plasma_electron_on_axis`,
`pres_plasma_thermal_on_axis`, `pres_plasma_thermal_vol_avg`, `kappa`, `tbeta`,
`radius_plasma_pedestal_density_norm`, `nd_plasma_pedestal_electron`,
`nd_plasma_electron_max_array[6]`, `temp_plasma_pedestal_kev`, and `cboot` — **32
reads**, against the live arm's fifteen plus `cboot`.

**`_audit/tokamak_boundary.md` reads this the other way and this port disagrees with it,
deliberately.** That document's `.tokamak.bootstrap_current` note says §F's trace enters
all fourteen functions at `i_bootstrap_current = 4`, and concludes the switch "selects a
*value from a computed vector*, not a subgraph — one node producing the family plus an
index, not an occupant per arm". The trace observation is correct; the conclusion is not
followed here, because:

1. `next_steps.md` §14.2's binding policy is unconditional — one occupant class per
   switch value — and the dispatch brief restates it. There is no "computes-then-selects"
   exception in it.
2. The thirteen unselected values are **dead work**: their outputs have no reader in
   `process/` outside `PlasmaBootstrapCurrent.output` (`:1271-1445`) and
   `core/io/plot/summary.py:8934-8946`. Measured by `grep` over `process/`.
3. Keeping the family would give the node the 32-read union above, sixteen of which the
   live computation never touches — including `alphaj` and `ind_plasma_internal_norm`,
   which are `plasma_current.py`'s outputs, so the union would manufacture two real
   inter-node edges for a printed number.

The same shape is recorded for `l_h_transition` and `density_limit` in
`tokamak_boundary.md`. **SETTLED (2026-08-26, wave coordinator): occupant per arm is the
policy for every computes-then-selects family**, this one and those two — §14.2 binds,
and `tokamak_boundary.md`'s "family plus an index" note on all three slots is superseded.
The ports of `l_h_transition` and `density_limit` should not re-derive this. See open
question 2.

### reporting-only writes, measured

`grep -rn` over `process/`, excluding `output*` methods and `data_structure/`:

| field | non-reporting readers |
|---|---|
| the thirteen `.current_drive.f_c_plasma_bootstrap_*` siblings | **none** |
| `.current_drive.bscf_gi_i`, `.bscf_gi_ii` | **none** |
| `.physics.j_plasma_bootstrap_sauter_profile` | **none** (only `bootstrap_current.py:1322`, in `output`) |
| `.physics.err242`, `.physics.err243` | **none** (only `:1406`, `:1410`, in `output`) |
| `.current_drive.f_c_plasma_pfirsch_schluter_scene` | **none** (only `:1402`; the `i_ps == 1` copy at `physics.py:539` is the switch, not a read) |
| `.current_drive.f_c_plasma_diamagnetic_hender`, `_scene` | **none** (only `plasma_current.py:1117`, `:1124`, `:1130`) |
| `.current_drive.f_c_plasma_internal` | **none outside `physics.py:558-588` itself** |

`err242`/`err243` and the sibling fractions are dropped, following
`plasma_current.md`'s call on `.physics.alphaj_wesson`.
`.physics.j_plasma_bootstrap_sauter_profile` **is** carried — see "## deviations" 3.

By contrast the four the port carries all have real readers, or are the unit's point:

| field | readers |
|---|---|
| `.current_drive.f_c_plasma_bootstrap` | `physics.py:547/553/559`, `current_drive.py:2989`; MFILE |
| `.current_drive.f_c_plasma_internal` | `physics.py:571-587` (this unit's own tail) |
| `.physics.f_c_plasma_auxiliary` | `current_drive.py:1837`, and `HcdPrimaryInjectedPower` in the port |
| `.physics.f_c_plasma_inductive` | `current_drive.py:2411`, `pfcoil.py:2233/3488`, `physics.py:775/941/1691/4878` |

## calls into other models

**One, and it is an object rather than a call.** `SauterBootstrapCurrent.
bootstrap_fraction_sauter` takes a `PlasmaProfile` and reads three arrays off its two
`Profile` sub-models. Those are `models/physics/profiles.py`'s (registry unit #21) and
`plasma_profiles.py`'s (unit #12) outputs, already ported and already minted as
`.physics.*` paths, so the port declares reads on them. No `Model.run()` is called from
anywhere in the ported closure.

No CoolProp anywhere in the chain.

## JAX-difficulty flags

- **`np.gradient` with a coordinate array** (`:1541-1543`). `jnp.gradient` accepts it and
  agrees only to `2.2e-16` — not agreement at `MACHINE_PRECISION` — so the port writes
  out `numpy`'s non-uniform interior stencil and `edge_order = 1` ends explicitly
  (`_gradient`). Measured **bit-identical** to `np.gradient` on this unit's grid. Exactly
  the trap `plasma_profiles._simpson` records for `scipy.integrate.simpson`: a
  uniform-grid shortcut that is right in value and wrong in `d/d(x[i])`.
- **One `x ** 0.5`-at-zero site, repaired.** `jnp.sqrt(ion_collisionality)` in
  `_calculate_l34_alpha_31_coefficient`'s `alpha` (`:2237-2238`) is `0` whenever the ion
  density is, and differentiates to `inf` there. Found by
  `Tier1Contract.test_gradient_finite_at_zero` on `nd_plasma_ions_total_vol_avg` — the
  only site the whole-function probe found — and fixed with `models/safe_math.safe_sqrt`,
  which is value-identical for every non-zero radicand. `_audit/next_steps.md` §9's class,
  and the repair that module exists for.
- **One unguarded-division site, reached by three arguments, registered rather than
  repaired.** See "## the three coefficient contracts and their register entries".
- **Two data-dependent `min`/`max` branches** in the `physics.py` tail (`:552`, `:574`,
  `:581`), ported as `jnp.minimum`/`jnp.maximum`. Both are `needs-lax-cond-or-where` in
  `traceability_policy.md`'s classification; both arms are finite everywhere, so nothing
  leaks through the untaken branch, and the subgradient at the kink is JAX's.
- **A dead `np.where` arm** in both `_beta_poloidal*_sauter` — `radial_elements != nr`
  is always true. Carried verbatim; **D2**.
- `@nb.njit` on eight of the reference functions is irrelevant to the port and only
  affects the harness: the reference side is compiled, the port side traced.

## suspected defects in PROCESS

**D1 — `zmain` is passed into a parameter named `zeff`, and raised to the fourth power.**
`_calculate_l34_alpha_31_coefficient` calls
`self._ion_collisionality_sauter(radial_elements, rmajor, inverse_q, sqeps, tempi,
amain, zmain, ni)` (`:2231-2233`) **positionally**, against a signature whose seventh
parameter is `zeff` (`:1786`). That value flows into `_ion_collisions_sauter`'s
`zeff[...] ** 4` (`:1770`). So the ion collision frequency is scaled by
`(1 + f_plasma_fuel_helium3) ** 4`, not by `n_charge_plasma_effective_vol_avg ** 4`.
Whether that is the intended physics (the Sauter paper's `nu_i*` uses the main-ion
charge, so it may well be) or a positional-argument slip, the *naming* is misleading and
an extractor reading the signature would get it wrong. Ported verbatim; the port renames
the parameter to `zmain` at the call site and says why in both docstrings.

**D2 — the second arm of both `_beta_poloidal*_sauter`'s `np.where` is unreachable.**
`radial_elements = np.arange(2, n_plasma_profile_elements)` (`:1532`), so its maximum is
`nr - 1` and `radial_elements != nr` is always `True`. The `6.4e-4 * pi * ...` arm at
`:2353` / `:2426` — which is the correct endpoint form, `2x` the interior coefficient
because there is no `[j] + [j-1]` pair to halve — never runs. Harmless today; it would
matter if the loop bound ever became `arange(2, n + 1)`. Carried verbatim.

**D3 — `PlasmaBootstrapCurrent.run` evaluates all fourteen scalings on every call**
(`:90-248`), including `bootstrap_fraction_nevins`'s `scipy.integrate.quad`
(`:606-710`), on a machine that will use exactly one of them. That is an adaptive
quadrature with a Python integrand, per solver evaluation, for a number nothing reads.
It is also `i_bootstrap_current`'s real cost, and the reason this port refuses the
"family plus an index" shape — see above.

**D4 — `SauterBootstrapCurrent` is a `Model` with an empty `run()` and an empty
`output()`** (`:1447-1454`), constructed and injected into `PlasmaBootstrapCurrent`
purely as a namespace for eleven functions. `PlasmaBootstrapCurrent.
bootstrap_fraction_sauter` (`:587-604`) then exists only to forward to it. Same shape as
`PlasmaCurrent.run()` being empty (`plasma_current.md`'s D-list); cosmetic.

**D5 — `i_bootstrap_current`'s range check is a `try`/`except ValueError` around the
whole selection** (`:251-262`), so an `IndexError`/`KeyError` from anywhere inside
`get_bootstrap_current_fraction_value` would surface as "Illegal value of
i_bootstrap_current". Not reachable today (the `model_map` is exhaustive over the enum).

## tier signal

**Tier 1 throughout.** Nothing in the ported closure iterates: the Sauter scaling is a
single vectorised pass over the profile grid and a rectangle sum, and the two `physics.py`
tail functions are arithmetic with two clamps. PROCESS's answer *is* ground truth and
value agreement is checkable at machine precision. Measured: the whole-function diff
agrees to `1.6e-16` relative at `n = 11` and to the last bit at `n = 51`
(see "## ported").

`bootstrap_fraction_nevins` (#7) would be the one tier-2 candidate in the file —
`scipy.integrate.quad` over a Python integrand — and it is UNPORTED.

## proposed signature(s)

Matched exactly by the port:

```
_gradient(profile_y, coordinate)
_coulomb_logarithm_sauter(radial_elements, tempe, ne)
_electron_collisions_sauter(radial_elements, tempe, ne)
_electron_collisionality_sauter(radial_elements, rmajor, zeff, inverse_q, sqeps,
                                tempe, ne)
_ion_collisions_sauter(radial_elements, zeff, ni, tempi, amain)
_ion_collisionality_sauter(radial_elements, rmajor, inverse_q, sqeps, tempi, amain,
                           zeff, ni)
_trapped_particle_fraction_sauter(radial_elements, sqeps)
_beta_poloidal_sauter(radial_elements, nr, rmajor, b_plasma_toroidal_on_axis, ne,
                      tempe, inverse_q, rho)
_beta_poloidal_total_sauter(radial_elements, nr, rmajor, b_plasma_toroidal_on_axis,
                            ne, ni, tempe, tempi, inverse_q, rho)
_calculate_l31_coefficient(radial_elements, number_of_elements, rmajor,
                           b_plasma_toroidal_on_axis, ne, ni, tempe, tempi,
                           inverse_q, rho, zeff, sqeps)
_calculate_l31_32_coefficient(... same ...)
_calculate_l34_alpha_31_coefficient(radial_elements, number_of_elements, rmajor,
                                    b_plasma_toroidal_on_axis, inverse_q, sqeps,
                                    tempi, tempe, amain, zmain, ni, ne, rho, zeff)
bootstrap_fraction_sauter(*, n_plasma_profile_elements, radius_plasma_profile_norm,
                          nd_plasma_electron_profile,
                          temp_plasma_electron_profile_kev, a_plasma_poloidal,
                          rminor, rmajor, nd_plasma_ions_total_vol_avg,
                          nd_plasma_electrons_vol_avg, temp_plasma_ion_vol_avg_kev,
                          temp_plasma_electron_vol_avg_kev,
                          n_charge_plasma_effective_vol_avg, q0, q95,
                          m_ions_total_amu, f_plasma_fuel_helium3,
                          b_plasma_toroidal_on_axis, plasma_current)
enforce_bootstrap_current_fraction_max(f_c_plasma_bootstrap,
                                       f_c_plasma_bootstrap_max)
calculate_plasma_current_fractions(f_c_plasma_bootstrap, f_c_plasma_diamagnetic,
                                   f_c_plasma_pfirsch_schluter,
                                   f_c_plasma_non_inductive)
```

Every one drops `triang` and `fit` relative to PROCESS's; `bootstrap_fraction_sauter`
additionally replaces the `PlasmaProfile` object with three explicit arrays. The last two
are new: PROCESS has no callable for them (see "## the chain is not one file").

## switches touched

### `i_bootstrap_current` (`physics_variables.py:818`, default `3`; `large_tokamak_eval:286` sets `4`)

`BootstrapCurrentFractionModel`, `bootstrap_current.py:26-67`. Fourteen values, and the
reads genuinely differ per arm — band (d) of `switch_kwarg_survey.md`, not a static
kwarg.

| value | name | reads | ported |
|---|---|---|---|
| 0 | `USER_INPUT` | selects `.current_drive.f_c_plasma_bootstrap` from itself (`:283`); exempt from the cap (`physics.py:549-551`) | **empty slot** |
| 1 | ITER IPDG89 | `aspect`, `beta_total_vol_avg`, `b_plasma_total`, `plasma_current`, `q95`, `q0`, `rmajor`, `vol_plasma` | no |
| 2 | Nevins | `alphan`, `alphat`, `beta_toroidal_vol_avg`, `b_plasma_toroidal_on_axis`, `nd_plasma_electrons_vol_avg`, `plasma_current`, `q95`, `q0`, `rmajor`, `rminor`, `temp_plasma_electron_vol_avg_kev`, `n_charge_plasma_effective_vol_avg` | no |
| 3 | Wilson | `alphaj`, `alphap`, `alphat`, `beta_thermal_poloidal_vol_avg`, `q0`, `q95`, `rmajor`, `rminor` | no |
| 4 | **Sauter** | the fifteen in "## data footprint", plus three profile arrays | **yes** |
| 5 | Sakai | `beta_poloidal_vol_avg`, `q95`, `q0`, `alphan`, `alphat`, `eps`, `ind_plasma_internal_norm` | no |
| 6 | Aries | `beta_poloidal_vol_avg`, `ind_plasma_internal_norm`, `nd_plasma_electron_on_axis`, `nd_plasma_electrons_vol_avg`, `eps` | no |
| 7 | Andrade | `beta_poloidal_vol_avg`, `pres_plasma_thermal_on_axis`, `pres_plasma_thermal_vol_avg`, `eps` | no |
| 8 | Hoang | `beta_poloidal_vol_avg`, `alphap`, `alphaj`, `eps` | no |
| 9 | Wong | `beta_poloidal_vol_avg`, `alphan`, `alphat`, `eps`, `kappa` | no |
| 10 | GI 1 | `beta_poloidal_vol_avg`, `alphap`, `alphat`, `eps`, `n_charge_plasma_effective_vol_avg`, `q95`, `q0` | no |
| 11 | GI 2 | as GI 1 without `q95`/`q0` | no |
| 12 | Sugiyama L | `eps`, `beta_poloidal_vol_avg`, `alphan`, `alphat`, `n_charge_plasma_effective_vol_avg`, `q95`, `q0` | no |
| 13 | Sugiyama H | as Sugiyama L plus `tbeta`, `radius_plasma_pedestal_density_norm`, `nd_plasma_pedestal_electron`, `nd_plasma_electron_max_array[6]`, `temp_plasma_pedestal_kev` | no |

Two notes the table cannot carry:

- **Value 4 is structurally different from the other thirteen.** It is the only arm that
  integrates over the plasma profiles; the rest are closed-form expressions in
  volume-averaged scalars. An occupant for any of them will not share this one's shape or
  its profile reads.
- **Value 0 is an empty slot, not an unported model** — the same shape
  `plasma_current.md` records for `i_alphaj == 0`. Under it
  `.current_drive.f_c_plasma_bootstrap` is a boundary input (it *is* an `IN.DAT`
  variable, `core/input.py:252`) and no node owns it.

### `i_diamagnetic_current` (`physics_variables.py:856`, default `0`; not set by `large_tokamak_eval`)

`PlasmaDiamagneticCurrentModel`, `plasma_current.py:1020-1055`; selection at
`plasma_current.py:1081-1094`. Three values:

| value | name | body | ported |
|---|---|---|---|
| 0 | `NONE` | no assignment at all; the field holds `current_drive_variables.py:77`'s `0.0` | **yes** (`NoDiamagneticCurrent`) |
| 1 | `HENDER_ST_FIT` | `diamagnetic_fraction_hender(beta_total_vol_avg)` (`:1138-1154`) | no |
| 2 | `SCENE_FIT` | `diamagnetic_fraction_scene(beta_total_vol_avg, q95, q0)` (`:1156-1175`) | no |

Value 0 is ported as a **zero-producing source node**, not left as a boundary input.
`.current_drive.f_c_plasma_diamagnetic` is not settable from `IN.DAT` and nothing else in
`process/` writes it, so it is a computed constant rather than an input —
`tokamak_boundary.md` §"The twelve that are simply inputs" is explicit that the boundary
is for variables PROCESS computes *nowhere*. Same argument, and the same shape, as
`current_drive.py`'s `NoSecondaryHcd`.

Values 1 and 2 are two-line pure `@nb.njit` staticmethods and are cheap whenever a device
needs them; they live in `plasma_current.py`, which `plasma_current.md` explicitly put out
of that unit's scope, so they are nobody's today.

### `i_pfirsch_schluter_current` (`physics_variables.py:895`, default `0`; not set by `large_tokamak_eval`)

Selection at `physics.py:538-541`, a bare `if == 1` with no `else`. Two values:

| value | body | ported |
|---|---|---|
| 0 | no assignment; the field holds `current_drive_variables.py:283`'s `0.0` | **yes** (`NoPfirschSchluterCurrent`) |
| 1 | copies `.current_drive.f_c_plasma_pfirsch_schluter_scene`, itself `ps_fraction_scene(beta_total_vol_avg)` (`physics.py:534-536`) | no |

Same zero-node argument as `i_diamagnetic_current == 0`. Value 1's producer,
`ps_fraction_scene`, is a module-level function in `physics.py`; it is computed
unconditionally on every run and is reporting-only at value 0. On `large_tokamak_eval` it
evaluates to `-2.9e-3` (`MFILE:6900`) — negative, and it would have been *added* to the
plasma-driven fraction had the switch been on.

## the three coefficient contracts and their register entries

**Was "owed"; now written, together with the six `_harness/boundary.py` entries they
need** (wave coordinator, 2026-08-26, granting an exception to the dispatch brief's
do-not-edit list for these additive entries alone).

`TestCalculateL31Coefficient`, `TestCalculateL3132Coefficient` and
`TestCalculateL34Alpha31Coefficient` each fail
`Tier1Contract.test_gradient_finite_at_zero`, and every failure is one site:
`_electron_collisionality_sauter` divides by
`|inverse_q[j] * sqeps[j]**3 * sqrt(tempe[j]) * 1.875e7|` (`bootstrap_current.py:
1728-1733`). Zero either `inverse_q[j]` or `sqeps[j]` and the collisionality is `+inf`;
every consumer has the form `f_trapped / (a + b * nu)`, which saturates to a finite `0`
while the tangent through `a / b` stays `nan`. `rmajor == 0` is the same site from the
numerator side, and it survives only into `l31`, where the trailing
`_beta_poloidal_total_sauter` keeps `(rmajor / ...) ** 2` finite; in `l31_32` the extra
`_beta_poloidal_sauter / _beta_poloidal_total_sauter` quotient cancels the factor that
made it singular. `_harness/boundary.py`'s **unguarded-division** class, not the
`x ** p` one, so it is registered rather than repaired — there is no guarded reciprocal
to write without first deciding what the electron collisionality *means* at infinite
safety factor or zero inverse aspect ratio.

Measured per contract at `n = 11` (9 radial components each):

| contract | registered arguments |
|---|---|
| `TestCalculateL31Coefficient` | `inverse_q`, `sqeps`, `rmajor` |
| `TestCalculateL3132Coefficient` | `inverse_q`, `sqeps` |
| `TestCalculateL34Alpha31Coefficient` | `inverse_q` |

**Six entries, where this record's first draft predicted four.** The first estimate came
from a scratch probe on a synthetic profile pair and missed `sqeps` on the first two
contracts; the entries above are taken from the failing test output on the contracts'
own declared samples. Worth recording as a small instance of the register's own rule —
an entry must be earned by a site that actually fails, and `test_gradient_finite_at_zero`
fails a contract whose registered site has *stopped* failing, so the set cannot be
padded.

The three functions are additionally covered in value and in gradient by
`TestBootstrapFractionSauter`, whose own zero-boundary probe is clean: the scalars that
would zero `inverse_q` or `sqeps` there (`q0`, `q95`, `rminor`) make the whole value
non-finite, which the check steps aside for.

## ported (2026-08-26)

Port: `functional_process/models/physics/bootstrap_current.py`. Tests:
`tests/functional_process/models/physics/test_bootstrap_current.py`. Ten `Tier1Contract`s
plus five plain tests. **`62 passed, 57 skipped`** on a plain run (gradient checks skip
by default); **`119 passed`** with `--fp-gradients` (~52 s); **`251 passed`** with
`--fp-gradients --fp-fuzz 4 --fp-fuzz-seed 11` (~64 s).

Also touched: `functional_process/_harness/boundary.py`, six additive
`DIVISION_BY_ZERO_AT_BOUNDARY` entries — see "## the three coefficient contracts and
their register entries". Nothing else in that file changed.

Measured agreement of the whole-function diff against PROCESS
(`SauterBootstrapCurrent.bootstrap_fraction_sauter` on a real `DataStructure`):
`1.6e-16` relative at `n_plasma_profile_elements = 11`, exact at `51`, `1.4e-16` at
`501`, where the answer is `0.4052168782500341` — the literal
`tests/unit/models/physics/test_physics.py::test_bootstrap_fraction_sauter` records.

### cottax nodes

| class | family | owns | reads |
|---|---|---|---|
| `SauterBootstrapCurrentFraction` | `BootstrapCurrentFractionScaling` | `.current_drive.f_c_plasma_bootstrap_sauter`, `.physics.j_plasma_bootstrap_sauter_profile`, `.current_drive.f_c_plasma_bootstrap` | `.physics.radius_plasma_profile_norm`, `.nd_plasma_electron_profile`, `.temp_plasma_electron_profile_kev`, `.a_plasma_poloidal`, `.rminor`, `.rmajor`, `.nd_plasma_ions_total_vol_avg`, `.nd_plasma_electrons_vol_avg`, `.temp_plasma_ion_vol_avg_kev`, `.temp_plasma_electron_vol_avg_kev`, `.n_charge_plasma_effective_vol_avg`, `.q0`, `.q95`, `.m_ions_total_amu`, `.f_plasma_fuel_helium3`, `.b_plasma_toroidal_on_axis`, `.plasma_current`, `.current_drive.cboot`, `.current_drive.f_c_plasma_bootstrap_max` |
| `NoDiamagneticCurrent` | `PlasmaDiamagneticCurrentFraction` | `.current_drive.f_c_plasma_diamagnetic` | none |
| `NoPfirschSchluterCurrent` | `PlasmaPfirschSchluterCurrentFraction` | `.current_drive.f_c_plasma_pfirsch_schluter` | none |
| `PlasmaCurrentFractions` | (none — unconditional) | `.current_drive.f_c_plasma_internal`, `.physics.f_c_plasma_auxiliary`, `.physics.f_c_plasma_inductive` | `.current_drive.f_c_plasma_bootstrap`, `.f_c_plasma_diamagnetic`, `.f_c_plasma_pfirsch_schluter`, `.physics.f_c_plasma_non_inductive` |

Nineteen + zero + zero + four = **23 declared reads**, against the **36** PROCESS's two
call sites read for the same outputs (32 for the fourteen-member family plus four in the
`physics.py` tail). Two of the removed edges are structural rather than merely redundant:
`.physics.alphaj` and `.physics.ind_plasma_internal_norm` are `plasma_current.py`'s
outputs and enter `PlasmaBootstrapCurrent` only to feed the Wilson/Hoang and Sakai/Aries
fractions, which nothing reads. A third, `.physics.triang`, is removed for a different
reason entirely — it feeds a branch that cannot run.

`n_plasma_profile_elements` is a static `eqx.field` on
`SauterBootstrapCurrentFraction`, not a read: it is a shape, and `profiles.py`'s
`ProfileGrid` already treats it the same way.

### registration instructions (for the consolidation pass)

Slot `.tokamak.bootstrap_current`, plus three slots that do not exist yet. In this
evaluation order (acyclic — a real `Graph` derives it, this is only the witness):

```python
from functional_process.models.physics.bootstrap_current import (
    NoDiamagneticCurrent,
    NoPfirschSchluterCurrent,
    PlasmaCurrentFractions,
    SauterBootstrapCurrentFraction,
)
```

- `.tokamak.bootstrap_current: BootstrapCurrentFractionScaling` — factory-filled, no
  default (a switch answers it). `i_bootstrap_current == 4` →
  `SauterBootstrapCurrentFraction(n_plasma_profile_elements=data.physics.
  n_plasma_profile_elements)`. `== 0` → **empty**
  (`.current_drive.f_c_plasma_bootstrap` becomes a boundary input, and it is an `IN.DAT`
  variable). Every other value → `UNPORTED`.
- `.tokamak.diamagnetic_current: PlasmaDiamagneticCurrentFraction` — factory-filled.
  `i_diamagnetic_current == 0` → `NoDiamagneticCurrent()`. `1`, `2` → `UNPORTED`.
  **New slot**; `tokamak_boundary.md` folds `PlasmaDiamagneticCurrent` into
  `.tokamak.plasma_current`, which `plasma_current.md` then declared out of its scope, so
  it currently has no home. Putting it here keeps it next to its only consumer.
- `.tokamak.pfirsch_schluter_current: PlasmaPfirschSchluterCurrentFraction` —
  factory-filled. `i_pfirsch_schluter_current == 0` → `NoPfirschSchluterCurrent()`;
  `1` → `UNPORTED`. **New slot.**
- `.tokamak.current_fractions: PlasmaCurrentFractions = PlasmaCurrentFractions()` —
  defaulted; nothing switches it. **New slot**, and the one that closes
  `HcdPrimaryInjectedPower`'s `.physics.f_c_plasma_auxiliary` boundary read.

`large_tokamak_eval.IN.DAT` fills all four (line 286; defaults at
`physics_variables.py:856` and `:895`).

**Boundary inputs this slot then needs** (none produced here):
`.current_drive.cboot` (`input.py:260`, default `1.0`),
`.current_drive.f_c_plasma_bootstrap_max` (`IN.DAT:121`, `0.95`; also scan variable 12),
`.physics.f_c_plasma_non_inductive` (iteration variable 44, `IN.DAT:283`),
`.physics.q0`, `.q95` (iteration variable 18), `.b_plasma_toroidal_on_axis` (2),
`.plasma_current`. The rest come from ported producers: `.rminor`/`.a_plasma_poloidal`
from `plasma_geometry.py`, `.rmajor` is iteration variable 3, the profile trio from
`profiles.py`/`plasma_profiles.py`, the composition four
(`nd_plasma_ions_total_vol_avg`, `nd_plasma_electrons_vol_avg`, `m_ions_total_amu`,
`f_plasma_fuel_helium3`, `n_charge_plasma_effective_vol_avg`) from `composition.py`, and
the two volume-averaged temperatures from `plasma_profiles.py`.

**`UNPORTED` entries for `indat.py`:** see the "switches touched" tables — thirteen
`i_bootstrap_current` values (0 being an empty slot rather than unported), two
`i_diamagnetic_current` values, one `i_pfirsch_schluter_current` value.

### not ported in this pass

- The thirteen other `i_bootstrap_current` scalings and their thirteen
  `.current_drive.f_c_plasma_bootstrap_*` fields. Deliberately *not*
  ported-but-unwired: each needs its own occupant and its own harness contract to be
  worth anything.
- `PlasmaBootstrapCurrent.output`, `SauterBootstrapCurrent.run`/`output` — reporting, and
  the last two are empty.
- `.physics.err242`, `.physics.err243`,
  `.current_drive.f_c_plasma_pfirsch_schluter_scene`,
  `.current_drive.f_c_plasma_diamagnetic_hender`/`_scene` — reporting-only, measured.
- `ps_fraction_scene` (`physics.py:534`) and `diamagnetic_fraction_hender`/`_scene`
  (`plasma_current.py:1138-1175`) — the non-zero arms of the two fraction switches.

### deviations from PROCESS

1. **`.physics.triang` is not read.** PROCESS threads it through four functions to a
   branch that cannot run. Measured bit-identical with the triangularity moved across
   zero (`test_reference_is_invariant_to_triangularity`), and asserted absent from the
   node's `VarPath` reads. See "## the invented `triang` edge".
2. **`_trapped_particle_fraction_sauter`'s `fit` parameter does not exist.** Only `fit ==
   0` is reachable; the other two branches are dead code and are not carried. A
   method-choice static kwarg carrying no `DataStructure` field is exactly what §14.2
   forbids.
3. **`SauterBootstrapCurrentFraction` owns three `VarPath`s including the *capped*
   fraction**, where PROCESS writes the uncapped one in `bootstrap_current.py` and caps
   it in `physics.py`. Folding the cap into the occupant is what keeps a node from
   reading what it owns and keeps `i_bootstrap_current` answered once; the exemption at
   `physics.py:549-551` makes the cap per-occupant behaviour anyway. It also carries
   `.physics.j_plasma_bootstrap_sauter_profile`, which is reporting-only — kept, against
   `plasma_current.md`'s precedent for reporting-only fields, because it is the pure
   function's second return value rather than a copy of an owned scalar, and because
   `profiles.py`'s `ProfileGrid` sets the precedent of declaring an unread output so the
   graph shows what the source computes. `Graph.prune` drops it.
4. **The `i_bootstrap_current` family is not computed.** See "## the family PROCESS
   computes and this port does not" — a deliberate disagreement with
   `tokamak_boundary.md`'s reading of this slot.
5. **`np.gradient` is written out rather than called.** Bit-identical; see
   "## JAX-difficulty flags".
6. **One `jnp.sqrt` became `safe_sqrt`.** Value-identical for every non-zero radicand;
   changes only the derivative at exactly zero, from `inf` to `0`. See
   "## JAX-difficulty flags".
7. **Two guards are not carried**: the `ProcessValueError` on an illegal
   `i_bootstrap_current` (`:258-262`) and the one on an illegal `fit` (`:2529`). Both are
   switch-domain checks, answered by which occupant exists —
   `naming_convention.md` § "switches are not ports".

**No expression is rewritten.** Every ported body spells PROCESS's arithmetic as PROCESS
spells it; the deviations above are about which fields are read and owned, which branches
exist, and two numerics substitutions that are measured value-identical. No PROCESS
defect above is fixed.

## open questions

1. **Does the `physics.py` current-fraction tail become its own unit?**
   `enforce_bootstrap_current_fraction_max` and `calculate_plasma_current_fractions` are
   ported here because `.physics.f_c_plasma_auxiliary` is a declared boundary read with
   exactly one producer and no other unit was going to claim it. They are also the two
   functions in this file whose harness reference is a **transcription** rather than a
   PROCESS call, because `Physics.run()` has no callable sub-shell — the weakest oracle
   in the unit, compensated by `test_reference_arm_matches_recorded_mfile`. If a later
   pass builds a `.tokamak.physics` unit that can call the block for real, both move
   there and both contracts should be rewritten against it. **Needs a decision before the
   `physics.py` remainder is ported, not before this lands.**
2. **Computes-then-selects — SETTLED: occupant per arm.** Decided by the wave
   coordinator (2026-08-26), so the next pass does not re-derive it. `tokamak_boundary.md`
   records three slots with this shape — `.tokamak.bootstrap_current`,
   `.l_h_transition`, `.density_limit` — and reads them as "one node producing the family
   plus an index"; §14.2's binding policy binds instead, at one occupant class per switch
   value, for all three. **This item overrides `tokamak_boundary.md`'s note on each of
   them.** Neither of the other two is ported yet, so nothing else changes today.
3. **`.physics.j_plasma_bootstrap_sauter_profile` stays owned — SETTLED** (wave
   coordinator, 2026-08-26), against `plasma_current.md`'s precedent of dropping
   reporting-only fields. It is the pure function's second return value rather than a
   copy of an owned scalar, and `profiles.py`'s `ProfileGrid` already declares an unread
   output so the graph shows what the source computes. `Graph.prune` drops it. If the
   MDA comparison objects to an array whose `DataStructure` default is an empty `list`,
   that is a `mda_harness.py` question, not a reason to stop owning it.
4. **Fuzz bounds here are hand-chosen, not taken from
   `bounds_from_iteration_variables`.** `rmajor`, `q95`, `b_plasma_toroidal_on_axis` and
   `f_c_plasma_non_inductive` *are* iteration variables (3, 18, 2, 44) with declared
   bounds; the profile arrays, `a_plasma_poloidal`, `m_ions_total_amu` and the rest are
   not. The array arguments are fuzzed as a ±15% multiplicative band around PROCESS's own
   pedestal profile, which keeps positivity (every `log` here needs it) without pinning
   the port to one profile family. Same gap `plasma_current.md`'s question 5 records.

## the two SCENE fits (2026-08-29, the ST closing wave)

`i_diamagnetic_current == 2` and `i_pfirsch_schluter_current == 1` now have occupants.
Both tracked spherical tokamaks select both.

| function | ports | reference used by the contract |
|---|---|---|
| `diamagnetic_fraction_scene` | `PlasmaDiamagneticCurrent.diamagnetic_fraction_scene`, `plasma_current.py:1158-1179` | the PROCESS `@staticmethod` itself |
| `ps_fraction_scene` | the module-level `ps_fraction_scene`, `physics.py:161-179` | the PROCESS function itself |

| class | family | owns | reads |
|---|---|---|---|
| `SceneDiamagneticCurrent` | `PlasmaDiamagneticCurrentFraction` | `.current_drive.f_c_plasma_diamagnetic` | `.physics.beta_total_vol_avg`, `.q95`, `.q0` |
| `ScenePfirschSchluterCurrent` | `PlasmaPfirschSchluterCurrentFraction` | `.current_drive.f_c_plasma_pfirsch_schluter` | `.physics.beta_total_vol_avg` |

### two strong oracles in a file that mostly has weak ones

Both PROCESS functions are pure, take exactly the port's arguments, and have recorded
unit-test points (`tests/unit/models/physics/test_physics.py::
test_diamagnetic_fraction_scene`, `::test_ps_fraction_scene`). No adapter, no pinned
arguments, no second reading of the source — which is worth naming next to
`TestCalculatePlasmaCurrentFractions`, whose reference is this file's weakest kind and
which sits one link downstream of both. The two contracts are the strongest evidence in
the unit and they are three lines of arithmetic; the weakest is the six-statement block
that consumes them. That asymmetry is a property of what PROCESS made callable, not of
what matters.

### the sibling `_scene`/`_hender` fields stay uncarried, and the wave confirms it

PROCESS computes `f_c_plasma_diamagnetic_hender`, `f_c_plasma_diamagnetic_scene` and
`f_c_plasma_pfirsch_schluter_scene` unconditionally and *then* selects. The measurement in
"## data footprint" said all three are reporting-only, and the two new occupants act on
it: each owns the **selected** field directly, with no intermediate and no copy — the same
call `WessonCurrentProfileIndex` made for `.physics.alphaj_wesson`. A family with one
owned `VarPath` per arm is what makes `Graph.prune` able to drop the arm that is not
chosen; owning the sibling too would have put a dead node in every ST graph.

### the first non-zero current fractions this port has ever produced

Until this wave every assembled machine took `NoDiamagneticCurrent` and
`NoPfirschSchluterCurrent`, both of which return the literal `0.0`. So
`calculate_plasma_current_fractions`' three-term sum
(`f_c_plasma_internal = bootstrap + diamagnetic + pfirsch_schluter`) had never been
exercised with anything but the bootstrap term, and the `min(., non_inductive)` clamp had
never been approached from a genuinely different value. The Pfirsch-Schlüter term is
**negative** (`-0.09 * beta`), so on an ST the sum is not even monotone in the three
inputs. Nothing is known to be wrong; it is stated because the first machine that
exercises it will be the first evidence either way, and that machine does not assemble
yet (see below).

`i_diamagnetic_current == 1` (Hender) stays UNPORTED: not live on any tracked input, and
a strictly smaller read set (`beta` alone), so it is its own occupant rather than a
constant inside the SCENE one.

**Tests**: `tests/functional_process/models/physics/test_bootstrap_current.py`,
`TestDiamagneticFractionScene` and `TestPsFractionScene`. **`139 passed`** with
`--fp-gradients` (was 133).

### these arms do not make either ST file assemble

Four of the six blockers on `spherical_tokamak_eval`/`st_regression` were closed by this
wave (these two, FIESTA in `plasma_current.md`, `i_beta_norm_max == 0` in `physics.md`).
The two that remain are unported model *packages*: `i_tf_turn_type == 2`, the whole CROCO
TF coil class (`tfcoil/superconducting.md`), and `pf_coil_system_arm`, which both files
miss on four of its seven refused dimensions at once (`pfcoil/geometry.md`).
