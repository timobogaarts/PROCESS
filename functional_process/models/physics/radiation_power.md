---
kind: model-unit
status: draft
confidence: high
---

## source

`process/models/physics/radiation_power.py`, whole file (243 lines: one `@dataclass`
`RadpwrData`, `calculate_radiation_powers()` L29-139, `psync_albajar_fidone()` L142-243).
Registry unit #20; scope was recorded as `calculate_radiation_powers()` and everything it
calls.

Called from two places, with **identical** argument lists:
`stellarator.py:2132` inside `st_phys()` (unit #1, chunk 1B — the in-scope caller), and
`physics.py:734` inside `PhysicsCalculations.physics()` (unit #22, tokamak).

### Scope correction: `impurity_radiation.py`'s `ImpurityRadiation` is a missing registry unit

`calculate_radiation_powers`' first act (L103-104) is

```python
imp_rad = impurity.ImpurityRadiation(plasma_profile, data_structure)
imp_rad.calculate_imprad()
```

`process/models/physics/impurity_radiation.py` is **756 LOC and appears nowhere in
`unit_registry.md`**. It is reached by a bare module import
(`import process.models.physics.impurity_radiation as impurity`, L12), not through a
`self.<attr>.<method>` injection — the same blind spot that missed units #19 and #20
themselves, and `profiles.py` (#21). This is now the **fourth** instance of that pattern:
the scoping grep systematically misses (a) one-level-deeper injection and (b) bare module
imports.

The two halves of that file are very different, and only one is in scope:

| half | what it is | in scope for #20? |
|---|---|---|
| `initialise_imprad`, `init_imp_element`, `read_impurity_file` (L27-376) | one-time startup I/O: parses 28 `.dat` files under `process/data/lz_non_corona_14_elements/` into `.impurity_radiation.{temp_impurity_keV_array, pden_impurity_lz_nd_temp_array, impurity_arr_zav, ...}` | **no** — never called from here; its product is a graph constant |
| `ImpurityRadiation`, `create_f_rad_core_profile`, `calculate_impurity_radiation_power_density` (L379-755) | the radiation model itself | **yes** — all of it |
| `calculate_average_charge_at_temp` + its `@njit` twin (L408-510) | temperature-dependent average charge | **no** — nothing on this path reaches it |

The true unit #20 is therefore `radiation_power.py` (243 LOC) **plus** the
`ImpurityRadiation` half of `impurity_radiation.py` (~380 LOC). Both were ported here
rather than the second half being deferred, because `calculate_radiation_powers` is six
lines of arithmetic wrapped around the impurity model — porting the wrapper alone would
have ported nothing, and would have left the `DataStructure` back door wide open, which
is the whole reason this unit was flagged.

**Registry action**: add `physics/impurity_radiation.py` as a new unit, scoped to
`initialise_imprad` / `init_imp_element` / `read_impurity_file` /
`calculate_average_charge_at_temp` — i.e. *the parts this unit did not need*. The
`ImpurityRadiation` half is covered here and should be cross-referenced, not re-audited.

## data footprint

The whole point of this unit is the two opaque object arguments. Both are read-only, and
between them they carry **eleven** values and nothing else.

### Read off `plasma_profile` (a `PlasmaProfile`; object attributes, no `VarPath`)

| attribute | read/write | classification | note |
|---|---|---|---|
| `neprofile.profile_y` | read | explicit-arg | electron density profile (m^-3); `impurity_radiation.py:693` |
| `teprofile.profile_y` | read | explicit-arg | electron temperature profile (keV); `impurity_radiation.py:694` |
| `neprofile.profile_x` | read | explicit-arg | normalised radius grid; read 4x (`:710`, `:715`, `:739`, `:744`) |
| `neprofile.profile_dx` | read | **not a dependency** | passed to `integrate.simpson(..., dx=)` at `:740` and `:745`, but `scipy` **ignores `dx` whenever `x=` is given**. Verified by the harness: `test_radiation_power.py::_plasma_profile` sets it to a deliberately wrong `1.0` and every case still agrees to `rtol=1e-12`. Not in the port's signature. |

`teprofile.profile_x`, `profile_integ`, and every other `Profile` attribute are **not**
read. Neither is any `PlasmaProfile` method — the object is used purely as a carrier for
three arrays.

The first two already have minted `VarPath`s from unit #12 and are **reused, not
re-minted**: `.physics.nd_plasma_electron_profile` and
`.physics.temp_plasma_electron_profile_kev` (see `plasma_profiles.md` § `ProfileFactors`).
This unit being their second independent consumer is the evidence those spellings were
right. `neprofile.profile_x` had no mint yet; **this record mints
`.physics.radius_plasma_profile_norm`** — same justification (no PROCESS storage; built
by `Profile.initialise_profile_x`/`normalise_profile_x`, `profiles.py:60-84`), named as a
sibling of the existing `.impurity_radiation.radius_plasma_core_norm` it is compared
against. Producer is unit #21.

### Read off `data_structure` (the `DataStructure` back door)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.impurity_radiation.f_nd_impurity_electron_array` | read | explicit-arg | `(14,)`. Read **twice**: `:651` for the species selection, `:565` per species for `n_i = f_nd n_e`. Entries `[2]`/`[3]` are iteration variables 125/126 — a genuine graph edge, also written by `physics.py:1288-1300` |
| `.impurity_radiation.temp_impurity_keV_array` | read | explicit-arg (constant) | `(14, 200)` L(Z,Te) abscissae (keV); file-loaded at startup, never written by a model |
| `.impurity_radiation.pden_impurity_lz_nd_temp_array` | read | explicit-arg (constant) | `(14, 200)` L(Z,Te) values (W m^3); ″ |
| `.impurity_radiation.impurity_arr_len_tab` | read | **dead** | `(14,)`, all entries 200. Used only to index the top of the table at `:591`/`:598`; the interpolation itself always uses the full 200-wide row, so PROCESS already requires `len_tab == 200`. Dropped from the port's signature (`[-1]` instead) |
| `.impurity_radiation.radius_plasma_core_norm` | read | explicit-arg | `:716` |
| `.impurity_radiation.f_p_plasma_core_rad_reduction` | read | explicit-arg | `:717` |
| `.physics.n_plasma_profile_elements` | read | **shape, not value** | `:655-665`, only to size four zero-initialised accumulators. Same `Out.static`-shaped problem as `plasma_profiles.md` § open question 6 |

**No writes.** `calculate_radiation_powers` writes nothing to `data_structure`:
`ImpurityRadiation` accumulates onto `self` only, and both callers assign the returned
`RadpwrData` themselves. The `data_structure` argument is a pure read-through, which is
the best possible outcome for the two pending units (#19 `fusion_reactions.py`, #22
`physics.py`) that share this back door.

### Written by the caller from the return value (not by this unit)

| VarPath | note |
|---|---|
| `.physics.pden_plasma_sync_mw` | `stellarator.py:2147`, `physics.py:750` |
| `.physics.pden_plasma_core_rad_mw` | ″ `:2148` / `:751`; **stellarator then clips at 0, tokamak does not** |
| `.physics.pden_plasma_outer_rad_mw` | ″ `:2149` / `:752`; same asymmetry |
| `.physics.pden_plasma_rad_mw` | ″ `:2150` / `:753` |

### Minted outputs (no PROCESS storage)

`pden_impurity_rad_total_mw` and `pden_impurity_core_rad_total_mw` exist only as
attributes of the `ImpurityRadiation` instance that `calculate_radiation_powers`
constructs and discards. Minted as `.impurity_radiation.pden_impurity_rad_total_mw` /
`.impurity_radiation.pden_impurity_core_rad_total_mw` — the area whose tables and
fractions produce them, and the spelling `ImpurityRadiation` already uses for them.

## proposed signature(s)

Five pure units. Every one is straight-line arithmetic over scalars and profile arrays;
there is no iteration anywhere in this unit.

```python
def calculate_impurity_radiation_power_density(
    nd_electron_profile, temp_electron_profile_kev, f_nd_impurity_electron,
    temp_impurity_kev, pden_impurity_lz_nd_temp,
):
    """`impurity_radiation.py:513-602`, for one species, with the two table rows and the
    fraction passed in rather than indexed out of `data`. `n_i n_e L(Z, Te)`, `L`
    interpolated log-log. Drops the dead `np.digitize` block and the dead `len_tab`.
    Reproduces the out-of-table clamp bug exactly -- see JAX-difficulty flags.
    -> pden_impurity_profile (W/m^3, profile-shaped)
    """

def create_f_rad_core_profile(
    profile_x, radius_plasma_core_norm, f_p_plasma_core_rad_reduction,
):
    """`impurity_radiation.py:379-405`. A step function; strict `<` as in the source.
    -> f_rad_core_profile (profile-shaped)
    """

def calculate_impurity_radiation_totals(
    profile_x, nd_electron_profile, temp_electron_profile_kev,
    f_nd_impurity_electron_array, temp_impurity_kev_array,
    pden_impurity_lz_nd_temp_array, radius_plasma_core_norm,
    f_p_plasma_core_rad_reduction,
):
    """The whole of `ImpurityRadiation.calculate_imprad()` (`:677-755`) as one function.
    The three per-species arguments arrive already stacked over the *selected* species
    (see "switches touched"). `2e-6 * simpson(rho * pden, x=rho)`.
    -> (pden_impurity_rad_total_mw, pden_impurity_core_rad_total_mw)  [MW/m^3]
    """

def psync_albajar_fidone(
    nd_plasma_electron_on_axis, rminor, b_plasma_toroidal_on_axis, aspect, alphan,
    alphat, tbeta, temp_plasma_electron_on_axis_kev, f_sync_reflect, rmajor, kappa,
    vol_plasma,
):
    """`radiation_power.py:142-243`, unchanged. Already pure and already explicit;
    `np.exp` -> `jnp.exp` is the entire diff.
    -> pden_plasma_sync_mw  [MW/m^3]
    """

def combine_radiation_powers(
    pden_impurity_rad_total_mw, pden_impurity_core_rad_total_mw, pden_plasma_sync_mw,
):
    """`radiation_power.py:106-139`'s own three lines. Synchrotron radiation is booked
    entirely to the core. `pden_plasma_sync_mw` is an argument, not a return, because
    `SynchrotronRadiationPower` already owns that `VarPath`.
    -> (pden_plasma_core_rad_mw, pden_plasma_outer_rad_mw, pden_plasma_rad_mw)
    """
```

plus a sixth, `calculate_radiation_powers(...)`, which is the composition of the three
above in PROCESS's own order and returns `RadpwrData`'s four fields as a tuple. It gets
**no node** — the graph says the same thing better as three nodes — but it does get a
harness case, and that case is the unit's headline result: it diffs PROCESS's *entire
entry point*, `PlasmaProfile` and `DataStructure` arguments intact, against a port that
receives only the eleven values tabulated above. Agreement at `rtol=1e-12` is what makes
the data-footprint table a measurement rather than a claim.

## cottax node

Three, in `radiation_power.py`, in dependency order.

- **`SynchrotronRadiationPower`** — `psync_albajar_fidone`. Twelve `.physics` reads, one
  `.physics` write, nothing minted, no switch, no alternative arm. **Ready for
  `total_process.COMMON`.**
- **`ImpurityRadiationTotals`** — `calculate_impurity_radiation_totals`. Three minted
  reads (two reused from unit #12, one new), five real `.impurity_radiation` reads, two
  minted writes. Carries a static `imp_indices: tuple[int, ...]` field.
  **Blocked** — see open question 2.
- **`PlasmaRadiationPowers`** — `combine_radiation_powers`. Structurally settled, but it
  cannot be registered before `ImpurityRadiationTotals` is: two of its three inputs would
  otherwise have no producer.

`create_f_rad_core_profile` gets no node: it is an internal helper of
`calculate_impurity_radiation_totals` with no PROCESS storage for its result.

## tier signal

**Tier 1**, all five signatures. No `scipy.optimize`, no internal iteration, no call into
another `Model`. `ImpurityRadiation` *looks* stateful — four zero-initialised accumulator
arrays, three zero-initialised scalars, a `map` over species — but the state is a
species-wise sum written imperatively; there is no loop whose result depends on its own
previous value. `sp.integrate.simpson` is a fixed quadrature rule, as in unit #12.

`calculate_radiation_powers` as a whole is also tier 1, which is unusual for a function
taking a `DataStructure`: the back door here is only used for reads of constants and of
values other nodes produce.

## switches touched

**None.** No `i_*` field and no legacy switch is read anywhere on this path. That is worth
stating explicitly — it is the first unit audited with a genuinely empty switch set.

There is, however, one thing that *behaves* like a switch without being one:

- **the impurity species selection**, `self.imp = np.nonzero(f_nd_impurity_electron_array
  > 1.0e-30)[0]` (`impurity_radiation.py:650-652`). A data-dependent gather: neither its
  shape nor its contents are known to a tracer. Resolved at graph-assembly time as
  `ImpurityRadiationTotals.imp_indices`, per `naming_convention.md` § "switches are not
  ports" — which species a machine has is a design decision, not a state variable.
  Not a switch record, since it is not a `data.<area>.<field>` enum; flagged here so the
  next unit that meets a threshold-on-an-array selection has a precedent.

## calls into other models

- `impurity.ImpurityRadiation(plasma_profile, data_structure)` then `.calculate_imprad()`
  — `process/models/physics/impurity_radiation.py`, L103-104. **Not** a `Model` in the
  `process.core.model.Model` sense (it does not subclass it, and its `run()`/`output()`
  are empty stubs), but it is constructed with an injected model and a `DataStructure` and
  is otherwise shaped exactly like one. Reached by bare module import — see § Scope
  correction.
- Nothing else. `psync_albajar_fidone` is local to the file.

## JAX-difficulty flags

- `np.interp` in log-log space (`impurity_radiation.py:548-560`) — **minor**.
  `jnp.interp` exists, clamps to the end values exactly as `np.interp` does, and
  differentiates. Agrees to 2e-16 relative on all three species checked.
- Boolean-mask assignment `pden_impurity_profile[mask] = ...` (`:581`, `:595`) —
  **workaround-known**, two `jnp.where`s. Both are discontinuous in
  `temp_electron_profile_kev` at the table edges; the harness samples stay inside.
- `rho < radius_plasma_core_norm` (`:402`) — **minor but sharp**. A step function of
  both `radius_plasma_core_norm` and each `profile_x[i]`; the derivative is 0 a.e. and
  undefined at the straddling grid point. PROCESS's default `radius_plasma_core_norm =
  0.6` lands **exactly** on a grid point of the default 201-point `profile_x`, so a
  finite-difference gradient there is meaningless. The harness cases use 0.65 and hold it
  fixed under fuzzing; noted rather than worked around.
- `np.nonzero(f_nd > 1e-30)` (`:650-652`) — **blocker if kept inside the node**, resolved
  by hoisting to a static field. See open question 2.
- `sp.integrate.simpson` (`:737`, `:742`) — **workaround-known**, and already solved:
  `plasma_profiles._simpson` is imported and reused rather than reimplemented, which is
  the whole point of that function having been written once with its non-uniform-rule bug
  already found and fixed.
- `np.zeros(n_plasma_profile_elements)` (`:654-665`) — **minor**; a static shape in a
  traced port, and only used as an accumulator this port replaces with a sum.
- `numba.njit` on `_calculate_average_charge_at_temp_compiled` (`:437`) —
  **not on this path**, so not a flag for this unit. It *is* one for the
  `impurity_radiation.py` unit the scope correction proposes.
- **`calculate_impurity_radiation_power_density`'s out-of-table clamps are wrong**, and
  the port reproduces the bug faithfully. Detail in open question 1.

## open questions

1. **PROCESS bug, not fixed here: the out-of-table clamps have the wrong units.**
   `impurity_radiation.py:581` and `:595` assign the *loss function* `L(Z, Te)` (W m^3,
   ~1e-33) directly into the *power density* array (W/m^3, ~1e6):

   ```python
   pden_impurity_profile[less_than_imp_temp_mask]    = pden_impurity_lz_nd_temp_array[i, 0]
   pden_impurity_profile[greater_than_imp_temp_mask] = pden_impurity_lz_nd_temp_array[i, -1]
   ```

   The two comments above them ("line radiation will dominate at lower temp", "the L(Z,Te)
   value at the lowest temperature is likely to be an overestimate") describe clamping `L`
   and *then* multiplying by `f_nd n_e^2`, which is not what the code does. The factor
   omitted is `f_nd * n_e^2` ~ 4e38. Measured at `Te = 50 keV` (above the 40 keV table
   top), `n_e = 8e19`, `f_nd = 0.05` for argon: PROCESS returns `1.685e-33` W/m^3 where
   the commented intent gives `5.4e5`. In effect the clamp sets impurity radiation to
   **zero** outside the table rather than to a conservative overestimate.

   The tables span 0.001-40 keV. On-axis temperatures above 40 keV are reachable, and
   `temp_plasma_electron_separatrix_kev` below 1 eV is not obviously impossible either, so
   this is not unreachable code. **Not silently fixed** — the port reproduces it exactly
   and says so in three places. Needs a decision from someone who owns the physics.

   Secondary consequence, and the reason the species selection cannot simply be replaced
   by "sum over all 14": an absent species contributes exactly 0 in-range, but out of
   range the clamp gives it a spurious `~1e-33` per grid point. Small, but it means
   `self.imp` is load-bearing rather than an optimisation.

2. **`imp_indices` blocks `ImpurityRadiationTotals` from `total_process.py` — but only
   just, and for a narrower reason than it first appears.** The static field is the right
   shape (a data-dependent gather cannot live inside a traced node); the question is
   whether the selection can change *during a solve*, which would silently change the
   node's shape. Three routes into `f_nd_impurity_electron_array`, checked:

   - **Iteration variables 125-136** (array indices 2-13, species 3-14). Bounds are
     `(1e-8, 0.01)` in `iteration_variables.py:104-200` — **22 orders of magnitude above
     the `1e-30` threshold.** The optimiser provably cannot deselect a species. This was
     the worst version of the risk and it is not real.
   - **Input parsing**, `init.py:381-384`, copies `f_nd_impurity_electrons` straight
     across. Fixed for the run, and exactly `0.0` for every unseeded species. Also fine.
   - **`physics.plasma_composition()`**, `physics.py:1288-1305`, *recomputes* indices 0
     (`H_`) and 1 (`He`) every evaluation — and on the stellarator path too
     (`stellarator.py:1910`, inside `st_phys`, ~220 lines before the call under audit).
     `H_` is `(n_p + (f_D + f_T) n_fuel + n_beam) / n_e` and cannot plausibly reach
     `1e-30`. **`He` is `f_He3 n_fuel / n_e + f_nd_alpha_thermal_electron`, which is
     exactly `0.0` whenever both terms are** — a D-T machine with no thermal-alpha
     fraction. Then helium leaves `self.imp` and the node's shape changes.

   So the residual blocker is one species, reachable only through a configuration
   (`f_plasma_fuel_helium3 == 0` **and** `f_nd_alpha_thermal_electron == 0`) that is fixed
   for a whole run and therefore *is* a graph-assembly-time fact — but nothing in
   `configuration.py` reads it, and nothing checks that `imp_indices` agrees with it.
   The fix is small and specific: `imp_indices` should be derived by the same code that
   resolves topology switches, from the post-`plasma_composition` fractions, with an
   assertion that no selected fraction is within orders of magnitude of `1e-30`. Not
   implemented, so the node stays out of `total_process.py`. This is a milder cousin of
   `plasma_profiles.md` § open question 1 rather than a new kind of problem.

3. **Two dead reads, both harmless, both dropped by the port.**
   `impurity_arr_len_tab` (used only to index the top of a table whose full width the
   interpolation already assumes), and the `np.digitize` block at `:541-544` and
   `:470-472`, which computes `indices`, clips it, and never uses it. The latter also
   indexes `indices` as an array, so `calculate_impurity_radiation_power_density` would
   raise on a scalar `temp_electron_profile_kev` despite its type hint (`np.array | float`)
   promising otherwise. Neither is a correctness issue today.

4. **The two call sites disagree about clipping at zero.** `stellarator.py:2152-2158`
   applies `max(..., 0)` to `pden_plasma_core_rad_mw` and `pden_plasma_outer_rad_mw`;
   `physics.py:750-753` does not. Both then multiply by `vol_plasma` to get the `p_*`
   totals, so the two devices' pipelines differ in a way that has nothing to do with
   stellarator physics. Either could be the mistake. The only route to a negative value is
   `f_p_plasma_core_rad_reduction > 1` (then `core > total` and `outer < 0`), which is not
   bounded anywhere I found — so the guard is not obviously dead. Left to the caller's
   node in both cases; flagged for unit #1 chunk 1B and unit #22.

5. **`profile_dx` is dead everywhere, not just here.** `Profile.calculate_profile_dx`
   (`profiles.py:86-95`) computes it, `Profile.calculate_profile_integ` (`:110`) and both
   `integrate_radiation_loss_profiles` calls pass it to `simpson` alongside `x=`, and
   `scipy` ignores it in every case. Worth one check in unit #21 for a site where it *is*
   used before concluding it can be deleted.

6. **`.physics.radius_plasma_profile_norm` is a mint of a pure function of a shape.**
   `profile_x` is `arange(n) / (n - 1)` — entirely determined by
   `n_plasma_profile_elements`. It could equally be a graph constant rather than an edge.
   Minted as an edge here to match unit #12's treatment of the other two profile arrays
   and because unit #21 will produce it; if `n_plasma_profile_elements` ends up fixed at
   graph-assembly time (as `plasma_profiles.md` § open question 6 expects), this mint
   should be revisited and probably folded into that constant.

7. **Not checked: whether `physics.py`'s call site prepares `plasma_profile` differently.**
   The argument lists are byte-identical between `stellarator.py:2132` and
   `physics.py:734`, and both pass `self.plasma_profile`, but whether the tokamak pipeline
   guarantees the same profile-array state at that point was not verified — it is unit
   #22's question, not this one's.
