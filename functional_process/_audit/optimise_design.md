# Optimise / driver design

Current state of the solver, driver and code-generation work. **Everything below is what is
true now**; the investigations that established it are in git history.

This file used to carry a dated sequence of numbered sections (§1 through §94) recording
each measurement as it happened -- 3 900 lines by the end. That narrative is gone, on the
same rule the rest of this directory follows: what is landed lives in the code, what was
refuted lives in `tried_and_rejected.md`, what is still open lives in `next_steps.md`, and
the reasoning stays in git (`git log -p -- functional_process/_audit/optimise_design.md`).
**Section numbers cited from docstrings are unchanged; they resolve into git history rather
than into this file.**

## Drivers

Two, both host-side NumPy libraries reached through `jax.pure_callback`
(`core/solver/drivers.py`). `VmconDriver` (`pyvmcon`, cvxpy/CLARABEL) is production;
`SlsqpDriver` (scipy) is a **second opinion** -- kept so that two solvers disagreeing
localises a problem. `scaled_problem` builds the problem once and both consume it, so a
difference between them is the solver, not the setup.

The published matrices are `cottax/reference_warm_matrix.txt` (CPU) and
`..._gpu.txt`; `_audit/performance.md` is the headline. Read them **warm**: a cold row is
~97 % compilation.

**Open**: `SlsqpDriver` and VMCON settle ~1e-04 apart on `stellarator_helias` SAND, stably
across ten +-ulp draws. Not the interpolation -- it survives both the C1 change and the
resolution alternative. Unexplained.

## Cost, and what actually drives it

- A cold row is **~97 % compilation, ~1 % arithmetic**. The `model` column of the cold
  matrix is mis-attributed trace/lower work and must not be divided by an iteration count.
- The largest block is ~2.3 M StableHLO characters = **28 k ops**, of which **41.6 % is
  pure shape plumbing** (`broadcast_in_dim` alone 27.9 %) -- a scalar-valued graph expressed
  in an array language. `jacfwd` roughly doubles it; XLA's fusion pass then *raises* the
  instruction count to 70 k.
- **Machine code is small**: 5.8 MB, ~87-165 bytes per post-optimisation instruction.
  `LoadedExecutable.serialize()`'s 11 MB is the optimised **HLO**, not code.
- **Per-program resident cost is not a well-defined quantity** -- attempts to measure it
  produced two retractions. Use pass-level peaks. Measured one-program-per-process, peak is
  **~157 MB fixed + ~11 KB per post-optimisation instruction**; of a 867 MB peak, ~54 % is
  transient LLVM workspace, ~30 % is held by the live executable, and the ~142 MB residue is
  **almost entirely glibc free-list** (`mallinfo2` says ~14 MB is live). A tuning problem,
  not a hard cost. `MALLOC_ARENA_MAX=1` is the only real lever: peak -32 %, compile +57 %.
- **Reverse mode** costs ~2.9x the compile time and ~3.3x the peak RSS of forward, for 14 %
  more HLO. Its runtime temp buffer is 5.2x forward's -- and 1.7 MB, four orders below
  compile memory. **`jax.checkpoint` is the wrong tool here**: it cuts the runtime peak 40 %
  and costs 86 MB of compile memory, because it expresses its trade as extra graph.

## GPU

Same answers, **median 2.41x slower** over the warm matrix, worst 9x on the largest
configuration -- launch and per-kernel fixed costs over a scalar graph, plus **float64 at
1/16.4 of float32 on this card** (76.6 against 1254.7 GFLOP/s), and PROCESS is float64
throughout. Treat those numbers as specific to a 35 W laptop Quadro.

**Batching is where the GPU wins**: `vmap` over the block evaluation crosses over at
**~256 points** and reaches 3.4x by 4 096, while the CPU curve *worsens* past 64 as the
working set outgrows cache. That is the shape of `process/core/scan.py`, and it is the
reason to want a jax-native or generated solver at all -- not single-solve speed.

## Code generation (`cottax/warp/`)

The graph is an explicit dataflow IR, so the block can be **emitted** rather than lowered.
Warp is the target: scalar-native (the shape plumbing never exists), source-to-source
adjoints, and `wp.launch(dim=n)` is the scan shape.

**Where it stands.** Of **853** live model functions, **757 (88.7 %) transpile**; 686 emit as
one module; **455 validate bit-identical** against their JAX originals (323 single-return,
132 multi), zero disagreements. `helias_5b`'s SAND Drive resolves **85 of 94** nodes under
the arity invariant, `stellarator_helias` 113 of 123.

**The first real kernels.** The maximal prefix-closed sub-DAG of each Drive -- excluded
leaves named, nothing stubbed -- compiled and compared against JAX evaluating the
**identical** sub-DAG:

| configuration | entries | conditions | agreement | fwd regs | bwd regs / spill |
|---|---:|---:|---:|---:|---|
| `helias_5b` | 18/89 | 1/11 | **0.000e+00** | 22 | 244 / **0** |
| `stellarator_helias` | 29/117 | 1/21 | **0.000e+00** | 24 | 255 / **320 B** |
| `large_tokamak_nof` | 37/150 | 2/33, incl. the objective | **0.000e+00** | 24 | 255 / **224 B** |

**Bit-identical on all three**, tokamak included. Faster than JAX at every batch that fits:
5.8x / 6.2x / 6.6x on CPU at batch 1 / 16 / 256, and 4.5x the best JAX number at 4 096 on
GPU.

**The adjoint spills at 29 nodes.** The forward pass stays cheap and spill-free (22 -> 24
registers at every size measured); the backward pass begins spilling to local memory at
**29 leaf calls** -- far earlier than the 40-60 guessed -- and stays spilled at 37. On
`sm_75` the adjoint cannot carry full occupancy past roughly two dozen leaf calls. **This is
the constraint on a Warp path, and it is specific**: emit the forward kernel freely, treat
the generated adjoint as size-limited until measured otherwise.

**Two expectations corrected by measurement:**

- **The 65 536 GPU failure was not card fragmentation** -- it is **JAX's own default GPU
  preallocation** competing with Warp in the same process. With
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, usage drops from ~3.7 GB to ~1.2 GB of 4 and the
  batch succeeds at 0.0014 us/point. Device memory is ~2 048 bytes/point, close to the
  ~2 300-2 700 the I/O implies.
- **Tokamaks do not come nearly free off stellarator leaves.** Only **29 of 144 (20 %)** of
  `large_tokamak_nof`'s distinct leaves are shared with `stellarator_helias`. The 92 %
  collapse is a **portfolio-wide** effect across all seven configurations, not a pairwise
  one -- this pair shares costs/power/availability plumbing, not the profile, build and
  coil-shape machinery that dominates each.

**Constraint nodes are not a judgement call after all.** They were excluded because
`constraint_11/32/34/35/65/82/83` forward to `eq`/`leq`/`geq`, which return **4** values
against a node owning **1**, and picking an element looked like a decision about intent.
It is not -- three independent places in the code say the same thing:

- `core/solver/constraints.py` -- `eq`/`leq`/`geq` all return
  `(residual, normalised_residual, constraint_value, constraint_bound)`, no branching;
- `sand.py`'s `_NormalisedResidual.__call__` -- `return self.fn(**arguments)[1]`, applied
  uniformly to **every** constraint node, one call path, no per-constraint branch;
- PROCESS itself -- `tmp_cc = result.normalised_residual`, and `-tmp_cc` is what VMCON has
  always been driven to zero on.

**Index 1, unconditionally.** The resolver **derives** it rather than hard-coding: it
AST-parses the wrapper's `__call__` and accepts the arity mismatch only when every `return`
is provably `self.fn(**kwargs)[K]` for one literal `K`. A structural fact about one wrapper
class, not a per-node guess -- which is the difference between this and what the arity
invariant rightly refuses.

Coverage, agreement unchanged at **0.000e+00** on every newly included condition:
`stellarator_helias` 29 -> **32** entries and 1 -> **4** conditions, `large_tokamak_nof`
37 -> **38** and 2 -> **3**. `helias_5b` does not move: its four constraint leaves are now
resolvable but their *upstream* chains are still blocked by `interp`, an unresolved `gamma`
and array-valued tables -- the two remaining gaps compound rather than add.

**And a cost that was measured rather than assumed**: forward registers jumped **24 -> 110**
(`stellarator_helias`) and **24 -> 80** for one to three constraint nodes, because the
emitter materialises all four returns even though three are discarded. **The fix is
monomorphisation** -- the same technique already used for function-valued parameters:
emit a specialised single-return variant per selected index. Not done.

**Array-valued parameters and `jnp.interp` now transpile.** The shape decision is
*derived*: the primary source is the bound `VarPath`'s own `DataStructure` field
annotation (`CostData.ucsc: list[float]` against `PhysicsData.rminor: float`) -- static,
no live value needed. A ground-truth value is a fallback only for a genuinely derived port
with no such field. The load-bearing detail is that "classify or refuse" is asked **only
of a parameter the body actually subscripts**: asking it of every dynamic parameter
regressed a large part of the closure, because most scalar parameters are graph-derived
and have no field of their own -- refused for a question their own body never asks.
`jnp.interp` is exact, one `wp_interp_N` per distinct table length built from
`jax._src.numpy.lax_numpy._interp`'s own body, with a static `for k in range(N)` -- the
only loop shape Warp differentiates correctly.

Agreement stays **0.000e+00**; `helias_5b` 18 -> **24**/89 entries and `stellarator_helias`
29 -> **35**/117. `large_tokamak_nof` is unchanged because it reaches none of these leaves.

**Two results that did not go the right way, recorded because they are the ones that
matter.** First, **no config gained a condition** -- 1/11, 1/21 and 2/33 are all unmoved.
Entry coverage rose and the SAND residual did not, which is the metric that decides
whether any of this is a residual yet. Second, **the backward spill grew with coverage**:
`helias_5b` 0 -> **592 B**, `stellarator_helias` 320 -> **912 B**. Whether the adjoint
spill is fatal was already open; this is the first evidence it scales with the thing we
are trying to increase, and it bears on SAND *optimisation* rather than evaluation.

`pchip_interp` (the C1 interpolant) was **not** attempted, for two separately sufficient
reasons: `_pchip_slopes` does whole-array slice arithmetic (`xp[1:] - xp[:-1]`, boolean
masks, `concatenate`), a different feature from indexing; and its tables
(`.stellarator.wp_width_r`/`.lhs`/`.rhs`) have **no ground-truth value at all**, being
produced upstream of this SAND block.

**What is actually blocking coverage is the resolver, not the transpiler.** "71 of 89
entries blocked" was hiding the shape of the problem, because the emittable subset is
grown PREFIX-CLOSED -- an entry joins only if every input is already available, so one
unemittable node near the root withholds its entire downstream cone and the raw count
measures cone size rather than work. Charging each blocked entry to its *root cause*
instead (`scratchpad/arity/frontier.py`) collapses 71 blocked entries on `helias_5b` to
**16 causes**, of which one -- `set_fusion_powers` -- gates 27 on its own, and 37 on
`stellarator_helias`.

And `set_fusion_powers` is not a transpiler failure. It is blocked on an input **nothing
in the Drive produces**. Tracing every such hole to the node that `owns` it, from the
graph's own `owners` map rather than by name:

| configuration | holes | owning nodes | all unresolved? |
|---|---|---|---|
| `helias_5b` | 23 | **5** | yes, 23/23 |
| `large_tokamak_nof` | 36 | 20 | yes, 36/36 |

Zero orphans on either -- every hole is owned by a node the *resolver* refused, so
transpiler capability is not the binding constraint. On `helias_5b` the entire remaining
gap is **five nodes**: `.buildings.sizing` (9 holes, and the whole costs subtree behind
them), `.physics.plasma_composition` (7), `.availability.electric_production` (4),
`.physics.impurity_radiation_totals` (2), `.power.delta_eta_step` (1).

The 38 refusals across all three configurations sort into six named causes, which is what
makes them work items rather than a backlog:

| cause | h5b | st_helias | lt_nof | total |
|---|---|---|---|---|
| `self.<attr>` passed as an argument | 1 | 1 | 11 | **13** |
| a local computed in the wrapper body (`Composition`) | 2 | 2 | 5 | **9** |
| array-valued or sliced argument | 1 | 2 | 5 | **8** |
| no call matching the node's declared arity | 0 | 0 | 4 | 4 |
| genuinely ambiguous -- several candidate calls | 1 | 1 | 1 | 3 |
| leaf parameter absent from the call | 0 | 0 | 1 | 1 |

`self.<attr>` is the largest and the most defensible to fix: `fn` is *bound*, so the
attribute has a readable value -- a fact about the object rather than a guess about intent.
It is only legitimate while the attribute is invariant across the Drive, which has to be
established and not assumed. `Composition` is the biggest lever on the stellarators (2 of
5 and 2 of 6), and the reason it was refused is worth keeping in view: binding a local to
a plausible same-named boundary variable would compute the wrong thing while looking like
float noise.

**Entries are the wrong denominator; conditions are the residual.** Array support added 6
entries on `helias_5b` and moved no condition, and the `self.<attr>` work resolved 5 more
node-instances and moved neither entries nor conditions -- "the denominator grew, the
numerator did not". Both are explained by the same measurement, which is per-CONDITION
rather than per-entry (`scratchpad/reach/cond_holes.py`): it walks each condition's cone
through the resolved **entries** from the unknown/boundary supply, so it reports what
actually decides `prefix_closure` instead of a structural over-approximation. (The
structural version over-counts badly -- it claimed all 5 refused nodes gate all 11
conditions, which cannot be right when one condition already emits.)

`helias_5b`, every condition and exactly what it still needs:

| condition | needs |
|---|---|
| `^cond^cond.physics.temp_plasma_ion_vol_avg_kev` | **nothing -- the 1/11 that emits** |
| `^cond.constraints.c16` | `.availability.electric_production`, and nothing else |
| `^cond^cond.power.delta_eta` | `.power.delta_eta_step`, and nothing else |
| `^cond.stellarator.wp_width_r_min` | `intersect_residual` (`pchip`), and nothing else |
| `^cond.physics.proton_rate_density` | `.physics.plasma_composition` |
| `^cond.physics.fusden_alpha_total` | `.physics.plasma_composition` |
| `^cond.fwbs.f_ster_div_single` | `plasma_composition` + `impurity_radiation_totals` |
| `^cond.constraints.c24`, `c84` | `plasma_composition` + `calculate_parabolic_profile_values` |
| `^cond.constraints.c2` | those two + `impurity_radiation_totals` |
| `^cond.numerics.objf` | everything -- 21 holes, 3 unemittable leaves |

Ranked by conditions gated, which is the order the work is worth doing in:

| blocker | gates | kind |
|---|---|---|
| `.physics.plasma_composition` | **7/11** | array-assembling helper |
| `calculate_parabolic_profile_values` | **4/11** | unresolved global `gamma` |
| `.physics.impurity_radiation_totals` | 3/11 | was `self.imp_indices`; now array subscript |
| `.availability.electric_production` | 2/11 | `Composition` |
| `buildings.sizing`, `delta_eta_step`, `intersect_residual`, `quench`, `tf_magnet_cost` | 1/11 each | mixed |

So three of the eleven conditions are **one node away each**, and `objf` -- the objective --
is last by construction. That is why entry coverage and condition coverage came apart, and
why the two leaves array support did add (`quench`, `tf_magnet_cost`) bought no condition:
both gate 1/11, and that one is `objf`, which needs all the others anyway.

**`Composition` nodes resolve, and the check that makes it trustworthy is the negative
control.** Four distinct nodes, eight instances: `.buildings.sizing` (a named local),
`.availability.electric_production` (two inline calls in argument position, materialised as
preludes), `.tokamak.build.tf_outboard_mid` (a tuple-unpacked local) and `.power.delta_eta_step`
(index 4 **derived** from the name's position in the unpack target list, exactly as the
constraint fix derives one from a literal subscript). Agreement 0.000e+00 throughout, and a
binding diff against the untouched resolver shows **0 changed and 0 lost** among the
89/117/150 previously-resolved entries -- purely additive.

The negative control is why 0.000e+00 counts as evidence here rather than decoration.
Binding `shh` to `.build.z_tf_inside_half` -- its own dominant argument, the most plausible
wrong substitute -- gives **2.409e-02**: comfortably passable as noise under any loose
tolerance, which is precisely the failure this whole class was refused to avoid. Binding a
prelude local to the same-named *parameter* instead gives 5.018e+00. Python's own shadowing
rule is honoured by statement position (`_Frame.local_at(name, pos)`), which matters because
`DeltaEtaStep.step` opens with `p_fw_blkt_coolant_pump_mw = calculate_...(..., p_fw_blkt_coolant_pump_mw)`,
where the argument is the parameter and every later reference is the local.

An unrelated real bug fell out: `_source_and_params` read only `fdef.args.args`, so a
keyword-only signature (`def _masses(self, *, len_tf_coil, ...)`) looked **zero-parameter**
and every argument failed to bind. That is why
`superconducting_tf_coil_areas_and_masses` was refused, and it was never a `Composition`
node at all.

**What is left on `helias_5b` is not a long tail -- it is one feature.** The two nodes
gating 7 of the 11 conditions, `.physics.plasma_composition` and
`.physics.impurity_radiation_totals`, need the **same** capability stack, so the remaining
residual splits cleanly:

| | conditions |
|---|---|
| already emitting | 1 |
| reachable without the species stack (`wp_width_r_min`, via `pchip`) | 1 |
| **gated by the species-array stack** | **9** |

**Even if every other refusal is resolved, this configuration caps at 2/11** -- a correction
to the 4/11 first recorded here, see below. That is the number that decides whether "SAND in
Warp" is reachable, and it is one feature away rather than nine.

The stack, scoped against the real bodies rather than guessed:

1. a **resolver** rule for a leaf passed *by value* into a same-class helper --
   `PlasmaCompositionIgnited.__call__` calls `self._composition(plasma_composition_ignited, ...)`,
   which `_find_value_passed` refuses by design; accepting it means treating `_composition`
   (a `cottax` method, not a `models` function) as the leaf, with `arm` monomorphised;
2. **fixed-length array as N named scalars** end-to-end -- `_composition` does
   `jnp.stack([...])` over 12 scalars plus 2 placeholders, then destructures
   `results[4][H_INDEX]` and splices `results[:4]`/`results[5:]`. N = 14 and every index is
   a literal, so it is SSA-expansible, but it needs slice-sums, `.at[i].set()` and
   literal-index reads -- a different feature from "type this parameter `wp.array`";
3. **unrolling `jax.vmap`** over the 14 species (mechanical once 2 exists);
4. **2-D tables plus a log-interp variant** -- `calculate_average_charge_at_temp` is
   `jnp.interp(jnp.log(x), jnp.log(xp), fp)` against each species' own 200-point table, from
   genuinely `(14, 200)` `DataStructure` fields, where `wp_interp_N` is 1-D only;
5. `impurity_radiation_totals` additionally needs a Simpson integration over a profile grid.

Two hazards specific to this stack, both of which produce a *plausible* wrong number rather
than a crash: a slice-sum reassociated into a different order changes the last bits, and an
off-by-one in a literal species index lands on a neighbour of similar magnitude. Bit-exact
0.000e+00 is the only signal that separates them from success.

**Reverse mode works, and the autodiff thesis survives contact with a real kernel.**
`WARP_ENABLE_BACKWARD=1` builds the full-coverage `helias_5b` kernel (43 entries, 112,829
jaxpr equations), `wp.Tape` records it, and six reverse passes give the full 6x9 Jacobian.
No adjoint-pass refusal on anything the forward pass accepted.

| comparison | worst rel |
|---|---|
| Warp adjoint vs `jax.jacfwd` | **5.692e-12** |
| Warp adjoint vs `jax.jacrev` | 6.833e-12 |
| **`jax.jacrev` vs `jax.jacfwd`, Warp not involved** | **1.141e-12** |

That last row is the attribution and the reason the first is acceptable: JAX's own two AD
modes disagree by 1.1e-12 on this problem, so ~1e-12 is the intrinsic reverse-vs-forward
accumulation spread and Warp sits within ~5x of it. The **sparsity pattern is exact** -- 38
of 54 entries are zero in JAX and zero in Warp, worst absolute difference 0.000e+00.

**The 60x compile figure was wrong: it is 11-14x**, and for an instructive reason -- the
*forward* baseline grew from 35.1 s to 180-259 s as coverage rose, so the multiplier
collapsed while the absolute backward cost (42 and 46 min) barely moved. Measured on two
configs, neither extrapolated. Run time is negligible either way: 0.3 ms forward, 0.9-1.6 ms
per reverse pass, and a module is compiled once per hash and cached (a second run: 1.1 s).

**Registers are the real warning, and they have moved a long way.** Against the recorded
22-24 forward / 244-255 backward with 0-912 B spill:

| kernel | regs | spill |
|---|---|---|
| forward, 10 entries | 255 | 20,920 B |
| forward, 27 entries | 255 | **111,248 B** |
| backward, 10 entries | 255 | 40,024 B |

**The forward kernel is now itself register-saturated** -- 255 is the sm_75 ceiling -- with
spill growing 5.3x from 10 to 27 entries. Reverse mode is not what broke this; forward did.

**And the GPU build, not the arithmetic, is what runs out of room.** At full coverage NVRTC
was OOM-killed twice on the 29.9 MB `.cu`, once at **12.5 GB RSS / 40 GB VM after 32 minutes
alone** on a 15 GB machine. That is a *host RAM* limit in the CUDA compiler, not the card's
4 GiB. There is therefore no full-coverage GPU number, and none was fabricated. Since the
backward build triples the generated C++ (3.10x on both configs), a full-coverage backward
CUDA build is well outside this machine.

**Batching flips the CPU/GPU verdict, which the single-solve comparison had settled the
other way.** On reduced cuts where CUDA does build (identical kernel both sides, agreement
0.000e+00):

| batch | 10 entries CPU -> GPU | 27 entries CPU -> GPU |
|---|---|---|
| 1 | 46.9 -> 1629.0 us/pt | 154.0 -> 3921.5 |
| 256 | 12.2 -> 9.4 | 96.9 -> 16.8 |
| 65536 | 11.97 -> **0.85** | 96.5 -> **2.60** |

GPU loses ~35x at batch 1 on launch overhead, crosses over between 16 and 256, and at 65536
is **14.1x faster at 10 entries and 37.1x at 27** -- **the advantage grows with kernel
size**. This is the first measurement of the batched case and it goes opposite to
`optimise_design` §68's single-solve result (13.55 -> 15.57 s cold, 0.136 -> 0.227 s warm).

**On CPU, Warp loses to JAX by 7.6x** -- 144.6 against 19.0 us/pt at 65536, verified
independently. Warp's CPU per-point cost is **flat** (one thread per point, run serially,
1.27 ns per jaxpr equation) so batching buys it nothing there, while JAX vectorises across
the batch. This **reverses** the earlier record of Warp being 5.8-6.6x faster on CPU, which
was taken at 18-37 entries.

One caveat kept in view rather than buried: in the reduced-cut runs only 1-2 conditions are
reachable and XLA dead-code-eliminates nearly everything (0.017 us/pt for 12k equations,
not physically possible), while Warp executes the whole kernel. **Those JAX columns are not
a baseline**; only the full-coverage row, where 6 conditions keep the work live, is an
honest Warp-vs-JAX comparison.

**The jaxpr path crossed over and the resolver is gone.** Conditions roughly tripled to
quadrupled on every configuration, all compiling in round 0 with **zero Warp-codegen
exclusions**:

| configuration | entries | conditions | worst rel. diff |
|---|---|---|---|
| `helias_5b` | 32 -> **43**/94 | 1 -> **6**/11 | 4.3e-16 |
| `stellarator_helias` | 49 -> **62**/123 | 7 -> **12**/21 | 6.7e-16 |
| `large_tokamak_nof` | 47 -> **101**/177 | 4 -> **16**/33 | 2.7e-15 |

Per-node validation over 8 draws: 92, 120 and 166 passing, **zero failures**. `-6129` lines;
the module is **3493** against roughly 9600.

**The second-order effect is the one worth recording.** With the identification taxonomy
gone, the work stopped being *cataloguing refusals* and became *finding real bugs* -- four
of them, every one producing a plausible number rather than crashing:

- `jnp.stack(axis=1)` read `dimension` (a `concatenate` parameter), defaulted to 0, and
  silently **transposed** a (22, 6) output;
- `wp.max`/`min`/`clamp` **swallow NaN** where XLA propagates, turning an honest NaN out of a
  secant root-find into a *finite wrong answer* at 3 of 8 draws;
- `wp.sign(0.0)` is +1 against XLA's 0 (66/484 arguments);
- `wp.asin`/`acos` clamp to [-1, 1] where XLA returns NaN (264/484).

The last three came from `prim_check.py`, which sweeps every mapped primitive against XLA
over hostile arguments and now reports no unexpected divergence. That check is the general
form of the `DIRECT`-map finding recorded above, and it is what the single-point agreement
gate could never have produced.

Three specifics made the crossover. `bitcast_convert_type` is **not** emitted as a general
bitcast -- Warp 1.17 exposes no float-to-int reinterpretation at all -- so the
`bitcast`-then-`shift_right_arithmetic 63` **pair** is fused into a `_signbit` helper,
licensed by a use-count proof that fires only when the bitcast has exactly one consumer.
The value-supply gap was `_assemble` computing the MDA output env and **discarding it**,
leaving the 201-point profile grid a scalar so `_simpson` raised. Array-valued intermediates
bind as **one fixed-length Warp vector per `VarPath`**, not N flattened columns: flattening
would put 610 parameters on `fusion_rates` alone, spends the scarce ABI resource on values
that never leave the kernel, and -- decisively -- a scalarised array cannot be subscripted,
which every runtime-indexed `gather` needs.

**Open, and it bears on the whole reason for choosing Warp**: adjoint codegen defaults
**off**, because it costs ~60x compile time (~35 min against 35.1 s on a 1.36 MB module).
Forward values are byte-identical, but the built module cannot be `wp.Tape`d. Being
measured.

**jaxpr is the right IR, and the argument is not aesthetic.** The whole session's cost has
been a taxonomy of *Python shapes* -- `self.<attr>`, non-literal exponents, switch-valued
index bounds, tuple unpacking, `del`, keyword-only arguments, branches. **Tracing dissolves
every one of them by construction**: a jaxpr contains only primitives applied to typed
values, with every static decision already resolved against the concrete configuration. Not
one entry in that taxonomy survives contact with `jax.make_jaxpr`.

Measured before committing to it (`scratchpad/jaxpr/census.py`), tracing each node's own
`__call__` separately so node structure -- and therefore the emitter, and one `@wp.func` per
node -- is preserved rather than flattened into one giant jaxpr:

| configuration | node definitions | traced | distinct primitives | total eqns | median eqns/node |
|---|---|---|---|---|---|
| `helias_5b` | 95 | **91 (96 %)** | 44 | 2704 | 12 |
| `large_tokamak_nof` | 178 | **164 (92 %)** | 54 | 8155 | 6 |

Against 48 % / 52 % for the AST expander, and 34 % / 26 % of *entries* for the resolver.
(`large_tokamak_nof` first read 86 %; the missing 11 were a scalar `1.0` handed to an
array-valued read by the census's own fallback, tracing as "0-dimensional but 1 indexed"
and looking exactly like a tracing failure. **Second time this session a wrong harness
manufactured a wall of plausible failures** -- the first was reading parameter names off the
`VarPath` instead of the signature. Supplying the declared shape gives 92 %, and raises that
config's equation count from 4724 to 8155, so the array-shaped share is larger than the
first number suggested.) The
primitive histogram is dominated by trivial scalar arithmetic (`mul` 664, `add` 524,
`select_n` 211, `convert_element_type` 210, `div` 190, `sub` 173); the genuinely array-shaped
tail -- `dynamic_slice`, `slice`, `broadcast_in_dim`, `stack`, `argmax`, `reduce_sum`, `scan`,
`cond` -- is small and is the real remaining work. A whole configuration is under 5000
primitive operations.

**The honest limits.** Tracing bakes in the configuration, so a jaxpr is per-config -- which
suits a code generator and would not suit a general library. Names are lost, so generated
code reads as `a:f64[] = mul b c`; keeping one jaxpr per node rather than one per graph is
what keeps that navigable. And it does **not** solve array intermediates: `_simpson`'s
201-element grids are as unrepresentable in a per-thread scalar kernel after tracing as
before. Tracing removes the accidental difficulty, not the essential one.

**cottax already exposes exactly the interface the port needs, and the resolver exists
because that was not noticed.** A node's `__call__` *is* a pure function of its declared
inputs -- the `From(...)` defaults are ordinary parameters:

```python
def __call__(self, rminor=From(physics), kappa=From(physics)):
    return calculate_z_plasma_xpoint(rminor, kappa)
```

Census of every node body in the Drive (`scratchpad/species/shape_census.py`):

| shape | `helias_5b` | `large_tokamak_nof` |
|---|---|---|
| **one line, `return f(...)`** | **82 (86 %)** | **142 (80 %)** |
| straight-line, 1-2 statements | 4 | 5 |
| `_NormalisedResidual` wrappers (`Assign`/`Expr`/`Return`) | 4 | 26 |
| builtin (`operator.sub` behind `Compare`) | 3 | 3 |
| **has a branch** | **1** (`.Objective`) | **1** |

One branch in the entire graph. So the translation target could simply **be** `__call__`,
with whatever it calls transpiled recursively into further `@wp.func`s -- Warp device
functions call device functions, so nothing needs inlining for its own sake. That is:
transpiler + recursion + emitter. **No resolver, and no expander except where arrays
genuinely demand one.**

The resolver was built to look *past* the wrapper and find `calculate_z_plasma_xpoint`,
which is a reasonable instinct if the goal is to reuse the per-unit audit records keyed to
`functional_process/models/**` -- but it is not required for code generation, and it is what
turned a translation problem into a program-identification problem. The whole refusal
taxonomy in this document is downstream of that one choice.

**The resolver was the wrong architecture, and it generated most of this work.** The
tracked transpiler is **154 lines** and does its job. The resolver is **820** and exists
solely to answer *"which underlying function IS this node?"* -- because `defn.fn` is a
callable `eqx.Module` instance whose `__call__` wraps the real computation. Every refusal
category in this document is an artefact of asking that question: `Composition`,
`self.<attr>`, "4 distinct calls -- genuinely ambiguous", the arity mismatches. None of them
is a property of the code being ported.

`scalarise_function` -- written for exactly one node -- asks a different question: *expand
this node's own body, inlining whatever it calls, into one flat function.* That question has
no such categories. Pointed at **every** node in the Drive with no resolver at all
(`scratchpad/species/inline_all.py`):

| configuration | node definitions | scalarise straight off |
|---|---|---|
| `helias_5b` | 95 | **46 (48 %)** |
| `large_tokamak_nof` | 178 | **92 (52 %)** |

Half the graph, first try. **And not one remaining obstacle is "I cannot identify the
leaf."** What is left is a short, named, mostly mechanical list:

| obstacle | h5b | lt_nof | nature |
|---|---|---|---|
| `self.names` is a tuple (constraint wrappers) | 4 | **26** | the sequence-static mechanism already exists |
| `x ** <non-literal>` (`** p`, `** 1.4`, `** alphat`) | 12 | 8 | **refused only for bit-exactness** -- see below |
| index bound is a switch, not a literal (`lsa - 1`) | 13 | -- | switches are config-static integers, constant-foldable |
| `self.topology`, `self.__self__` | 2 | 8 | structured/bound attributes |
| builtin `operator.sub` (`Compare`/`Pairwise`) | 3 | 3 | one special case |
| genuine control flow, array intermediates, `y.shape` | ~15 | ~40 | real work |

The `**` row deserves particular attention: `scalarise` refuses a non-literal exponent
**only because `wp.pow` is not bitwise `lax.integer_pow`**. Under the relaxed `1e-12` gate
that objection evaporates and those ~20 nodes become a one-line change. A policy chosen for
tidiness was itself blocking a fifth of the remainder.

**So the honest answer to "what is the real holdup" is: an architectural choice, not the
physics.** The bodies really are pure `jnp`. Committing early to find-the-leaf turned a
translation problem into a program-identification problem, and then most of the effort went
into the taxonomy of ways identification fails. The right shape is inline-everything with a
small, explicit refusal list.

Two honest limits on this measurement: it counts `scalarise_function` **succeeding**, not the
result being verified correct -- each expansion still needs checking against `node_def.fn`,
which is exactly what that module was designed to allow. And a node that scalarises may still
depend on one that does not.

**`plasma_composition` resolves, no condition flipped, and the "one feature away" claim was
wrong.** The node is not resolved by finding its leaf but by **building** one:
`scalarise.py` expands the node's whole `__call__` -- helper, arm, `jax.vmap` and all -- into
a flat straight-line function of exactly its declared parameters, monomorphised per
occupant. Fixed-length arrays become N named scalars (`jnp.stack`, slice-sums,
`.at[i].set()`, literal-index reads, `vmap` unrolled to 14 calls); per-species tables become
one `wp.array2d` parameter each, chosen over 14 separate parameters so the species index
stays *data* rather than part of 28 generated identifiers where an off-by-one would silently
permute rows. Verified 18/18 outputs bit-identical against the real `node_def.fn`, and
**4000/4000 points on a swept range**.

**Two measured corrections to the scoping, both of which say the frontier analysis
systematically under-estimates.**

1. On `helias_5b` and `stellarator_helias` all seven "`plasma_composition`-gated" conditions
   are *also* gated by the `gamma` leaf -- which is an **ancestor of `plasma_composition`
   itself**. On those two configs this work could not have flipped a condition even in
   principle.
2. On `large_tokamak_nof`, where `plasma_composition` does gate 9 conditions alone, none
   flipped either, because resolving it **exposed a new frontier behind it**: `_simpson`'s
   `y.shape[...]` (17 conditions), `gammaln` in `calculate_pedestal_on_axis_temperatures`
   (20), `jnp.min` (20), `REACTION_CONSTANTS_DD2` (13), `jnp.full_like` (12), and per-condition
   constraint arity mismatches. `c84`/`c24` on `helias_5b` turn out to be blocked by a
   constraint arity mismatch **independently of `gamma`**.

This is the third time a "what is left" estimate has been revised upward, and the pattern is
now clear enough to state as a property of the method rather than as three separate
mistakes: **prefix-closure can only see one layer deep.** Resolving the frontier reveals the
next one, so every remaining-work estimate derived from it is a lower bound. The honest unit
of progress is the refusal census driven to zero, not a coverage fraction.

**`_simpson` refused with a measured reason**, which is the right shape for a refusal: its
`jnp.sum` is over 100 elements, and XLA's reduction is bitwise left-to-right sequential only
up to ~14 (0/2000 mismatches at n=12 and n=14; **1464/2000 at n=64**). A sequential expansion
would be arithmetically right and bitwise wrong. Under the relaxed `1e-12` gate this
particular objection weakens; the second reason does not -- its operands are 201-element
*intermediate* arrays, which a per-thread scalar kernel has no representation for.

**Sweeping found a bug a single point did not.** XLA fuses `jnp.interp`'s final step
`fp[i-1] + (delta/dx)*df` into an FMA; Warp emits the unfused pair -- genuinely different
arithmetic, not just different rounding. It produced a 1-ulp error in one species' `<Z>` at
**1 of 400 sweep points**. Fixed by emitting an exact Dekker FMA in both interp helpers; the
pre-existing 1-D helper had the same latent bug. And the corollary, found the same way:
`_xla_exp` is exact only on **[-8, 8]** -- 19999/20000 on [-20, 5], and **3896/20000** near
underflow at [-745, -700]. The earlier 8000/8000 figure was measured precisely on the range
where it holds.

**A latent baseline defect surfaced**: `calculate_divertor_geometry_conventional` emits
`wp.asin` (77 % exact) and was **inside the passing baseline**, so `large_tokamak_nof`'s
0.000e+00 was passing through an unverified transcendental at a single point. Under the
relaxed gate this is acceptable at ~1 ulp and the function stays; under the old exact gate it
had simply never been tested.

**The `DIRECT` math map is wrong for 6 of its 17 entries, and the 0.000e+00 record
survived by luck.** `transpile.py` maps `jnp.<f>` straight onto `wp.<f>` for seventeen
functions. Swept 4000 points each against the JAX counterpart, CPU, float64
(`scratchpad/transcendental/sweep2.py`):

| exact | differs |
|---|---|
| `sin` `cos` `tan` `log` `sqrt` `atan` `floor` `ceil` `sign` `round` `pow` -- 100 % | `tanh` **38.5 %** (4 ulp), `sinh` 48.6 % (3), `cosh` 54.7 % (2), `asin` 77.0 % (1), `acos` 92.8 % (1), `exp` **87.0 %** (1) |

Reachability analysis over every generated kernel shows only `sin`, `cos`, `log`, `sqrt`
and `pow` are actually called from an included condition -- **all five in the exact
column**. That is why every leaf so far reported 0.000e+00: not because the mapping is
sound, but because no reached leaf had yet used one of the six that are not. `tanh` is the
worst of them and is an unremarkable thing for a physics leaf to call.

**This also corrects the claim that "no leaf had ever exercised a transcendental other than
`sqrt`".** `wp.sin` appears 144 times across the generated modules and is reached 4 times;
`cos`, `log` and `pow` are reached too. They agree with JAX, which is a different and much
weaker statement than never having been tested.

The deeper point is about the gate rather than the map: **agreement is measured at a single
input point.** A one-point test would pass `exp` with probability 0.87 and `tanh` with 0.385.
The one function anyone swept over a range turned out to disagree on 13 % of it. Every
bit-exactness claim in this document rests on single-point evidence and should be read that
way until the gate samples a range.

**Fix, not yet applied**: the six unsafe names must move out of `DIRECT` into a refusal, so
a leaf using one is rejected rather than silently mis-emitted -- the same "refuse rather than
guess" rule the resolver already follows. A verified `_xla_exp` exists (below); the other
five have no replacement yet. Deferred only because an agent is mid-flight against this file
and one emitted kernel currently reaches `wp.exp`.

**Correction: the per-condition table above was a lower bound, and "three conditions are
one node away" was wrong.** `cond_holes.py` stopped *at* each hole and named the node owning
it, without recursing into that node's own reads -- so it reported the frontier, not the
requirement. Resolving the named node does not make the condition emittable if the node
itself reads another hole. Recursing through the refused nodes (`cond_closure.py`) gives the
set that must **all** be resolved:

| blocker | gates (transitive) | was reported |
|---|---|---|
| `.physics.plasma_composition` | **9/11** | 7/11 |
| `calculate_parabolic_profile_values` (`gamma`) | **9/11** | 4/11 |
| `.physics.impurity_radiation_totals` | **5/11** | 3/11 |
| `.buildings.sizing`, `.availability.electric_production` | 2/11 each | 1/11 |
| `intersect_residual` (`pchip`) | 1/11 | 1/11 |

Only **one** condition, `^cond.stellarator.wp_width_r_min`, is genuinely reachable without
the species stack, and it needs `pchip`. `c16` and `delta_eta` looked one node away and are
not: their named blockers read `plasma_composition` and `impurity_radiation_totals`
themselves. The `gamma` leaf turns out to matter as much as `plasma_composition`.

**`delta_eta_step` is not the ambiguous case its own error message claims, and the
message was about to justify building a registry for it.** The refusal names four calls --
`calculate_p_{div,shld}_heat_deposited_mw`, `calculate_p_fw_blkt_{coolant_pump,heat_deposited}_mw`
-- and calls the choice among them "genuinely ambiguous". Reading `DeltaEtaStep.step`
(`cottax/power/thermal_cryo.py`) shows none of the four is the leaf: they bind four prelude
locals, and the producing call is `calculate_delta_eta`, which returns 5 values with the
body taking index 4 via `_, _, _, _, delta_eta_next = ...`. The arity filter dropped it
(5 against 1 declared output) and then reported the survivors as an ambiguity -- **the
message names four calls precisely because it discarded the right one**. Ordering artefact,
not a property of the node. It is a `Composition` plus an unpack-index derivation exactly
analogous to the constraint fix, differing only in being spelled as an unpack target
position rather than a subscript. A registry for truly ambiguous nodes may still be worth
having; this is not one of them.

**A `self.<attr>` sequence-static mechanism exists** and is exercised end-to-end against
Warp at 0.000e+00, resolving `imp_indices`, `coefficients` and `den_helium_at_nodes`.
Constancy was established from declarations rather than intent: `eqx.field(static=True)`
puts a field in the pytree **treedef**, so it cannot be traced or mutated by a
transformation, and a different value is a structurally different graph -- checkable in one
line rather than inferred from control flow. `self.topology` was left alone deliberately:
it is a structured record, and every leaf that takes it uses it only to slice array
parameters, so all seven nodes are refused on independent array grounds and rendering it
would buy nothing. The mechanism refuses `jnp`/`numpy` arrays outright even inside a
static field, where they are genuinely constant -- a refusal, not a wrong read, pending
array support.

**Open**: widening past ~38 nodes; whether the adjoint spill is fatal or merely costly;
`jnp.interp`/`searchsorted` and array-valued lookup tables; and the arity-mismatched
constraint nodes (`constraint_11/32/34/35/65/82/83` forward to `eq`/`leq`/`geq`, four
returns against one owned output) which are **excluded and named rather than guessed at** --
choosing which tuple element a node owns is a judgement about intent, not about code.

## Two rules this file paid for

**Refuse rather than guess.** The transpiler raises on anything it does not recognise and
the function goes to a hand-written registry. A generator that silently mistranslates one
physics formula is worse than one that covers less and says so. The same rule caught three
silent mis-resolutions once the resolver was given an **arity invariant** (a resolved leaf's
return arity must equal its node's declared output count).

**A harness that skips what it cannot drive will report success on the subset it reaches.**
Five separate results in this file's history were wrong that way -- an unconditional return
annotation, a validator calling `float()` on a tuple, a "compiled" claim that was module
loading, a path rewrite pointing at the wrong copy, and `pkgutil.walk_packages` silently
never descending into namespace packages, hiding four subpackages and a third of the
codebase. **Every one inflated confidence by hiding a subset rather than by being wrong
about what it measured.** The measurements that survived enumerate from the graph or the
filesystem directly.
