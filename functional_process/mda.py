"""Turning `indat.GRAPH` into something that can actually be run.

Most of `indat.GRAPH`'s SCCs already declare a problem and need only a driver:
the structural `FixedPointFunction`/`ImplicitFunction` self-loop pairs, the coil
island (`Intersect` and its own `^problem`) included. The rest are
**raw cross-node cycles with no declared problem at all** -- `Blocking` finds them;
nobody has said what solves them. A `Drive` refuses such a block outright
(`cottax.evaluate.Drive.__check_init__`: *"block ... declares no problem: it is run,
not driven"*), so the graph is not runnable until something says what closes them.

The stellarator has two -- `Divertor`/`AFwTotalWithPowerflow` (`ipowerflow != 0`
only) and the density/fusion/composition cycle around `DensityProfile`/`FusionRates`/
`PlasmaComposition`/`ParabolicOnAxisDensities` -- and the tokamak adds its own;
`CUTS`'s docstring names each with its measurement.

This module does two things: cut those raw cycles into declared `FixedPoint` problems
(via `cottax.rewrites.FixedPointCut`, using `Graph.closing_readers` to find the cut and
an empirical check that the cut set actually breaks the whole cycle -- neither is
guesswork, both are computed), and assign a driver to every block automatically, by
problem type -- and, with the driver, `Supply` the starting guesses the graph itself
computes (`SUPPLIED_STARTS`, `supply_starts`).

**Deliberately no node/SCC/cut counts here.** They moved on every porting wave, and a
docstring stating last wave's number is worse than one stating none -- five different
node counts, all present tense, once coexisted in a single audit document. The cycles
and their cuts are named in `CUTS` below and re-derived by
`test_mda.py::test_each_raw_cycle_is_fully_broken_by_its_own_cuts_and_no_fewer`, which
fails if either membership changes.
"""

from cottax.blocking import Blocking
from cottax.evaluate import Schedule
from cottax.interfaces.pytree_namespace_module import resolve
from cottax.problem import (
    Driven,
    FixedPoint,
    Optimise,
    RootFind,
    Start,
    driver_vars,
    unknowns_of,
)
from cottax.rewrites import Assign, Cut, FixedPointCut, Supply, Undrive
from cottax.graph import Graph
from cottax.spec import ProblemNode, NodePath, VarPath
from cottax.tools.path import path_map, written
import jax.numpy as jnp
from jax.tree_util import GetAttrKey

from functional_process.core.solver.drivers import (
    PicardDriver,
    SeededNewtonDriver,
    VmconDriver,
)
from functional_process.indat import GRAPH
from functional_process.paths import fwbs, pf_coil, physics, tfcoil, times

CUTS = (
    resolve(physics.proton_rate_density, VarPath),
    resolve(physics.fusden_alpha_total, VarPath),
    resolve(physics.f_temp_plasma_electron_density_vol_avg, VarPath),
    resolve(fwbs.f_ster_div_single, VarPath),
    resolve(tfcoil.dx_tf_wp_primary_toroidal, VarPath),
    resolve(times.t_plant_pulse_burn, VarPath),
    resolve(pf_coil.ind_pf_cs_plasma_mutual, VarPath),
    resolve(pf_coil.n_pf_coil_turns, VarPath),
    resolve(tfcoil.dr_tf_plasma_case, VarPath),
)
"""The variables cut to turn each raw cross-node cycle into a declared `FixedPoint`.

**`fusden_alpha_total` is the density/fusion cycle's *second* cut**, added when
`FusionTotalsNoBeam` gave `.physics.fusden_total`/`.fusden_alpha_total`/`.p_dt_total_mw`
their first producers (`_audit/boundary_inputs_audit.md` §4c (b7)/(b8)). That edge
(`FusionRates -> FusionTotalsNoBeam -> PlasmaComposition`) runs parallel to the one
`proton_rate_density` already cut, so one cut no longer breaks the cycle: `Blocking`
raised *"still cyclic with its problem(s) removed"* until this was added. Which second
cut to use was **measured, not chosen** -- of all 42 variables owned inside the enlarged
6-node cycle, `.physics.fusden_alpha_total` is the only one that makes the cycle acyclic
when paired with `proton_rate_density`, and no single variable does it alone.

**Watch this one on a cold start.** `PlasmaComposition` branches on
`fusden_alpha_total < 1e-6` as a "not yet calculated" bootstrap
(`composition.py:203-210`), so cutting it makes a Picard iterate drive a
*branch predicate*, not just a value. Seeded from a converged run (every harness here)
the branch never flips; from a cold `DataStructure` it starts on the other side.

**`f_temp_plasma_electron_density_vol_avg` is the density/fusion cycle's *third* cut, and
only on a pedestal machine.** With `i_plasma_pedestal = 1` (`large_tokamak_eval.IN.DAT`)
the profile slot is occupied by `pedestal_profile_values`/`pedestal_on_axis_densities`
plus `ne_profile_integral`, and that arm closes a **second** ring inside the same cycle
that the parabolic arm does not have:

    pedestal_on_axis_densities --.physics.nd_plasma_electron_on_axis-->
    density_profile            --.physics.nd_plasma_electron_profile--> (and via
    ne_profile_integral        --.physics.nd_plasma_electron_profile_integral-->)
    pedestal_profile_values    --.physics.f_temp_plasma_electron_density_vol_avg-->
    plasma_composition         --.physics.nd_plasma_ions_total_vol_avg--> (back to the
                                 first)

so the two cuts above leave a 5-node cycle and `Blocking` raises the same *"still cyclic
with its problem(s) removed"*. **Measured** the same way as the second cut: with
`proton_rate_density` and `fusden_alpha_total` already cut, exactly four of the cycle's
owned variables are single cuts that finish the job -- the four edges of that ring,
`.physics.f_temp_plasma_electron_density_vol_avg`,
`.physics.nd_plasma_electron_on_axis`, `.physics.nd_plasma_electron_profile` and
`.physics.nd_plasma_ions_total_vol_avg` -- and none of the other 45 does.

The tie-break, again, is PROCESS's own stale read, and here PROCESS says so **in a
comment and in a bootstrap flag**. `Physics.run` calls `plasma_composition()`
(`physics.py:254`) *before* `plasma_profile.run()` (`:370`), and `plasma_composition`
reads `f_temp_plasma_electron_density_vol_avg` at `physics.py:1387` under
`physics.py:1377-1386`: *"f_temp_plasma_electron_density_vol_avg now calculated in
plasma_profiles, after the very first call of plasma_composition; use old parabolic
profile estimate in this case"*, guarded by `physics.first_call`. That is precisely a
fixed-point iterate with a hand-written first guess -- the same shape as
`fusden_alpha_total`'s `< 1e-6` bootstrap, and the same caveat: seeded from a converged
run the guard is long past, cold it is not.

**Inert on the stellarator**, and for a structural reason rather than a gate:
`.physics.f_temp_plasma_electron_density_vol_avg` is owned there by
`parabolic_profile_values`, whose `closing_readers` set is *empty* --
`plasma_composition` reads it, but the parabolic arm has no path back, so the variable
is not on a cycle at all and `cut_graph` skips it.

`proton_rate_density` and `f_ster_div_single` were each, when added, the *only*
single-variable cut (out of every variable owned inside their own cycle) that made that
cycle's subgraph fully acyclic on its own -- checked directly, not assumed, by cutting
each candidate in turn and checking `.is_acyclic` on the result. `proton_rate_density`
(owned by `FusionRates`, read by `PlasmaComposition`) was sufficient for the
density/fusion/pedestal/composition loop as it stood then, before `FusionTotalsNoBeam`
enlarged it; `f_ster_div_single` (owned by `Divertor`, read by
`AFwTotalWithPowerflow`) still is, on its own, for the divertor/first-wall loop -- see
`_audit/next_steps.md` §5 for the cycle's own discovery. Neither is a `Feasibility`/
`Optimise` question: both are genuine "PROCESS iterates this to a fixed point"
couplings, so `FixedPointCut` (not `RootFindCut`) is the right closure -- matching
PROCESS's own `Caller.call_models`, which re-runs its whole pipeline up to 10 times
and checks idempotence, the same shape as a Picard iteration over these two blocks.

**`dx_tf_wp_primary_toroidal` is the tokamak build/winding-pack cycle's cut**, and it is
the one entry here that exists on no stellarator graph at all. The cycle is four nodes:

    tf_global_geometry            --.superconducting_tfcoil.tan_theta_coil-->
    superconducting_tf_wp_geometry --.tfcoil.dx_tf_wp_primary_toroidal-->
    build.wp_conductor_max_width  --.tfcoil.dx_tf_wp_conductor_max-->
    build.tf_outboard_mid         --.build.r_tf_outboard_mid-->  (back to the first)

**Measured, not chosen**, the same way the density loop's second cut was: of the 22
variables owned inside this cycle, exactly four have closing readers -- the four edge
variables above -- and **each of the four, cut alone, makes the cycle acyclic and
`Blocking` accept**. So sufficiency does not pick between them and something else has to.
`test_mda.py::test_the_tokamak_build_winding_pack_cycle_is_cut_where_process_reads_stale`
records that table and the tie-break.

The tie-break is **PROCESS's own stale-read**. `Caller._call_models_once` runs
`build.run()` (`caller.py:288`) *before* `cicc_sctfcoil.run()` (`:306`), and
`Caller.call_models` re-runs that whole order up to 10 times until it stops moving --
a Gauss-Seidel sweep. Three of the four edges are read *fresh* inside one such sweep:
`tan_theta_coil` is written at `tfcoil/base.py:145` and read at
`superconducting.py:1613`, both inside one `cicc_sctfcoil.run()`;
`dx_tf_wp_conductor_max` is a local inside one call of
`plasma_outboard_edge_toroidal_ripple` (`build.py:1570` to `:1593`); and
`.build.r_tf_outboard_mid` is written by `build` (`build.py:1901`/`:1939`) and read by
the TF coil model (`tfcoil/base.py:137`/`:205`) *later in the same pass*.

Exactly one edge crosses the pass boundary: `build.py:1570` reads
`self.data.tfcoil.dx_tf_wp_primary_toroidal` (`build.py:1929`/`:1971`) -- the
winding-pack toroidal width the TF coil model wrote on the **previous** pass
(`superconducting.py:186`, `:1630`/`:1664`/`:1715`). Cutting there makes one Picard
iterate of this block exactly one PROCESS pass, so the port's fixed point is reached by
the same recurrence PROCESS uses rather than by a differently-rotated one that merely
shares its fixed point. It also
happens to be the friendliest of the four to seed: `.tfcoil.dx_tf_wp_primary_toroidal` is
a real `DataStructure` field, so a harness reads the converged value straight off `data`,
where `.tfcoil.dx_tf_wp_conductor_max` -- the next-best candidate on stale-read grounds,
since it is what `build`'s stale read is immediately turned into -- is a mint and needs
its `mda_harness.KNOWN_MINT_VALUES` entry to be seedable at all.

**Applies only where the cycle exists.** `cut_graph` skips a cut whose variable has no
closing readers in the graph it was given, so this entry is inert on the stellarator
(whose TF winding pack is `models/stellarator/coils/`, a different set of nodes
entirely): same digest, same blocks, same pins -- the same gating the `ipowerflow`
paragraph below describes, for a different reason.

**`t_plant_pulse_burn`, `ind_pf_cs_plasma_mutual` and `n_pf_coil_turns` together cut
one nine-node SCC since 2026-08-27** -- registering `pfcoil.vsec` and its turn-current
feed (`cold_boundary.md` producer 4; `models/pfcoil/volt_seconds.py`) merged the
volt-second/burn-time ring and the PF coil ring the two paragraphs below describe:
`burn_time` now reads `.pf_coil.vs_cs_pf_total_burn` from `pf_coil.volt_seconds`,
which reads the inductance matrix and `turn_currents`' waveform products, while
`flux_swing` still reads `.physics.vs_plasma_ramp_required` from
`plasma_inductance.volt_seconds`, which reads `.times.t_plant_pulse_burn` back from
`burn_time`. **Measured on the merged cycle** (27 owned variables, 18 with closing
readers): *no* single cut suffices any more; exactly two *pairs* do
(`c_pf_cs_coils_peak_ma` or `f_j_cs_start_end_flat_top`, each with
`t_plant_pulse_burn`), and both pair-members on the PF side are the stale edges the PF
paragraph below already rejected as not being what PROCESS carries; the standing trio
is sufficient and each of its three is necessary given the other two.
So the merged cycle keeps exactly the cuts its two halves had, `cut_graph` groups them
into **one** `FixedPoint` over three unknowns
(`^problem.times.t_plant_pulse_burn.cycle`), and the per-half tie-breaks below --
PROCESS's own cross-pass stale read for `t_plant_pulse_burn`, PROCESS's own
`first_call` seed pair for the PF unknowns -- carry over unchanged.
`test_mda.py::test_the_merged_pf_volt_second_burn_time_cycle_keeps_its_cuts` records
the table.

**`t_plant_pulse_burn` cuts the volt-second/burn-time half**, a two-node ring waves
2/3's consolidation created by registering both of its nodes:
`.tokamak.plasma_inductance.volt_seconds` reads `.times.t_plant_pulse_burn` (for the
flat-top volt-second requirement) and owns `.physics.v_plasma_loop_burn`, which
`.tokamak.pulse.burn_time` reads to produce the burn time. **Measured, pre-merge**
(the merged-cycle table above supersedes the sufficiency census, not the tie-break):
the ring's nine owned variables contained exactly two with closing readers -- the two
edges -- and *each* was a sufficient single cut, so sufficiency did not pick and the
tie-break is,
again, PROCESS's own stale read. `Caller._call_models_once` runs `physics.run()`
(`caller.py:290`, volt-seconds inside it) *before* `pulse.run()` (`caller.py:322`), so
within one pass the burn time the volt-second requirement reads is the **previous**
pass's -- and PROCESS says so in its own comment (`physics.py:4882-4884`: *"tburn ...
on first iteration will not be correct if the pulsed reactor option is used, but the
value will be correct on subsequent calls"*), while `v_plasma_loop_burn` is read fresh
in the same pass. Cutting where PROCESS carries the stale value makes one Picard
iterate one PROCESS pass; `test_mda.py::
test_the_volt_second_burn_time_cycle_is_cut_where_process_reads_stale` records the
table. `.times.t_plant_pulse_burn` is a real `DataStructure` field, so the harness
seeds the guess straight off converged `data`.

**`ind_pf_cs_plasma_mutual` + `n_pf_coil_turns` cut the PF coil half, and they are
PROCESS's own seeds.** The five-node ring the pfcoil registration created (now the PF
side of the merged SCC above) --

    time_point_currents --.pf_coil.c_pf_cs_coil_*_ma--> waveform
    waveform            --.pf_coil.c_pf_cs_coils_peak_ma--> sizes
    sizes               --geometry + .pf_coil.n_pf_coil_turns--> inductance
    inductance          --.pf_coil.ind_pf_cs_plasma_mutual--> flux_swing
    flux_swing          --.pf_coil.f_j_cs_start_end_flat_top--> time_point_currents
    (plus the parallel edges sizes -> flux_swing via `n_pf_coil_turns` and the coil
    geometry arrays sizes -> inductance)

-- is `currents.md` § "The cycle" made real. **Measured, pre-merge**: of the ring's
fifteen owned variables, eleven had closing readers; exactly two were sufficient
*single* cuts
(`.pf_coil.c_pf_cs_coils_peak_ma`, `.pf_coil.f_j_cs_start_end_flat_top`), and neither
is what PROCESS carries across a pass. The pair chosen instead is **exactly the
loop-carried state PROCESS bootstraps**: `pfcoil.py:605-608` seeds
`ind_pf_cs_plasma_mutual[:, :] = 1.0` and `n_pf_coil_turns[:] = 100.0` on `first_call`
and leans on `Caller.call_models` re-running the pipeline -- so those two are the
iteration's unknowns by PROCESS's own declaration, each is necessary given the other
(measured: dropping either leaves the ring cyclic), and both are real `DataStructure`
fields the harness can seed from converged `data` (or from PROCESS's own literals on a
cold start). The two single cuts that sufficiency alone would allow are each *one*
stale edge of the three PROCESS actually carries (`sizes` reads the previous pass's
peak currents too), so neither reproduces PROCESS's recurrence either -- preferring
the declared seeds keeps the port's fixed-point unknowns the ones PROCESS names.
`test_mda.py::test_the_pf_coil_cycle_is_cut_at_the_variables_process_seeds` records
the table and the tie-break. `cut_graph` groups the pair into **one** `FixedPoint`
problem over two unknowns, driven Picard -- `FixedPointCut` -> `PicardDriver`, per the
round-2 brief's decision (validation against PROCESS first; a `RootFind` on the
`n_pf_coil_turns` residual is the recorded later upgrade, deliberately not done now).

**On a machine with no central solenoid the merged SCC is four nodes, not nine, and
`n_pf_coil_turns` drops out of this table entirely.** Measured 2026-08-31 on the arm-2
graph (`indat._pf_coil_system_arm == 2`, `iohcl = 0`, eight PF coils) of
`spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT`. Measured **twice**, and the
second time is what makes it evidence: first on a stand-in graph, with the then-last
unported occupant (`tf_stress_arm == (0, 1, 0)`, `extended_plane_strain`) replaced by
the plane-stress node, and then again once that occupant actually landed, on the real
assembly. The two graphs agree in every respect this paragraph reports -- 234 nodes,
the same six cycles, the same five landing cuts, `schedule()` building -- so the
stand-in probe was sound, which is worth knowing the next time one is needed. Both
files give the same answer:

    .tokamak.pf_coil.inductance  --.pf_coil.ind_pf_cs_plasma_mutual-->
    .tokamak.pf_coil.volt_seconds --.pf_coil.vs_cs_pf_total_burn-->
    .tokamak.pulse.burn_time      --.times.t_plant_pulse_burn-->
    .tokamak.plasma_inductance.volt_seconds --.physics.ind_plasma--> (back to the first)

`cs_coil.flux_swing` is gone with the namespace, and `sizes`, `waveform`,
`time_point_currents` and `turn_currents` leave the SCC with it: `next_steps.md` §20.2
predicted exactly that and it holds. `.pf_coil.n_pf_coil_turns` is still *owned* (by
`pf_coil.sizes`) and still read by `pf_coil.inductance`, but has **no closing readers**
-- there is no path back to `sizes` any more -- so `cut_graph` skips it, as it skips
`.fwbs.f_ster_div_single` (the divertor arm has no cycle here) and
`.tfcoil.dx_tf_wp_primary_toroidal` (picture-frame TF, so `build.tf_outboard_mid` does
not close the winding-pack ring). Six of `CUTS`'s eight entries are owned on arm 2 and
five land: the three density/fusion cuts, plus `t_plant_pulse_burn` and
`ind_pf_cs_plasma_mutual`, which `cut_graph` groups into one two-unknown
`^problem.times.t_plant_pulse_burn.cycle`. `schedule()` builds on both files.

**Sufficient, but not minimal, and that is the one thing arm 2 changes about this
table.** The four-node SCC is two rings sharing the `burn_time -> t_plant_pulse_burn ->
plasma_inductance.volt_seconds` edge (the long one above, and the two-node
`v_plasma_loop_burn` ring), so **`.times.t_plant_pulse_burn` alone breaks it** -- it is
the only one of the SCC's five closing-reader variables that does, measured by cutting
each in turn on `graph.subgraph(cycle)` and checking `.is_acyclic`. Cutting
`.pf_coil.ind_pf_cs_plasma_mutual` as well is therefore redundant, and redundant in a
way worth naming rather than tidying: PROCESS's `pfcoil()` calls `induct` before `vsec`
in one pass, so `vsec` reads the matrix *fresh*, and cutting it makes the port's Picard
carry a stale value across an iterate where PROCESS does not. On the reference tokamak
that cut is necessary (the pre-merge PF ring closes through it) and is PROCESS's own
`first_call` seed; on arm 2 it is neither. The entry stays, because `CUTS` is one table
serving every machine and `cut_graph`'s "cut where there are closing readers" rule is
what makes it portable -- but note that
`test_mda.py::_assert_every_raw_cycle_is_cut_sufficiently_and_minimally`, which asserts
sufficiency *and* minimality, would fail on an arm-2 graph, and that is a property of
this table rather than of the port. Deciding between "drop redundant cuts in
`cut_graph`" and "let a machine carry a redundant unknown" is open, and should be taken
when an ST file assembles for real rather than against a stand-in.

**`dr_tf_plasma_case` cuts the inboard-TF radial-build cycle, and it is the entry that
made `st_regression.IN.DAT` assemble at all** (2026-08-31). The earlier reading of this
cycle -- *"the block declares one problem either way, so `Blocking` accepts it and no new
entry is needed here"* -- was **wrong**, and wrong in a way worth naming: it read the
four-node shape off `large_tokamak_nof` and assumed the ST had it too. It does not. The
TF plasma-side case slot has two occupants, and only one of them declares a loop:
`DrTfPlasmaCaseFromInput` (`i_f_dr_tf_plasma_case == False`) is a `FixedPointFunction`
whose `^problem.tokamak.cicc_superconducting_tf_coil.dr_tf_plasma_case` sits *inside* the
SCC, while `DrTfPlasmaCaseFromFraction` (`True`, and `st_regression`'s arm) is a plain
`ExplicitFunction`. So on `st_regression` the SCC is three nodes with **no problem in
it**, and `Blocking` raised *"declares no problem ... there is nothing to hand a
driver"* -- the one reference machine that built no schedule.

The cycle is three nodes and two rings sharing one return edge:

    build.dr_tf_inboard_winding_pack --.build.dr_tf_inboard--> build.tf_inboard_radii
    build.tf_inboard_radii           --.build.r_tf_inboard_in-->
    cicc_superconducting_tf_coil.dr_tf_plasma_case
                                     --.tfcoil.dr_tf_plasma_case--> (back to the first)

plus the short ring `dr_tf_inboard_winding_pack --.build.dr_tf_inboard-->
dr_tf_plasma_case --.tfcoil.dr_tf_plasma_case-->` back, which skips `tf_inboard_radii`.
It exists only where iteration variable 140 is active: that is what puts
`DrTfInboardFromWindingPack` in the slot and makes `.build.dr_tf_inboard` a *produced*
variable rather than a run input (`models/build.py`, `process/models/build.py:1683-1688`).

**Measured, then tie-broken.** Of the SCC's seven owned variables exactly three have
closing readers, and two of those are sufficient single cuts:
`.build.dr_tf_inboard` (it is the forward edge of *both* rings) and
`.tfcoil.dr_tf_plasma_case` (the return edge of both); `.build.r_tf_inboard_in` breaks
only the long ring and leaves the short one, so it is not sufficient. Sufficiency does
not pick between the first two, and the tie-break is the standing one -- **PROCESS's own
stale read**. `Caller._call_models_once` runs `build.run()` (`caller.py:288`) *before*
`cicc_sctfcoil.run()` (`:306`), and of the three edges only one crosses that pass
boundary: `process/models/build.py:1685-1687` reads `self.data.tfcoil.dr_tf_plasma_case`
-- the value `tf_global_geometry` wrote on the **previous** pass
(`process/models/tfcoil/base.py:325`, clamped at `:333-340`). The other two are read
fresh inside one sweep: `.build.dr_tf_inboard` is written at `build.py:1684` and read at
`base.py:325`/`:333` later in the same pass, and `.build.r_tf_inboard_in` is written at
`build.py:1719-1725` and read at `base.py:333` in that same pass. Cutting the one edge
PROCESS actually carries makes one Picard iterate exactly one PROCESS pass, rather than a
differently-rotated recurrence that merely shares the fixed point -- the same argument,
and the same conclusion, as `dx_tf_wp_primary_toroidal` above. `.tfcoil.dr_tf_plasma_case`
is also a real `DataStructure` field (`tfcoil_variables.py:77`), so a harness seeds the
guess straight off converged `data` with no `KNOWN_MINT_VALUES` entry.

**Where it lands, and why the other three tokamaks do not move.** `cut_graph` skips a cut
whose SCC already declares a problem (see its own comment): on `large_tokamak_nof` the
same three nodes merge with `DrTfPlasmaCaseFromInput`'s self-loop into a four-node block
that already has one, and on `large_tokamak_eval`/`low_aspect_ratio_DEMO` (140 not in
`ixc`, so `.build.dr_tf_inboard` is a boundary input) the only cycle is that two-node
self-loop. All three skip. `spherical_tokamak_eval.IN.DAT` -- the other arm-2 machine, and
the natural place to expect a shared cut -- turns out **not** to share it: 140 is not in
its `ixc` either, so `.build.dr_tf_inboard` is unproduced, `.tfcoil.dr_tf_plasma_case`
has no closing readers, and there is no TF cycle there at all. On the stellarators the
variable is owned by `.stellarator.coils.coil_casing` with no closing readers. So this
entry is live on exactly one of the seven reference machines, and inert -- checked, not
assumed -- on the other six.

**The second cycle is `ipowerflow != 0`-only** (`next_steps.md` §5:
`AFwTotalWithPowerflow` is the `ipowerflow != 0` arm; `AFwTotalNoPowerflow` -- the
`ipowerflow == 0` arm -- does not read `.fwbs.f_ster_div_single` at all, so there is
no cycle to cut there). `driven_graph` below only attempts a cut whose variable
actually has closing readers in the `graph` it was given, so a graph built with
`ipowerflow == 0` skips this cut cleanly rather than raising `Cut`'s own "no readers"
refusal. The default `GRAPH` (this module's default argument) has `ipowerflow == 1`,
so both cuts apply there.
"""


def cut_graph(graph=GRAPH):
    """`graph` (default: `indat.GRAPH`, the default-configuration graph), with
    every raw cycle in `CUTS` that actually exists in `graph` cut into a declared
    `FixedPoint` problem. Every remaining multi-node SCC now declares exactly one
    problem -- confirmed below, not assumed, by `driven_graph()`'s own doctest-free
    but exercised construction (`schedule()` calling `Blocking.scc` on this graph
    would raise if not).

    Takes `graph` rather than always using the module's default so a caller checking
    this against a specific real `IN.DAT` can pass `indat.graph_for(
    configuration_matching_that_file)` instead -- which nodes exist at all (and
    therefore which of `CUTS` even applies -- see that tuple's own docstring on the
    `ipowerflow`-gated second cycle) can differ by configuration, so this should be
    run against whichever graph is actually being checked, not silently default to
    the wrong one.
    """
    # Cuts are grouped by the cycle they break, and each group becomes **one**
    # `FixedPointCut` -- i.e. one `FixedPoint` problem over however many unknowns that
    # cycle needed. Applying them one at a time instead mints one problem per cut, and
    # `Blocking` then refuses the block outright: *"declares 2 problems -- one driver
    # answers one problem, so `Combine` them into a single problem over every unknown,
    # or nest one inside the other. Which of those is a modelling decision"*. It is,
    # and this is the decision: PROCESS iterates its whole pipeline to idempotence, so
    # the two cut variables of the density/fusion cycle are two unknowns of one Picard
    # iteration, not two nested loops.
    #
    # Every `closing_readers` call is made on the **uncut** graph, before any of the
    # group is applied, so the readers a cut re-routes are the ones the original cycle
    # had rather than ones a sibling cut already moved.
    by_cycle: dict = {}
    cycles = [frozenset(c) for c in graph.cycles]
    declared = frozenset(graph.declared)
    for var in CUTS:
        if var not in graph.owners:
            # Not produced in this configuration at all -- `closing_readers` refuses
            # an unowned variable outright, and unowned is the strongest form of "no
            # cycle to cut here": `.times.t_plant_pulse_burn` is a *plain boundary
            # input* of the stellarator graph (its producer, `.tokamak.pulse.
            # burn_time`, is a tokamak node), where `dx_tf_wp_primary_toroidal` is
            # merely acyclic there.
            continue
        readers = graph.closing_readers(var)
        if not readers:
            continue  # this cycle does not exist in this configuration
        owner = graph.owners[var]
        key = next((i for i, c in enumerate(cycles) if owner in c), var)
        if key is not var and any(n in declared for n in cycles[key]):
            # **The SCC already declares its own problem, so it needs no cut.**
            # `Blocking` allows a block exactly one problem -- *"one driver answers one
            # problem, so `Combine` them into a single problem over every unknown, or
            # nest one inside the other"* -- and `cut_graph`'s whole job is to give a
            # problem to an SCC that has none. Where a `FixedPointFunction`'s declared
            # self-loop already sits inside the SCC, that job is done: the self-loop's
            # driver re-runs every other node of the block on each iterate, which is
            # exactly what a cut here would buy. Adding one anyway mints a *second*
            # problem in the same block and `Blocking` refuses it outright.
            #
            # This is the same shape as the `closing_readers` skip above -- a cut
            # applies where the cycle it names actually needs breaking -- and it is what
            # lets `.tfcoil.dr_tf_plasma_case` be one table entry serving every machine:
            # on `st_regression.IN.DAT` the TF case slot is `DrTfPlasmaCaseFromFraction`
            # (an `ExplicitFunction`, no loop) and the three-node SCC has no problem, so
            # the cut lands; on `large_tokamak_nof`/`_eval`/`low_aspect_ratio_DEMO` the
            # slot is `DrTfPlasmaCaseFromInput` (a `FixedPointFunction`) and its
            # `^problem.tokamak.cicc_superconducting_tf_coil.dr_tf_plasma_case` is in the
            # SCC, so the cut is skipped and those three graphs are bit-for-bit what they
            # were. Measured: without this guard all three raise *"declares 2
            # problems"*.
            continue
        by_cycle.setdefault(key, []).append(Cut(var=var, readers=readers))
    for cuts in by_cycle.values():
        # One cut keeps its historical name (`^problem.physics.proton_rate_density`);
        # several need an explicit `place`, since no single variable names what closes
        # them. Named after the first cut's own place with a `.cycle` component, which
        # is unique (a variable is cut at most once) and reads as what it is:
        # `^problem.physics.proton_rate_density.cycle`.
        place = (
            None
            if len(cuts) == 1
            else NodePath((*cuts[0].var.keys, GetAttrKey("cycle")))
        )
        graph = FixedPointCut(tuple(cuts), place=place).apply(graph)

    # Every problem gets `Start` ports, one per unknown, read from `^guess.<place>`.
    #
    # `cottax.evaluate.AbstractDriver` takes its starting values as *declared driver
    # data* rather than reading them off the unknowns' own names: `Drive.role_data`
    # walks the driver's `requires` and looks up the ports the problem declares, and
    # `Drive.__check_init__` refuses both directions -- a driver requiring a kind the
    # problem lacks, and a kind declared but not consumed. Every driver this port
    # **The driver is part of the graph now.** `Assign` retypes each problem into a
    # `Driven` -- problem plus algorithm -- and *mints* the ports that algorithm needs
    # from its own `requires`: a Newton wants a `Start`, so `^guess.<place>` appears per
    # unknown; a Picard wants nothing and nothing appears. That is one op where this used
    # to need two (`Initialise` to declare the ports, then a separate `{problem: driver}`
    # map handed to `schedule_for`), and it removes the failure mode between them --
    # ports declared before the algorithm was known could be required-but-undeclared or
    # declared-but-unconsumed, and both are now unrepresentable rather than refused.
    #
    # It stays here rather than in `schedule()` because the minted ports are real
    # boundary inputs: a caller measuring this graph's boundary, or drawing it, must see
    # them. Assigning is a modelling decision and is recorded in `Plan.ops` like any
    # other, so it survives `subgraph`/`prune` without a side table.
    return graph


def starts_for(graph, problem):
    """`(unknown, guess_port)` pairs for `problem`, in `owns` order.

    A driven problem's starting values are no longer read off the unknowns' own names:
    `Initialise` (applied by `driven_graph`) gives every problem one `Start` port per
    unknown, named `^guess.<place>`, and `Drive.role_data` reads the start from *there*.
    So a caller seeding a run writes `^guess.physics.temp_plasma_ion_vol_avg_kev`, not
    `.physics.temp_plasma_ion_vol_avg_kev` -- the latter is the answer, and writing it
    would be seeding the output.

    The pairing is read off the node rather than re-derived with `Initialise.start_of`:
    `Start`s pair with the *unknowns* by declaration order
    (`cottax.problem._check_driver_data`), so the node itself is the authority on which
    guess belongs to which unknown, and `strict=True` fails loudly if that ever stops
    being true.

    **`unknowns_of`, not `owns`.** A driven node also owns whatever its driver reports
    (`cottax.problem.DriverOut` -- `^driver_out.steps.<place>` and friends), and a report
    is not something a start pairs with. Asking `owns` here would zip a `Start` against a
    step count and `strict=True` would refuse a perfectly good graph -- loudly, which is
    why this is a one-line fix and not a lurking wrong answer.

    **A `Supply`-ed start is not returned**, because there is nothing for a caller to
    seed: `supply_starts` points those ports at a node that computes the guess, so the
    graph produces them and a seeder writing one would be overwriting a computed value
    with a `DataStructure` field. Every caller of this function is a seeding site
    (`mda_harness`, `sand_harness`, `mdf.seed`/`prime`, `run_sand_harness`), so the
    filter belongs here rather than in each of them. `Drive.role_data` does not use this
    function -- it asks the node -- so the driver still reads the supplied value.
    """
    node = graph[problem]
    starts = driver_vars(node, Start)
    if not starts:
        # **No driver, or a driver that needs no start: no ports, and that is legitimate
        # now.** `Assign` mints driver data from the algorithm's own `requires`, so a
        # problem that has not been assigned one has no `Start` to pair with -- where the
        # old `Initialise` gave every problem a port before any algorithm was chosen, and
        # a missing port could only mean a bug. `strict=True` below still catches the
        # real error, a partially-ported problem.
        return ()
    return tuple(
        (unknown, start)
        for unknown, start in zip(unknowns_of(node), starts, strict=True)
        if start not in graph.owners
    )


def guess_sources(graph) -> dict:
    """`{guess_port: unknown}` over every problem in `graph`.

    The inverse lookup every seeding site needs: given a `^guess.*` input, which
    unknown's value belongs in it. Nothing in a `DataStructure` is spelled `^guess.*`,
    so a seeder that grounds these ports by their own name silently writes `0.0` into
    every starting guess -- which is not a slow solve but an impossible one, and is
    exactly the failure `run_sand_harness._seed`'s own docstring describes.
    """
    return {
        guess: unknown
        for problem in graph.declared
        for unknown, guess in starts_for(graph, problem)
    }


SUPPLIED_STARTS = {
    ".stellarator.wp_width_r_min": ".stellarator.wp_width_r_min_guess",
}
"""`unknown -> the graph-owned variable that is its starting guess`, applied by
`supply_starts` as a `cottax.rewrites.Supply` on the problem's `Start` port.

**This entry used to be a `ROOT_FIND_SEEDS` row and the difference is the point**
(`_audit/next_steps.md` §14.5). `intersect`'s `xin` is computed by PROCESS itself
(`(r_coil_minor / 10) ** 2`, `coils/calculate.py:452-458`) and the port discarded it, so
the seed had to re-derive it from `.stellarator.r_coil_minor` read out of *the block's
context* -- which held `r_coil_minor` only because the invented `.tfcoil.j_tf_wp` edge
dragged the pre-intersect node into the block. Splitting `i_tf_sc_mat` into occupants
removes that edge and the seed loses its source. The occupant owns the guess now, and
`Supply` points the `Start` at it: PROCESS's own number, reaching the driver as an
ordinary graph edge, correct per material because the occupant is per material, and one
fewer `guess` entry on the boundary.

Keyed by `path_str()` on both sides so the table reads as a table; resolved against the
graph's own owners at use, which is the check that the producer is really there.
"""

GIVEN_STARTS = {
    ".pf_coil.ind_pf_cs_plasma_mutual": 1.0,
    ".pf_coil.n_pf_coil_turns": 100.0,
}
"""`unknown -> the value its `^guess.*` port is **given**`, applied by `given_start`.

**PROCESS's own numbers, and its own reason.** `process/models/pfcoil.py:600-608`:

    # If this is the first visit to the routine the inductance matrix
    # ind_pf_cs_plasma_mutual and the turns array have not yet been calculated,
    # so we set them to (very) approximate values to avoid strange behaviour...
    if self.data.pf_coil.first_call:
        self.data.pf_coil.ind_pf_cs_plasma_mutual[:, :] = 1.0e0
        self.data.pf_coil.n_pf_coil_turns[:] = 100.0e0
        self.data.pf_coil.first_call = False

That guard is a *starting guess* wearing mutable state, and this port drops mutable
state -- so the value came out with the mechanism. These two are exactly the cut
variables of `^problem.times.t_plant_pulse_burn.cycle`, and without them that block's
first Picard pass runs the PF chain at **zero turns and zero mutual inductance**: both
volt-second producers come out `0.0`, and `pulse.calculate_burn_time`'s
`abs(vs_cs_pf_total_burn) / v_plasma_loop_burn` is `0.0 / 0.0`. The iterate is `nan`,
`optx.fixed_point` stops on a non-finite iterate and returns it, `mdf.prime` writes it
into the `^guess` port, and every later call reads it back -- the block frozen at a
poisoned value for the whole solve (`_audit/optimise_design.md` §31.29). All three
tokamak configurations fail on it.

**Why a *given* value and not a `Supply` edge.** `SUPPLIED_STARTS` is the better
mechanism and does not apply here: cottax refuses a `Start` produced by the block it
starts (*"the driver reads its data before the block runs, so a producer inside the
block cannot have run yet"*, `supply_starts`), and both producers -- `.tokamak.pf_coil.
volt_seconds` and `.tokamak.plasma_inductance.volt_seconds` -- are **inside** this
block. cottax states the honest answer for that case itself: such a machine keeps its
`^guess.*` boundary input. So the guess is given, explicitly, from a table that says
where the number came from.

**Why not guard the division instead.** `calculate_burn_time` is a faithful port --
PROCESS computes `0.0 / 0.0` there too. It never *evaluates* it at zero turns, because
it initialises by running `first_call`. Wrapping the division in `safe_divide` would
make this port's answer differ from PROCESS's at points where PROCESS is fine, to paper
over a seed we chose. The seed is the defect.

Values are broadcast to the shape the cold `DataStructure` carries, so the table says
what a coil is worth and never how many there are. Keys are the **unminted** path: the
unknown of a `FixedPointCut` is `^hat.pf_coil.n_pf_coil_turns`, and what PROCESS writes
a number for is `.pf_coil.n_pf_coil_turns`.
"""


def given_start(unknown, fallback):
    """`GIVEN_STARTS`' value for `unknown`, shaped like `fallback`, or `fallback`.

    `fallback` is what the `DataStructure` answered, which supplies the shape and is the
    answer whenever the table says nothing. A table entry replaces the *value* only: a
    dataclass default of `np.zeros(22)` becomes `100.0` twenty-two times, and nothing
    here decides how many PF coils a machine has.
    """
    from cottax.tools.minting import unminted  # noqa: PLC0415

    # Keyed on the **quantity**, not on the minted copy. A `FixedPointCut`'s unknown is
    # `^hat.pf_coil.n_pf_coil_turns`; the number PROCESS writes is for
    # `.pf_coil.n_pf_coil_turns`, and a table that had to spell the mint would be a table
    # about cottax's naming rather than about the machine. `unminted` is the same
    # normalisation `mda_harness._ground_truth` already applies for the same reason.
    given = GIVEN_STARTS.get(unminted(unknown).path_str())
    if given is None:
        return fallback
    return jnp.full_like(jnp.asarray(fallback, dtype=float), given)


ROOT_FIND_SEEDS = {
    # PROCESS's own starting value, `d = np.full(4, 1e-6)`
    # (`process/models/vacuum.py:379`) -- a flat constant there, so a flat constant
    # here. Every `VarPath` of this node is minted, so cold or warm there is nothing in
    # `data` to seed it from: this is its *only* starting guess, not a fallback.
    ".vacuum.d_duct": lambda context: (1.0e-6,),
}
"""Fallback starting guesses for `RootFind` unknowns, as `f(context) -> tuple`, used
only when the value seeded from `data` is unusable (see `SeededNewtonDriver`).

One entry left. `.stellarator.wp_width_r_min`'s moved to `SUPPLIED_STARTS` above, which
is a strictly better answer for a guess PROCESS computes: a `Supply`-ed start needs no
`data`, no block context and no fallback at all. `d_duct`'s cannot follow it -- nothing
computes that constant, PROCESS writes the literal -- so the mechanism stays.

Keyed by `path_str()` rather than by `VarPath` so the table reads as a table; resolved
against the block's own context at use, which is also the check that the block really
does close over what the guess needs.
"""


def _var(context, path_str):
    """The `VarPath` in `context` spelled `path_str`.

    Raises
    ------
    KeyError
        If the block does not close over it -- a seed that silently fell back would be
        indistinguishable from no seed at all.
    """
    for var in context:
        if var.path_str() == path_str:
            return var
    raise KeyError(
        f"{path_str} is not in this block's context, so no starting guess can be "
        f"derived from it"
    )


def _root_find_seed(conditions):
    """A `SeededNewtonDriver`'s starting guess for whichever block it is driving.

    Matched on the block's *unknown*, not on the problem: for a
    `FixedPointCut`/`ImplicitFunction` the problem's name is minted from the unknown's
    place -- so `^problem['Intersect']` is answered by the unknown it owns, resolved
    from the context at call time.

    **Module level, and that is load-bearing.** This used to be a factory
    (`_root_find_seed(problem)` returning a closure) whose `problem` argument the body
    never read: a *fresh function object* on every assembly, equal to no other. A
    `SeededNewtonDriver` is a frozen `equinox.Module`, so it compares its `seed` field by
    identity, and one such field made every re-assembled `Schedule` compare unequal to
    the structurally identical one before it. That is the whole key of
    `sand_harness._SCHEDULE_WHOLE`, so an assemble-and-solve loop rebuilt and recompiled
    the schedule's two XLA programs every time round: **2 compiles per re-assembly, 0
    after this** (`_audit/optimise_design.md` §37 is the same lesson applied to
    `host_cache._BOUND`, and this is the last instance of it in the port). A module-level
    function has one identity for the life of the process, which is what a cache key
    wants.

    Raises
    ------
    KeyError
        If no unknown of this block names an entry in `ROOT_FIND_SEEDS` -- a driver that
        silently started from nothing would be indistinguishable from a seeded one.
    """
    for var in conditions.unknowns:
        entry = ROOT_FIND_SEEDS.get(var.path_str())
        if entry is not None:
            return entry(conditions.context)
    raise KeyError(
        f"no starting guess for {written(conditions.unknowns)}, and the one seeded "
        f"from `data` was unusable -- add an entry to `ROOT_FIND_SEEDS`"
    )


def driven_graph(graph=GRAPH, **driver_options):
    """`cut_graph` with an algorithm attached to every problem: the runnable graph.

    **Split from `cut_graph` because the two are different decisions and one caller needs
    only the first.** Cutting a cycle is structure -- it says these nodes are coupled and
    this variable closes the loop. Assigning a driver is an algorithm choice. They used
    to be one function because a driver lived in a side map and could be chosen last;
    now it lives *in the graph*, and `Combine` refuses to join two problems that carry
    one (*"combining two problems discards the algorithm answering each -- `Undrive`
    first"*). `sand` joins its `FixedPoint`s into one `Optimise`, so it must build on the
    cut graph and assign afterwards. That refusal is what made the seam visible; it was
    always there.
    """
    graph = cut_graph(graph)
    return assign_drivers(graph, default_drivers(graph, **driver_options))


def supply_starts(graph: Graph) -> Graph:
    """Point every `Start` port `SUPPLIED_STARTS` names at the node that computes it.

    `Assign` opens `^guess.<place>` as a boundary input, one per unknown; `Supply` is
    cottax's counterpart -- *"this is how a low-fi model's output becomes the start
    instead"* -- and an ordinary `Rewire`, so the port stays a `Start` and cannot
    silently become something else. A guess that PROCESS itself computes therefore
    arrives as an ordinary graph edge rather than as a driver-side fallback that has to
    re-derive it out of the block's context (`SUPPLIED_STARTS`' own docstring for why
    that mattered here, and `_audit/next_steps.md` §14.5 for the whole story).

    **A producer inside the block is skipped, not supplied.** cottax refuses a `Start`
    produced by the block it starts (*"the driver reads its data before the block runs,
    so a producer inside the block cannot have run yet"*), and that refusal would be
    raised by `schedule_for`, far from here. On this graph the case is real and is
    exactly the switch value the split is about: `Bi2212WindingPackIntersectInputs` reads
    `.tfcoil.j_tf_wp`, so with that occupant the guess's producer is *in* the coils SCC.
    Such a machine keeps its `^guess.*` boundary input, which is the honest answer -- the
    guess is not available before the solve that computes it.

    Applied by `assign_drivers`/`reassign_drivers` rather than by `driven_graph`, because
    `mdf`, `sand` and `sand_harness` each cut and assign for themselves and every one of
    them needs the port pointed somewhere.
    """
    for problem in tuple(graph.declared):
        node = graph[problem]
        if not isinstance(node, Driven):
            continue
        onto = {}
        for unknown, start in starts_for(graph, problem):
            target = SUPPLIED_STARTS.get(unknown.path_str())
            if target is None:
                continue
            producer = next(
                (
                    (var, owner)
                    for var, owner in graph.owners.items()
                    if var.path_str() == target
                ),
                None,
            )
            if producer is None:
                continue  # no occupant of that slot produces it in this machine
            var, owner = producer
            if owner in graph.descendants([problem]):
                continue  # inside the block -- see this function's own docstring
            onto[start] = var
        if onto:
            graph = Supply(problem, path_map(onto)).apply(graph)
    return graph


def assign_drivers(graph: Graph, drivers: dict) -> Graph:
    """`Assign` each driver onto its problem, then `supply_starts`.

    The two-line idiom every call site needs now that `schedule_for` takes no drivers:
    choose (`default_drivers`), then attach. Kept as a function rather than inlined
    because *re*-assigning is a different operation -- see `reassign_drivers` -- and a
    caller that conflates them silently replaces one algorithm with another.

    `supply_starts` runs here, and not one layer up, because the ports it re-points do
    not exist until `Assign` has minted them: which data a solve needs supplied is a
    property of its algorithm (`Assign`'s own docstring), so the two are one step.
    """
    for problem, driver in drivers.items():
        graph = Assign(problem, driver).apply(graph)
    return supply_starts(graph)


def reassign_drivers(graph: Graph, drivers: dict) -> Graph:
    """Replace the algorithm on problems that already carry one: `Undrive`, then `Assign`.

    `Assign` refuses a problem that is already `Driven`, deliberately -- *"replacing one
    algorithm with another silently is not a rewrite"* -- so swapping a driver is two
    recorded ops, not an overwrite. `mdf` needs exactly this: it runs one blocking under
    two algorithms (a seeded eager solve and a traceable one), which used to be two
    `{problem: driver}` maps over one graph and is now two graphs, because the algorithm
    is part of the graph.
    """
    for problem, driver in drivers.items():
        if isinstance(graph[problem], Driven):
            graph = Undrive(problem).apply(graph)
        graph = Assign(problem, driver).apply(graph)
    return supply_starts(graph)


def default_drivers(
    graph: Graph,
    bounds=(),
    callback=None,
    condition_scale=(),
    max_iter=None,
    optimiser=VmconDriver,
) -> dict:
    """One driver per **problem**, chosen mechanically by problem type

    Takes a `Graph` rather than a `Blocking`: since `Assign` puts the driver *in* the
    graph, the choice has to be made before there is a blocking to speak of -- and it
    never needed one, because the problem's own type is what decides. A node that already
    carries a driver (`Driven`) is skipped rather than re-assigned, which `Assign` would
    refuse anyway: replacing one algorithm with another silently is not a rewrite.

    Historic shape -- a Newton
    for every `RootFind` (`Intersect`, `DuctDiameterRootFind`), a Picard for every
    `FixedPoint` (the 8 structural self-loops plus the two cuts above), and a
    `VmconDriver` for an `Optimise` (only `functional_process.sand` registers one). No
    bespoke per-block choice: every block in this graph is one of exactly these shapes.

    **The `Optimise` arm is where the equality/inequality split is settled.**
    `ConditionMap` carries a flat condition tuple with no type information
    (`~/jaxgraph/src/cottax/evaluate.py:135-178`), so a driver cannot ask which of its
    conditions is the objective. It does not have to guess either: this function has the
    problem's own definition in hand, so the counts are **read off `Optimise.equalities`/
    `Optimise.inequalities`** and never counted by a caller. That is the whole of
    `_audit/optimise_design.md` §4.1's "positional contract" worry, removed by asking the
    node instead of the reader.

    `bounds`/`callback`/`condition_scale`/`max_iter` are forwarded to the `VmconDriver`
    and ignored by the others -- all four are algorithm choices with no home on
    `Optimise` (see `VmconDriver`'s docstring on why bounds are not extra inequality
    constraints, and on why the residual equalities need scaling that PROCESS's own
    constraints must not get).

    **`max_iter=None` keeps `VmconDriver`'s own default of 100**, which is PROCESS's
    `n_iteration_max` for PROCESS's own 8-variable problem. A SAND block is a *different*
    and larger problem -- the stellarator's is 14 unknowns against 21 conditions -- and
    PROCESS's cap has no standing over it, which is why the caller may say. Measured, on
    the cached reference run: SAND C2 needs **326** SQP iterations and C3 **258**, both
    landing on the same optimum (`objf 1.2177573`), where the pre-round-2 graph's larger
    SAND block (22 unknowns / 16 equalities, before ten `FixedPoint`s dissolved into
    ordinary nodes) needed 131. `_audit/optimise_design.md` §12 records the measurement
    and why the count moved.

    **`optimiser` names the class, not an instance**, because the counts are read off
    `Optimise.equalities`/`Optimise.inequalities` *here* and a caller handing in a
    built driver would have to have counted them itself -- the one thing the paragraph
    above says never to do. `SlsqpDriver` is the other class that fits: its fields are a
    strict subset of `VmconDriver`'s, so the same call constructs either. Anything
    passed here must therefore accept `n_equality`/`n_inequality`/`bounds`/`callback`/
    `condition_scale` and, if `max_iter` is given, `max_iter`.

    A second optimiser is worth having at all because on a problem with no PROCESS
    answer to compare against -- every SAND arm -- it is the closest thing to an oracle
    available: agreement is evidence, and a disagreement separates "this problem is
    degenerate" from "this solver handles degeneracy badly"
    (`_audit/optimise_design.md` §42).

    Raises
    ------
    TypeError
        If a block declares a problem type none of the three answers -- e.g. a
        `Feasibility`, which this graph does not currently register any of.
    """
    drivers = {}
    for problem, definition in graph.definitions.items():
        if not isinstance(definition, ProblemNode) or isinstance(definition, Driven):
            continue
        if isinstance(definition, RootFind):
            drivers[problem] = SeededNewtonDriver(seed=_root_find_seed)
        elif isinstance(definition, FixedPoint):
            drivers[problem] = PicardDriver()
        elif isinstance(definition, Optimise):
            # Not passed as `max_iter=max_iter`: `None` here means *say nothing*, and
            # `VmconDriver.max_iter` is an `int` field with a default it would then be
            # handed instead of keeping.
            said = {} if max_iter is None else {"max_iter": max_iter}
            drivers[problem] = optimiser(
                n_equality=len(definition.equalities),
                n_inequality=len(definition.inequalities),
                bounds=bounds,
                callback=callback,
                condition_scale=condition_scale,
                **said,
            )
        else:
            raise TypeError(
                f"{problem!r} declares a {type(definition).__name__}, and "
                f"default_drivers has no default driver for that problem type -- "
                f"assign one explicitly"
            )
    return drivers


def schedule(graph=GRAPH) -> Schedule:
    """`graph` (default: `indat.GRAPH`), block by block, every cyclic block
    driven by its default driver. `Drive`/`Schedule`'s own construction is what
    checks this is actually runnable -- a `Schedule` that builds is a `Schedule` that
    can be called. See `driven_graph`'s own docstring for why `graph` is a parameter,
    not always the module default.
    """
    driven = driven_graph(graph)
    blocking = Blocking.scc(driven)
    return Schedule(blocking)
