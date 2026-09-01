# The uncut graph — a census of PROCESS's own coupling, 2026-09-01

**What this file is.** A measurement, not a design decision: how many strongly connected
components `indat.GRAPH` (and its tokamak counterpart) has **before** `mda.cut_graph`/
`mda.driven_graph` touch it at all — no cut, no assigned driver, no problem beyond what
`indat.py` already declared structurally (the `FixedPointFunction`/`ImplicitFunction`
self-loops). That is a direct measurement of the open question `CLAUDE.md` records under
"Logical mapping" (`Caller.call_models`/`Blocking`, `SCC` row): *"how much of PROCESS turns
out to be genuinely cyclic is an open empirical question … one confirmed SCC among 44 ported
nodes so far, with more expected once the orchestration layer … is reached."* The port has
grown a great deal since that sentence was written (154–245 nodes now, depending on
machine), and this file is the current answer on the two reference machines that assemble
without `process` at solve time and without a live PROCESS run.

**What this file is not.** It is not a claim about run order. `render_xdsm.grouped`'s own
docstring, and `mda.py`'s module docstring, both say the same thing for a good reason: *"a
raw cycle has no order, so `Blocking.scc` on the undriven graph would be reporting on a
schedule nothing can run."* That objection stands, unmodified, and this file does not
dispute it — see § 2 for exactly what does and does not survive it.

**Where the pictures are.** `render_xdsm.grouped_uncut()` (new; `functional_process/
render_xdsm.py`), invoked as `$PY -m functional_process.render_xdsm grouped_uncut
[--machine [IN.DAT]]`, writes:

- `dsm_provenance_uncut.html` / `dsm_scc_uncut.html` — the reference stellarator
  (`stellarator_helias.IN.DAT`)
- `dsm_provenance_uncut_tokamak.html` / `dsm_scc_uncut_tokamak.html` — the reference
  tokamak (`large_tokamak_eval.IN.DAT`, `boundary.TOKAMAK_INPUT_FILE`)

into `functional_process/` (gitignored, like the existing `dsm_*.html` pair — `.gitignore:65`
`*.html`). Titles say "UNCUT graph" and the SCC page's title says explicitly "(mutual
coupling, not a run order)"; the pages themselves carry no other prose, by the same
convention `visualization/grouping.py`'s module docstring states for `grouped`'s pair.

---

## 1. The predicted obstacle, and why it did not occur [measured]

The brief for this file warned that `Blocking.scc` on an undriven graph, or the downstream
helpers (`grouping_report`, `structure_order`, `render_grouped_dsm_html`), might raise the
same refusal `st_regression` hits elsewhere in this tree: *"coupled block [...] declares no
problem, so it has no conditions and there is nothing to hand a driver"*
(`Graph.needs_driver`, `~jaxgraph/src/cottax/graph.py:678-684`).

**It does not raise, and the reason is structural, not luck.** Traced through
`~jaxgraph/src/cottax/blocking.py` and `graph.py`:

- `Blocking.scc(graph)` calls `graph.strongly_connected_components`
  (`graph.py:449-464`), which is `nx.condensation` of the read/own edge graph followed by
  `nx.topological_sort` of the condensation. Neither step reads `DeclaredNode`-ness, a
  problem, or a driver — it is pure graph topology.
- `grouping_report`, `structure_order` and `_matrix_struct` (`visualization/grouping.py`)
  read `blocking.blocks` and `blocking.graph` only. None of them touches
  `Blocking.problems`/`Blocking.problem_types` — the two `@cached_query`s
  (`blocking.py:227-269`) that call `Graph.needs_driver`/`Graph.problem_type` and are the
  only things in the class that *can* raise the refusal above.
- The refusal is real and does exist on this exact graph — `mda.py`'s own module
  docstring names it as the reason `cut_graph` exists at all — but it lives behind a
  property this call path never asks for. `Blocking.scc` partitions and orders; it does not
  ask "can this be driven."

Verified directly (2026-09-01, `process_port` env): `Blocking.scc(machine_graph()[0])`
(the reference stellarator, no cut, no driver) returns without error, `len(blocking.blocks)
== 144`. Same for the tokamak: `223` blocks, no error. `tests/functional_process/
test_render_xdsm.py::test_blocking_scc_needs_no_driven_graph_at_all` patches
`mda.driven_graph` to raise `AssertionError` if called at all, then runs
`render_xdsm.grouped_uncut()` end to end (both files written) — the patched function is
never reached.

**So there was nothing to route around.** No placeholder problem, no fake driver, no
partial rendering — `grouped_uncut` calls exactly `machine_graph` → `Blocking.scc` →
`grouping_report`/`structure_order`/`render_grouped_dsm_html`, the same four calls
`grouped` makes minus `mda.driven_graph`.

---

## 2. What the SCC ordering means here, precisely

`Graph.strongly_connected_components` (`graph.py:449-464`) returns components ordered so
that if component A depends on component B, A appears after B — the condensation's own
topological order, well-defined for *any* directed graph, cyclic or not, driven or not.
Two different claims live inside that one order and this file is only making one of them:

- **Which block precedes which is a real, checkable fact about the uncut graph.** It says
  nothing about drivers or algorithms — it is the same claim `topological_order` makes for
  an acyclic graph, generalised to survive a cycle by collapsing it to one vertex first.
- **The order of members *inside* a genuinely coupled block is not a claim about anything.**
  `strongly_connected_components`' tie-break for members within one component is binding
  order (`graph.py:458-462`, `rank = {name: i for i, name in enumerate(self.nodes)}`) —
  there for lack of anything else to break the tie by, not because binding order is when
  those nodes run. Reading that as a schedule is exactly the error `grouped`'s docstring
  warns about, and this file does not make it. The SCC *membership* — which nodes fall in
  the same block at all — is the fact this file is about; the row order inside a coupled
  block's rows is drawn (nothing else to draw), and reads as "these nodes are mutually
  coupled," never as "in this order."

This is why `dsm_scc_uncut.html`'s title ends "(mutual coupling, not a run order)" rather
than reusing `grouped`'s "ordered by structure (SCC)" — the words are the only prose these
pages carry, so they are where the distinction has to live.

---

## 3. The census, uncut vs. cut, both machines [measured]

All figures below are `Blocking.scc(...)`/`grouping_report(...)` read directly off
`functional_process.render_xdsm.machine_graph` (uncut) and
`functional_process.mda.driven_graph` (cut/driven), 2026-09-01, `process_port` env. "Real"
membership excludes a minted problem paired with its own node (`BlockGrouping.real`) — the
same filter `grouped`'s own report already applies, so these numbers are directly
comparable to `grouped`'s printed summary.

| | reference stellarator (`stellarator_helias.IN.DAT`) | reference tokamak (`large_tokamak_eval.IN.DAT`) |
|---|---|---|
| declared nodes, uncut | 154 | 245 |
| declared nodes, driven | 156 (+2) | 248 (+3) |
| total blocks, uncut | **144** | **223** |
| total blocks, driven | **144** (unchanged) | **223** (unchanged) |
| SCCs with >1 member, uncut | 6 | 7 |
| — of which "really" coupled (>1 non-minted member) | **2** | **3** |
| — of which a bare structural self-loop (declared node + its own minted problem, `real == 1`) | 4 | 4 |
| cross-group edges, uncut / driven | 229 / 229 | 412 / 415 |
| cross-subsystem edges, uncut / driven | 147 / 147 | 173 / 173 |

**The headline: cutting changes zero block counts, on either machine.** Every
`FixedPointCut` `mda.cut_graph` applies lands *inside* an SCC that `Blocking.scc` already
found on the uncut graph — cutting mints one problem node and rewires the cycle's readers
onto it, which keeps the same set of nodes in one strongly connected component (the new
problem node both reads from and is read by the cycle's members) rather than splitting or
merging blocks. So each of the genuinely-coupled blocks above grows by exactly one member
(the minted problem) between the uncut and driven column, `real` is unchanged, and every
other block — including the four structural self-loops, which existed before any cut ran —
is bit-for-bit the same object. `tests/functional_process/test_render_xdsm.py::
test_cutting_changes_no_block_count_only_the_coupled_blocks_own_size` pins this.

The four structural self-loops present on **both** machines (`.physics.profiles.
ion_vol_avg_temperature`, `.vacuum.duct_diameter_root_find`, `.power.delta_eta_step`, and
one more each: `.stellarator.coils.intersect` on the stellarator, `.tokamak.
cicc_superconducting_tf_coil.dr_tf_plasma_case` on the tokamak) are `FixedPointFunction`/
`ImplicitFunction` declarations from `indat.py` itself — they are not what `mda.CUTS`
exists for, and `grouping_report`'s `real` filter is exactly what keeps them out of the
"genuinely coupled" count above (each is 2 members, 1 real).

---

## 4. Every genuinely-coupled uncut SCC, named [measured]

Node names spelled with `SPELLING` (`xDSMFormatterFlat`), the same formatter `render_xdsm`
draws with, so these match what the pages show.

### 4.1 Reference stellarator — 2 blocks

| size | members | subsystem span |
|---|---|---|
| 6 | `.physics.fusion_power_totals_mw`, `.physics.fusion_rates`, `.physics.fusion_totals_no_beam`, `.physics.plasma_composition`, `.physics.profiles.density_profile`, `.physics.profiles.parameterisation.parabolic_on_axis_densities` | inside `physics` (nests: `physics`, `physics.profiles`, `physics.profiles.parameterisation` — `BlockGrouping.container == ('physics',)`, does not cross) |
| 2 | `.stellarator.divertor`, `.stellarator.fw_area` | inside `stellarator` (single group — not `spans`, so neither crosses nor nests) |

Zero blocks cross a subsystem boundary on this machine (`report.crossing == ()`), matching
`grouped`'s own driven-graph figure exactly — cutting these two loops does not create or
remove any cross-subsystem coupling, only gives each an algorithm.

### 4.2 Reference tokamak — 3 blocks

| size | members | subsystem span |
|---|---|---|
| 4 | `.tokamak.build.tf_outboard_mid`, `.tokamak.build.wp_conductor_max_width`, `.tokamak.cicc_superconducting_tf_coil.superconducting_tf_wp_geometry`, `.tokamak.cicc_superconducting_tf_coil.tf_global_geometry` | inside `tokamak` (nests: `tokamak.build`, `tokamak.cicc_superconducting_tf_coil`) |
| 8 | `.physics.fusion_power_totals_mw`, `.physics.fusion_rates`, `.physics.fusion_totals_no_beam`, `.physics.plasma_composition`, `.physics.profiles.density_profile`, `.physics.profiles.ne_profile_integral`, `.physics.profiles.parameterisation.pedestal_on_axis_densities`, `.physics.profiles.parameterisation.pedestal_profile_values` | inside `physics` (nests: `physics`, `physics.profiles`, `physics.profiles.parameterisation`) — the stellarator's 6-node cycle, enlarged by the pedestal profile arm (`i_plasma_pedestal = 1`) exactly as `mda.CUTS`' own docstring describes |
| 9 | `.tokamak.cs_coil.flux_swing`, `.tokamak.pf_coil.inductance`, `.tokamak.pf_coil.sizes`, `.tokamak.pf_coil.time_point_currents`, `.tokamak.pf_coil.turn_currents`, `.tokamak.pf_coil.volt_seconds`, `.tokamak.pf_coil.waveform`, `.tokamak.plasma_inductance.volt_seconds`, `.tokamak.pulse.burn_time` | inside `tokamak` (nests: `tokamak.cs_coil`, `tokamak.pf_coil`, `tokamak.plasma_inductance`, `tokamak.pulse`) — the merged PF-coil/volt-second/burn-time cycle `mda.CUTS` describes since 2026-08-27 |

Zero blocks cross a subsystem boundary here either (`report.crossing == ()`, 3/3
`nests`) — every genuinely coupled loop this port has found so far, on either reference
machine, stays inside one top-level subsystem. That is a fact about the two machines
measured, not a structural guarantee: nothing in `Blocking.scc` or `grouping_report`
prevents a future node from closing a loop across subsystems, and this is exactly the
number `report.crossing` exists to keep watching.

None of the tokamak's three blocks is the stellarator's `stellarator.divertor`/
`stellarator.fw_area` pair — that cycle does not exist on the tokamak graph at all (see
§ 5, `.fwbs.f_ster_div_single`), and none of the stellarator's blocks is either of the
tokamak's TF-build or PF-coil cycles, neither of which the stellarator graph has the nodes
for (`.tfcoil.dx_tf_wp_primary_toroidal` is owned by a different, acyclic node there;
`.pf_coil.ind_pf_cs_plasma_mutual`/`.pf_coil.n_pf_coil_turns` are not owned on the
stellarator graph at all).

---

## 5. Which `mda.CUTS` entry opens which uncut SCC [measured]

For every variable in `mda.CUTS`, its owner on each declared graph, which uncut block that
owner sits in, and `graph.closing_readers(var)` — the same test `cut_graph` itself applies
before deciding whether a cut is live, inert (no closing readers: the cycle the variable
would close does not exist in this configuration) or skipped (the SCC already declares its
own problem, so a second one would make `Blocking` refuse the block outright — see
`mda.cut_graph`'s own comment, `mda.py:423-445`).

| `CUTS` variable | stellarator | tokamak |
|---|---|---|
| `.physics.proton_rate_density` | **live** → 6-node `physics` block | **live** → 8-node `physics` block |
| `.physics.fusden_alpha_total` | **live** → same 6-node `physics` block | **live** → same 8-node `physics` block |
| `.physics.f_temp_plasma_electron_density_vol_avg` | inert (0 closing readers — `parabolic_profile_values`'s arm has no path back, exactly as `CUTS`'s own docstring records) | **live** → same 8-node `physics` block (the third density cut, needed only on the pedestal arm) |
| `.fwbs.f_ster_div_single` | **live** → 2-node `stellarator.divertor`/`.fw_area` block | inert (0 closing readers — `large_tokamak_eval.IN.DAT` is the `ipowerflow == 0` arm, `AFwTotalNoPowerflow`, which does not read this variable at all) |
| `.tfcoil.dx_tf_wp_primary_toroidal` | inert (0 closing readers — owned by `.stellarator.coils.winding_pack_total_size_post`, a different, acyclic node; the cycle is TF-coil-architecture-specific) | **live** → 4-node TF build/winding-pack block |
| `.times.t_plant_pulse_burn` | inert (0 closing readers — owned by `.initialisation.stellarator_pulse_times`, a boundary-side node on this machine) | **live** → 9-node PF-coil/volt-second/burn-time block |
| `.pf_coil.ind_pf_cs_plasma_mutual` | not owned at all (no PF-coil-with-inductance arm on this stellarator graph) | **live** → same 9-node PF-coil block |
| `.pf_coil.n_pf_coil_turns` | not owned at all | **live** → same 9-node PF-coil block |
| `.tfcoil.dr_tf_plasma_case` | inert (0 closing readers — owned by `.stellarator.coils.coil_casing`, no cycle there) | **skipped** — owner *is* `^problem.tokamak.cicc_superconducting_tf_coil.dr_tf_plasma_case` itself: `DrTfPlasmaCaseFromInput` (`i_f_dr_tf_plasma_case == False`, `large_tokamak_eval`'s arm) is already a `FixedPointFunction` self-loop sitting inside this 2-node block, so `cut_graph` finds a declared node already in the SCC and skips — matching `CUTS`'s own docstring ("`large_tokamak_nof`/`_eval`/`low_aspect_ratio_DEMO` … all skip") |

**Every live cut lands on exactly one of § 4's blocks, and every block in § 4 has at least
one live cut** — the two enumerations agree in both directions, on both machines, which is
the check this table exists to make: `mda.CUTS` is neither over- nor under-complete against
what `Blocking.scc` finds on the uncut graph, for these two configurations.

`.tfcoil.dr_tf_plasma_case`'s **live-and-cutting** case — `st_regression.IN.DAT`, the one
file `CUTS`'s own docstring says this entry was added *for* (`i_f_dr_tf_plasma_case ==
True`, `DrTfPlasmaCaseFromFraction`, no self-loop in the slot, so the SCC has no problem and
needs the cut) — is **not** one of this file's two required configurations and was not
re-measured here; `mda.py:307-364` already carries that machine's own account in detail, at
the time it was added.

---

## 6. What this leaves open

- **Only two of the seven reference machines are measured here** (the brief's floor: the
  reference stellarator and at least one tokamak). `mda.CUTS`' docstring documents
  configuration-dependent branches for several other reference files (`spherical_tokamak_
  eval`/`st_regression`'s arm-2 PF-coil SCC losing a member and dropping `n_pf_coil_turns`
  entirely, §"On a machine with no central solenoid"; `st_regression`'s TF-case SCC actually
  needing `dr_tf_plasma_case`) that this census does not independently re-verify — that
  work already exists in `mda.py` itself and duplicating it here would not add information.
- **Whether a *crossing* (not merely nesting) coupled block exists anywhere in the port is
  still unmeasured beyond these two machines.** Zero on both machines checked; `mda.CUTS`'
  own docstring for `dx_tf_wp_primary_toroidal`/`dr_tf_plasma_case` suggests the TF-coil
  cycles are architecture-specific enough that a third machine could plausibly show a
  different member set, though nothing found here predicts a cross-subsystem one.
- **This file does not re-derive `mda.CUTS`'s sufficiency/minimality proofs** (§the merged
  cycles, the stale-read tie-breaks) — those live in `mda.py` and `test_mda.py` and are
  cited, not repeated.
