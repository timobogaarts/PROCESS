---
kind: model-unit
status: reviewed
confidence: high
---

**Ported.** `mass.py` / `test_mass.py`, one composed tier-1 function, tests passing
(fuzz only — no matching PROCESS unit test found).

## source
`process/models/stellarator/coils/mass.py` (146 lines, full file in scope). One public
orchestrator (`calculate_coils_mass`) chaining 8 private steps (`casing`,
`ground_insulation`, `superconductor`, `copper`, `conduit_steel`, `conduit_insulation`,
`total_conductor`, `total_coil`), each a `data`-in-`data`-out function called only from
`calculate_coils_mass` itself (grepped: none of the 8 called from `coils/calculate.py`
or elsewhere). Called once, from `coils/calculate.py` (registry unit #9).

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.len_tf_coil` | read | explicit-arg | read by 5 of the 8 steps, same value each time |
| `.tfcoil.a_tf_coil_inboard_case` | read | explicit-arg | `casing` |
| `.tfcoil.den_tf_coil_case` | read | explicit-arg | `casing` |
| `.tfcoil.m_tf_coil_case` | write, then read again by `total_coil` | local-intermediate | unconditional, unbranched, same-orchestrator produce-then-consume — see module docstring |
| `.tfcoil.den_tf_wp_turn_insulation` | read | explicit-arg | `ground_insulation`, `conduit_insulation` |
| `.tfcoil.m_tf_coil_wp_insulation` | write, then read again by `total_coil` | local-intermediate | same pattern |
| `.tfcoil.n_tf_coil_turns` | read | explicit-arg | `superconductor`, `copper`, `conduit_steel` |
| `.tfcoil.a_tf_turn_cable_space_no_void` | read | explicit-arg | `superconductor`, `copper` |
| `.tfcoil.f_a_tf_turn_cable_space_extra_void` | read | explicit-arg | `superconductor`, `copper` |
| `.tfcoil.f_a_tf_turn_cable_copper` | read | explicit-arg | `superconductor`, `copper` |
| `.tfcoil.a_tf_wp_coolant_channels` | read | explicit-arg | `superconductor`, `copper` (source comment: "0 for a stellarator... but keep this term for now") |
| `.tfcoil.dcond[i_tf_sc_mat - 1]` | read | explicit-arg (see note) | `superconductor` (`process/models/stellarator/coils/mass.py:88`) — a *data-table lookup*, not a formula branch: `dcond` (nine fixed material densities, `process/data_structure/tfcoil_variables.py:157-170`) is indexed by the same `i_tf_sc_mat` switch `coils.md` flags for `jcrit_from_material`. The **pure function** takes it as one already-indexed scalar, `den_tf_sc_material`, and does not itself decide the index. The **node** reads it at its real `VarPath`, `.tfcoil.dcond[0]` — see cottax-node section. |
| `.tfcoil.m_tf_coil_superconductor` | write, then read again by `total_conductor` | local-intermediate | |
| `.tfcoil.m_tf_coil_copper` | write, then read again by `total_conductor` | local-intermediate | |
| `.tfcoil.a_tf_turn_steel` | read | explicit-arg | `conduit_steel` |
| `.fwbs.den_steel` | read | explicit-arg | `conduit_steel` |
| `.tfcoil.m_tf_wp_steel_conduit` | write, then read again by `total_conductor` | local-intermediate | |
| `.tfcoil.a_tf_coil_wp_turn_insulation` | read | explicit-arg | `conduit_insulation` (docstring: "already contains `n_tf_coil_turns`") |
| `.tfcoil.m_tf_coil_wp_turn_insulation` | write, then read again by `total_conductor` **and** `total_coil` | local-intermediate | read by two different downstream steps, still same-orchestrator and unbranched |
| `.tfcoil.m_tf_coil_conductor` | write, then read again by `total_coil` | local-intermediate | |
| `.tfcoil.n_tf_coils` | read | explicit-arg | `total_coil` |
| `.tfcoil.m_tf_coils_total` | write | explicit-arg | final output |
| `a_tf_wp_with_insulation`, `a_tf_wp_no_insulation` | read | explicit-arg | already explicit args of `calculate_coils_mass`/`ground_insulation` in the source, not `data`-mediated |

No `implicit-io`, `implicit-io-via-callee`, or `redundant-duplicate-write` — every
internal write is read back exactly once more, in the same orchestrating call, with no
intervening branch or loop (textbook `local-intermediate`, same as
`stellarator_D_structure.md`'s `aintmass` chain).

## proposed signature(s)

```python
def calculate_coils_mass(
    a_tf_wp_with_insulation,
    a_tf_wp_no_insulation,
    len_tf_coil,
    a_tf_coil_inboard_case,
    den_tf_coil_case,
    den_tf_wp_turn_insulation,
    n_tf_coil_turns,
    a_tf_turn_cable_space_no_void,
    f_a_tf_turn_cable_space_extra_void,
    f_a_tf_turn_cable_copper,
    a_tf_wp_coolant_channels,
    den_tf_sc_material,
    a_tf_turn_steel,
    den_steel,
    a_tf_coil_wp_turn_insulation,
    n_tf_coils,
) -> tuple[float, float, float, float, float, float, float, float]:
    # (m_tf_coil_case, m_tf_coil_wp_insulation, m_tf_coil_superconductor,
    #  m_tf_coil_copper, m_tf_wp_steel_conduit, m_tf_coil_wp_turn_insulation,
    #  m_tf_coil_conductor, m_tf_coils_total)
    ...
```
As ported (`mass.py`) — the 8 source sub-functions are inlined as one straight-line
function rather than kept as 8 separate `data`-chained ones, since every intermediate is
`local-intermediate` (no reason to route through 8 node boundaries for values that never
diverge from what was just computed).

## cottax node
`CoilsMass`, an `ExplicitFunction` wrapping `calculate_coils_mass` unchanged, registered
in `functional_process/total_process.py:504`.

`den_tf_sc_material`'s `FromExactly` reads **`.tfcoil.dcond[0]`, a real `DataStructure` field
element** — not the invented `.tfcoil.den_tf_sc_material` an earlier pass minted. Changed
during the MDA-harness triage of `_audit/next_steps.md` §8.1, where that mint was one of
the three "ungrounded inputs" the harness reported; it is the only one of the sixteen
triaged paths that turned out to have a real field behind it. Evidence and reasoning:

- `dcond` is a real field with nine entries and a documented meaning
  (`process/data_structure/tfcoil_variables.py:157-170`, "density of superconductor type
  given by i_tf_sc_mat/…"). Nothing anywhere in `process/` is named `den_tf_sc_material`
  (grepped).
- `_audit/naming_convention.md` § "Array elements" already prescribes the
  `SequenceKey`-indexed `VarPath` for exactly this shape, and
  `physics/radiation_power.py:619-660` already binds
  `.impurity_radiation.f_nd_impurity_electron_array[0..13]` that way, so this needs no new
  mechanism.
- The index is static because `i_tf_sc_mat` is a topology switch
  (`_audit/naming_convention.md` § "switches are not ports") *and* because a `FromExactly`
  default is fixed at class-definition time. `CoilsMass` is therefore the
  `i_tf_sc_mat == 1` (ITER Nb3Sn) arm: PROCESS's own default
  (`process/data_structure/tfcoil_variables.py:246`), the value
  `tests/regression/input_files/stellarator_helias.IN.DAT:235` sets, and the value
  `total_process.py` already hardcodes for the same switch on the sibling node
  `WindingPackIntersectInputs`. A different material needs a sibling class overriding just
  this one `FromExactly`, in the style of `coils.py`'s eight `jcrit_from_material` node classes.

The node's other two invented-looking `From`s, `.tfcoil.a_tf_wp_with_insulation` and
`.tfcoil.a_tf_wp_no_insulation`, are **genuine mints and stay minted**: PROCESS keeps both
as Python locals in `winding_pack_total_size`
(`process/models/stellarator/coils/calculate.py:496-501`, with the source's own comment
"not global"), and this port's `WindingPackTotalSizePost`
(`coils/calculate.py:1136-1137`) is their producer. Note the trap recorded in
`_audit/next_steps.md` §8.1: fields with *exactly these two names* do exist, at
`.superconducting_tfcoil.*` (`process/data_structure/superconducting_tf_coil_variables.py:35,40`),
but they are written only by the tokamak resistive TF model
(`process/models/tfcoil/resistive.py:310,334`), which never runs for a stellarator, so
rebinding to them would compare against `DataStructure()`'s bare `0.0`.

## tier signal
**Tier 1.** No internal solve, no calls into other models, no switches inside this file
(the material switch lives one level up, in the caller that resolves `dcond[i_tf_sc_mat
- 1]` before this function is reached), no data-dependent control flow.

## switches touched
`.tfcoil.i_tf_sc_mat` — **not** touched by the pure function (which takes the resolved
scalar), but resolved by the *node*, at graph-assembly time, to pick `dcond`'s index.
Split-by-default per `_audit/traceability_policy.md`: `CoilsMass` is the
`i_tf_sc_mat == 1` arm. The reads-set difference between arms is exactly one array index
on one otherwise-identical read, so the arms are one-line siblings, not separate ports.
See the cottax-node section and `coils.md`'s `i_tf_sc_mat` entry.

## calls into other models
None. `constants.DEN_COPPER` (a module-level constant import) is the only external
reference, not a model call.

## JAX-difficulty flags
None. Plain scalar arithmetic throughout.

## open questions
1. **[RESOLVED — MDA triage, `_audit/next_steps.md` §8.1.]** Was: "is
   `.tfcoil.den_tf_sc_material` the right permanent name for the resolved lookup, or
   should whoever designs the `i_tf_sc_mat` node pick something else?" Answer: neither —
   the question presupposed a mint was needed. `dcond` is a real field, so the node reads
   `.tfcoil.dcond[0]` and no name is invented and no lookup node exists. The general rule
   this establishes: a mint is warranted only once it is confirmed that *no* real field —
   **including an array element** — holds the value. (The original wording cross-referenced
   a matching concern in `stellarator_F_tf_nuclear_heating.md`; no such note is present in
   that file today, so that pointer is dropped rather than repeated.)
