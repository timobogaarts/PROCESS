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
| `.tfcoil.dcond[i_tf_sc_mat - 1]` | read | explicit-arg (see note) | `superconductor` — a *data-table lookup*, not a formula branch: `dcond` (8 fixed material densities, `tfcoil_variables.py`) is indexed by the same `i_tf_sc_mat` switch `coils.md` flags for `jcrit_from_material`. Ported as `den_tf_sc_material`, the already-indexed scalar — the port does not itself decide which index, matching `stellarator_D_structure.md`'s treatment of pre-resolved locals. **Invented name**: no PROCESS field is named this; it stands for a value one array-index step removed from any real `data` field (see cottax-node section). |
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
    a_tf_wp_with_insulation, a_tf_wp_no_insulation, len_tf_coil,
    a_tf_coil_inboard_case, den_tf_coil_case, den_tf_wp_turn_insulation,
    n_tf_coil_turns, a_tf_turn_cable_space_no_void,
    f_a_tf_turn_cable_space_extra_void, f_a_tf_turn_cable_copper,
    a_tf_wp_coolant_channels, den_tf_sc_material, a_tf_turn_steel, den_steel,
    a_tf_coil_wp_turn_insulation, n_tf_coils,
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
in `functional_process/total_process.py`. `den_tf_sc_material`'s `Input` reads
`.tfcoil.den_tf_sc_material` — **an invented `VarPath`, not an existing PROCESS field**
(see data-footprint table). Whoever declares the real `i_tf_sc_mat` lookup node (see
`coils.md`'s open question 1) should mint its output under this exact name so this node's
input resolves to a real edge instead of a dangling read.

## tier signal
**Tier 1.** No internal solve, no calls into other models, no switches inside this file
(the material switch lives one level up, in the caller that resolves `dcond[i_tf_sc_mat
- 1]` before this function is reached), no data-dependent control flow.

## switches touched
None directly — see `.tfcoil.dcond[...]` row above and `coils.md`'s `i_tf_sc_mat` entry.

## calls into other models
None. `constants.DEN_COPPER` (a module-level constant import) is the only external
reference, not a model call.

## JAX-difficulty flags
None. Plain scalar arithmetic throughout.

## open questions
1. Same invented-name concern as `stellarator_F_tf_nuclear_heating.md`: is
   `.tfcoil.den_tf_sc_material` the right permanent name for the resolved lookup, or
   should whoever designs the `i_tf_sc_mat` node pick something else — flagged for
   consolidation, not decided unilaterally here.
