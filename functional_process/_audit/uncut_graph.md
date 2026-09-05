# The uncut graph — a census of PROCESS's own coupling

Answers `CLAUDE.md`'s "Logical mapping" open question — how much of PROCESS's coupling is
genuinely cyclic — directly, by running `Blocking.scc` on the graph **before**
`mda.cut_graph`/`mda.driven_graph` touch it (no cut, no assigned driver): pure topology,
not a run order (SCC *membership* is a fact; the order of nodes *within* one coupled
block is only a tie-break, not a schedule — don't read it as one).

## Result, on the two reference machines (2026-09-01)

| | stellarator | tokamak |
|---|---|---|
| declared nodes, uncut / driven | 154 / 156 | 245 / 248 |
| total blocks, uncut / driven | 144 / 144 | 223 / 223 |
| genuinely coupled SCCs (>1 real member) | **2** | **3** |
| bare structural self-loops (`FixedPointFunction`/`ImplicitFunction`, declared as such) | 4 | 4 |
| blocks crossing a subsystem boundary | **0** | **0** |

**Cutting changes zero block counts on either machine** — every `mda.CUTS` entry lands
inside an SCC `Blocking.scc` already finds on the uncut graph; a cut mints one problem
node into the same block rather than splitting or merging anything. And **every live cut
lands on exactly one coupled block, and every coupled block has at least one live cut** —
`mda.CUTS` is neither over- nor under-complete against what the uncut graph shows, on
both machines checked.

**Every genuinely-coupled block found so far, on either reference machine, stays inside
one top-level subsystem** — a fact about the two machines measured, not a structural
guarantee (nothing prevents a future node from closing a cross-subsystem loop). The
stellarator's two blocks are a 6-node `physics` fusion/profile cycle and a 2-node
`stellarator.divertor`/`fw_area` pair; the tokamak's three are a 4-node TF build/
winding-pack cycle, an 8-node `physics` cycle (the stellarator's, enlarged by the
pedestal profile arm), and a 9-node PF-coil/volt-second/burn-time cycle. Per-node
membership and the `mda.CUTS`-to-block mapping: git history.

Not measured beyond these two machines: whether a genuinely *crossing* (not merely
nesting) coupled block exists anywhere in the port, and the other five reference
machines' configuration-dependent variants (`mda.CUTS`'s own docstring documents several,
e.g. `st_regression`'s TF-case SCC needing a cut the tokamak reference doesn't).
