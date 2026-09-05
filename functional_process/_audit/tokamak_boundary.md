# Tokamak boundary — what a second device was missing, by name

**Superseded — tokamak support has since fully landed.** This was the pre-implementation
sizing exercise: of the tokamak's 314 boundary reads, 239 were already the stellarator's
own (same tracked work), leaving a real device-specific debt of 75, of which 58 were
actual missing producers (an 11-slot work list), 4 belonged to a shared subsystem's
pedestal arm, 12 were things PROCESS itself computes nowhere on a tokamak (permanent
boundary, not debt), and 1 was an already-intentionally-empty slot.

The one structural finding worth keeping now that the device is built: **the cyclic
structure survived the device swap.** The tokamak and stellarator graphs carry the same
six-node `physics` fusion/density SCC (only the profile-arm occupant differs), and their
remaining cycles are both mostly two-node `^problem` self-loops on driven nodes — a
second device changed *which* nodes exist, not the shape of the coupling. Consistent with
`uncut_graph.md`'s later, more complete measurement of the same question.
