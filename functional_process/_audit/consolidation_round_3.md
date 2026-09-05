# Consolidation round 3 — the 2026-08-27 wave day's bookkeeping and the ST endgame

**Done.** End-of-session handoff brief for the 2026-08-27 wave day: registry/docs debt
from that day's landings (registered tokamak units, the SAND max-iteration cap fix, the
driver benchmark, `low_aspect_ratio_DEMO`, the ST frontier chain — pulse-ramp, ECRH-13,
TART divertor/TF shape/masses, `i_tf_sc_mat`'s 18-leaf family, double-null, D-shaped
fw/blkt/vv, hcpb centrepost), two pending agent merges, and a queued "ST closing wave"
enumerating the last unported switch values blocking `spherical_tokamak_eval` and
`st_regression` (`i_plasma_current == 9` / FIESTA, `i_diamagnetic_current == 2`,
`i_pfirsch_schluter_current == 1`, `pf_coil_system_arm == -3`, one `i_tf_sc_mat = 9`
site). All of it landed — the registry/docs debt is closed per `next_steps.md` §16, and
the ST closing wave's outcome (the FIESTA arm landed, handled by a domain fix rather than
a tolerance) is in `next_steps_archive.md` §18. The one operational lesson from this
round still worth carrying forward: agent worktrees are cut at the session-start commit,
not current HEAD, so every brief must start with a base check and `git merge main`, and
cross-branch semantic conflicts (one wave's landing invalidating another's assumption)
need the meta-tests re-run after every merge, not just a clean `git merge`.
