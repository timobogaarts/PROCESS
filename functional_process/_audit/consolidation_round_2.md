# Consolidation round 2 — waves 2/3 registration + fixes (2026-08-26/27)

**Done.** This was the dispatch brief for a single-agent registration pass: eleven
tokamak units (`plasma_current`, `bootstrap_current`, `l_h_transition`, `density_limit`,
`scrape_off_layer`, the five `pfcoil` units into `.tokamak.cs_coil`/`.tokamak.pf_coil`,
`plasma_inductance`, `pulse`'s `burn_time`, `shield`), plus three orchestrator-decided
fixes (the `PlasmaComposition` ignition split threaded from the factory instead of
hardcoded; a `boundary._main` error-message fix; `ComparisonReport.summary`'s truncated
error listing). The outcome — what actually landed, the two new raw cycles registration
created and how each was cut (volt-seconds/burn-time by a single sufficient cut on
`.times.t_plant_pulse_burn`; the 5-node PF coil ring by the pair PROCESS itself seeds on
`first_call`, `ind_pf_cs_plasma_mutual` + `n_pf_coil_turns`, driven Picard — RootFind on
the residual deliberately deferred), and the full harness numbers — is the historical
record and lives in `next_steps_archive.md` §15, not here.
