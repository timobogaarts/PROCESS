# `st_fwbs` S1 + S5 — audit record

Companion to `stellarator_E_fwbs_synthesis.md`, which found six real sub-computations
inside `Stellarator.st_fwbs` (S1-S6). This record covers the two the synthesis already
classified as **portable now, no audit blocker, just execution** — no new judgment made
here, only the porting itself. See the synthesis record for the other four (S2/S3/S4
blocked or in progress elsewhere this wave; S6 out of scope).

## source

- **S1** `fw_blanket_shield_geometry_setup` —
  `process/models/stellarator/stellarator.py:515-605`, inside `st_fwbs`.
- **S5** `cryostat_and_vv_geometry` — `process/models/stellarator/stellarator.py:1282-1330`,
  inside `st_fwbs`.

Neither calls into another model (`self.hcpb`/`self.physics`/etc.) — confirmed by
grepping both line ranges for any `self.<submodel>.` reference: none. Both are pure
`self.data` arithmetic.

## data footprint

**S1** — `explicit-arg` reads: `.costs.abktflnc`, `.physics.pflux_fw_neutron_mw`,
`.costs.life_plant`, `.first_wall.a_fw_total`, `.physics.rminor`,
`.build.dr_fw_plasma_gap_inboard`/`dr_fw_plasma_gap_outboard`/`dr_fw_inboard`/
`dr_fw_outboard`, `.heat_transport.ipowerflow` (switch, kept as a plain traced argument —
both branches share the same output shape, only the subtracted loss terms differ),
`.physics.a_plasma_surface`, `.fwbs.fhole`/`f_ster_div_single`/`f_a_fw_outboard_hcd`,
`.build.dr_blkt_inboard`/`dr_blkt_outboard`, `.fwbs.fvolsi`/`fvolso`,
`.build.dr_shld_inboard`/`dr_shld_outboard`, `.physics.p_neutron_total_mw`,
`.stellarator_config.stella_config_neutron_peakfactor`. `explicit-arg` writes:
`.fwbs.life_fw_fpy`, `.first_wall.a_fw_inboard`/`a_fw_outboard`,
`.build.a_blkt_total_surface`/`a_blkt_inboard_surface`/`a_blkt_outboard_surface`,
`.fwbs.vol_blkt_inboard`/`vol_blkt_outboard`/`vol_blkt_total`,
`.build.a_shld_total_surface`/`a_shld_inboard_surface`/`a_shld_outboard_surface`,
`.fwbs.vol_shld_total`, `.fwbs.pnucloss`, `.fwbs.wallpf`. `local-intermediate`:
`vol_shld_inboard`/`vol_shld_outboard` — computed, summed into `vol_shld_total`, never
themselves written to `data` — not part of the port's return, same convention
`stellarator_D_structure.py` uses for its own dropped locals.

**S5** — `explicit-arg` reads: `.build.r_tf_outboard_mid`/`dr_tf_outboard`,
`.fwbs.dr_pf_cryostat`, `.physics.rmajor`, `.build.dr_cryostat`,
`.build.dr_fw_plasma_gap_inboard`/`dr_fw_inboard`/`dr_blkt_inboard`/`dr_shld_inboard`/
`dr_fw_plasma_gap_outboard`/`dr_fw_outboard`/`dr_blkt_outboard`/`dr_shld_outboard`,
`.physics.rminor`, `.build.dr_vv_inboard`/`dr_vv_outboard`, `.physics.a_plasma_surface`,
`.fwbs.fvoldw`/`den_steel`. `explicit-arg` writes: `.fwbs.r_cryostat_inboard`,
`.fwbs.vol_cryostat`, `.fwbs.vol_vv`, `.fwbs.m_vv`, `.fwbs.dewmkg`.

**Real downstream dependency**: `.fwbs.dewmkg` (S5's output) is already read by chunk
1D's registered `StructureMasses` node (`stellarator_D_structure.py`,
`dewmkg=Input(lambda s: s.fwbs.dewmkg)`) — a genuine graph edge, not just a coincidence
of naming.

## proposed signature(s)

See `stellarator_fwbs_s1_s5.py`'s `calculate_fw_blanket_shield_geometry` (S1) and
`calculate_cryostat_and_vv_geometry` (S5) docstrings for the full parameter list.

## cottax node

`FwBlanketShieldGeometry` (S1) and `CryostatAndVvGeometry` (S5), both `ExplicitFunction`,
in `stellarator_fwbs_s1_s5.py`. Not registered in `total_process.py` — registration is
reserved for the consolidation pass, same discipline as every other unit this wave.

## tier signal

Both tier-1: explicit, no internal iteration, no `self.data` access inside the ported
functions (all reads/writes happen at the node boundary).

## switches touched

`.heat_transport.ipowerflow` (S1 only) — kept as a plain traced argument (`jnp.where`),
not a topology switch: both branches mint the exact same output set, only the formula
for `a_blkt_total_surface` differs by which loss fractions are subtracted.

## calls into other models

None.

## JAX-difficulty flags

None. Both are pure closed-form arithmetic (S5 uses `jnp.pi`, otherwise ordinary
multiplication/division/`jnp.where`).

## open questions

None — this record exists to close out the two "portable now" pieces the synthesis
already fully specified, not to raise new ones.

## verification note

Both harness reference adapters call PROCESS's real `Stellarator.st_fwbs(output=False)`
end to end (not a hand-reimplemented formula) — `st_fwbs` has no standalone callable for
just S1 or S5, so the adapter constructs a `Stellarator` instance without going through
`__init__` (which would need all eleven injected sub-models built just to reach two
arithmetic blocks that touch none of them) and sets `.data`/`first_call_stfwbs` directly.
`blktmodel = 0` throughout so S2 (`blanket_neutronics`, gated on `blktmodel == 1`, and
the unit with the documented zero-argument bug — see `hcpb.md`) never executes. At
`ipowerflow == 1`, a later, unrelated block of `st_fwbs` (coolant pumping power, owned by
neither S1 nor S5) does execute and needs several more fields populated purely so the
call completes without raising — those values are physically plausible defaults, not
independently verified, since nothing S1/S5 actually own depends on them.
