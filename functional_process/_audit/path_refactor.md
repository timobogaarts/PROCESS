# The two path refactors — variable ports, then node names

**Status: both parts done.** Part A converted the declaration surface from
`FromExactly(lambda s: s.area.field)` / `Output(lambda s: s.area.field)` to the sugared
`From(area)` / `OutputInto(area)` wherever the parameter/attribute name equals the field
name (2078 of 2157 sites, mechanically, by an AST codemod that has since been deleted per
its own docstring); the escape hatch (43 sites, all array-element addressing — 42 elements
of `impurity_radiation.f_nd_impurity_electron_array`, one of `tfcoil.dcond[...]`) was
renamed `Input`→`FromExactly` rather than kept as an alias, and cottax's `Area` was made
subscriptable so those sites need no lambda either — postcondition
`grep -rn "lambda s:" functional_process --include='*.py'` → **0**, still true. 36 sites
needed an actual rename because the local parameter name differed from the field name; two
of those were substantive rather than cosmetic (`pure_formulas.py`'s `ElectronThermalEnergy`
vs. `IonThermalEnergy` both bound electrons-or-ions to the same local name
`nd_plasma_vol_avg`; `confinement_time.py` conflated `temp_plasma_electron_vol_avg_kev`
with the plural field). Verified inert throughout: `pytest functional_process` stayed at
3704 passed and the MDA harness stayed byte-identical (499/34/3/0 · 557/0 · 61/0/3/0).

Part B (hierarchical `NodePath` node names, `physics.profiles.DensityProfile` instead of a
flat class name) is **superseded in design** by `model_tree_design.md`, which is the
current reference — node identity there is a `NodePath`, i.e. the snake_case slot path
through an `eqx.Module` tree (`.physics.profiles.density_profile`), not the class-name
scheme this file originally designed. One result from here is still load-bearing and not
restated elsewhere: **a switch arm cannot be placed as one `place` per `Switch`** — tried
and wrong, because a single arm can populate two unrelated places at once (`.stellarator.
istell`'s `value=6` arm declares both a `stellarator.*` node and a
`physics.confinement_time.*` node). The fix that does work is to make an arm a whole
subtree, merged the same way `COMMON` merges into the root graph — `model_tree_design.md`
carries this forward as settled design, not as an open question.

Two withdrawn/incorrect designs worth not re-trying: a bare `Area` as a parameter default
with no `From`/`Input` wrapper (cottax refuses this, `TypeError` at declaration); and
`Input` as a working alias for `FromExactly` (deliberately not kept — it would silently
restore the two-spellings-for-one-thing ambiguity the whole conversion exists to remove,
and a codemod bug emitting `Input` instead of `From` would have passed every test in the
repo while doing so).
