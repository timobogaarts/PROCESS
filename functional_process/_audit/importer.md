# `functional_process/importer.py` — the legacy `IN.DAT` reader

**Status: built and green.** A deliberate skeleton: `read_indat(path) -> Imported` parses
values, presence and the raw problem statement, PROCESS-free at runtime (checked with
`process` blocked at `sys.meta_path`, imports in 60 ms, pulls in neither `numpy` nor
`jax`). No sentinel resolution, no `init.py`/`st_init`/`initialise_imprad` derivation, no
validation raise — those are `init_audit.md`'s next layer, not this one.
`vocabulary/input_variables.py` vendors PROCESS's 865-row input table (asserted equal to
`process.core.input.INPUT_VARIABLES` in tests, not retyped by hand); zero disagreements,
zero unknown names, zero unparsed lines on all seven tracked configurations.

Two things worth not rediscovering:

- **15 `DataStructure` fields default to `NaN`**, so a naive pre/post-init field diff
  reports them as differing from themselves; comparisons need `equal_nan=True`.
- **Array presence is not the same as array value.** `a = 1,2,3` makes PROCESS
  zero-fill the whole array first; `a(2) = 1.0` does not — `ArrayInput` therefore stores
  a sparse `{index: value}` plus a `zero_filled` flag rather than a dense array, since the
  importer holds no shapes to make a dense form lossless on its own.

Five other initialisation sources run before or instead of this parser and are not
duplicated by it (see `init_audit.md`): `SingleRun.set_filenames`,
`initialise_iteration_variables`, and `initialise_imprad` (reads impurity data files off
disk, `main.py:430`, before `init_process` runs at all).
