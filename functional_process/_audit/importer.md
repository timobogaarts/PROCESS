# `functional_process/importer.py` — the legacy `IN.DAT` reader

**Status:** built and green, 2026-08-31. A skeleton on purpose: values, presence, the
`raw` namespace and the problem statement. No sentinel resolution, no `init.py` /
`st_init` / `initialise_imprad` derivation, no validation raise, no node. Those are the
next layer and each already has its classification in `init_audit.md`.

`next_steps.md` §24 is the spec; this file is what was built and what it measured.

## What exists

| file | what |
|---|---|
| `functional_process/importer.py` | `read_indat(path) -> Imported`; `ArrayInput`, `Assignment`, `Problem` |
| `functional_process/vocabulary/input_variables.py` | **865** vendored `name -> InputDecl(module, type, array)` rows |
| `tests/functional_process/test_importer.py` | 56 tests: table equality, the seven-file oracle, the grammar |

`Imported` carries `values: {(area, field): scalar | str | ArrayInput}`, `present:
frozenset[str]`, `assignments`, `problem`, `unknown`, `errors`; and the methods
`named(name)`, `get(area, field)`, `scalars()`, `raw_values()`, `present_paths`.

**PROCESS-free at runtime, and checked rather than asserted** (§23): the test runs a
subprocess with `process` blocked at `sys.meta_path` and reads `st_regression.IN.DAT`
in it. The whole module imports in 60 ms — it pulls in neither `numpy` nor `jax`.

## The vendored table, and the count correction

§23.2's rule: vendor for runtime, assert equality in tests. `TestVendoredTable` asserts
the name set and every row (`module`, `type`, `array`) against
`process.core.input.INPUT_VARIABLES`, plus two facts the design leans on — that exactly
`ixc`/`icc` set no field, and that **no** row sets `target_name`.

**[measured] PROCESS declares 865 input variables, not the 873 §22.4 quotes.** 863 name a
`DataStructure` area; `ixc` and `icc` name none. 726 float, 136 int, 3 str; 31 arrays. No
row uses `target_name`, no row uses `additional_validation`, no `module` is dotted, and
the only two `additional_actions` are `ixc`/`icc`. **`range` and `choices` are
deliberately not vendored** — they are input *validation*, which is the next layer, and a
vendored field with no consumer is drift surface with no test pressure behind it. The
generator recipe is in the vendored module's docstring; it was introspected, never typed.

## The oracle: pre-`init_process`, and why nothing else would do

`init_audit.md`'s method, one stage earlier: `process.core.init.parse_input_file` is
wrapped inside a **real `SingleRun`**, the wrapper `deepcopy`s the `DataStructure` and
keeps the parser's own return dict, then raises to abort the run before
`set_active_constraints`. Everything after that point is the derivation layer this module
does not implement, and running it would put derived values into the oracle: `init.py`
**destroys three genuine inputs** under the double-null branch and `st_init` overwrites
**nine** values a stellarator `IN.DAT` set explicitly. A post-init comparison cannot tell
a parse bug from a derivation.

The parser's return dict is the complete statement of which names PROCESS read, so name
coverage is an equality, not a subset.

### [measured] Per file, all seven (`IFE` out of scope everywhere)

| configuration | names | places | arrays (elements) | `ixc` | `icc` | `i_figure_merit` | missed | disagreeing |
|---|---|---|---|---|---|---|---|---|
| `stellarator_helias` | 122 | 120 | 3 (29) | 8 | 14 | 6 | 0 | 0 |
| `helias_5b` | 104 | 102 | 3 (19) | 3 | 5 | 7 | 0 | 0 |
| `large_tokamak_nof` | 139 | 137 | 8 (60) | 20 | 26 | 1 | 0 | 0 |
| `large_tokamak_eval` | 141 | 139 | 7 (49) | 2 | 25 | *unset* | 0 | 0 |
| `low_aspect_ratio_DEMO` | 183 | 181 | 8 (57) | 19 | 25 | -14 | 0 | 0 |
| `spherical_tokamak_eval` | 156 | 154 | 7 (48) | 3 | 18 | *unset* | 0 | 0 |
| `st_regression` | 159 | 157 | 7 (48) | 14 | 18 | -5 | 0 | 0 |

`names - places = 2` on every file, and it is the same two every time: `ixc` and `icc`
address no field. `i_figure_merit` is genuinely absent on both `_eval` files — they are
evaluation runs, and the importer reports `None` rather than PROCESS's post-init default,
because resolving an unset value is the next layer's job.

**Zero disagreements, zero unknown names, zero unparsed lines, on all seven.** That is
the expected result and not a strong claim: the line grammar, the `d`->`e` exponent fix,
the comma-list zero-fill and the 1-based index were transcribed from
`parse_input_file` rather than reinvented, precisely so that they could not drift. What
the run *does* establish is that transcription is complete — no name, no array element
and no cast is handled differently.

### [measured] The reverse direction, and the four other initialisation sources

`test_no_parsed_field_is_missed` diffs the pre-init `DataStructure` against a bare
`DataStructure()` and demands every differing field be one the importer has an entry for.
Thirteen fields are named as exempt, and each is another initialisation source running
**before** the parse — this is the reverse-direction confirmation of `init_audit.md` §5's
count of five:

- `globals.fileprefix`, `globals.output_prefix` — `SingleRun.set_filenames`.
- `numerics.lablxc`, `numerics.boundl`, `numerics.boundu` — `initialise_iteration_variables`,
  `init_process`'s first line. A file may then overwrite a bound, and all seven do.
- eight `impurity_radiation.*` arrays — `initialise_imprad` (`main.py:430`), the fifth
  source. It reads impurity data files off disk, and it runs *before* `init_process`, not
  inside it.

A trap worth recording: **15 `DataStructure` fields default to `NaN`**, so a naive
field-level diff reports them as differing from themselves. The comparison uses
`equal_nan=True`.

## What this makes fixable (not fixed here)

`indat.py:4420-4428` infers presence for `i_f_dr_tf_plasma_case` and
`tfc_sidewall_is_fraction` by scanning `switches_from_indat` for names that **are not
declared PROCESS inputs**, so the scan can only ever return `0` while `init.py` takes the
`True` arm on 4 of 7 configurations; that is the cause of `st_regression`'s missing
`.tfcoil.dx_tf_side_case_min` producer (`init_audit.md` §3). `Imported.named()` is the
question that code needs to ask and cannot ask today — presence is a property of the
text, and §24.2 item 1 is why no node can recover it. **`indat.py` is owned by another
agent; nothing there was touched.**

`test_presence_is_recorded_for_a_default_valued_name` pins the shape:
`dx_tf_side_case_min = 0.0` is indistinguishable from the dataclass default *by value*,
and only `named()` separates them.

## The `raw` namespace

`raw_values()` returns the same values keyed `.raw.<area>.<field>`. A namespacing, not new
data — it exists so §24.2 item 2's eight sentinel resolutions can be nodes with a read and
a **distinct** write instead of self-loops, and so the two sentinels that look like
answers (`eyoung_ins` at `1e8`, `eyoung_cond_axial` at `6.6e8`) cannot be mistaken for a
user's number the way a flat defaults table mistakes them.

## Deliberate gaps, named

1. **Arrays are sparse.** `ArrayInput` holds `{index: value}` plus a `zero_filled` flag,
   because the importer holds no defaults and no shapes. The flag is not cosmetic:
   `a = 1,2,3` makes PROCESS zero the whole array first, `a(2) = 1.0` does not.
   `dense(length)` is offered for a caller that knows a length, and is only lossless for
   the zero-filled form. **[measured]** 2-4 comma-list arrays per tokamak, none on either
   stellarator.
2. **No resolution.** An unset `i_figure_merit` or `n_equality_constraints` is `None`, not
   `-1` and not a resolved count.
3. **No raises.** `parse_input_file` raises on an unrecognised name; the importer collects
   it in `unknown`. An unparsed or uncastable line lands in `errors`. Refusing to read a
   file is a validation decision, and validation is the next layer.
4. **Multi-dimensional arrays are stored by their Fortran flat index.** PROCESS's
   `set_array_variable` ravels column-major before indexing; the importer records the
   index and the *test* does the `.T.ravel()`, because the importer has no shapes. No
   in-scope file exercises this.
5. **Nothing consumes it yet.** `provider.py` still resolves scalars from the file text
   with its own scanner and still answers arrays "named, value not resolvable" (§22.6).
   Pointing it at `read_indat` is the next step and would close that gap by construction.
