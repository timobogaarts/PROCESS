---
kind: model-unit
status: reviewed
confidence: high
---

**Ported.** `structure.py` / `test_structure.py`, both
tier-1 contracts passing (legacy + fuzz). Open question 1 resolved for this unit only
(not as a general policy): `fncmass`/`gsmass` are not ported as functions — they are
unconditional literals with no inputs, so per `naming_convention.md`'s treatment of
switch-selected topology, they are the absence of a node, not a degenerate one. Whoever
wires the graph should read `.structure.fncmass`/`.structure.gsmass` as inline `0.0`
literals wherever they're consumed downstream, with the source's own comments carried
forward as the reason (see the port's module docstring).

## source
`process/models/stellarator/stellarator.py`, lines 320-421: `st_strc()`. Chunk 1D of
unit #1 (see `../../_audit/unit_registry.md`). Full method, 102 lines, no split needed.

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.e_tf_magnetic_stored_total_gj` | read | explicit-arg | only feeds `m_struc`/`msupstr`, a reporting-only comparison value, see open questions |
| `.stellarator_config.stella_config_coilsurface` | read | explicit-arg | |
| `.stellarator.f_st_rmajor` | read | explicit-arg | |
| `.stellarator.r_coil_minor` | read | explicit-arg | |
| `.stellarator_config.stella_config_coil_rminor` | read | explicit-arg | |
| `.tfcoil.dx_tf_inboard_out_toroidal` | read | explicit-arg | |
| `.tfcoil.len_tf_coil` | read | explicit-arg | |
| `.tfcoil.n_tf_coils` | read | explicit-arg | |
| `.physics.b_plasma_toroidal_on_axis` | read | explicit-arg | |
| `.fwbs.den_steel` | read | explicit-arg | |
| `.tfcoil.m_tf_coils_total` | read | explicit-arg | |
| `.fwbs.dewmkg` | read | explicit-arg | |
| `.structure.fncmass` | write | explicit-arg | unconditional constant `0.0` — see open questions on constant-valued "producers" |
| `.structure.gsmass` | write | explicit-arg | unconditional constant `0.0`; source comment `# ? Not sure about this.` — author's own uncertainty marker, flagging for review, not resolving |
| `.structure.aintmass` | write, then read again later in the same call (for `clgsmass`, `coldmass`) | implicit-io (see note) | textbook case of the "trivial self-produced intermediate" pattern — see open questions, this isn't the risky kind of implicit-io |
| `.structure.clgsmass` | write | explicit-arg | derived from `aintmass` |
| `.structure.coldmass` | write | explicit-arg | derived from `aintmass` + two other reads |

No `implicit-io-via-callee` or `redundant-duplicate-write` instances in this chunk.

## proposed signature(s)

**As ported** (`e_tf_magnetic_stored_total_gj` dropped as anticipated, `fncmass`/
`gsmass` dropped per the resolution above):

```python
def calculate_structure_masses(
    stella_config_coilsurface: float,
    f_st_rmajor: float,
    r_coil_minor: float,
    stella_config_coil_rminor: float,
    dx_tf_inboard_out_toroidal: float,
    len_tf_coil: float,
    n_tf_coils: float,
    b_plasma_toroidal_on_axis: float,
    den_steel: float,
    m_tf_coils_total: float,
    dewmkg: float,
) -> tuple[float, float, float]:
    # returns (aintmass, clgsmass, coldmass)
    ...
```

The "previous scaling law, kept as reference, not fully trusted" comparison value
(`msupstr`, printed but never stored to `data` and never fed into `aintmass`/`clgsmass`/
`coldmass`) is its own tiny, separable, reporting-only function:

```python
def calculate_intercoil_mass_scaling_reference(
    e_tf_magnetic_stored_total_gj: float,
) -> float:  # msupstr, for output comparison only
    ...
```

## cottax node

**Actually written**, in `structure.py` (`StructureMasses`, an
`ExplicitFunction` over the pytree namespace — see `schema.md`'s "cottax node" section
for why), and registered in `functional_process/total_process.py`:

```python
class StructureMasses(ExplicitFunction):
    aintmass = OutputInto(structure)
    clgsmass = OutputInto(structure)
    coldmass = OutputInto(structure)

    def __call__(
        self, stella_config_coilsurface=From(stellarator_config), ..., dewmkg=From(fwbs)
    ):
        return calculate_structure_masses(stella_config_coilsurface, ..., dewmkg)
```
`calculate_intercoil_mass_scaling_reference` is not wrapped — it feeds no other node
(reporting-only, see above), so it has no place in the graph as declared here; it stays a
plain function, not an `ExplicitFunction`.

## tier signal

**Tier 1.** No internal solve, no calls into other models, no data-dependent Python
control flow on a traced quantity. The `output: bool` parameter only gates printing —
every write happens unconditionally before the `if output:` block, so the pure
computational core should drop `output` and the printing entirely (consistent with the
"reporting is not in scope" stance already established for `output()`-shaped code
elsewhere in this audit).

## switches touched

None. `output` is a plain Python bool controlling printing, not a `data.<area>.i_*`
switch — no entry needed in `switches.md`.

## calls into other models

None — self-contained arithmetic on already-available `data` fields.

## JAX-difficulty flags

None found. No external calls, no dynamic shapes, no control flow on a traced value. The
`if output:` branch is pure Python-level dead code for the pure core (see tier signal),
not a `needs-lax-cond-or-where` case — nothing about the *computation* branches on data,
only the *printing* does, and printing isn't ported.

## open questions

1. **Constant-valued "producers".** `.structure.fncmass` and `.structure.gsmass` are
   unconditionally set to `0.0` — not computed from anything. Should a node that always
   produces a literal even exist as a graph node/port, or should these be inlined as
   literals wherever `structure.fncmass`/`structure.gsmass` are read downstream, with a
   comment carrying forward the docstring's explanation ("many masses are simply set to
   zero to avoid double-counting of structural components that are specified differently
   for tokamaks") and `gsmass`'s specific author-uncertainty marker? No precedent yet in
   this audit for a genuinely constant field; flagging for a policy decision rather than
   guessing.
2. **`aintmass`'s implicit-io classification feels too strong for what it is.** The
   traceability-policy definition of `implicit-io` ("read mid-loop, depends on state
   written earlier in the same call") technically covers this, but the actual pattern —
   compute A, store it in `self.data` because that's the codebase's universal idiom for
   *any* value including pure locals, then read it back three lines later in the same
   straight-line function — is mechanically identical to an ordinary Python local
   variable and carries none of the risk the classification exists to flag (no branching,
   no cross-call aliasing, no possibility the value differs from what was just written).
   If this classification is applied literally across the rest of the audit, most
   Fortran-derived functions in this codebase will show several "implicit-io" entries
   that are actually trivial, diluting the signal for the genuinely risky cases (loop-
   dependent reads, cross-model aliasing via `implicit-io-via-callee`). Suggest a fifth
   classification — something like `local-intermediate` — for same-function,
   unconditional, un-branched produce-then-consume, so `implicit-io` stays reserved for
   cases that actually need a careful read. Not resolved here; used `implicit-io` with
   this note since it's the closest existing fit, per this audit's established practice
   of not inventing labels unilaterally.
3. Whether `m_struc`/`msupstr`'s "previous scaling law... which we do not really trust
   yet" comparison is still wanted in the ported output at all, or was a one-time
   validation check that's now stale — a question for you, not resolvable from the code
   alone.

## Derivative-safe power laws (`safe_pow` / `safe_sqrt`)

1 fractional power law in this file has been rewritten from `x ** p` / `jnp.sqrt(x)` to
`models/safe_math.py`'s `safe_pow(x, p)` / `safe_sqrt(x)`.

**Why.** For `0 < p < 1` the function is continuous at `x == 0` and its derivative is
not: `d/dx x**p = p * x**(p-1) -> +inf`. JAX's JVP then returns `inf` along the
direction that perturbs `x` and `nan` (`inf * 0`) along every other, so the *value* is
right everywhere and the *Jacobian row* is poisoned. That is the defect class
`_audit/next_steps.md` §9 records; the most recent instance produced 46 non-finite
Jacobian cells and stalled a cold optimiser start at zero SQP steps, reported by the
solver as "the problem seems to be non-convex".

**Value identity, checked not asserted.** `safe_pow`/`safe_sqrt` dispatch on `x == 0`
and evaluate the identical expression otherwise, so every `x != 0` result is bit-for-bit
what it was, and the `x == 0` result is `0.0 ** p` / `sqrt(0.0)` -- again exactly what
the bare expression returns. Verified two ways: a hex-exact diff of every Tier-1
contract's output over every declared sample plus eight fresh fuzz draws (3655 points,
zero differing bits), and `run_mda_harness.py` unchanged at 492 agreements / 34
disagreements. PROCESS itself does not raise at `x == 0` here -- it is plain Python
`float.__pow__` / `numpy.sqrt`, both of which return `0.0` -- and the reference was
re-evaluated at each boundary point to confirm it returns the port's number.

**What changed is only the derivative at exactly `x == 0`**, which becomes `0` instead
of `inf`/`nan` -- the same convention JAX already uses at `jnp.maximum`'s kink.

`Tier1Contract.test_gradient_finite_at_zero` (`--fp-gradients`) now checks the whole
class automatically: it zeroes each differentiable argument in turn and requires a
finite Jacobian wherever the value is finite.
