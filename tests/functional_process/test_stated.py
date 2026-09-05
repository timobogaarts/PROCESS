"""`models/stated` and `indat.STATED_VALUES`, on all seven configurations.

**The gate the family needs and `carried.py` never had.** A carried value was a field of
a declaration, so it could not go missing; a *stated* value is a boundary read, and a
read with no `STATED_VALUES` row does not fail -- `sand_harness.ground_truth` falls
through to `unminted`, which reads the place the statement is *for* off the seed state.
On a cold `DataStructure` that is `build_variables.py`'s `0.811 m` central solenoid for a
stellarator that has none, i.e. exactly the wrong number §28.7 measured, arriving in
silence. `test_every_stated_port_has_a_value` is the refusal.

The other half is that the value must not depend on *which* state supplied it. Six of the
fourteen declarations state a resolution rather than a literal, and `STATED_VALUES`
re-applies `indat.resolve_*` to the raw value and switch it reads back off the state --
so it is handed an already-resolved value on the PROCESS path and a raw one on the native
path. Every one of those resolutions is idempotent on its own answer, which is what makes
the two agree; `test_the_native_and_process_states_state_the_same_thing` checks it rather
than trusting it.

`test_initialisation.py` holds the oracle that these numbers are PROCESS's own; this file
holds the structure that they arrive at all.
"""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from functional_process import indat  # noqa: E402
from functional_process.boundary import STATED, boundary  # noqa: E402
from functional_process.cold_start import cold_state  # noqa: E402
from functional_process.indat import graph_for, machine_from_indat  # noqa: E402
from functional_process.cottax.stated import StatesValues, stated_port  # noqa: E402
from functional_process.native import native_state  # noqa: E402
from functional_process.provider import CONFIGURATIONS, stem  # noqa: E402


def _stated_reads(input_file):
    """Every `^stated.*` boundary read of one configuration's assembled graph."""
    graph = graph_for(machine_from_indat(input_file))
    return {var.path_str() for kind, var in boundary(graph) if kind == STATED}


@pytest.mark.parametrize(
    "input_file", CONFIGURATIONS, ids=[stem(c) for c in CONFIGURATIONS]
)
def test_every_stated_port_has_a_value(input_file):
    """No stated read falls through to `ground_truth`'s `unminted` fallback.

    The failure this refuses is silent by construction, which is why it is asked per
    configuration rather than once: a declaration that only the spherical tokamaks
    occupy would otherwise be checked by nothing.
    """
    missing = sorted(_stated_reads(input_file) - set(indat.STATED_VALUES))
    assert missing == [], f"{stem(input_file)}: no STATED_VALUES row for {missing}"


def test_no_stated_value_is_dead():
    """Every `STATED_VALUES` row is read by at least one configuration.

    The converse of the test above, and it is what keeps the table from accumulating
    rows for declarations that have been deleted -- a table nobody prunes is a table
    nobody trusts.
    """
    read = set()
    for input_file in CONFIGURATIONS:
        read |= _stated_reads(input_file)
    assert sorted(set(indat.STATED_VALUES) - read) == []


@pytest.mark.parametrize(
    "input_file", CONFIGURATIONS, ids=[stem(c) for c in CONFIGURATIONS]
)
def test_the_native_and_process_states_state_the_same_thing(input_file):
    """A stated value does not depend on which seed state answered for it.

    The six resolutions read a raw value and a switch back off the state, so PROCESS's
    post-`init_process` seed hands them their own answer and `native.NativeState` hands
    them the file's raw one. Idempotence is what makes those agree, and it is a property
    of `indat.resolve_*` rather than of this table -- so it is measured here, on the
    machines that exercise it, rather than argued in a docstring.
    """
    seed = cold_state(input_file).seed
    native = native_state(input_file)
    for path in sorted(_stated_reads(input_file)):
        value = indat.STATED_VALUES[path]
        # Exact, by intent: the claim is that the two are the *same* number
        # (`resolve_*` idempotent), not that they round alike.
        assert float(value(seed)) == float(value(native)), path  # noqa: RUF069


def test_a_stating_declaration_holds_no_data():
    """There is nowhere left to put a number.

    The structural claim the whole change rests on: a `StatesValues` has no field, so it
    cannot carry an array (which `cottax` refuses in a graph) and it cannot carry a
    Python scalar (which XLA folds and `eqx.filter_jit` keys on). Asked of every
    occupant of every configuration's machine, since a subclass added later is exactly
    what would reintroduce one.
    """
    seen = set()
    for input_file in CONFIGURATIONS:
        machine = machine_from_indat(input_file)
        for leaf in jax.tree_util.tree_leaves(
            machine, is_leaf=lambda x: isinstance(x, StatesValues)
        ):
            if isinstance(leaf, StatesValues):
                seen.add(type(leaf).__name__)
                # No pytree leaf, so no array can hide here; and the definition
                # hashes, which is `graph._check_bindings`' own question asked one
                # object earlier. `_signature_of`/`_read_of` are cottax `ClassVar`s
                # and are neither.
                assert jax.tree_util.tree_leaves(leaf) == []
                assert hash(leaf.node_definition) == hash(leaf.node_definition)
    assert len(seen) >= 8, sorted(seen)


def test_reads_are_derived_from_writes():
    """One read per output, at the output's own place -- and no other read.

    `inputs` is a function of `outputs`, so a stating node names no variable of its own
    and there is no second place a stray edge could enter.
    """
    for input_file in CONFIGURATIONS:
        graph = graph_for(machine_from_indat(input_file))
        for name, node in graph.definitions.items():
            reads = [
                v.path_str() for v in node.reads if v.path_str().startswith("^stated")
            ]
            if not reads:
                continue
            assert reads == [stated_port(out).var.path_str() for out in node.outputs], (
                name.path_str()
            )
