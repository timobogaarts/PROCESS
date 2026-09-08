"""`cottax/wraps.py`: the alternative declaration form, against the one it parallels.

Every assertion here is a comparison with a live `ExplicitFunction` node rather than a
claim about the new form on its own. That is the point: the form is only worth having if
a node written in it is indistinguishable from the same node written the other way, and
"indistinguishable" is `inputs`, `outputs` and the value.
"""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from cottax.interfaces.pytree_namespace_module import (  # noqa: E402
    ExplicitFunction,
    From,
    FromExactly,
    ModelNamespace,
    Output,
    OutputInto,
    to_graph,
)

from functional_process.cottax.paths import physics  # noqa: E402
from functional_process.cottax.physics.density_limit import (  # noqa: E402
    EnforcedDensityLimitGreenwald,
    GreenwaldDensityLimit,
)
from functional_process.cottax.wraps import WrapsFunction  # noqa: E402
from functional_process.models.physics.density_limit import (  # noqa: E402
    calculate_greenwald_density_limit,
    select_enforced_density_limit_greenwald,
)


class WrapsGreenwald(WrapsFunction):
    """`GreenwaldDensityLimit`, with the reads declared rather than forwarded.

    Carries the rename case: the pure function's parameter is `c_plasma` and the field is
    `.physics.plasma_current`, which the other form spells by renaming in the forwarding
    call and this one spells with `FromExactly`.
    """

    fn = calculate_greenwald_density_limit

    c_plasma = FromExactly(physics.plasma_current)
    rminor = From(physics)

    nd_plasma_electron_max_array_7 = Output(physics.nd_plasma_electron_max_array[6])


class WrapsEnforced(WrapsFunction):
    """`EnforcedDensityLimitGreenwald` -- a read of one array element."""

    fn = select_enforced_density_limit_greenwald

    nd_plasma_electron_max_array_7 = FromExactly(
        physics.nd_plasma_electron_max_array[6]
    )

    nd_plasma_electrons_max = OutputInto(physics)


PAIRS = [
    (GreenwaldDensityLimit, WrapsGreenwald),
    (EnforcedDensityLimitGreenwald, WrapsEnforced),
]
IDS = [written.__name__ for written, _ in PAIRS]


@pytest.mark.parametrize(("written", "declared"), PAIRS, ids=IDS)
def test_the_two_forms_declare_the_same_ports(written, declared):
    """Same reads, same writes, in the same order."""
    assert [str(i.var) for i in declared().inputs] == [
        str(i.var) for i in written().inputs
    ]
    assert [str(o.var) for o in declared().outputs] == [
        str(o.var) for o in written().outputs
    ]


@pytest.mark.parametrize(("written", "declared"), PAIRS, ids=IDS)
def test_the_declared_form_assembles(written, declared):
    """`to_graph` cannot tell them apart: a synthesised `__call__` is a `__call__`."""
    assert to_graph(declared()).definitions


def test_the_body_is_the_wrapped_function():
    """Calling the node calls `fn`, with the ports bound by name."""
    assert WrapsGreenwald()(c_plasma=1.2e7, rminor=2.0) == (
        calculate_greenwald_density_limit(c_plasma=1.2e7, rminor=2.0)
    )


def test_a_read_that_does_not_match_the_function_is_refused():
    """The check the two-place form cannot make.

    Today a parameter added to a `models/` function and forgotten in its node is silent
    until something reads the port that was never declared. Here it is a `TypeError` at
    class-creation time, naming the parameter.
    """
    with pytest.raises(TypeError, match=r"missing \['rminor'\]"):

        class Missing(WrapsFunction):
            fn = calculate_greenwald_density_limit

            c_plasma = FromExactly(physics.plasma_current)

            nd_plasma_electron_max_array_7 = Output(
                physics.nd_plasma_electron_max_array[6]
            )


def test_a_read_that_is_not_a_parameter_is_refused():
    """The other direction: a declared read the function has no parameter for."""
    with pytest.raises(TypeError, match=r"declared but not a parameter: \['kappa'\]"):

        class Extra(WrapsFunction):
            fn = calculate_greenwald_density_limit

            c_plasma = FromExactly(physics.plasma_current)
            rminor = From(physics)
            kappa = From(physics)

            nd_plasma_electron_max_array_7 = Output(
                physics.nd_plasma_electron_max_array[6]
            )


def test_an_arm_overrides_only_the_formula():
    """The switch-arm case, and the reason `fn` is an attribute and not a class keyword.

    An arm that shares its family's reads and writes and differs only in which formula
    answers them says exactly that: one line. A keyword in the class header would have to
    be repeated by every arm, and a call form could not subclass at all.
    """

    class Family(WrapsFunction):
        """A family head."""

        fn = calculate_greenwald_density_limit

        c_plasma = FromExactly(physics.plasma_current)
        rminor = From(physics)

        nd_plasma_electrons_max = OutputInto(physics)

    class Arm(Family):
        """An arm: same ports, its own formula."""

        fn = calculate_greenwald_density_limit

    assert [str(i.var) for i in Arm().inputs] == [str(i.var) for i in Family().inputs]
    assert [str(o.var) for o in Arm().outputs] == [str(o.var) for o in Family().outputs]
    assert Arm()(c_plasma=1.2e7, rminor=2.0) == calculate_greenwald_density_limit(
        c_plasma=1.2e7, rminor=2.0
    )


def test_a_declared_node_takes_its_nested_name_from_its_slot():
    """`.physics.greenwald_density_limit`, exactly as the hand-written form does.

    The node's own class name is only the fallback for a declaration nothing placed
    (`_class_name`); a node in a `ModelNamespace` slot is named by the slot, and nesting
    the namespaces nests the name. This is the property a call form
    (`X = wrap(...)`) would have had to supply by hand, and it is why the form stayed a
    class.
    """

    class Written(ExplicitFunction):
        """The same node, hand-written."""

        def __call__(
            self,
            plasma_current=From(physics),  # noqa: B008 -- this IS the declaration
            rminor=From(physics),  # noqa: B008
        ):
            return calculate_greenwald_density_limit(
                c_plasma=plasma_current, rminor=rminor
            )

        nd7 = Output(physics.nd_plasma_electron_max_array[6])

    class Declared(WrapsFunction):
        """The same node, declared."""

        fn = calculate_greenwald_density_limit

        c_plasma = FromExactly(physics.plasma_current)
        rminor = From(physics)

        nd7 = Output(physics.nd_plasma_electron_max_array[6])

    def machine(node_class):
        class Physics(ModelNamespace):
            """The subsystem."""

            greenwald_density_limit: node_class = node_class()

        class Machine(ModelNamespace):
            """The device."""

            physics: Physics = Physics()

        return Machine()

    for node_class in (Written, Declared):
        graph = to_graph(machine(node_class))
        assert [n.path_str() for n in graph.nodes] == [
            ".physics.greenwald_density_limit"
        ], node_class.__name__
