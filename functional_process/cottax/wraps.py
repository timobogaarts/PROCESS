"""An alternative declaration form: reads as class attributes, body as a bare function.

**Not a replacement for `ExplicitFunction`.** That form declares its reads as `__call__`
parameter defaults and writes the body itself, which is right wherever the body is
anything but a forwarding call. This one is for the case that dominates this port: a
node whose whole body is `return f(a, b, c)`, where the parameter list is written
twice -- once as the signature that declares the ports, once as the call forwarding them.

The asymmetry that causes it is in cottax and is worth naming, because it is the whole
argument for this module. `NodalDeclaration.outputs` reads **class attributes**
(`_declared_outputs_on_cls`); `NodalDeclaration.inputs` reads a **method signature**
(`_params`). Outputs are therefore declared once and never duplicated; reads are declared
in one place and spent in another. `WrapsFunction` makes reads symmetric with writes:

    class Volume(WrapsFunction, fn=calculate_plasma_volume):
        rminor = From(physics)
        rmajor = From(physics)
        kappa  = From(physics)

        vol_plasma = OutputInto(physics)

Every port and every write is on its own line, greppable and visible, and each name is
written once. `FromExactly` stays the escape hatch it already is, and here it also
covers the case where the function's parameter is spelled unlike the field --
`c_plasma = FromExactly(physics.plasma_current)` -- which `ExplicitFunction` handles by
renaming in the forwarding call. Measured here: 35 of 3,450 ports, 1.0 %.

**It needs no change to cottax.** The class-level reads are used to synthesise a
`__call__` whose `__signature__` is exactly what the hand-written one would have been, so
`_params`, `_read_of`, `inputs` and `node_definition` are the ones that already exist and
nothing downstream can tell the difference. That is deliberate while this is a prototype:
if the form proves out, `inputs` growing a `_declared_reads_on_cls` branch upstream is
the same idea with one less layer, and this module becomes a compatibility shim
rather than a mechanism.
"""

import inspect

from cottax.interfaces.pytree_namespace_module import ExplicitFunction, FromExactly


def _declared_reads_on_cls(cls: type) -> dict[str, FromExactly]:
    """Declared reads, base-first and in declaration order, keyed by attribute name.

    A deliberate transcription of cottax's `_declared_outputs_on_cls`, including the
    subclass-overrides-a-base rule: a family arm that reads a different place for the
    same parameter says so by redeclaring the attribute.
    """
    seen: dict[str, FromExactly] = {}
    for klass in reversed(cls.__mro__):
        seen.update(
            (attr, value)
            for attr, value in vars(klass).items()
            if isinstance(value, FromExactly)
        )
    return seen


class WrapsFunction(ExplicitFunction):
    """`ExplicitFunction` whose reads are class attributes and whose body is `fn`."""

    def __init_subclass__(cls, fn=None, **kwargs):
        """Synthesise `__call__` from the declared reads, in `fn`'s parameter order.

        Raises
        ------
        TypeError
            If the declared reads and `fn`'s parameters are not the same set. This is
            the check the two-place form cannot make: today a parameter added to a
            `models/` function and forgotten in its node is silent until something reads
            the missing port.
        """
        super().__init_subclass__(**kwargs)
        if fn is None:
            return                      # an intermediate base; its subclasses name `fn`

        reads = _declared_reads_on_cls(cls)
        expected = [
            p.name
            for p in inspect.signature(fn).parameters.values()
            if p.kind
            in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        ]
        if set(reads) != set(expected):
            missing = sorted(set(expected) - set(reads))
            extra = sorted(set(reads) - set(expected))
            raise TypeError(
                f"{cls.__name__} declares reads that do not match "
                f"{fn.__name__}{inspect.signature(fn)}: "
                + (f"missing {missing}; " if missing else "")
                + (f"declared but not a parameter: {extra}" if extra else "")
            )

        order = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)] + [
            inspect.Parameter(
                name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=reads[name]
            )
            for name in expected
        ]

        def call(self, *args, **kwargs):
            bound = inspect.signature(type(self).__call__).bind(self, *args, **kwargs)
            bound.apply_defaults()
            return fn(**{k: v for k, v in bound.arguments.items() if k != "self"})

        call.__signature__ = inspect.Signature(order)
        call.__name__ = "__call__"
        call.__doc__ = f"`{fn.__module__}.{fn.__name__}`, ports declared above."
        cls.__call__ = call
        cls.fn = staticmethod(fn)
