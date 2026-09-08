"""The per-tier contracts a ported unit subclasses."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from functional_process.cottax._harness.boundary import (
    DIVISION_BY_ZERO_AT_BOUNDARY,
    registered_reason,
)
from functional_process.cottax._harness import sample_store
from functional_process.cottax._harness.finite_difference import (
    PROCESS_EPSFCN,
    ZeroPerturbationError,
    fd_gradient_with_error,
)
from functional_process.cottax._harness.sampling import fuzz_samples
from functional_process.cottax._harness.tolerance import (
    MACHINE_PRECISION,
    DeclaredDeviation,
    Tolerance,
)


def _as_array(value):
    """Normalise a scalar, array or tuple return into a flat 1-D float array."""
    return np.concatenate([
        np.ravel(np.asarray(leaf, dtype=float)) for leaf in jax.tree.leaves(value)
    ])


def _as_traced_array(value):
    """`_as_array`'s traced twin: the same leaf order, in `jnp` so `jacfwd` sees
    through.
    """
    return jnp.concatenate([
        jnp.ravel(jnp.asarray(leaf, dtype=float)) for leaf in jax.tree.leaves(value)
    ])


def _component_label(name, shape, index):
    """`rho` for a scalar argument, `temperatures[2]` / `kt[1, 7]` for an array one."""
    if shape == ():
        return name
    return f"{name}[{', '.join(str(i) for i in np.unravel_index(index, shape))}]"


class PortContract:
    """Shared declaration surface for every tier."""

    audit_record = None
    reference = None
    ported = None
    samples = ()
    static_argnames = ()

    def __init_subclass__(cls, **kwargs):
        """Wrap bare functions assigned to `reference`/`ported` in `staticmethod`, and
        resolve `samples = FROM_FILE` against the module's sample store."""
        super().__init_subclass__(**kwargs)
        for attr in ("reference", "ported"):
            value = cls.__dict__.get(attr)
            if callable(value) and not isinstance(value, staticmethod):
                setattr(cls, attr, staticmethod(value))
        # Resolved here rather than lazily so that `SomeContract.samples` is an ordinary
        # list for every reader -- conftest's parametrisation, `signoff`, and the cases
        # that build one point out of another's.
        if cls.__dict__.get("samples") is sample_store.FROM_FILE:
            cls.samples = sample_store.load(cls.__module__, cls.__name__)

    @classmethod
    def diff_argnames(cls, sample):
        """Arguments to differentiate with respect to, for one sample."""
        return tuple(k for k in sample.kwargs if k not in cls.static_argnames)

    def test_unit_is_identified(self):
        """The case names the unit it is a case for."""
        assert self.audit_record is not None, (
            f"{type(self).__name__} must declare `audit_record`"
        )


class Tier1Contract(PortContract):
    """Explicit pure functions: no internal iteration, no `self.data` access."""

    pytestmark = pytest.mark.tier1

    value_tolerance = MACHINE_PRECISION
    declared_deviation: DeclaredDeviation | None = None
    """Set when the port **deliberately** does not compute PROCESS's expression."""

    epsfcn = PROCESS_EPSFCN
    gradient_safety = 25.0
    """Multiplier on the finite difference's own error bar."""

    gradient_floor = 0.0
    """Extra allowance, as a fraction of the largest derivative in the same column."""

    reference_domain_errors = ()
    """Exceptions the PROCESS reference raises to signal an out-of-domain input."""

    def _reference_or_domain_error(self, kwargs):
        """Evaluate the reference, distinguishing a domain error from a real failure."""
        try:
            return _as_array(self.reference(**kwargs)), None
        except self.reference_domain_errors as exc:
            return None, exc

    def test_value_agreement(self, sample):
        """Port and PROCESS agree to float64 round-off."""
        expected, domain_error = self._reference_or_domain_error(dict(sample.kwargs))
        actual = _as_array(self.ported(**sample.kwargs))

        if domain_error is not None:
            assert not np.all(np.isfinite(actual)), (
                f"PROCESS rejects this point ({type(domain_error).__name__}: "
                f"{domain_error}) but the port returned finite {actual}. A traced port "
                f"cannot raise, so it must return non-finite here instead"
            )
            return

        assert actual.shape == expected.shape, (
            f"output size mismatch: port produced {actual.size} values, PROCESS "
            f"{expected.size} (both counted flattened — see `_as_array`)"
        )
        against = (
            self.value_tolerance
            if self.declared_deviation is None
            else self.declared_deviation.bound
        )
        bad = against.mismatches(actual, expected)
        detail = [
            f"  output[{i}]: port={a!r} process={e!r} |diff|={err:g} allowed={allowed:g}"
            for i, a, e, err, allowed in bad
        ]
        header = (
            f"value mismatch at {against.describe()}:"
            if self.declared_deviation is None
            else f"value mismatch OUTSIDE the declared deviation -- "
            f"{self.declared_deviation.describe()}:"
        )
        assert not bad, "\n".join([header, *detail])

    def test_declared_deviation_is_real(self, sample):
        """A declared deviation must be **exercised**, or it is a loosened tolerance."""
        if self.declared_deviation is None:
            pytest.skip("no declared deviation")
        exercised = getattr(type(self), "_deviation_exercised", False)
        expected, domain_error = self._reference_or_domain_error(dict(sample.kwargs))
        if domain_error is None:
            actual = _as_array(self.ported(**sample.kwargs))
            if actual.shape == expected.shape and self.value_tolerance.mismatches(
                actual, expected
            ):
                type(self)._deviation_exercised = True
                exercised = True
        assert exercised or sample is not self.samples[-1], (
            f"{type(self).__name__} declares a deviation "
            f"({self.declared_deviation.reason}) but no sample disagrees with PROCESS "
            f"by more than {self.value_tolerance.describe()}. A declared deviation that "
            f"is never exercised is a loosened tolerance wearing a label -- delete it, "
            f"or add a sample that reaches the regime it exists for"
        )

    def test_declared_deviation_is_documented(self, audit_root):
        """A declared deviation names its reason and cites a record that exists."""
        if self.declared_deviation is None:
            pytest.skip("no declared deviation")
        deviation = self.declared_deviation
        assert deviation.reason.strip(), "a declared deviation must say why"
        cited = deviation.record.split("#")[0].strip()
        record = audit_root / cited
        assert record.is_file() or (audit_root.parent / cited).is_file(), (
            f"{type(self).__name__}'s declared deviation cites {deviation.record}, "
            f"which is not a file under {audit_root} or {audit_root.parent}"
        )

    def test_outputs_finite(self, sample):
        """The port's value is free of NaN/Inf on an in-domain point."""
        _, domain_error = self._reference_or_domain_error(dict(sample.kwargs))
        if domain_error is not None:
            pytest.skip(f"point is outside PROCESS's domain: {domain_error}")

        value = _as_array(self.ported(**sample.kwargs))
        assert np.all(np.isfinite(value)), f"non-finite output: {value}"

    @pytest.mark.gradient
    def test_gradient_finite(self, sample):
        """The port's gradient is free of NaN/Inf on an in-domain point."""
        _, domain_error = self._reference_or_domain_error(dict(sample.kwargs))
        if domain_error is not None:
            pytest.skip(f"point is outside PROCESS's domain: {domain_error}")

        jacobians = self._jacobians(sample)
        for name, jac in jacobians.items():
            assert np.all(np.isfinite(jac)), (
                f"non-finite d(output)/d({name}) = {jac} at a point where the value "
                f"itself is finite — the classic symptom of a jnp.where whose untaken "
                f"branch evaluates to NaN"
            )

    @pytest.mark.gradient
    def test_gradient_agreement(self, sample):
        """`jacfwd` of the port matches PROCESS's finite difference, within its error.
        """
        _, domain_error = self._reference_or_domain_error(dict(sample.kwargs))
        if domain_error is not None:
            pytest.skip(f"point is outside PROCESS's domain: {domain_error}")

        jacobians = self._jacobians(sample)
        failures = []
        for name in self.diff_argnames(sample):
            argument = np.asarray(sample.kwargs[name], dtype=float)
            jac = jacobians[name]

            for component, x in enumerate(argument.ravel()):
                try:
                    reference, error_bar = fd_gradient_with_error(
                        self._reference_along(sample, name, component),
                        x,
                        self.epsfcn,
                    )
                except ZeroPerturbationError:
                    continue

                label = _component_label(name, argument.shape, component)
                # The floor is relative to the largest derivative in *this column*, so
                # it says "round-off at the scale of what this input actually moves"
                # rather than fixing an absolute number that would mean different things
                # for a stress in Pa and a deflection in m. Zero by default; see
                # `gradient_floor`.
                floor = self.gradient_floor * float(np.max(np.abs(reference)))
                allowed = self.gradient_safety * error_bar + floor
                for i, (got, want, tol) in enumerate(
                    zip(jac[:, component], reference, allowed, strict=True)
                ):
                    if not abs(got - want) <= tol:
                        failures.append(
                            f"  d(output[{i}])/d({label}): jacfwd={got!r} "
                            f"process_fd={want!r} |diff|={abs(got - want):g} "
                            f"allowed={tol:g} (fd error bar {error_bar[i]:g} "
                            f"x safety {self.gradient_safety:g}, floor {floor:g})"
                        )

        header = (
            f"gradient mismatch vs PROCESS finite difference (epsfcn={self.epsfcn:g}):"
        )
        assert not failures, "\n".join([header, *failures])

    def _reference_along(self, sample, name, component):
        """The PROCESS reference as a function of one flat component of one argument."""
        shape = np.shape(sample.kwargs[name])
        held = np.asarray(sample.kwargs[name], dtype=float).ravel()

        def along(value):
            perturbed = held.copy()
            perturbed[component] = value
            kwargs = dict(sample.kwargs)
            kwargs[name] = perturbed.reshape(shape) if shape else float(perturbed[0])
            return _as_array(self.reference(**kwargs))

        return along

    def _jacobians(self, sample):
        """`jacfwd` of the port with respect to every differentiable argument at once.
        """
        names = self.diff_argnames(sample)
        if not names:
            return {}
        shapes = {name: np.shape(sample.kwargs[name]) for name in names}

        def f(*flats):
            kwargs = dict(sample.kwargs)
            for name, flat in zip(names, flats, strict=True):
                kwargs[name] = flat.reshape(shapes[name])
            return _as_traced_array(self.ported(**kwargs))

        flats = tuple(
            jnp.asarray(np.asarray(sample.kwargs[name], dtype=float).ravel())
            for name in names
        )
        jacobians = jax.jacfwd(f, argnums=tuple(range(len(names))))(*flats)
        return {
            name: np.asarray(jac, dtype=float)
            for name, jac in zip(names, jacobians, strict=True)
        }

    @pytest.mark.gradient
    def test_gradient_finite_at_zero(self):
        """No argument may be finite in value and non-finite in gradient at `x == 0`."""
        sample = self._boundary_sample()
        failures = []
        excused = set()

        for name in self.diff_argnames(sample):
            base = np.asarray(sample.kwargs[name], dtype=float)
            shape = base.shape
            reason = registered_reason(type(self).__name__, name)
            for component in range(base.size):
                if base.ravel()[component] == 0.0:
                    continue
                value, jacobian = self._value_and_jacobian_at_zero(
                    sample, name, component, shape
                )
                if not np.all(np.isfinite(value)):
                    continue
                if np.all(np.isfinite(jacobian)):
                    continue
                if reason is not None:
                    excused.add(name)
                    continue
                failures.append(
                    f"  d(output)/d({_component_label(name, shape, component)}) = "
                    f"{jacobian} at a point where the value {value} is finite"
                )

        assert not failures, "\n".join([
            "non-finite gradient at a zero-valued argument, where the value itself is "
            "finite -- the `x ** p` (0 < p < 1) / `jnp.sqrt` trap of "
            "`_audit/next_steps.md` §9. Fix it with `models/safe_math.py`'s `safe_pow` "
            "/ `safe_sqrt`, or register it in `_harness/boundary.py` with the reason:",
            *failures,
        ])

        stale = {
            argument
            for (contract, argument) in DIVISION_BY_ZERO_AT_BOUNDARY
            if contract == type(self).__name__
        } - excused
        assert not stale, (
            f"{type(self).__name__} registers {sorted(stale)} in "
            f"`_harness/boundary.py` as non-finite at the zero boundary, but they are "
            f"finite now. Delete the entries -- a register that outlives its defect is "
            f"an excuse, not a record"
        )

    def _boundary_sample(self):
        """The single deterministic point `test_gradient_finite_at_zero` probes from."""
        samples = list(getattr(self, "samples", ()))
        if samples:
            return samples[0]
        bounds = getattr(self, "fuzz_bounds", None)
        assert bounds, (
            f"{type(self).__name__} declares neither `samples` nor `fuzz_bounds`, so "
            f"there is no point to probe the zero boundary from"
        )
        return fuzz_samples(bounds, 1, 0, fixed=getattr(self, "fuzz_fixed", None))[0]

    def _value_and_jacobian_at_zero(self, sample, name, component, shape):
        """Value and `jacfwd` of the port with one flat component of `name` set to zero.
        """
        flat = np.asarray(sample.kwargs[name], dtype=float).ravel().copy()
        flat[component] = 0.0
        kwargs = dict(sample.kwargs)

        def f(x):
            kwargs[name] = x.reshape(shape) if shape else x.reshape(())
            return _as_traced_array(self.ported(**kwargs))

        at_zero = jnp.asarray(flat)
        return (
            np.asarray(f(at_zero), dtype=float),
            np.asarray(jax.jacfwd(f)(at_zero), dtype=float),
        )

    def _jacobian(self, sample, name):
        """`_jacobians(sample)[name]` — kept for call sites that want a single argument.
        """
        return self._jacobians(sample)[name]


class Tier2Contract(PortContract):
    """Units whose PROCESS implementation closes an internal loop."""

    pytestmark = pytest.mark.tier2

    def __init_subclass__(cls, **kwargs):
        """`eqx.filter_jit`-wrap `ported`, once, at class-definition time."""
        super().__init_subclass__(**kwargs)
        ported = cls.__dict__.get("ported")
        if ported is not None:
            fn = ported.__func__ if isinstance(ported, staticmethod) else ported
            cls.ported = staticmethod(eqx.filter_jit(fn))

    residual = None
    """`(solution, **kwargs) -> array` — the unit's defining equations."""

    residual_tolerance = Tolerance(
        rtol=0.0,
        atol=1e-8,
        reason="absolute, physical: a converged driver should zero its own residual",
    )

    def test_ported_residual_small(self, sample):
        """The port's answer actually solves the unit's defining equations."""
        assert self.residual is not None, (
            f"{type(self).__name__} must declare `residual` — a tier-2 unit has no "
            f"pass criterion without one"
        )
        solution = self.ported(**sample.kwargs)
        res = _as_array(self.residual(solution, **sample.kwargs))
        bad = self.residual_tolerance.mismatches(res, np.zeros_like(res))
        assert not bad, (
            f"port's residual is not small at "
            f"{self.residual_tolerance.describe()}: {res}"
        )

    def test_ported_residual_no_worse_than_process(self, sample):
        """The port is at least as converged as PROCESS is at its own stopping point."""
        ported_res = _as_array(
            self.residual(self.ported(**sample.kwargs), **sample.kwargs)
        )
        process_res = _as_array(
            self.residual(self.reference(**sample.kwargs), **sample.kwargs)
        )
        assert np.linalg.norm(ported_res) <= np.linalg.norm(process_res) * (1 + 1e-9), (
            f"port residual {np.linalg.norm(ported_res):g} is worse than PROCESS's "
            f"{np.linalg.norm(process_res):g} at its own stopping point"
        )
