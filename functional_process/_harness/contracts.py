"""The per-tier contracts a ported unit subclasses.

A ported unit does not write test functions. It declares what it is — the PROCESS
reference, the port, the points to check, the audit record it came from — and inherits
the checks its tier demands:

    class TestSudoDensityLimit(Tier1Contract):
        audit_record = "models/stellarator/density_limits.md"
        reference = _reference_sudo_density_limit
        ported = calculate_sudo_density_limit
        samples = [...]

Tier is expressed by which class you subclass, which is the same field the unit's audit
record already carries (`## tier signal`). Porting a unit therefore means reading its
record and picking a base class, with no second decision to keep in sync.

The tiers differ in *which tests exist*, not merely in tolerance. `Tier2Contract` has no
value-agreement test at all, because for a unit whose PROCESS implementation is an
unchecked fixed-iteration loop, PROCESS's answer is not ground truth and diffing against
it would fail a correct port. Making that test structurally absent is stronger than
documenting that it should not be written.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from functional_process._harness.boundary import (
    DIVISION_BY_ZERO_AT_BOUNDARY,
    registered_reason,
)
from functional_process._harness.finite_difference import (
    PROCESS_EPSFCN,
    ZeroPerturbationError,
    fd_gradient_with_error,
)
from functional_process._harness.sampling import fuzz_samples
from functional_process._harness.tolerance import (
    MACHINE_PRECISION,
    DeclaredDeviation,
    Tolerance,
)


def _as_array(value):
    """Normalise a scalar, array or tuple return into a flat 1-D float array.

    Flattened in leaf order, structure discarded: every check below compares component
    by component and has no use for the shape. That is what lets a unit returning
    `(4, 30)` arrays, or a tuple of four `(4,)` ones, be declared with no per-unit
    flattening adapter — the reference and the port are flattened by the same rule, so
    their components line up if and only if they return the same thing.
    """
    return np.concatenate([
        np.ravel(np.asarray(leaf, dtype=float)) for leaf in jax.tree.leaves(value)
    ])


def _as_traced_array(value):
    """`_as_array`'s traced twin: the same leaf order, in `jnp` so `jacfwd` sees through.

    Separate from `_as_array` rather than parameterised by an array module, because the
    two are used at different times — one on a concrete PROCESS return, one inside a
    trace — and `np.asarray` on a tracer is exactly the mistake this keeps out of reach.
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
    """Shared declaration surface for every tier.

    Attributes
    ----------
    audit_record :
        Path of the unit's audit record, relative to `functional_process/_audit/units/`
        — the record tree, which mirrors the package layout, as does the
        `tests/functional_process/` tree this case lives in. Records and cases each moved
        out of the package into their own mirror, so this stays the package-relative path
        it always was; only the root it resolves against moved. Resolved by
        `conftest.py`'s `audit_root`. Checked for existence, so a port whose record was
        moved or never written fails loudly.
    reference :
        The PROCESS-side callable, adapted to the port's signature. Where PROCESS's
        function takes a `DataStructure`, the adapter that binds one lives in the unit's
        test module — writing it is the point at which the audit's "close the `data`
        back-door" claim gets tested rather than asserted.
    ported :
        The pure JAX callable under test.
    samples :
        Evaluation points. See `_harness.sampling`.
    static_argnames :
        Arguments that are switches or preconditions rather than continuous inputs.
        Excluded from differentiation, per `_audit/naming_convention.md` § "switches are
        not ports".
    """

    audit_record = None
    reference = None
    ported = None
    samples = ()
    static_argnames = ()

    def __init_subclass__(cls, **kwargs):
        """Wrap bare functions assigned to `reference`/`ported` in `staticmethod`.

        Without this, `self.reference` would bind as a method and silently pass the
        contract instance as the first physics argument. Requiring an explicit
        `staticmethod(...)` in every subclass would work too, but it is a footgun that
        costs a confusing failure the first time someone forgets.
        """
        super().__init_subclass__(**kwargs)
        for attr in ("reference", "ported"):
            value = cls.__dict__.get(attr)
            if callable(value) and not isinstance(value, staticmethod):
                setattr(cls, attr, staticmethod(value))

    @classmethod
    def diff_argnames(cls, sample):
        """Arguments to differentiate with respect to, for one sample."""
        return tuple(k for k in sample.kwargs if k not in cls.static_argnames)

    def test_audit_record_exists(self, audit_root):
        """The unit's audit record must exist.

        `_audit/test_harness.md` makes the audit a precondition of the test, not a
        parallel activity: the true signature is not known until the record's
        implicit-io classifications are resolved. This check is what stops that from
        being an honour system.
        """
        assert self.audit_record is not None, (
            f"{type(self).__name__} must declare `audit_record`"
        )
        record = audit_root / self.audit_record
        assert record.is_file(), f"audit record not found: {record}"


class Tier1Contract(PortContract):
    """Explicit pure functions: no internal iteration, no `self.data` access.

    Four checks, each a separate test node so a failure names itself:

    - value agreement at machine precision (no solver is involved on either side);
    - value finiteness, on every run, eager, no `jacfwd`;
    - gradient finiteness and gradient agreement against PROCESS's own finite
      difference (to within that difference's self-estimated error) — both require a
      `jacfwd` trace+compile of the port and are **opt-in**, `--fp-gradients`, skipped
      otherwise. Gradient finiteness is what catches a `jnp.where` that returns the
      right number while leaking a NaN through the untaken branch; a value-only diff
      cannot see that, and it is the failure mode the rewrite is most exposed to — but
      compiling every ported unit's autodiff graph on every routine run is the
      dominant cost of this harness, so it is gated the same as the finite-difference
      comparison rather than running unconditionally.

    **An argument may be an array.** Its components are differentiated one at a time —
    the PROCESS side by perturbing that one entry, the port side by one `jacfwd` column
    per argument whose columns are those same entries, batched across *every*
    differentiable argument in one `jacfwd` call (`_jacobians`) rather than one call per
    argument — so a function vectorised over species or over a quadrature grid is checked
    exactly as densely as a scalar one, with each failure naming the entry
    (`temperatures[2]`, `kt[1, 7]`). The cost is linear in the number of components: a
    `(4, 30)` argument means 120 columns and ~4 reference calls each, which is why an
    array-heavy unit fuzzes at the same count but takes seconds rather than
    milliseconds.

    Returns are flattened the same way (`_as_array`), so a port returning a tuple of
    arrays needs no adapter to be compared against a reference returning one array.
    """

    pytestmark = pytest.mark.tier1

    value_tolerance = MACHINE_PRECISION
    declared_deviation: DeclaredDeviation | None = None
    """Set when the port **deliberately** does not compute PROCESS's expression.

    `None` for every ordinary unit, and `value_tolerance` then means what it says. When
    set, `test_value_agreement` is checked against `declared_deviation.bound` instead --
    and `test_declared_deviation_is_real` requires the deviation to be *exercised*, so
    this is strictly more demanding than leaving it unset, not less. See
    `_harness/tolerance.DeclaredDeviation` for why this is not a tolerance knob.
    """

    epsfcn = PROCESS_EPSFCN
    gradient_safety = 25.0
    """Multiplier on the finite difference's own error bar.

    The error estimate is a leading-order extrapolation, not a bound; a plain factor of
    1 would flag correct ports wherever the neglected `O(h^4)` term is not negligible.

    **Raised from 10 after measurement, not after a failure.** Two `neoclassics.py`
    contracts failed at fuzz points where the port was demonstrably right — refining the
    step showed `jacfwd` is the `h -> 0` limit of PROCESS's own difference, agreeing to
    3e-11 relative at `epsfcn = 1e-4`, while `epsfcn = 1e-3` sits where truncation and
    cancellation are comparable. Both needed about 1.8x more headroom than 10 gave, from
    two unrelated causes (one round-off dominated, one truncation dominated); 25 covers
    the measured worst case with ~40% margin. See `finite_difference` for the numbers.

    This costs almost nothing in detection power. A wrong derivative is wrong by an
    `O(1)` *relative* amount, not by a small multiple of the reference's own error bar —
    the `scipy.integrate.simpson` bug this harness caught in
    `models/physics/plasma_profiles.py` was off by factors of 2 to 30 -- many orders
    of magnitude outside the bar either way. `test_gradient_agreement`'s job is
    separating "wrong" from "right", not grading a correct port's last digit.
    """

    gradient_floor = 0.0
    """Extra allowance, as a fraction of the largest derivative in the same column.

    **Zero by default, and nothing that does not set it changes behaviour.** It exists
    for the one case `gradient_safety` structurally cannot cover: PROCESS's error bar is
    a Richardson extrapolation of its *own* truncation error, so when PROCESS's answer
    is bit-for-bit flat in an input the bar is exactly `0.0` and no multiplier of it is
    anything but `0.0`. That is not "PROCESS is certain"; it is "PROCESS happened to
    round to the same float four times".

    The case that needed it (`models/tfcoil/stress.py`'s `plane_stress`,
    `test_stress.py`): `sigr` at the innermost radius is a boundary condition, so both
    implementations return exactly `0.0` and PROCESS's difference is exactly `0.0` --
    while the port's `jacfwd`, propagating a tangent through the same cancelling
    expression, returns `-3.8e-10` against derivatives of order `1e6` elsewhere in that
    column. Nothing is wrong; two linear solves on a matrix PROCESS's own comment calls
    "often very ill-conditioned" (`process/models/tfcoil/base.py:4404-4412`) simply do
    not cancel to the same bit.

    Scaled to the column rather than absolute so that one number means the same thing
    for a stress in Pa and a deflection in m. A wrong derivative is wrong by an `O(1)`
    relative amount, so a floor of `1e-8` still separates wrong from right by eight
    orders of magnitude -- the same argument `gradient_safety` above makes.
    """

    reference_domain_errors = ()
    """Exceptions the PROCESS reference raises to signal an out-of-domain input.

    Where PROCESS raises (e.g. `ProcessValueError` on a negative square root), the port
    is expected to return a non-finite value instead of raising — a traced function
    cannot raise on a data-dependent condition. Declaring the exception type here turns
    that expectation into an assertion instead of letting fuzz samples outside the
    domain fail the run.
    """

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
        """A declared deviation must be **exercised**, or it is a loosened tolerance.

        Skipped for every unit that declares none. For a unit that does, this asserts
        that *some* sample genuinely disagrees with PROCESS by more than the ordinary
        tier-1 tolerance -- so a `DeclaredDeviation` cannot be left behind after the
        deviation is removed, and cannot be added to quieten a unit that would have
        passed anyway.
        """
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
        """The port's value is free of NaN/Inf on an in-domain point.

        Eager, no `jacfwd` — this is the check that runs on every default invocation
        (import the unit, call it, look at the result), which is what keeps a plain
        `pytest tests/functional_process` a fast "did I break an import/signature" pass
        rather than a full recompile of every ported unit's autodiff graph. The gradient
        half of this same idea — a `jnp.where` whose untaken branch is NaN — is
        `test_gradient_finite` below, gated the same way as `test_gradient_agreement`.
        """
        _, domain_error = self._reference_or_domain_error(dict(sample.kwargs))
        if domain_error is not None:
            pytest.skip(f"point is outside PROCESS's domain: {domain_error}")

        value = _as_array(self.ported(**sample.kwargs))
        assert np.all(np.isfinite(value)), f"non-finite output: {value}"

    @pytest.mark.gradient
    def test_gradient_finite(self, sample):
        """The port's gradient is free of NaN/Inf on an in-domain point.

        Split out from `test_outputs_finite` and gated behind `--fp-gradients`
        alongside `test_gradient_agreement`: both require a `jacfwd` trace+compile of
        the port, which is the expensive part of this harness (see `_jacobians`), and
        a routine "did I break something unrelated" run has no use for it. This is
        still the check that catches a `jnp.where` whose untaken branch evaluates to
        NaN — a value-only diff cannot see that — it just no longer pays its compile
        cost on every run.
        """
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

        A function can agree in value everywhere and still be wrong in derivative, and
        the derivative is what the solver consumes -- but for an *explicit* pure
        function
        whose value already agrees, autodiff is hard to get wrong, and this is by far the
        most expensive check here (four reference evaluations per argument component, on
        top of `_jacobians`' own compile). So it is **opt-in**: `--fp-gradients`, skipped
        otherwise, same as `test_gradient_finite` — neither differentiates on a routine
        run.
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
        """The PROCESS reference as a function of one flat component of one argument.

        Every other component of that argument, and every other argument, is held fixed.
        That is what `Evaluators.fcnvmc2` does to one iteration variable at a time, so an
        array argument is differentiated component by component rather than along some
        aggregate direction — the reference stays PROCESS's own scheme, and a failure
        names the entry it is in.

        A scalar argument is handed back as a plain `float`, not a 0-d array, so a
        reference adapter that does anything but arithmetic with it sees what it always
        saw.
        """
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

        Returns `{name: (outputs, components) matrix}`, one entry per
        `diff_argnames(sample)` — same shape per entry as the old per-argument
        `_jacobian`, but computed as **one** `jax.jacfwd(..., argnums=...)` trace over
        every argument together, instead of one trace (and one XLA compile) per
        argument name in a Python loop.

        That loop was the actual cost of this harness: differentiating an
        `n`-argument unit used to mean `n` separate compiles of essentially the same
        computation, each paying CPU XLA's fixed per-program overhead on top of
        whatever the function itself costs. Multi-`argnums` `jacfwd` batches every
        argument's tangent directions into one program instead, so it pays that fixed
        overhead once — measured 2.7x faster on an 11-argument unit
        (`FusionRates`), and the saving grows with argument count. Component-level
        batching (a `(4, 30)` argument's 120 columns) was already handled inside a
        single `jacfwd` call and is unaffected by this change.
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
        """No argument may be finite in value and non-finite in gradient at `x == 0`.

        The class-closing check for `_audit/next_steps.md` §9's trap: `x ** p` with
        `0 < p < 1` (`jnp.sqrt` included) at exactly zero is value-correct and
        differentiates to `inf`/`nan`, so every other test in this file passes while a
        solver's whole Jacobian row is poisoned. See `_harness/boundary.py` for why the
        criterion is "value finite implies gradient finite" and why the register there
        holds a *different* class (unguarded division) rather than instances of this one.

        Deliberately **not** parametrized over `sample`. The defect is a property of the
        function's structure at a boundary point, not of the sample it was reached from,
        so one deterministic point per contract -- the first declared sample, or one
        fuzz draw at seed 0 for a fuzz-only unit -- exercises the same code path that
        every other sample would, at a fraction of the cost. That cost is not small: one
        `jacfwd` trace per argument *component*, because each zeroed component is a
        different input point and cannot be batched into one trace the way
        `_jacobians` batches directions at a single point.

        An argument component already `0.0` at the sample point is skipped -- there is
        no boundary to move to -- and so is any component whose *value* goes non-finite
        when zeroed, which is an out-of-domain point that `test_outputs_finite` owns.
        """
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

        The other arguments are held at the sample's own values, exactly as
        `_reference_along` holds them for the finite-difference comparison -- so a
        failure names one argument, not a direction in the whole input space.
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

        `test_gradient_finite`/`test_gradient_agreement` use `_jacobians` directly so a
        multi-argument unit pays one compile, not one per argument; this wrapper is for
        the rarer case (`test_harness_sensitivity.py`) that only wants one column-group
        and has no other argument to batch it with.
        """
        return self._jacobians(sample)[name]


class Tier2Contract(PortContract):
    """Units whose PROCESS implementation closes an internal loop.

    Deliberately has **no** value-agreement test. PROCESS's answer here is often not a
    converged one — the motivating case, `power_at_ignition_point`, calls `st_phys`
    exactly twice with no convergence check at all — so a properly convergent port is
    *expected* to land somewhere numerically different, and diffing values would fail
    correct work for reasons that have nothing to do with the port.

    The pass criterion is residual-based instead: plug both answers back into the unit's
    own defining equations, and require the port to be no worse. That sidesteps "whose
    stopping point is right" entirely. See `_audit/test_harness.md` § Tier 2.
    """

    pytestmark = pytest.mark.tier2

    def __init_subclass__(cls, **kwargs):
        """`eqx.filter_jit`-wrap `ported`, once, at class-definition time.

        A tier-2 unit's internal solve (bisection, Newton, ...) typically closes over
        its own data arguments as free variables inside a `lax.while_loop`/`lax.scan`
        it builds internally (e.g. `optx.root_find`'s solver state). Traced without an
        enclosing `jax.jit`, those closed-over arrays get embedded as literal constants
        in the program XLA compiles -- so a *different* sample with the same shape is a
        *different* program, and every single call recompiles from scratch, however
        many times the same unit runs. Measured on `intersect`: four same-shape,
        different-data calls cost 0.44/0.29/0.28/0.28s unjitted (no call ever got
        cheaper), versus 0.24s once and ~0s for the rest once jitted.

        Doing this here, once, rather than inside a test method, is what makes the
        cache actually pay off: pytest gives each test item its own contract instance,
        so a wrapper built inside `test_ported_residual_small` would be a fresh,
        never-reused `eqx.filter_jit` object every sample -- as cold as not jitting at
        all. Built once at class-body-execution time and stored as a class attribute,
        every sample and both test methods below share the one compiled cache.
        """
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
