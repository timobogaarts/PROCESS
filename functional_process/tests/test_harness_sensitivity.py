"""The gradient check still catches a known-real bug at the current safety factor.

`Tier1Contract.gradient_safety` was raised from 10 to 25 after two `neoclassics.py`
contracts failed at fuzz points where the port was demonstrably correct. Loosening a
check to make failures go away is exactly the move that quietly turns a test suite into
decoration, so this module pins the other side of the trade: a derivative error that
really happened, reintroduced deliberately, must still fail.

The bug is real and was caught by this harness for real. `scipy.integrate.simpson(y, x=)`
uses the general non-uniform rule whenever `x` is passed, even on a uniform grid. The
uniform shortcut -- `h = (x[-1] - x[0]) / n` with `1, 4, 2, ..., 4, 1` weights -- returns
**identical values** on a uniform grid but derivatives with respect to each `x[i]` that
are wrong by factors of 2 to 30, because it only ever looks at the two endpoints. No
value comparison at any tolerance can see it. See
`models/physics/plasma_profiles.py`'s `_simpson`.

If this module ever fails, the gradient check has been blunted past the point of doing
its job, and `gradient_safety` is the first thing to look at.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from functional_process.cottax.physics.plasma_profiles import (
    calculate_pedestal_profile_values,
)
from functional_process.tests.models.physics import (
    test_plasma_profiles as reference_case,
)


def _simpson_uniform_shortcut(y, x):
    """The bug: correct on a uniform grid, blind to every interior `x[i]`."""
    n_intervals = y.shape[0] - 1
    h = (x[-1] - x[0]) / n_intervals
    weights = jnp.ones_like(y).at[1:-1:2].set(4.0).at[2:-1:2].set(2.0)
    return h / 3.0 * jnp.sum(weights * y)


def _broken_pedestal_profile_values(profile_x, ne_profile_y, te_profile_y, **kwargs):
    """`calculate_pedestal_profile_values` with the uniform-shortcut integrator."""
    integ1 = _simpson_uniform_shortcut(
        profile_x * ne_profile_y * te_profile_y, profile_x
    )
    integ2 = _simpson_uniform_shortcut(profile_x * ne_profile_y, profile_x)

    correct = calculate_pedestal_profile_values(
        profile_x=profile_x,
        ne_profile_y=ne_profile_y,
        te_profile_y=te_profile_y,
        **kwargs,
    )
    # Only the two density-weighted temperatures depend on the integrals; the rest of
    # the tuple is untouched, so the injected error is as small as this bug really was.
    return (integ1 / integ2, correct[1], *correct[2:])


class _BrokenPedestalContract(reference_case.TestPedestalProfileValues):
    """The real contract, with only the integrator swapped for the buggy one."""

    __test__ = False  # not collected in its own right; driven by the tests below
    ported = _broken_pedestal_profile_values


@pytest.fixture
def sample():
    """The pedestal reference point, reused from the real contract's declaration."""
    return reference_case.TestPedestalProfileValues.samples[0]


def test_values_alone_cannot_see_the_bug(sample):
    """The premise: the buggy integrator agrees on *values* to machine precision.

    Without this, the gradient check catching it would prove nothing -- a value test
    would already have.
    """
    case = _BrokenPedestalContract()
    case.test_value_agreement(sample)


@pytest.mark.gradient
def test_gradient_check_still_catches_it(sample):
    """And the derivative check fails it anyway, at the current `gradient_safety`.

    Gated like every other `jacfwd`-driven check: it exists to prove
    `gradient_safety` still separates a real bug from noise, not to be paid for on a
    routine run that touches something unrelated.
    """
    case = _BrokenPedestalContract()
    with pytest.raises(AssertionError, match=r"gradient mismatch"):
        case.test_gradient_agreement(sample)


@pytest.mark.gradient
def test_the_bug_is_far_outside_the_error_bar(sample):
    """Not a marginal catch: the margin is orders of magnitude, not a factor of two.

    This is what makes raising `gradient_safety` from 10 to 25 safe. A wrong derivative
    is wrong by an `O(1)` *relative* amount; the reference's own error bar is a round-off
    and truncation quantity many orders smaller. The two populations do not overlap, so
    where the threshold sits between them barely matters.
    """
    case = _BrokenPedestalContract()
    correct = np.asarray(
        case._jacobian.__func__(
            reference_case.TestPedestalProfileValues(), sample, "profile_x"
        )
    )
    broken = np.asarray(case._jacobian(sample, "profile_x"))

    interior = slice(1, -1)  # the endpoints are the only entries the shortcut sees
    scale = np.abs(correct[0, interior]).max()
    worst = np.abs(broken[0, interior] - correct[0, interior]).max()

    assert worst / scale > 0.1, (
        f"the injected derivative error is only {worst / scale:.2e} of the derivative's "
        f"own scale -- too small to be the clean separation this test claims"
    )
