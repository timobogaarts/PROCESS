---
kind: model-unit
status: draft
confidence: medium
---

**Ported.** `models/pfcoil/stresses.py` / `tests/functional_process/models/pfcoil/
test_stresses.py`: `_ellipk`, `_ellipe`, `calculate_cs_hoop_stress`,
`calculate_cs_radial_stress`, `calculate_cs_self_peak_midplane_axial_stress`,
`calculate_tresca_stress`, `calculate_von_mises_stress`, `calculate_cs_stresses` —
tier-1, seven contracts. One cottax node: `CSCoilStresses` (`.tokamak.cs_coil.stresses`).

## source

`process/models/pfcoil.py:3398-3521` — the `i_pf_conductor == SUPERCONDUCTING` arm of
`ohcalc`'s stress block. Two spans of that range are deliberately **not** ported and the
reason is under "open questions".

## what it ports

Wilson's hoop and radial stresses, the elliptic-integral axial self-stress, and the
Tresca/von Mises combinations. Every stress is evaluated at the **beginning of pulse**,
at three different radii, each PROCESS's own choice.

**This module closes the blocker `models/pfcoil/namespace.py::CSCoil` named in so many
words** — "`ohcalc`'s `scipy.special` ellipk/ellipe calls". `_ellipk`/`_ellipe` are the
arithmetic-geometric mean: traceable, differentiable, twelve unrolled iterations (the
AGM converges quadratically), agreeing with `scipy.special` to `2.2e-16` / `1.1e-15`
over `m = 1e-8 … 0.9999`.

**Two ports of "the elliptic integrals" now live in this package, and that is
deliberate.** `fields.py`'s Green's-function kernel transcribes PROCESS's own Abramowitz
& Stegun rational fits, because PROCESS evaluates *those* inline and reproducing PROCESS
means reproducing its approximation error. `stresses.py` uses the AGM, because PROCESS
calls the exact library there. Substituting either for the other would be a divergence
dressed as a port.

**A finding worth keeping: PROCESS's `np.isclose(x, 0.0)` snap makes the CS's inner
radial stress exactly derivative-free in the two radii.** The guard zeroes a window of
half-width `1e-8` in two shape terms, so inside it the function is constant and the
port's autodiff says zero; PROCESS's own finite difference at `epsfcn = 1e-3` says
`-9.9e7`, because a `1e-3` step cannot see a `1e-8`-wide feature. The FD is not a valid
oracle for that point, so `TestCalculateCSRadialStress.diff_argnames` drops those two
arguments at that one sample and every other check stands. Any optimiser reaching for
that sensitivity gets zero from the port and a step-size-dependent number from an FD.

**+7 MDA-harness agreements**, all seven exact, including the axial stress and force
that go through the AGM.

## open questions

1. **The 21-point vertical profile of the axial self-stress** (`:3436-3465`,
   `.pf_coil.stress_z_cs_self_profile`) is not ported. Nothing in the graph and no
   active constraint reads it, and it carries an `np.isnan` sweep — a data-dependent
   mask over a fixed grid, portable but not free. Left out with this line rather than
   silently.
2. ~~**The CS fatigue call** (`:3486-3499`, `cs_fatigue.ncycle`) is not ported: it is a
   whole `Model` of its own and `.tokamak.cs_fatigue` is still an empty slot.~~
   **CLOSED 2026-08-30** — ported as the `Model` it is (`models/cs_fatigue.py::
   CsFatigue`, `.tokamak.cs_fatigue`), on `cs_fatigue.md`'s own staircase policy. The
   clause worth retracting is not this one but the sentence beside it in the module
   docstring: *"neither is read by any active constraint"*. Constraint 90 reads
   `.cs_fatigue.n_cycle` and **is** active on `low_aspect_ratio_DEMO`, where it was
   violated by exactly `+1.000000` with a zero gradient row for want of a producer.
   `.pf_coil.stress_hoop_cs_inner`, which this unit owns, is that node's one physics
   read — so this module is upstream of a live constraint by one hop, and did not know
   it.

## record provenance

Written 2026-08-27 inside `pfcoil/fields.md` § "the CS chain", because the wave that
wrote this module was asked to leave `unit_registry.md` alone while two sibling agents
had it open. Split out to its own record with its own registry row on 2026-08-29; the
material is unchanged apart from the heading levels.
