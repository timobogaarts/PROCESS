"""The coils' superconductor properties -- `superconpf`'s ITER Nb3Sn and NbTi arms.

Audit record: `functional_process/_audit/units/models/pfcoil/superconductor.md`, which
the wave that wrote this module could not create -- `unit_registry.md` was held open by
two sibling agents, so the material went into `pfcoil/fields.md` § "the CS chain" and
this docstring said a row was owed. Both were done on 2026-08-29 and the material moved
unchanged.

Ports the `ohcalc` block at `process/models/pfcoil.py:3577-3684` -- two `superconpf`
calls, at the end of flat-top and at the beginning of pulse, and the four
`.pf_coil.*` fields they produce that a constraint reads.

**Two arms of `superconpf`, not `superconpf`, and two is the whole of what can be
reached.** `.pf_coil.i_cs_superconductor` selects among eight critical-surface
parameterisations (`superconpf`, `pfcoil.py:4641-4924`), but `indat._pf_coil_system_arm`
has already refused every value but `1` and `5` before this module's slot is built --
they are the two halves of the `(i_pf_superconductor, i_cs_superconductor)` pair the
package's occupant sets are keyed on. Both are written here: `1` (`ITER_NB3SN`,
`large_tokamak_eval.IN.DAT:245`) and `5` (`WST_NB3SN`,
`low_aspect_ratio_DEMO.IN.DAT:845`). `indat.CS_SUPERCONDUCTOR` is therefore **total**
over the values that reach it, and has no `UNPORTED` entries at all.

The other six read genuinely different constants and, in three cases, different
variables -- `BI2212` computes its own temperature margin and never calls the root find
below, `HAZELTON_ZHAI_REBCO` reads the three `.superconducting_tfcoil.d*_hts_tape*`
fields nothing else touches -- so each would need its own occupant if the pair predicate
were ever widened. The critical-surface fits themselves are all already ported in
`functional_process/models/physics/superconductors.py` and are not re-ported here.

**The two occupants differ in one call and nothing else**, which is why they share a
base class the way `masses.PFCoilMassesCsWstNb3Sn` shares `PFCoilMasses`: identical
reads, identical outputs, one substituted critical-surface function and its two
constants.

**The temperature margin is a root find, and it is `superconpf`'s only non-arithmetic
step.** PROCESS solves `j_crit_sc(T) - j_sc = 0` for `T` with `scipy.optimize.newton`'s
secant iteration (`pfcoil.py:4906-4921`; `fprime=None`, `x1 = 2*T_op`, `tol = 1e-6`,
`rtol = 1e-6`, `maxiter = 50`, `disp=False`).

**Ported 2026-08-30, and the shared driver this module was waiting for is the TF coil's**
(`models/tfcoil/superconducting.py::solve_current_sharing_temperature`). That is why the
margin was deferred rather than written twice: `.tfcoil.temp_tf_superconductor_margin`
(constraint 36) is the same `scipy.optimize.newton` call on the same critical-surface
functions, it landed first, and it is imported here rather than re-derived. So the pair
is a `jax.lax.fori_loop` inside an `ExplicitFunction`, exactly as the TF coil's is, and
**not** the declared `ImplicitFunction`/`RootFind` pair this docstring predicted -- the
same decision `models/vacuum/vacuum.py::solve_duct_diameter` and the TF margin both
already made, for the reason `solve_current_sharing_temperature`'s own docstring gives:
the endpoint *is* the quantity constraint 60 compares against PROCESS's, so reproducing
scipy's stopping rule is what makes that a value test.

**The PF coils' strand critical current density joined the module on 2026-08-30**, and
it is the one thing here that is not `ohcalc`'s. `pfcoil()`'s own loop calls `superconpf`
once per PF coil (`process/models/pfcoil.py:871-904`) and keeps two of its four returns;
`masses.md` records both as computed-and-discarded because no mass depends on a critical
current. One of them is not discarded any more: `.pf_coil.j_crit_str_pf`
(`:900-904`) is Account 222.2's `PER_KAM` strand cost, and it was a field PROCESS
computes (`1.1018e9` A/m^2 on `large_tokamak_nof`) that nothing in the graph owned.
`PFStrandCriticalCurrentDensity` below owns it. `.pf_coil.j_pf_wp_critical`, the other
return, is still unowned -- see the note at the foot of this file.
"""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.models.pfcoil import (
    N_PF_COILS,
    SPHERICAL_TOKAMAK_TOPOLOGY,
    PFCoilTopology,
)
from functional_process.paths import pf_coil, superconducting_tfcoil, tfcoil
from functional_process.pfcoil.superconductor import (
    calculate_cs_critical_current_densities,
    calculate_cs_critical_current_density_iter_nb3sn,
    calculate_cs_critical_current_density_wst_nb3sn,
    calculate_cs_strand_critical_current_density,  # noqa: F401 -- re-exported for tests
    calculate_cs_temperature_margin_from_full_width_current,
    calculate_cs_temperature_margin_iter_nb3sn,
    calculate_cs_temperature_margin_wst_nb3sn,
    calculate_pf_strand_critical_current_density,
    calculate_pf_strand_critical_current_density_hazelton_zhai_rebco,  # noqa: F401 -- re-exported for tests
    calculate_pf_strand_critical_current_density_hazelton_zhai_rebco_topology,
)


class CSCriticalCurrentDensitiesIterNb3Sn(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.critical_current`, `i_cs_superconductor == 1`.

    Owns the two critical current densities constraints 26 and 27 compare against the
    operating ones, the two conductor-level densities behind them, and the strand
    density the costing model reads.

    Added 2026-08-27 for `optimise_design.md` §11.5:
    `.pf_coil.j_cs_critical_flat_top_end` and `.j_cs_critical_pulse_start` were
    boundary zeros against PROCESS's converged `3.780e7` / `3.841e7` A/m^2, so both
    constraints compared a real operating current density against a frozen zero.

    **`.pf_coil.temp_cs_superconductor_margin` is still not owned here**, and that is
    now a placement decision rather than a deferral: it is
    `CSTemperatureMarginIterNb3Sn` below, `.tokamak.cs_coil.temperature_margin`, landed
    2026-08-30. Two nodes rather than one for the same reason the TF coil has
    `cicc_superconductor_properties` and `tf_superconductor_temperature_margin` as two
    slots -- and because the margin declares a read this node argues it must not:
    `.pf_coil.c_pf_cs_coils_peak_ma`, dead work for the critical current density and the
    only input the root find needs beyond what is already here.

    The paragraph this replaces recorded the margin as owed to a shared driver with the
    TF coil's, "one commit rather than two". That is what happened: the driver is
    `models/tfcoil/superconducting.py::solve_current_sharing_temperature` and the CS arm
    imports it.

    `_critical_surface` is the one thing the WST Nb3Sn sibling below overrides.
    """

    j_cs_critical_flat_top_end = OutputInto(pf_coil)
    j_cs_critical_pulse_start = OutputInto(pf_coil)
    j_cs_conductor_critical_flat_top_end = OutputInto(pf_coil)
    j_cs_conductor_critical_pulse_start = OutputInto(pf_coil)
    j_crit_str_cs = OutputInto(pf_coil)

    _critical_surface = staticmethod(calculate_cs_critical_current_density_iter_nb3sn)
    """The arm's critical-current function. A `staticmethod` rather than an
    `eqx.field(static=True)` integer: the switch is answered by *which class is bound*,
    and the class is where the answer belongs (`_audit/next_steps.md` §14.2)."""

    def __call__(
        self,
        b_cs_peak_flat_top_end=From(pf_coil),
        b_cs_peak_pulse_start=From(pf_coil),
        f_a_cs_void=From(pf_coil),
        fcuohsu=From(pf_coil),
        str_cs_con_res=From(tfcoil),
        temp_cs_superconductor_operating=From(pf_coil),
        a_cs_cable_space=From(pf_coil),
        a_cs_poloidal=From(pf_coil),
    ):
        return calculate_cs_critical_current_densities(
            critical_surface=self._critical_surface,
            b_cs_peak_flat_top_end=b_cs_peak_flat_top_end,
            b_cs_peak_pulse_start=b_cs_peak_pulse_start,
            f_a_cs_void=f_a_cs_void,
            fcuohsu=fcuohsu,
            str_cs_con_res=str_cs_con_res,
            temp_cs_superconductor_operating=temp_cs_superconductor_operating,
            a_cs_cable_space=a_cs_cable_space,
            a_cs_poloidal=a_cs_poloidal,
        )


class CSCriticalCurrentDensitiesWstNb3Sn(CSCriticalCurrentDensitiesIterNb3Sn):
    """cottax node: `.tokamak.cs_coil.critical_current`, `i_cs_superconductor == 5`.

    `low_aspect_ratio_DEMO.IN.DAT:845`'s value. One overridden `staticmethod` and
    nothing else -- identical reads, identical outputs, `superconpf`'s WST Nb3Sn
    critical surface in place of the ITER one. Subclassing rather than a second full
    class for the same reason `masses.PFCoilMassesCsWstNb3Sn` does it: the slot is a
    **place**, and the node bound at `.tokamak.cs_coil.critical_current` keeps its name
    whichever occupant fills it.

    **Written because refusing it would have broken a file that already assembled.**
    `indat._pf_coil_system_arm`'s arm `1` exists precisely for
    `low_aspect_ratio_DEMO`'s `(3, 5)` superconductor pair, so a registry with only the
    ITER occupant would have turned an assembling machine into a `NotImplementedError`
    -- caught by `test_machine.test_a_switch_that_decides_two_slots_decides_both`, which
    builds that machine. A new slot may not narrow the set of files the port accepts.
    """

    _critical_surface = staticmethod(calculate_cs_critical_current_density_wst_nb3sn)


class CSTemperatureMarginIterNb3Sn(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.temperature_margin`, `i_cs_superconductor == 1`.

    Owns `.pf_coil.temp_cs_superconductor_margin`, constraint 60's read and the last of
    the four constraints `optimise_design.md` §11.5 found reading an `ohcalc` field with
    no producer. Landed 2026-08-30 as a missing producer measured on
    `large_tokamak_nof`: the port had `0.0` against PROCESS's `3.4208032` K, and
    constraint 60 (`temp_cs_superconductor_margin >= temp_cs_superconductor_margin_min`)
    was therefore comparing a frozen zero against a real bound.

    **A slot of its own, not a fifth output of `critical_current`.** The two nodes come
    out of the same PROCESS call -- `ohcalc` invokes `superconpf` twice and takes four of
    its returns -- but they have different read sets: only the margin needs
    `.pf_coil.c_pf_cs_coils_peak_ma`, whose edge `CSCriticalCurrentDensitiesIterNb3Sn`
    explicitly declines to invent because the critical-surface arms never use it. Same
    shape as the TF coil's two slots (`cicc_superconductor_properties` and
    `tf_superconductor_temperature_margin`), which split the same way for the same
    reason.

    `_critical_surface` is the one thing the WST Nb3Sn sibling below overrides -- and
    unlike `CSCriticalCurrentDensitiesIterNb3Sn`'s, it is the *arm function*, not an
    arm-specific wrapper, because the margin's two arms differ only in which fit and
    which pair of constants `_cs_temperature_margin_pair` is handed.
    """

    temp_cs_superconductor_margin = OutputInto(pf_coil)

    _critical_surface = staticmethod(calculate_cs_temperature_margin_iter_nb3sn)

    def __call__(
        self,
        b_cs_peak_flat_top_end=From(pf_coil),
        b_cs_peak_pulse_start=From(pf_coil),
        c_pf_cs_coils_peak_ma=From(pf_coil),
        a_cs_cable_space=From(pf_coil),
        f_a_cs_void=From(pf_coil),
        fcuohsu=From(pf_coil),
        str_cs_con_res=From(tfcoil),
        temp_cs_superconductor_operating=From(pf_coil),
    ):
        return calculate_cs_temperature_margin_from_full_width_current(
            critical_surface=self._critical_surface,
            b_cs_peak_flat_top_end=b_cs_peak_flat_top_end,
            b_cs_peak_pulse_start=b_cs_peak_pulse_start,
            c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma,
            a_cs_cable_space=a_cs_cable_space,
            f_a_cs_void=f_a_cs_void,
            fcuohsu=fcuohsu,
            str_cs_con_res=str_cs_con_res,
            temp_cs_superconductor_operating=temp_cs_superconductor_operating,
        )


class CSTemperatureMarginWstNb3Sn(CSTemperatureMarginIterNb3Sn):
    """cottax node: `.tokamak.cs_coil.temperature_margin`, `i_cs_superconductor == 5`.

    `low_aspect_ratio_DEMO.IN.DAT:845`'s value, and written for the same reason its
    `critical_current` sibling was: `indat._pf_coil_system_arm`'s arm `1` exists for that
    file's `(3, 5)` superconductor pair, so a registry with only the ITER occupant would
    turn a machine that assembles today into a `NotImplementedError`.
    """

    _critical_surface = staticmethod(calculate_cs_temperature_margin_wst_nb3sn)


class PFStrandCriticalCurrentDensity(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.strand_critical_current`.

    Owns `.pf_coil.j_crit_str_pf`, Account 222.2's `PER_KAM` strand cost. Landed
    2026-08-30 as a missing producer measured on `large_tokamak_nof`: PROCESS computes
    `1.1017899e9` A/m^2 there and the port had `0.0`, so `.costs.c2222` could not be
    registered without moving the hole from the account onto the field
    (`_audit/cost_boundary_inputs.md` §13.2).

    **The one node in this package that is `pfcoil()`'s rather than `ohcalc`'s**, which
    is why it sits in `.tokamak.pf_coil` and its four neighbours here sit in
    `.tokamak.cs_coil`. It is in this module rather than in `masses.py` because what it
    computes is a critical current, and `masses.md`'s scope argument is precisely that no
    mass depends on one -- so `superconpf`'s PF call belongs beside `superconpf`'s CS
    calls, not beside the mass loop it happens to share a `for` statement with.

    **Not a family, unlike its two CS neighbours.** They ask
    `.pf_coil.i_cs_superconductor` again because `_pf_coil_system_arm`'s arm `1` spans
    two of its values; `.pf_coil.i_pf_superconductor` is pinned to `3` by that same
    predicate on **both** arms (`_pf_coil_system_deviations`' `-6`), so there is one
    critical surface and nothing for a slot to decide. An instance default, like every
    other slot in `PFCoil`.

    **Two per-index reads, not two whole arrays.** `.pf_coil.b_pf_coil_peak[5]` and
    `.pf_coil.bpf2[5]` are the last PF coil's, the only ones the surviving scalar depends
    on (see `calculate_pf_strand_critical_current_density`); they match
    `fields.PFCoilPeakField`'s per-index `Output`s exactly, the same shape
    `masses.PFCoilMasses` uses for the six it needs.
    """

    j_crit_str_pf = OutputInto(pf_coil)

    def __call__(
        self,
        b_pf_coil_peak_last=FromExactly(pf_coil.b_pf_coil_peak[N_PF_COILS - 1]),
        bpf2_last=FromExactly(pf_coil.bpf2[N_PF_COILS - 1]),
        tftmp=From(tfcoil),
        fcupfsu=From(pf_coil),
    ):
        return calculate_pf_strand_critical_current_density(
            b_pf_coil_peak=b_pf_coil_peak_last,
            bpf2=bpf2_last,
            temp_pf_peak_field=tftmp,
            fcupfsu=fcupfsu,
        )


class PFStrandCriticalCurrentDensityHazeltonZhaiRebco(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.strand_critical_current`,
    `i_pf_superconductor == 9`.

    Both spherical tokamaks' value (`spherical_tokamak_eval.IN.DAT:235`,
    `st_regression.IN.DAT:1670`), and the reason `indat._pf_coil_system_deviations`'
    `-6` fired on them.

    **A sibling rather than a subclass**, unlike `CSCriticalCurrentDensitiesWstNb3Sn`:
    that family overrides one `staticmethod` and keeps its ports, and this arm cannot,
    because `hijc_rebco` needs the tape's three dimensions and `jcrit_nbti` does not.
    Three extra reads is a different node.

    **The two peak fields are read whole here and per index on the conventional arm**,
    and that is not a style choice either: `PFCoilPeakFieldNoCentralSolenoid` owns both
    arrays whole (there is no CS node to own the last slot), and cottax matches reads by
    equality -- a `FromExactly(b_pf_coil_peak[7])` against a whole-array owner is
    refused outright as a read that would silently become a boundary input. The index
    the calculation actually wants moves with the topology, so it is taken inside.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)
    """Static -- which coil the `pfcoil()` loop finishes on, and therefore which slot of
    the two peak-field arrays the surviving scalar came from."""

    j_crit_str_pf = OutputInto(pf_coil)

    def __call__(
        self,
        b_pf_coil_peak=From(pf_coil),
        bpf2=From(pf_coil),
        tftmp=From(tfcoil),
        fcupfsu=From(pf_coil),
        dr_tf_hts_tape=From(superconducting_tfcoil),
        dx_tf_hts_tape_rebco=From(superconducting_tfcoil),
        dx_tf_hts_tape_total=From(superconducting_tfcoil),
    ):
        return calculate_pf_strand_critical_current_density_hazelton_zhai_rebco_topology(
            b_pf_coil_peak=b_pf_coil_peak,
            bpf2=bpf2,
            tftmp=tftmp,
            fcupfsu=fcupfsu,
            dr_tf_hts_tape=dr_tf_hts_tape,
            dx_tf_hts_tape_rebco=dx_tf_hts_tape_rebco,
            dx_tf_hts_tape_total=dx_tf_hts_tape_total,
            topology=self.topology,
        )


# `.pf_coil.j_pf_wp_critical` -- left unowned, and named here so a reader who greps for
# it finds the reason rather than an omission. `ohcalc:3670-3675` writes the
# beginning-of-pulse winding-pack critical density into the CS slot and then copies it
# straight back out into `.pf_coil.j_cs_critical_pulse_start` (`:3677-3679`), so the CS
# slot is the same number under two names and the second name is the one every consumer
# reads. The six PF slots are `pfcoil()`'s (`:877`), and those `superconpf` calls **are**
# reached now -- `PFStrandCriticalCurrentDensity` above takes their third return for the
# last coil. Their first return is still not taken, and the reason has changed from
# "unported" to "no reader": `j_pf_wp_critical` is an array whose six entries nothing in
# this graph consumes (PROCESS itself only prints them, `outpf:2570-2603`), and owning it
# would mean six critical-surface evaluations for an output no edge leaves. It stays on
# `boundary.computed_by_process`'s list of fields PROCESS writes and the port does not,
# where it is invisible to `unproduced_but_computed` precisely because nothing reads it.
