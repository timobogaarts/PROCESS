# Tokamak call surface — the scope rule, traced

**Superseded — tokamak support has since fully landed; this was pre-implementation
scoping.** Traced (not assumed) by running `Caller._call_models_once` under a profiler on
the reference tokamak file. Kept: the scope rule itself, and the CoolProp finding, both
still relevant now the port is built.

## The scope rule

> All of `process/core/caller.py::_call_models_once`'s reachable call graph with
> `istell == 0` and `ife == 0` — 38 files under `process/models/**`, entered through 31
> top-level `Model` calls plus the sub-models `Models.__init__` injects — is **338
> distinct functions**, not whole files. `models/ife.py`, `models/stellarator/**`,
> `models/tfcoil/resistive.py`, `models/blankets/dcll.py`, `models/costs/costs_2015.py`,
> `models/engineering/{pumping,materials}.py` and all of `models/geometry/**` are
> **outside** it on the tokamak reference run. `models/geometry/**` (11 files, 1537 LOC)
> is a trap for a whole-directory glob specifically: it lives under `process/models/` but
> is imported only by the plotting module, never the solve, and is reached zero times.

## The CoolProp finding — one module reached, not six

An earlier estimate named six CoolProp-bound modules; that overcounts by five and is a
**weak** signal ("some module in the neighbourhood reaches CoolProp", not "this module's
own live path does"). Traced precisely on the tokamak reference run: **exactly one module
reaches CoolProp — `tfcoil/quench.py`, via the TF-coil quench-protection chain (450
CoolProp calls in one run), which constraints 34/35/36/74/75 all read, unported, no
registry row.** Five modules import `FluidProperties` at all (not six —
`engineering/pumping.py` only mentions "CoolProp" in a docstring); of those, three
(`fw.py`, `blanket_library.py`, `hcpb.py`) are reached by the trace but every one of their
CoolProp call sites sits behind `i_p_coolant_pumping == MECHANICAL` or
`i_blkt_coolant_type == WATER`, neither of which this reference run selects — dormant,
not dead. **CoolProp-bound-and-live is 1 module / 436 lines; CoolProp-bound-and-dormant
(live the moment either switch moves) is 3 more modules / 2077 lines.** Both numbers
matter — quoting only the first understates what a second input file on the same device
would cost. The stellarator reference run reaches CoolProp zero times, so this was a new
obstacle for the tokamak specifically, not inherited.
