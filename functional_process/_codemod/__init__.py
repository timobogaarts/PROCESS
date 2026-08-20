"""Part A one-shot tooling. Delete this package when Part A closes.

`_audit/path_refactor.md` Part A converts the port's declaration surface from the
`lambda s: s.<area>.<field>` escape hatch to cottax's `From(area)`/`OutputInto(area)`
sugar. Two scripts live here and neither is imported by anything the port ships:

- `convert_declarations.py` -- the AST codemod (§A.3) and its census (§0).
- `check_ports_identical.py` -- the per-file inertness proof (§A.4).

Both are throwaway by design: once every declaration is converted and
`grep -rn "lambda s:" functional_process --include='*.py'` returns nothing, this
directory has no remaining caller and should go with the closing commit.
"""
