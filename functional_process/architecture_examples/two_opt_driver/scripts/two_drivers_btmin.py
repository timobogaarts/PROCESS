import os
HERE = os.path.dirname(os.path.abspath(__file__))
import os, sys, time, dataclasses
BT_MIN = float(os.environ.get("BT_MIN", "5.5"))
exec(open(HERE + "/two_drivers2.py").read().split("F = xDSMFormatterFlat()")[0].split("folded = fold(build_graph(SPLIT_A))")[0])
import equinox as eqx
class _BtMin(eqx.Module):
    bt_min: float = eqx.field(static=True)
    def __call__(self, bt):            # cottax convention g <= 0: violated when bt < bt_min
        return (self.bt_min - bt) / self.bt_min
bt_path = iteration_variable_path(2)
cnodes[NodePath((GetAttrKey("ConstraintBtMin"),))] = ImplementedFunction(
    reads=(bt_path,), owns=(cond("bt_min"),), fn=_BtMin(BT_MIN))
SPLIT_BT = {"OptMagnet": ([2, 3, 56, 59], rmajor_obj, [32, 34, 35, 65, 82, 83, "bt_min"]),
            "OptPlasma": ([4, 6, 10, 109], coe, [2, 16, 8, 17, 18, 24, 62, 67])}
folded = fold(build_graph(SPLIT_BT))
b = Blocking.scc(folded)
print(f"BT_MIN={BT_MIN}: problems ->", [p.spelling for p in b.problems if p is not None])
assigned = assign_drivers(folded, default_drivers(folded, bounds=design_bounds(ref.ixc)))
tail = open(HERE + "/two_drivers_solve.py").read().split("\n", 3)[3]   # after the exec line
exec(tail.replace('print("constraints:"', 'print("bt_min condition:", float(np.asarray(out[cond("bt_min")])))\nprint("constraints:"'))
