// !(x == y) -> x != y
def NotCmpEqPattern
  : Pat<(Clift_LogicalNotOp (Clift_CmpEqOp $lhs, $rhs)),
        (Clift_CmpNeOp $lhs, $rhs)>;
