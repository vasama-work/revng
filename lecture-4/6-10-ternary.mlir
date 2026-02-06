// 1 ? 2 : 3;
clift.expr {
  %0 = clift.imm 1 : !int32_t
  %1 = clift.imm 2 : !int32_t
  %2 = clift.imm 3 : !int32_t
  %3 = clift.ternary %0, %1, %2 : (!int32_t, !int32_t)
  clift.yield %3 : !int32_t
}
