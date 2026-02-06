my.function {

  %SSA_VALUE = my.primary : () -> (!my.type)

  // Operand types must be specified explicitly:
  %123 = my.unary %SSA_VALUE : (!my.type) -> (i32)

  // Operations can have multiple operands and results:
  %0, %1 = my.binary %SSA_VALUE, %123 (!my.type, i32) -> (i32, i32)

}
