//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_EXPRESSION_ROOT_SIMPLIFICATION_REWRITES
#define MLIR_CLIFT_EXPRESSION_ROOT_SIMPLIFICATION_REWRITES

include "mlir/IR/PatternBase.td"

include "revng/Clift/Clift.td"
include "revng/CliftTransforms/ExpressionHelpers.td"

// *(T*)&(x = y) -> x = y
// Constraint: sizeof(T) == sizeof(x)
// Constraint: used at the root of an expression statement.
def DiscardedAssignmentPattern
  : Pat<(Clift_IndirectionOp:$res (Clift_CastOp:$new_ptr (Clift_AddressofOp:$old_ptr (Clift_AssignOp:$assignment $lhs, $rhs)),
                                                         $kind)),
        (replaceWithValue $assignment),
        [(IsBitcastKind $kind),
         (Constraint<CPred<"pointeesAreOfSameSize($0.getType(), $1.getType())">> $new_ptr, $old_ptr),
         (Constraint<CPred<"isDiscardedExpression($0)">> $res)]>;

// (T)x -> x
// Constraint: T is an integer type.
// Constraint: x is a boolean expression.
// Constraint: used at the root of a control flow condition.
def ConditionCastPattern
  : Pat<(Clift_CastOp:$res $arg, $kind),
        (replaceWithValue $arg),
        [(Constraint<CPred<"isIntegerType($0.getType())">> $res),
         (Constraint<CPred<"isBooleanExpression($0)">> $arg),
         (Constraint<CPred<"isUsedInConditional($0)">> $res)]>;

#endif
