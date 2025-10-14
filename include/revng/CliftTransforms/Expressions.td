//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_EXPRESSION_REWRITES
#define MLIR_CLIFT_EXPRESSION_REWRITES

include "mlir/IR/PatternBase.td"

include "revng/Clift/Clift.td"
include "revng/CliftTransforms/ExpressionHelpers.td"

//===--------------------------- Math identities --------------------------===//

// - -x -> x
def NegNegPattern
  : Pat<(Clift_NegOp (Clift_NegOp $arg)),
        (replaceWithValue $arg)>;

// x + 0 -> x
def AddZeroPattern
  : Pat<(Clift_AddOp $arg, (Clift_ImmediateOp $value)),
        (replaceWithValue $arg),
        [(Constraint<CPred<"$0.getValue().isZero()">> $value)]>;

// x * 0 -> 0
def MulZeroPattern
  : Pat<(Clift_MulOp $arg, (Clift_ImmediateOp:$res $value)),
        (replaceWithValue $res),
        [(Constraint<CPred<"$0.getValue().isZero()">> $value)]>;

// x * 1 -> x
def MulOnePattern
  : Pat<(Clift_MulOp $arg, (Clift_ImmediateOp $value)),
        (replaceWithValue $arg),
        [(Constraint<CPred<"$0.getValue().isOne()">> $value)]>;

// x / 1 -> x
def DivOnePattern
  : Pat<(Clift_DivOp $arg, (Clift_ImmediateOp $value)),
        (replaceWithValue $arg),
        [(Constraint<CPred<"$0.getValue().isOne()">> $value)]>;

//===------------------------- Comparison negation ------------------------===//

// !(x == y) -> x != y
def NotCmpEqPattern
  : Pat<(Clift_LogicalNotOp (Clift_CmpEqOp $lhs, $rhs)),
        (Clift_CmpNeOp $lhs, $rhs)>;

// !(x != y) -> x == y
def NotCmpNePattern
  : Pat<(Clift_LogicalNotOp (Clift_CmpNeOp $lhs, $rhs)),
        (Clift_CmpEqOp $lhs, $rhs)>;

// !(x < y) -> x >= y
def NotCmpLtPattern
  : Pat<(Clift_LogicalNotOp (Clift_CmpLtOp $lhs, $rhs)),
        (Clift_CmpGeOp $lhs, $rhs)>;

// !(x > y) -> x <= y
def NotCmpGtPattern
  : Pat<(Clift_LogicalNotOp (Clift_CmpGtOp $lhs, $rhs)),
        (Clift_CmpLeOp $lhs, $rhs)>;

// !(x <= y) -> x > y
def NotCmpLePattern
  : Pat<(Clift_LogicalNotOp (Clift_CmpLeOp $lhs, $rhs)),
        (Clift_CmpGtOp $lhs, $rhs)>;

// !(x >= y) -> x < y
def NotCmpGePattern
  : Pat<(Clift_LogicalNotOp (Clift_CmpGeOp $lhs, $rhs)),
        (Clift_CmpLtOp $lhs, $rhs)>;

//===---------------------------- Miscellaneous ---------------------------===//

// (T)123 -> 123
// Constraint: T is a primitive integer type
def CastImmediatePattern
  : Pat<(Clift_CastOp:$res (Clift_ImmediateOp $value), $kind),
        (createImmediateOp (getType $res),
                           (NativeCodeCall<"truncateIntegerValue($0, $1)"> $value, $res)),
        [(Clift_AnyPrimitiveIntegerType $res)]>;

// (enum E)123 -> A
// Constraint: A is an enumerator of E whose value is 123.
def EnumImmediatePattern
  : Pat<(Clift_CastOp:$res (Clift_ImmediateOp $value), $kind),
        (createImmediateOp (getType $res), $value),
        [(IsBitcastKind $kind),
         (Constraint<CPred<[{ hasEnumeratorValue($0.getType(),
                                                 $1.getValue().getZExtValue()) }]>> $res, $value)]>;

// *&x -> x
def IndirectionAddressofPattern
  : Pat<(Clift_IndirectionOp (Clift_AddressofOp $arg)),
        (replaceWithValue $arg)>;

// &*x -> x
def AddressofIndirectionPattern
  : Pat<(Clift_AddressofOp (Clift_IndirectionOp $arg)),
        (replaceWithValue $arg)>;

// (T)x -> x
// Constraint: the type of x is equivalent to T.
def RedundantCastPattern
  : Pat<(Clift_CastOp:$res $arg, $kind),
        (replaceWithValue $arg),
        [(TypesAreEquivalent $res, $arg)]>;

// (T)(U)x -> (T)x
// Constraint: the cast kinds are the same.
// Constraint: the cast kind is collapsible (implemented in .cpp file).
def CollapsibleCastPattern
  : Pat<(Clift_CastOp:$res (Clift_CastOp $arg, $inner_kind), $outer_kind),
        (Clift_CastOp $arg, $inner_kind),
        [(ValuesAreEqual $inner_kind, $outer_kind),
         (Constraint<CPred<"isCollapsibleCastKind($0.getValue())">> $inner_kind)]>;

// *(T*)p -> (T)*p
// Constraint: p is of type U* and U is convertible to type T using a bitcast.
// Constraint: The value category of the expression is not significant.
def ReadTypePunnedPattern
  : Pat<(Clift_IndirectionOp:$res (Clift_CastOp:$new_ptr $old_ptr, $kind)),
        (createCastOp (getPointeeType $new_ptr),
                      (Clift_IndirectionOp $old_ptr),
                      (getBitcastKind)),
        [(IsBitcastKind $kind),
         (Constraint<CPred<"readTypePunnedConstraint($0, $1)">> $res, $old_ptr)]>;

// *(T*)p = x ->         *p = (T)x
//            -> *(T*)&( *p = (T)x )
// Constraint: p is of type U* and x is convertible to type U using a bitcast.
//
// Replacement depends on whether the value is discarded.
def AssignTypePunnedPattern
  : Pattern<(Clift_AssignOp:$old_res (Clift_IndirectionOp:$old_ref (Clift_CastOp:$new_ptr $ptr,
                                                                                          $kind)),
                                     $value),
            [(Clift_AssignOp:$new_res (Clift_IndirectionOp $ptr),
                                             (createCastOp (getPointeeType $ptr),
                                                           $value,
                                                           $kind)),
             (NativeCodeCall<"assignTypePunnedResult($_builder, $0, $1, $2, $3)"> $old_res,
                                                                                  $new_res,
                                                                                  $new_ptr,
                                                                                  $old_ref)],
            [(IsBitcastKind $kind),
             (Constraint<CPred<"assignTypePunnedConstraint($0, $1)">> $ptr, $value)]>;

// (T*)(i + k_1) -> (T*)i + k_2
// Constraint: k_1 % sizeof(T) == 0
// k_2 = k_1 / sizeof(T)
def PointerArithmeticAddImmPattern
  : Pat<(Clift_CastOp:$res (Clift_AddOp $ptr,
                                        (Clift_ImmediateOp:$imm $k)),
                           $kind),
        (Clift_PtrAddOp (createCastOp (getType $res), $ptr, $kind),
                        (createImmediateOp (getType $imm),
                                           (NativeCodeCall<"ptrOffsetDivMod($0, $1).Div"> $k, $res))),
        [(IsBitcastKind $kind),
         (Clift_AnyPointerType $res),
         (Constraint<CPred<"ptrOffsetDivMod($0, $1).Mod == 0">> $k, $res)]>;

// (T*)(i + n * k_1) -> (T*)i + n * k_2
// Constraint: k_1 % sizeof(T) == 0
// k_2 = k_1 / sizeof(T)
def PointerArithmeticAddMulImmPattern
  : Pat<(Clift_CastOp:$res (Clift_AddOp $ptr,
                                        (Clift_MulOp $n,
                                                     (Clift_ImmediateOp:$imm $k))),
                           $kind),
        (Clift_PtrAddOp (createCastOp (getType $res), $ptr, $kind),
                        (Clift_MulOp $n,
                                     (createImmediateOp (getType $imm),
                                                        (NativeCodeCall<"ptrOffsetDivMod($0, $1).Div"> $k, $res)))),
        [(IsBitcastKind $kind),
         (Constraint<CPred<"ptrOffsetDivMod($0, $1).Mod == 0">> $k, $res)]>;

// *(p + i) -> p[i]
def PointerSubscriptPattern
  : Pat<(Clift_IndirectionOp (Clift_PtrAddOp Clift_AnyPointerType:$ptr, $offset)),
        (Clift_SubscriptOp $ptr, $offset)>;

// (&x)->m -> x.m
def DirectAccessPattern
  : Pat<(Clift_AccessOp (Clift_AddressofOp $arg), $indirect, $member_index),
        (Clift_AccessOp $arg, (NativeCodeCall<"mlir::UnitAttr(nullptr)">), $member_index),
        [(Constraint<CPred<"not $0">> $indirect)]>;

#endif
