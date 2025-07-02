//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_LEGALIZATION_EXPRESSION_REWRITES
#define MLIR_CLIFT_LEGALIZATION_EXPRESSION_REWRITES

include "mlir/IR/PatternBase.td"

include "revng/mlir/Dialect/Clift/IR/CliftOps.td"

def ValuesAreEqual : Constraint<CPred<[{ $0 == $1 }]>>;

def ValuesAreNotEqual : Constraint<CPred<[{ $0 == $1 }]>>;

def TypesAreEquivalent
  : Constraint<CPred<[{ clift::equivalent($0.getType(), $1.getType()) }]>>;

def TypesAreNotEquivalent
  : Constraint<CPred<[{ not clift::equivalent($0.getType(), $1.getType()) }]>>;

def IsBitcastKind
  : Constraint<CPred<[{ $0.getValue() == clift::CastKind::Bitcast }]>>;


def getType : NativeCodeCall<[{ $0.getType() }]>;

def getPointeeType : NativeCodeCall<[{
  mlir::cast<clift::PointerType>($0.getType()).getPointeeType()
}]>;

def getBitcastKind : NativeCodeCall<[{ clift::CastKind::Bitcast }]>;

def createImmediateOp : NativeCodeCall<[{
  $_builder.create<clift::ImmediateOp>($_loc, $0, $1)
}]>;

def createCastOp : NativeCodeCall<[{
  $_builder.create<clift::CastOp>($_loc, $0, $1, $2)
}]>;

def createAddressofOp : NativeCodeCall<[{
  $_builder.create<clift::AddressofOp>($_loc, $0, $1)
}]>;

//===--------------------------- Math identities --------------------------===//

// *x -> *(T*)x
// Constraint: the type of x is pointerN_t(T)
def PointerNIndirectionPattern
  : Pat<(Clift_IndirectionOp $arg),
        (Clift_IndirectionOp (createPointerResizeCastOp $arg))>;

// x[y] -> ((T*)x)[y]
// Constraint: the type of x is pointerN_t(T)
def NegNegPattern
  : Pat<(Clift_NegOp (Clift_NegOp $arg)),
        (replaceWithValue $arg)>;

// 

#endif