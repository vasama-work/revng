#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

#include "revng/Clift/CliftOpTraits.h"
#include "revng/Clift/CliftTypes.h"

namespace mlir::clift {

class LabelAssignmentOpInterface;

namespace impl {

/// Returns the break or continue label value (if any), depending on the
/// specified index. (Index=0 for break, Index=1 for continue).
template<typename LoopOpT>
mlir::Value getLoopLabel(LoopOpT Op, unsigned Index) {
  unsigned Mask = Op.getLabelMask();
  unsigned Flag = 1 << Index;

  if ((Mask & Flag) == 0)
    return nullptr;

  // The operand index is given by the value of the lower flag (if any):
  return Op->getOperand(Mask & Flag >> 1);
}

LabelAssignmentOpInterface getLabelAssignment(mlir::Value Label);

} // namespace impl
} // namespace mlir::clift

// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesBasic.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesLabel.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesJump.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesFlow.h.inc"

namespace mlir::clift {

bool isLvalueExpression(mlir::Value Value);

} // namespace mlir::clift
