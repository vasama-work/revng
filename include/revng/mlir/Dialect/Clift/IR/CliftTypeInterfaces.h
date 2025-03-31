#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

namespace mlir::clift {

class FieldAttr;

} // namespace mlir::clift

#include "mlir/IR/Types.h"

#include "revng/mlir/Dialect/Clift/IR/CliftTypeInterfaces1.h.inc"
#include "revng/mlir/Dialect/Clift/IR/CliftTypeInterfaces2.h.inc"

// Prevent reordering:
#include "revng/mlir/Dialect/Clift/IR/CliftTypeInterfaces.h.inc"
