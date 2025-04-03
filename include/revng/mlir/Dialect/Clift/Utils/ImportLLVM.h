#pragma once

#include "llvm/IR/Module.h"

#include "revng/Model/Binary.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"

namespace mlir::clift {

mlir::OwningOpRef<clift::ModuleOp> importLLVM(mlir::MLIRContext *Context,
                                              const model::Binary &Model,
                                              const llvm::Module *Module);

} // namespace mlir::clift
