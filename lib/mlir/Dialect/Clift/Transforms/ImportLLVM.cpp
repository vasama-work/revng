//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IRReader/IRReader.h"

#include "mlir/Pass/Pass.h"

#include "revng/mlir/Dialect/Clift/Transforms/Passes.h"
#include "revng/mlir/Dialect/Clift/Utils/ImportLLVM.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTIMPORTLLVM
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

struct ImportLLVMPass : clift::impl::CliftImportLLVMBase<ImportLLVMPass> {
  void runOnOperation() override {
    llvm::LLVMContext LLVMContext;

    llvm::SMDiagnostic Diag;
    auto LLVMModule = llvm::parseIRFile(LLVMIRPath, Diag, LLVMContext);
    if (LLVMModule == nullptr) {
      Diag.print("", llvm::errs());
      signalPassFailure();
      return;
    }

    auto M = clift::importLLVM(&getContext(), *Model, LLVMModule.get());
    getOperation().getRegion().front().push_back(M.release().getOperation());
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
clift::createImportLLVMPass() {
  return std::make_unique<ImportLLVMPass>();
}
