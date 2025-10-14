//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Passes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTEXPRESSIONROOTSIMPLIFICATION
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;
using namespace clift;

namespace {

static bool pointeesAreOfSameSize(mlir::Type L, mlir::Type R) {
  auto LT = mlir::cast<PointerType>(L).getPointeeType();
  auto RT = mlir::cast<PointerType>(R).getPointeeType();
  return LT.getByteSize() == RT.getByteSize();
}

static bool isDiscardedExpression(mlir::Value Value) {
  mlir::Operation *User = getOnlyUser(Value);
  revng_assert(User != nullptr);

  if (auto Yield = mlir::dyn_cast<YieldOp>(User)) {
    mlir::Operation *Op = Yield->getParentOp();

    if (mlir::isa<ExpressionStatementOp>(Op))
      return true;

    if (auto For = mlir::dyn_cast<ForOp>(Op)) {
      mlir::Region *R = Yield->getParentRegion();
      if (R == &For.getExpression())
        return true;
    }
  }
  return false;
}

static bool isUsedInConditional(mlir::Value Value) {
  mlir::Operation *User = getOnlyUser(Value);
  revng_assert(User != nullptr);

  if (auto Yield = mlir::dyn_cast<YieldOp>(User)) {
    mlir::Operation *Op = Yield->getParentOp();

    if (mlir::isa<BranchOpInterface>(Op))
      return true;

    if (auto Loop = mlir::dyn_cast<LoopOpInterface>(Op)) {
      if (auto For = mlir::dyn_cast<ForOp>(Loop.getOperation()))
        return Yield->getParentRegion() == &For.getCondition();

      return true;
    }

    return false;
  }

  if (auto E = mlir::dyn_cast<TernaryOp>(User))
    return Value == E.getCondition();

  return mlir::isa<LogicalNotOp, LogicalAndOp, LogicalOrOp>(User);
}

#include "revng/CliftTransforms/ExpressionRootSimplification.h.inc"

template<typename T>
using PassBase = impl::CliftExpressionRootSimplificationBase<T>;

struct ExpressionRootSimplificationPass
  : PassBase<ExpressionRootSimplificationPass> {

  mlir::LogicalResult initialize(mlir::MLIRContext *Context) override {
    mlir::RewritePatternSet Set(Context);
    populateWithGenerated(Set);

    Patterns = mlir::FrozenRewritePatternSet(std::move(Set),
                                             disabledPatterns,
                                             enabledPatterns);

    return mlir::success();
  }

  void runOnOperation() override {
    if (mlir::applyPatternsAndFoldGreedily(getOperation(), Patterns)
          .failed())
      signalPassFailure();
  }

  mlir::FrozenRewritePatternSet Patterns;
};

} // namespace

PassPtr<FunctionOp> clift::createExpressionRootSimplificationPass() {
  return std::make_unique<ExpressionRootSimplificationPass>();
}
