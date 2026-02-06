#include "mlir/IR/PatternMatch.h"

/// Inverts if-statements whose then-branches are empty.
struct EmptyIfInversionPattern : mlir::OpRewritePattern<IfOp> {
  using OpRewritePattern::OpRewritePattern;

  void initialize() { setDebugName("empty-if-inversion"); }

  mlir::LogicalResult
  matchAndRewrite(IfOp If, mlir::PatternRewriter &Rewriter) const override {

    if (not clift::isEmptyRegionOrBlock(If.getThen()))
      return mlir::failure();

    if (clift::isEmptyRegionOrBlock(If.getElse()))
      return mlir::failure();

    invertIfStatement(Rewriter, If);
    return mlir::success();

  }
};
