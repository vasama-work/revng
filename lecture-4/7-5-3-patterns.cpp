#include "revng/Clift/CliftOpHelpers.h"

void clift::invertIfStatement(mlir::PatternRewriter &Rewriter, IfOp If) {
  mlir::Region *Then = &If.getThen();
  mlir::Region *Else = &If.getElse();
  revng_assert(not Else->empty());

  invertBooleanExpression(Rewriter, If.getLoc(), If.getCondition());

  Rewriter.updateRootInPlace(If.getOperation(), [&]() {
    mlir::Block *ThenBlock = Then->empty() ? nullptr : &Then->front();
    mlir::Block *ElseBlock = &Else->front();

    if (ThenBlock != nullptr)
      Then->getBlocks().remove(ThenBlock);

    Else->getBlocks().remove(ElseBlock);
    Then->getBlocks().push_back(ElseBlock);

    if (ThenBlock != nullptr)
      Else->getBlocks().push_back(ThenBlock);
  });
}
