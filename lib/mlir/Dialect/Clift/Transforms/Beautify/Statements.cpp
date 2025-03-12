//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"

#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng/mlir/Dialect/Clift/Transforms/Rewrites.h"

namespace clift = mlir::clift;

namespace {

// WIP: Merge with the one in CBackend.cpp.
template<typename Operation = mlir::Operation *>
static Operation getOnlyOperation(mlir::Region &R) {
  revng_assert(R.hasOneBlock());
  mlir::Block &B = R.front();
  auto Beg = B.begin();
  auto End = B.end();

  if (Beg == End)
    return {};

  mlir::Operation *Op = &*Beg;

  if (++Beg != End)
    return {};

  if constexpr (std::is_same_v<Operation, mlir::Operation *>) {
    return Op;
  } else {
    return mlir::dyn_cast<Operation>(Op);
  }
}

// WIP: Merge with the one in CliftOps.cpp.
static clift::YieldOp getExpressionYieldOp(mlir::Region &R) {
  if (R.empty())
    return {};

  mlir::Block &B = R.front();

  if (B.empty())
    return {};

  return mlir::dyn_cast<clift::YieldOp>(B.back());
}


template<typename CallableT>
static void replaceExpression(mlir::PatternRewriter &Rewriter,
                              mlir::Region &Region,
                              CallableT &&Callable) {
  auto Yield = getExpressionYieldOp(Region);
  revng_assert(Yield);

  mlir::OpBuilder::InsertionGuard Guard(Rewriter);
  Rewriter.setInsertionPoint(Yield.getOperation());

  mlir::Value Value = std::forward<CallableT>(Callable)(Yield.getValue());

  // WIP: Is this assign legal? Should we notify the rewriter somehow?
  Yield->getOpOperand(0).assign(Value);
}

template<typename CallableT>
static void mergeExpressionInto(mlir::PatternRewriter &Rewriter,
                                mlir::Region &SourceRegion,
                                mlir::Region &TargetRegion,
                                CallableT &&Callable) {
  auto SourceYield = getExpressionYieldOp(SourceRegion);
  revng_assert(SourceYield);

  auto TargetYield = getExpressionYieldOp(TargetRegion);
  revng_assert(TargetYield);

  mlir::OpBuilder::InsertionGuard Guard(Rewriter);
  Rewriter.setInsertionPoint(TargetYield.getOperation());

  mlir::Value Value = std::forward<CallableT>(Callable)(SourceYield.getValue(),
                                                        TargetYield.getValue());

  // WIP: Is this assign legal? Should we notify the rewriter somehow?
  TargetYield->getOpOperand(0).assign(Value);
}

static void inlineBlockBefore(mlir::PatternRewriter &Rewriter,
                              mlir::Block* Src,
                              mlir::Block* Dst,
                              mlir::Block::iterator Pos);


template<typename OpT = mlir::Operation *, typename PredicateT>
static OpT getTrailingOp(mlir::Region &Region, PredicateT &&Predicate) {
  if (Region.empty())
    return {};

  revng_assert(Region.hasOneBlock());
  mlir::Block &Block = Region.front();

  if (Block.empty())
    return {};

  mlir::Operation *Op = Block.back();
  if constexpr (std::is_same_v<OpT, mlir::Operation *>) {
    if (Callable(Op))
      return Op;
  } else {
    if (auto Op2 = mlir::dyn_cast<OpT>(Op)) {
      if (Callable(Op2))
        return Op2;
    }
  }

  return {};
}

static clift::StatementOpInterface
getTrailingJumpStatement(mlir::Region &Region) {
  return getTrailingOp<clift::StatementOpInterface>(Region, [](auto Op) {
    return Op->hasTrait<clift::NoFallThrough>();
  });
}

static mlir::Type getBooleanType(mlir::MLIRContext *Context) {
  return clift::PrimitiveType::get(Context,
                                   clift::PrimitiveKind::SignedKind,
                                   1,
                                   mlir::BoolAttr::get(Context, false));
}

static void invertIfStatement(mlir::PatternRewriter &Rewriter,
                              clift::IfOp If) {
  revng_assert(not If.getElse().empty());

  mlir::Region &Then = If.getThen();
  mlir::Region &Else = If.getElse();

  mlir::Block *ThenBlock = &Then.front();
  mlir::Block *ElseBlock = &Else.front();

  // WIP: Is this legal? Do we need to notify the rewriter somehow?
  ThenBlock.remove();
  ElseBlock.remove();

  Then.push_back(ElseBlock);
  Else.push_back(ThenBlock);

  replaceExpression(Rewriter, If.getCondition(), [&](mlir::Value Value) {
    auto BooleanType =
      clift::PrimitiveType::get(Rewriter.getContext(),
                                clift::PrimitiveKind::SignedKind,
                                1,
                                mlir::BoolAttr::get(Rewriter.getContext(),
                                                    false));

    auto Op = Rewriter.create<clift::LogicalNotOp>(If.getLoc(),
                                                   BooleanType,
                                                   Value);

    return Op.getResult();
  });
}


struct NestedIfCombiningPattern : mlir::RewritePattern {
  NestedIfCombiningPattern(mlir::MLIRContext *Context) :
    // WIP: Think more about the benefit
    RewritePattern("clift.if", 3, Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Root,
                  mlir::PatternRewriter &Rewriter) const override {
    auto OuterIf = mlir::cast<clift::IfOp>(Root);
    auto InnerIf = getOnlyOperation<clift::IfOp>(OuterIf.getThen());
    if (not InnerIf) {
      return notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if does not contain a nested clift.if.";
      });
    }

    mlir::Region &InnerElse = InnerIf.getElse();
    mlir::Region &OuterElse = OuterIf.getElse();

    if (not InnerElse.empty()) {
      if (OuterElse.empty()) {
        return notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
          Diag << "The inner clift.if has a non-empty else.";
        });
      }

      auto Goto = getOnlyOperation<clift::GoToOp>(InnerElse);
      if (not Goto) {
        return notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
          Diag << "The inner clift.if does not contain a nested clift.goto.";
        });
      }

      if (not isJumpToStartOf(Goto, OuterElse)) {
        return notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
          Diag << "The clift.goto does not jump to the outer clift.if else.";
        });
      }
    } else if (not OuterElse.empty()) {
      return notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
        Diag << "The outer clift.if has a non-empty else.";
      });
    }

    mergeExpressionInto(Rewriter,
                        InnerIf.getCondition(),
                        OuterIf.getCondition(),
                        [&](mlir::Value InnerValue, mlir::Value OuterValue) {
      mlir::Location Loc = Rewriter.getFusedLoc(OuterIf.getLoc(),
                                                InnerIf.getLoc());

      auto Op = Rewriter.create<clift::LogicalAndOp>(Loc,
                                                     getBooleanType(),
                                                     OuterValue,
                                                     InnerValue);

      return Op.getResult();
    });

    mlir::Region &InnerThen = InnerIf.getThen();
    mlir::Region &OuterThen = OuterIf.getThen();
    mlir::Block *InnerThenBlock = &InnerThen.front();

    // WIP: Should we notify the rewriter here?
    InnerThenBlock->remove();
    Rewriter.eraseBlock(OuterThen.front());
    OuterThen.push_back(InnerThenBlock);

    return mlir::success();
  }
};

struct EmptyElseEliminationPattern : mlir::RewritePattern {
  EmptyElseEliminationPattern(mlir::MLIRContext *Context) :
    // WIP: Think more about the benefit
    RewritePattern("clift.if", 3, Context) {}

  mlir::LogicalResult match(mlir::Operation *Root) const override {
    auto If = mlir::cast<clift::IfOp>(Root);

    if (not If.getElse().empty()) {
      return notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if has a non-empty else region.";
      });
    }

    return mlir::success();
  }

  void rewrite(mlir::PatternRewriter &Rewriter) const override {
    auto If = mlir::cast<clift::IfOp>(Root);
    Rewriter.eraseBlock(&If.getElse().front());
  }
};

struct TerminalIfElseUnwrappingPattern : mlir::RewritePattern {
  TerminalIfElseUnwrappingPattern(mlir::MLIRContext *Context) :
    // WIP: Think more about the benefit
    RewritePattern("clift.if", 3, Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Root,
                  mlir::PatternRewriter &Rewriter) const override {
    auto If = mlir::cast<clift::IfOp>(Root);

    if (If.getElse().empty()) {
      return notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if has an empty else-region.";
      });
    }

    if (getTrailingJumpStatement(If.getElse())) {
      invertIfStatement(Rewriter, If);
    } else if (not getTrailingJumpStatement(If.getThen())) {
      return notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if contains no trailing jumps.";
      });
    }

    mlir::Block *ElseBlock = If.getElse().front();

    inlineBlockBefore(Rewriter,
                      ElseBlock,
                      Root->getBlock(),
                      Root->getIterator());

    // WIP: Do we need to notify the rewriter about this?
    ElseBlock->erase();
  }
};

#if 0
struct OptimizedWhileConversionPattern : mlir::RewritePattern {
  OptimizedWhileConversionPattern(mlir::MLIRContext *Context) :
    // WIP: Think more about the benefit
    RewritePattern("clift.do_while", 3, Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Root,
                  mlir::PatternRewriter &Rewriter) const override {
    
  }
};
#endif

struct LoopDetectionPattern : mlir::RewritePattern {
  LoopDetectionPattern(mlir::MLIRContext *Context) :
    // WIP: Think more about the benefit
    RewritePattern("clift.goto", 3, Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Root,
                  mlir::RewriteRewriter &Rewriter) const override {
    
  }
};

} // namespace
