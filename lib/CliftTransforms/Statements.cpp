//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#pragma clang optimize off

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Passes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTBEAUTIFYSTATEMENTS
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;
using namespace clift;

namespace {

//===------------------ Future PatternRewriter functions ------------------===//

static void inlineBlockBefore(mlir::Block *Src,
                              mlir::Block *Dst,
                              mlir::Block::iterator Pos) {
  Dst->getOperations().splice(Pos, Src->getOperations());
}

#if 0
static void moveBlockBefore(mlir::PatternRewriter &Rewriter,
                            mlir::Block *Block,
                            mlir::Region *Region,
                            mlir::Region::iterator Pos) {
  revng_assert(Block->getParent() != Region);
  Rewriter.updateRootInPlace(If.getOperation(), [&]() {
    Region->getBlocks().splice(Pos,
                              Block->getParent()->getBlocks(),
                              Block->getIterator());
  });
}
#endif

//===------------------------------- Helpers ------------------------------===//

using clift::BlockPosition;

static bool isConstantCondition(mlir::Region &R,
                                std::optional<bool> Value = std::nullopt) {
  if (auto Yield = clift::getYieldOp(R)) {
    if (auto Immediate = Yield.getValue().getDefiningOp<clift::ImmediateOp>())
      return not Value or static_cast<bool>(Immediate.getValue()) == *Value;
  }
  return false;
}

#if 0
class FallthroughRange {
  class sentinel {
  public:
    explicit sentinel() = default;
  };

  class iterator {
  public:
    explicit iterator(mlir::Block::iterator Pos) : Pos(Pos) {}

    mlir::Operation &operator*() const {
      revng_assert(Op != nullptr);
      return *Op;
    }

    iterator &operator++() & {
      revng_assert(Op != nullptr);

      auto Pos = BlockPosition::get(Op);
      auto &[B, I] = Pos;

      while (I == B->end()) {
        mlir::Operation *ParentOp = B->getParentOp();
        if (mlir::isa<clift::FunctionOp, clift::LoopOpInterface>(ParentOp))
          return setEnd();

        
      }

      Op = Pos.getOperation();
      return *this;
    }

    [[nodiscard]] iterator operator++(int) & {
      iterator It = *this;
      ++*this;
      return It;
    }

    friend bool operator==(iterator const& It, sentinel) {
      return Op == nullptr;
    }

  private:
    mlir::Operation *Op;

    iterator &setEnd() {
      Op = nullptr;
      return *this;
    }
  };

public:
  explicit FallthroughRange(mlir::Block *Block, mlir::Block::iterator Pos)
    : Block(Block), Pos(Pos) {}

  iterator begin() const {
    return iterator(Pos != Block->end() ? &*Pos : nullptr);
  }

  sentinel end() const { return sentinel(); }

private:
  mlir::Block *Block;
  mlir::Block::iterator Pos;
};

static FallthroughRange walkOperationsAfter(mlir::Operation *Op) {
  return FallthroughRange(std::next(Op->getIterator()));
}
#endif

[[maybe_unused]] // WIP
static void
removeBlock(mlir::Block *Block) {
  revng_assert(Block->getParent() != nullptr);
  Block->getParent()->getBlocks().remove(Block);
}

template<typename CallableT>
static void replaceExpression(mlir::PatternRewriter &Rewriter,
                              mlir::Region &Region,
                              CallableT &&Callable) {
  auto Yield = clift::getYieldOp(Region);
  revng_assert(Yield);

  mlir::OpBuilder::InsertionGuard Guard(Rewriter);
  Rewriter.setInsertionPoint(Yield.getOperation());

  mlir::Value Value = std::forward<CallableT>(Callable)(Yield.getValue());

  // WIP: Is this assign legal? Should we notify the rewriter somehow?
  Yield->getOpOperand(0).set(Value);
}

template<typename CallableT>
static void mergeExpressionInto(mlir::PatternRewriter &Rewriter,
                                mlir::Region &SourceRegion,
                                mlir::Region &TargetRegion,
                                CallableT &&Callable) {
  auto SourceYield = clift::getYieldOp(SourceRegion);
  revng_assert(SourceYield);

  auto TargetYield = clift::getYieldOp(TargetRegion);
  revng_assert(TargetYield);

  mlir::OpBuilder::InsertionGuard Guard(Rewriter);
  Rewriter.setInsertionPoint(TargetYield.getOperation());

  mlir::Value SourceValue = SourceYield.getValue();
  mlir::Value TargetValue = TargetYield.getValue();

  SourceYield->erase();
  inlineBlockBefore(&SourceRegion.front(),
                    &TargetRegion.front(),
                    TargetYield->getIterator());

  mlir::Value Value = std::forward<CallableT>(Callable)(SourceValue,
                                                        TargetValue);

  // WIP: Is this assign legal? Should we notify the rewriter somehow?
  TargetYield->getOpOperand(0).set(Value);
}

static void moveBlocks(mlir::Region &Src, mlir::Region &Dst) {
  Dst.getBlocks().splice(Dst.end(), Src.getBlocks());
}

static mlir::Type getBooleanType(mlir::MLIRContext *Context) {
  return clift::PrimitiveType::get(Context,
                                   clift::PrimitiveKind::SignedKind,
                                   1);
}

#if 0
[[maybe_unused]] // WIP
static bool
isJumpToStartOf(clift::GotoOp Goto, mlir::Region &Region) {
  if (Region.empty())
    return false;

  for (mlir::Operation &Op : Region.front()) {
    auto AssignLabel = mlir::dyn_cast<clift::AssignLabelOp>(&Op);

    if (not AssignLabel)
      break;

    if (Goto.getLabel() == AssignLabel.getLabel())
      return true;
  }

  return false;
}
#endif

#if 0
static BlockPosition findJumpTarget(mlir::Operation *Op) {
  if (mlir::isa<clift::ReturnOp>(Op)) {
    auto F = Op->getParentOfType<clift::FunctionOp>();
    return BlockPosition::getEnd(F.getBody());
  }

  if (mlir::isa<clift::LoopContinueOp>(Op)) {
    auto L = Op->getParentOfType<clift::LoopOpInterface>();
    return BlockPosition::getEnd(L.getLoopRegion());
  }

  if (mlir::isa<clift::SwitchBreakOp>(Op)) {
    auto S = Op->getParentOfType<clift::SwitchOp>();
    return getFallthroughTarget(BlockPosition::getNext(S.getOperation()));
  }

  if (auto G = mlir::dyn_cast<clift::GotoOp>(Op))
    return getJumpTarget(G);

  return {};
}
#endif

static BlockPosition skipLabels(BlockPosition Position) {
  if (Position) {
    auto &[B, I] = Position;
    while (I != B->end() and mlir::isa<clift::AssignLabelOp>(*I))
      ++I;
  }
  return Position;
}


static void invertBooleanExpression(mlir::PatternRewriter &Rewriter,
                                    mlir::Location Loc,
                                    mlir::Region &R) {
  replaceExpression(Rewriter, R, [&](mlir::Value Value) {
    auto BooleanType = getBooleanType(Rewriter.getContext());
    auto Op = Rewriter.create<clift::LogicalNotOp>(Loc,
                                                   BooleanType,
                                                   Value);
    return Op.getResult();
  });
}

static void invertIfStatement(mlir::PatternRewriter &Rewriter, clift::IfOp If) {
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

//===--------------------- Statement rewrite patterns ---------------------===//

#if 0
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
      return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if does not contain a nested clift.if.";
      });
    }

    mlir::Region &InnerElse = InnerIf.getElse();
    mlir::Region &OuterElse = OuterIf.getElse();

    if (not InnerElse.empty()) {
      if (OuterElse.empty()) {
        return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
          Diag << "The inner clift.if has a non-empty else.";
        });
      }

      auto Goto = getOnlyOperation<clift::GotoOp>(InnerElse);
      if (not Goto) {
        return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
          Diag << "The inner clift.if does not contain a nested clift.goto.";
        });
      }

      if (not isJumpToStartOf(Goto, OuterElse)) {
        return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
          Diag << "The clift.goto does not jump to the outer clift.if else.";
        });
      }
    } else if (not OuterElse.empty()) {
      return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
        Diag << "The outer clift.if has a non-empty else.";
      });
    }

    mergeExpressionInto(Rewriter,
                        InnerIf.getCondition(),
                        OuterIf.getCondition(),
                        [&](mlir::Value InnerValue, mlir::Value OuterValue) {
      mlir::Location Loc = Rewriter.getFusedLoc(OuterIf.getLoc(),
                                                InnerIf.getLoc());

      auto BooleanType = getBooleanType(Rewriter.getContext());
      auto Op = Rewriter.create<clift::LogicalAndOp>(Loc,
                                                     BooleanType,
                                                     OuterValue,
                                                     InnerValue);

      return Op.getResult();
    });

    mlir::Region &InnerThen = InnerIf.getThen();
    mlir::Region &OuterThen = OuterIf.getThen();
    mlir::Block *InnerThenBlock = &InnerThen.front();

    // WIP: Should we notify the rewriter here?
    removeBlock(InnerThenBlock);
    Rewriter.eraseBlock(&OuterThen.front());
    OuterThen.push_back(InnerThenBlock);

    return mlir::success();
  }
};
#endif

struct IfAndCombiningPattern : mlir::OpRewritePattern<clift::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  void initialize() {
    setDebugName("if-and-combining");
  }

  mlir::LogicalResult
  matchAndRewriteImpl2(clift::IfOp OuterIf,
                       mlir::Region &OuterBranch1,
                       mlir::Region &OuterBranch2,
                       clift::IfOp InnerIf,
                       mlir::Region &InnerBranch1,
                       mlir::Region &InnerBranch2,
                       mlir::PatternRewriter &Rewriter) const {
    auto Goto = getOnlyOperation<clift::GotoOp>(InnerBranch2);
    if (not Goto)
      return mlir::failure();

    auto ElseTarget = OuterBranch2.empty() ?
                        BlockPosition::getNext(OuterIf) :
                        BlockPosition::getBegin(OuterBranch2);

    if (getJumpTarget(Goto) != ElseTarget)
      return mlir::failure();

    bool IsOuterInverted = &OuterBranch1 == &OuterIf.getElse();
    bool IsInnerInverted = &InnerBranch1 == &InnerIf.getElse();

    if (IsOuterInverted ^ IsInnerInverted) {
      invertBooleanExpression(Rewriter,
                              InnerIf.getLoc(),
                              InnerIf.getCondition());
    }

    auto Merge = [&](mlir::Value Inner, mlir::Value Outer) -> mlir::Value {
      auto BooleanType = getBooleanType(Rewriter.getContext());

      return Rewriter.create<LogicalAndOp>(InnerIf.getLoc(),
                                           BooleanType,
                                           Outer,
                                           Inner);
    };

    mergeExpressionInto(Rewriter,
                        InnerIf.getCondition(),
                        OuterIf.getCondition(),
                        Merge);

    if (not InnerBranch1.empty()) {
      inlineBlockBefore(&InnerBranch1.front(),
                        &OuterBranch1.front(),
                        OuterBranch1.front().begin());
    }

    Rewriter.eraseOp(InnerIf);

    return mlir::success();
  }

  mlir::LogicalResult
  matchAndRewriteImpl1(clift::IfOp OuterIf,
                       mlir::Region &OuterBranch1,
                       mlir::Region &OuterBranch2,
                       mlir::PatternRewriter &Rewriter) const {
    if (auto InnerIf = getOnlyOperation<clift::IfOp>(OuterBranch1)) {
      if (matchAndRewriteImpl2(OuterIf,
                               OuterBranch1,
                               OuterBranch2,
                               InnerIf,
                               InnerIf.getThen(),
                               InnerIf.getElse(),
                               Rewriter).succeeded())
        return mlir::success();

      if (matchAndRewriteImpl2(OuterIf,
                               OuterBranch1,
                               OuterBranch2,
                               InnerIf,
                               InnerIf.getElse(),
                               InnerIf.getThen(),
                               Rewriter).succeeded())
        return mlir::success();
    }

    return mlir::failure();
  }

  mlir::LogicalResult
  matchAndRewrite(clift::IfOp OuterIf,
                  mlir::PatternRewriter &Rewriter) const override {
    if (matchAndRewriteImpl1(OuterIf,
                             OuterIf.getThen(),
                             OuterIf.getElse(),
                             Rewriter).succeeded())
      return mlir::success();

    if (matchAndRewriteImpl1(OuterIf,
                             OuterIf.getElse(),
                             OuterIf.getThen(),
                             Rewriter).succeeded())
      return mlir::success();

    return mlir::failure();
  }
};

struct LabelCombiningPattern : mlir::OpRewritePattern<clift::AssignLabelOp> {
  using OpRewritePattern::OpRewritePattern;

  void initialize() {
    setDebugName("label-combining");
  }

  mlir::LogicalResult
  matchAndRewrite(clift::AssignLabelOp AssignLabel,
                  mlir::PatternRewriter &Rewriter) const override {
    mlir::Block::iterator Pos = std::next(AssignLabel->getIterator());

    if (Pos == AssignLabel->getBlock()->end())
      return mlir::failure();

    auto NextAssignLabel = mlir::dyn_cast<clift::AssignLabelOp>(&*Pos);
    if (not NextAssignLabel)
      return mlir::failure();

    Rewriter.replaceAllUsesWith(NextAssignLabel.getLabel(),
                                AssignLabel.getLabel());

    Rewriter.eraseOp(NextAssignLabel.getOperation());

    return mlir::success();
  }
};

struct BranchEqualizationPattern
  : mlir::OpInterfaceRewritePattern<clift::BranchOpInterface> {

  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  void initialize() {
    setDebugName("branch-equalization");
  }

  mlir::LogicalResult
  matchAndRewrite(clift::BranchOpInterface Branch,
                  mlir::PatternRewriter &Rewriter) const override {
    mlir::Region *FallthroughRegion = nullptr;

    for (mlir::Region &R : Branch.getBranchRegions()) {
      if (clift::isIndirectlyNoFallthrough(R))
        continue;

      if (FallthroughRegion)
        return mlir::failure();

      FallthroughRegion = &R;
    }

    if (not FallthroughRegion)
      return mlir::failure();

    mlir::Block *Outer = Branch->getBlock();
    mlir::Block::iterator Beg = std::next(Branch->getIterator());
    mlir::Block::iterator End = Outer->end();

    // Skip backwards over any trailing labels:
    while (Beg != End and mlir::isa<clift::AssignLabelOp>(&*std::prev(End)))
      --End;

    if (Beg == End)
      return mlir::failure();

    //if (not mlir::cast<clift::StatementOpInterface>(std::prev(End)).isIndirectlyNoFallthrough())
    //  return mlir::failure();

    if (FallthroughRegion->empty())
      FallthroughRegion->emplaceBlock();

    mlir::Block *Inner = &FallthroughRegion->front();

    Inner->getOperations().splice(Inner->end(),
                                  Outer->getOperations(),
                                  Beg,
                                  End);

    return mlir::success();
  }
};

struct EmptyIfInversionPattern : mlir::OpRewritePattern<clift::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::IfOp If,
                  mlir::PatternRewriter &Rewriter) const override {
    if (not clift::isEmptyRegionOrBlock(If.getThen()))
      return mlir::failure();

    if (clift::isEmptyRegionOrBlock(If.getElse()))
      return mlir::failure();

    invertIfStatement(Rewriter, If);
    return mlir::success();
  }
};

struct EmptyElseEliminationPattern : mlir::OpRewritePattern<clift::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::IfOp If,
                  mlir::PatternRewriter &Rewriter) const override {
    if (not clift::hasEmptyBlock(If.getElse())) {
      return Rewriter.notifyMatchFailure(If, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if has a non-empty else-region.";
      });
    }

    Rewriter.eraseBlock(&If.getElse().front());
    return mlir::success();
  }
};

struct TerminalIfElseUnwrappingPattern : mlir::OpRewritePattern<clift::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::IfOp If,
                  mlir::PatternRewriter &Rewriter) const override {
    if (clift::isEmptyRegionOrBlock(If.getElse())) {
      return Rewriter.notifyMatchFailure(If, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if has an empty else-region or block.";
      });
    }

    auto ThenFallthrough = not clift::isIndirectlyNoFallthrough(If.getThen());
    auto ElseFallthrough = not clift::isIndirectlyNoFallthrough(If.getElse());

    if (ThenFallthrough and ElseFallthrough) {
      return Rewriter.notifyMatchFailure(If, [&](mlir::Diagnostic &Diag) {
        Diag << "Both branches of the clift.if fall through.";
      });
    }

    if (not ThenFallthrough and not ElseFallthrough) {
      return Rewriter.notifyMatchFailure(If, [&](mlir::Diagnostic &Diag) {
        Diag << "Neither branch of the clift.if falls through.";
      });
    }

    if (ThenFallthrough)
      invertIfStatement(Rewriter, If);

    mlir::Block *ElseBlock = &If.getElse().front();
    Rewriter.updateRootInPlace(If.getOperation(), [&]() {
      inlineBlockBefore(ElseBlock,
                        If->getBlock(),
                        std::next(If->getIterator()));
    });

    Rewriter.eraseBlock(ElseBlock);
    return mlir::success();
  }
};

struct TrivialJumpEliminationPattern
  : mlir::OpInterfaceRewritePattern<clift::JumpStatementOpInterface> {

  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  void initialize() {
    setDebugName("trivial-jump-elimination");
  }

  static BlockPosition getFallthroughTarget(BlockPosition Position) {
    auto &[B, I] = Position;

    while (true) {
      Position = skipLabels(Position);

      if (I != B->end())
        break;

      mlir::Operation *ParentOp = B->getParentOp();
      if (not mlir::isa<clift::BranchOpInterface>(ParentOp))
        break;

      Position = BlockPosition::getNext(ParentOp);
    }

    return Position;
  }

  mlir::LogicalResult
  matchAndRewrite(clift::JumpStatementOpInterface Jump,
                  mlir::PatternRewriter &Rewriter) const override {
    auto JumpTarget = getFallthroughTarget(clift::getJumpTarget(Jump));
    auto FallTarget = getFallthroughTarget(BlockPosition::getNext(Jump));

    if (JumpTarget != FallTarget)
      return mlir::failure();

    Rewriter.eraseOp(Jump);
    return mlir::success();
  }
};

struct WhileConditionHoistingPattern : mlir::OpRewritePattern<clift::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  void initialize() {
    setDebugName("while-condition-hoisting");
  }

  mlir::LogicalResult
  matchAndRewrite(clift::WhileOp While,
                  mlir::PatternRewriter &Rewriter) const override {
    if (not isConstantCondition(While.getCondition(), true))
      return fprintf(stderr, "LINE: %d\n", __LINE__), mlir::failure();

    auto If = clift::getLeadingOp<clift::IfOp>(While.getBody());
    if (not If)
      return fprintf(stderr, "LINE: %d\n", __LINE__), mlir::failure();

    auto BreakTarget = BlockPosition::getNext(While);
    auto HasBreak = [&BreakTarget](mlir::Region &R) {
      if (auto Jump = getOnlyOperation<JumpStatementOpInterface>(R))
        return getJumpTarget(Jump) == BreakTarget;
      return false;
    };

    bool ThenHasBreak = HasBreak(If.getThen());
    bool ElseHasBreak = HasBreak(If.getElse());

    if (not (ThenHasBreak ^ ElseHasBreak))
      return fprintf(stderr, "LINE: %d\n", __LINE__), mlir::failure();

    if (ThenHasBreak)
      invertBooleanExpression(Rewriter, If.getLoc(), If.getCondition());

    mlir::Region &BreakRegion = ThenHasBreak ? If.getThen() : If.getElse();
    mlir::Region &OtherRegion = ThenHasBreak ? If.getElse() : If.getThen();

    Rewriter.eraseOp(getOnlyOperation(BreakRegion));

    While.getCondition().getBlocks().clear();
    moveBlocks(If.getCondition(), While.getCondition());

    if (not OtherRegion.empty()) {
      inlineBlockBefore(&OtherRegion.front(),
                        If->getBlock(),
                        std::next(If->getIterator()));
    }

    Rewriter.eraseOp(If);

    return mlir::success();
  }
};

struct DoWhileConversionPattern : mlir::OpRewritePattern<clift::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::WhileOp While,
                  mlir::PatternRewriter &Rewriter) const override {
    if (not isConstantCondition(While.getCondition(), true))
      return mlir::failure();

    auto If = clift::getTrailingOp<clift::IfOp>(While.getBody());
    if (not If)
      return mlir::failure();

    auto IsBreak = [&While](mlir::Region &R) -> bool {
      auto Last = clift::getOnlyOperation<JumpStatementOpInterface>(R);
      return Last and getJumpTarget(Last) == BlockPosition::getNext(While);
    };

    bool ThenBreak = IsBreak(If.getThen());
    bool ElseBreak = IsBreak(If.getElse());

    if (not ThenBreak and not ElseBreak)
      return mlir::failure();

    if (ThenBreak) {
      // With the break in the true branch, the condition must be inverted.
      invertBooleanExpression(Rewriter, If.getLoc(), If.getCondition());
    }

    mlir::Region &OtherRegion = ThenBreak ? If.getElse() : If.getThen();
    if (not OtherRegion.empty()) {
      mlir::Block *Body = &While.getBody().front();
      inlineBlockBefore(&OtherRegion.front(), Body, Body->end());
    }

    Rewriter.setInsertionPointAfter(While);
    auto DoWhile = Rewriter.create<clift::DoWhileOp>(While.getLoc());

    moveBlocks(If.getCondition(), DoWhile.getCondition());
    moveBlocks(While.getBody(), DoWhile.getBody());

    Rewriter.eraseOp(While);
    Rewriter.eraseOp(If);

#if 0
    auto Goto = clift::getTrailingOp<clift::GotoOp>(If.getThen());
    if (not Goto or getJumpTarget(Goto) != BlockPosition::getNext(While))
      return mlir::failure();

    // With the break in the true branch, the condition must be inverted.
    invertBooleanExpression(Rewriter, If.getLoc(), If.getCondition());

    Rewriter.eraseOp(Goto);

    Rewriter.setInsertionPointAfter(While);
    auto DoWhile = Rewriter.create<clift::DoWhileOp>(While.getLoc());

    moveBlocks(If.getCondition(), DoWhile.getCondition());
    moveBlocks(While.getBody(), DoWhile.getBody());

    auto &OuterOperations = While->getBlock()->getOperations();
    OuterOperations.splice(std::next(DoWhile->getIterator()),
                           If.getThen().front().getOperations());

    Rewriter.eraseOp(While);
    Rewriter.eraseOp(If);

#endif
    return mlir::success();
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

struct BeautifyStatementsPass
  : clift::impl::CliftBeautifyStatementsBase<BeautifyStatementsPass> {
  mlir::LogicalResult initialize(mlir::MLIRContext *Context) override {
    mlir::RewritePatternSet Set(Context);

    Set.add(clift::MakeLabelOp::canonicalize);

    Set.add<IfAndCombiningPattern>(Context);
    Set.add<LabelCombiningPattern>(Context);
    // Set.add<NestedIfCombiningPattern>(Context);
    Set.add<BranchEqualizationPattern>(Context);
    Set.add<EmptyIfInversionPattern>(Context);
    Set.add<EmptyElseEliminationPattern>(Context);
    //Set.add<TerminalIfElseUnwrappingPattern>(Context);
    Set.add<TrivialJumpEliminationPattern>(Context);
    //Set.add<WhileConditionHoistingPattern>(Context);
    Set.add<DoWhileConversionPattern>(Context);

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

clift::PassPtr<clift::FunctionOp> clift::createBeautifyStatementsPass() {
  return std::make_unique<BeautifyStatementsPass>();
}
