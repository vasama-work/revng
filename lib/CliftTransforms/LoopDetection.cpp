#include <ranges>

#include "llvm/ADT/MapVector.h"

#include "mlir/Pass/Pass.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Passes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTLOOPDETECTION
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;
using namespace clift;

namespace {

static bool isBackwardGoto(GotoOp Goto, AssignLabelOp Label) {
  mlir::Block *LB = Label->getBlock();

  mlir::Operation *Op = Goto.getOperation();
  while (Op->getBlock() != LB) {
    mlir::Operation *ParentOp = Op->getParentOp();

    if (not mlir::isa<BranchOpInterface>(ParentOp))
      return false;

    Op = ParentOp;
  }

  mlir::Block::iterator Pos = Op->getIterator();
  for (auto I = Label->getIterator(), E = LB->end(); I != Pos; ++I) {
    if (I == E)
      return false;
  }

  return true;
}

static mlir::Block *extractAsBlock(mlir::Block *Block,
                                   mlir::Block::iterator Begin,
                                   mlir::Block::iterator End) {
  mlir::Block *NewBlock = new mlir::Block();

  NewBlock->getOperations().splice(NewBlock->begin(),
                                   Block->getOperations(),
                                   Begin,
                                   End);

  return NewBlock;
}

static void createLoop(FunctionOp Function,
                       AssignLabelOp LoopLabel,
                       llvm::ArrayRef<GotoOp> Gotos) {
  mlir::Location LoopLoc = mlir::UnknownLoc::get(Function->getContext());

  mlir::Block *OuterBlock = LoopLabel->getBlock();

  // True if the last goto is in the scope of the label.
  bool GotoInLabelScope = Gotos.back()->getBlock() == OuterBlock;
  // llvm::errs() << "GotoInLabelScope: " << GotoInLabelScope << "\n";

  auto FindLoopEnd = [&](mlir::Operation *BoundingOp) -> mlir::Block::iterator {
    while (BoundingOp->getBlock() != OuterBlock)
      BoundingOp = BoundingOp->getParentOp();

    return std::next(BoundingOp->getIterator());
  };

  mlir::Block *InnerBlock = extractAsBlock(OuterBlock,
                                           std::next(LoopLabel->getIterator()),
                                           FindLoopEnd(Gotos.back()));

  mlir::OpBuilder Builder(Function.getBody());

  auto CreateLabel = [&](mlir::Block *Block, mlir::Block::iterator Pos) {
    mlir::OpBuilder::InsertionGuard Guard(Builder);

    Builder.setInsertionPointToStart(&Function.getBody().front());
    mlir::Value Label = Builder.create<MakeLabelOp>(LoopLoc);

    Builder.setInsertionPoint(Block, Pos);
    Builder.create<AssignLabelOp>(LoopLoc, Label);

    return Label;
  };

  Builder.setInsertionPoint(OuterBlock, std::next(LoopLabel->getIterator()));
  auto Loop = Builder.create<WhileOp>(LoopLoc);

  Builder.setInsertionPointToStart(&Loop.getCondition().emplaceBlock());
  auto IntType = PrimitiveType::get(Builder.getContext(),
                                           PrimitiveKind::SignedKind,
                                           //PrimitiveKind::GenericKind,
                                           /*Size=*/4);
  Builder.create<YieldOp>(LoopLoc,
                                 Builder.create<ImmediateOp>(LoopLoc,
                                                                    IntType,
                                                                    1));

  Loop.getBody().push_back(InnerBlock);

  if (GotoInLabelScope) {
    // A loop-goto at the end of the loop body can be erased.
    revng_assert(Gotos.back()->getBlock() == InnerBlock);
    revng_assert(std::next(Gotos.back()->getIterator()) == InnerBlock->end());
    Gotos.back()->erase();

    // The last loop-goto has been erased and is no longer needed:
    Gotos = Gotos.drop_back(1);
  } else {
    // If the last loop-goto is conditional, a break must be inserted at the end
    // of the loop body.
    mlir::Value BreakLabel = CreateLabel(OuterBlock,
                                         std::next(Loop->getIterator()));

    Builder.setInsertionPointToEnd(InnerBlock);
    Builder.create<GotoOp>(LoopLoc, BreakLabel);
  }

  // If there still exist any loop-gotos, a continue label is created at the end
  // of the loop body and the targets of all remaining loop-gotos are replaced.
  if (not Gotos.empty()) {
    mlir::Value ContinueLabel = CreateLabel(InnerBlock, InnerBlock->end());

    for (GotoOp Goto : Gotos)
      Goto.setOperand(ContinueLabel);
  }
}

static void createLoops(FunctionOp Function) {
  // Maps each label assignment to the backwards gotos targeting it.
  llvm::MapVector<AssignLabelOp, llvm::SmallVector<GotoOp, 2>> Labels;

  Function->walk([&](mlir::Operation *Op) {
    if (auto Label = mlir::dyn_cast<AssignLabelOp>(Op)) {
      Labels.insert({ Label, {} });
    } else if (auto Goto = mlir::dyn_cast<GotoOp>(Op)) {
      mlir::Operation *Assignment = Goto.getLabelAssignmentOp();
      if (auto A = mlir::dyn_cast<AssignLabelOp>(Assignment)) {
        if (auto Iterator = Labels.find(A); Iterator != Labels.end()) {
          if (isBackwardGoto(Goto, Iterator->first))
            Iterator->second.push_back(Goto);
        }
      }
    }
  });

  for (const auto &[Label, Gotos] : std::views::reverse(Labels)) {
    if (not Gotos.empty())
      createLoop(Function, Label, Gotos);
  }
}

struct LoopDetectionPass
  : impl::CliftLoopDetectionBase<LoopDetectionPass> {

  void runOnOperation() override {
    createLoops(getOperation());
  }
};

} // namespace

PassPtr<FunctionOp> clift::createLoopDetectionPass() {
  return std::make_unique<LoopDetectionPass>();
}
