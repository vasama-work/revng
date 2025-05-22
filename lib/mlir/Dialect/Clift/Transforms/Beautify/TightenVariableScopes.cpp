//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"

#include "mlir/Pass/Pass.h"

#include "revng/mlir/Dialect/Clift/IR/CliftOpHelpers.h"
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTTIGHTENVARIABLESCOPES
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

// Stores the position and nesting level for each local variable definition
struct LocalVariableLocation {
  clift::BlockPosition Position;
  unsigned NestingLevel = 0;
};

// Custom walk function that provides nesting level
template<typename CallbackT>
static void walkWithNestingLevel(mlir::Region *Region,
                                 CallbackT Callback,
                                 unsigned NestingLevel = 0) {
  for (mlir::Block &Block : *Region) {
    for (mlir::Operation &Op : Block) {
      // Post-order walk through the nested regions, incrementing the nesting
      // level for inner region
      for (mlir::Region &NestedRegion : Op.getRegions()) {
        walkWithNestingLevel(&NestedRegion, Callback, NestingLevel + 1);
      }

      // Then call the callback on the operation itself
      Callback(&Op, NestingLevel);
    }
  }
}

// Merges the current location for a variable definition with the new use
// position
static void updateVariableLocation(LocalVariableLocation &VarLoc,
                                   clift::BlockPosition NewPosition,
                                   unsigned NewLevel) {
  mlir::Region *CurrentRegion = VarLoc.Position.Block->getParent();
  mlir::Region *NewPosRegion = NewPosition.Block->getParent();

  unsigned CurrentLevel = VarLoc.NestingLevel;

  // If the new position is at a lower nesting level, we surely need to
  // go up from the current region at least until the levels are equal,
  // in order to find the common ancestor
  while (CurrentLevel > NewLevel) {
    CurrentRegion = CurrentRegion->getParentRegion();
    CurrentLevel--;
  }

  // Conversely, if the new position is at a higher nesting level,
  // we need to go up from it until we reach the current region's level
  while (NewLevel > CurrentLevel) {
    NewPosRegion = NewPosRegion->getParentRegion();
    NewLevel--;
  }

  // We can now check if the current region is identical to the
  // parent region of the new position
  while (CurrentRegion != NewPosRegion) {
    // If they are not the same, we need to go up from both of them
    // until we reach the common ancestor region
    CurrentRegion = CurrentRegion->getParentRegion();
    NewPosRegion = NewPosRegion->getParentRegion();
    CurrentLevel--;
  }

  // Now we can find the common ancestor operation that is the closest to the
  // current variable location by walking up the tree until we match the
  // nesting level
  mlir::Operation *CurrentOp = VarLoc.Position.getOperation();
  for (unsigned i = 0; i < (VarLoc.NestingLevel - CurrentLevel); ++i) {
    CurrentOp = CurrentOp->getParentOp();
  }

  // Update the variable location to the newfound common ancestor
  VarLoc.Position = clift::BlockPosition::get(CurrentOp);
  VarLoc.NestingLevel = CurrentLevel;
}

struct TightenVariableScopePass
  : clift::impl::CliftTightenVariableScopesBase<TightenVariableScopePass> {
  void runOnOperation() override {
    clift::FunctionOp FunctionOp = getOperation();

    // Store the function's local variables in a map, associated with their
    // optimal position in the MLIR tree
    llvm::SmallDenseMap<clift::LocalVariableOp, LocalVariableLocation>
      LocalVariables;

    auto WalkCallback = [&](mlir::Operation *Op, unsigned OpNestingLevel) {
      // For each operand of the operation, we check if it is a local
      // variable.
      for (mlir::Value Operand : Op->getOperands()) {
        auto LocalVarOp = Operand.getDefiningOp<clift::LocalVariableOp>();

        // If the defining op is not a local variable, we can skip it
        if (not LocalVarOp) {
          continue;
        }

        // Local variables with an initializer cannot be moved
        if (not LocalVarOp.getInitializer().empty()) {
          continue;
        }

        auto [Iterator, Inserted] = LocalVariables.try_emplace(LocalVarOp);

        if (Inserted) {
          // If the local variable was not already in the map, mark
          // its optimal position as right before the parent operation
          // of the current user.
          Iterator->second
            .Position = clift::BlockPosition::get(Op->getParentOp());
          Iterator->second.NestingLevel = OpNestingLevel - 1;
        } else {
          // If the local variable is already in the map, update its
          // location to find the common ancestor position that covers
          // the new user
          auto OpPosition = clift::BlockPosition::get(Op);
          updateVariableLocation(Iterator->second, OpPosition, OpNestingLevel);
        }
      }
    };

    // Walk the function body to identify local variable uses, along with their
    // nesting levels.
    walkWithNestingLevel(&FunctionOp.getBody(), WalkCallback);

    // Move each local variable to its optimal position
    for (const auto &[LocalVarOp, Location] : LocalVariables) {
      // Get the target operation where we want to insert the variable
      mlir::Operation *TargetOp = Location.Position.getOperation();

      // Check if moving would be a no-op (already in the right place)
      if (TargetOp->getPrevNode() == LocalVarOp) {
        continue;
      }

      // Move the local variable declaration to the optimal position
      LocalVarOp->moveBefore(TargetOp);
    }
  }
};

} // namespace

clift::PassPtr<clift::FunctionOp> clift::createTightenVariableScopePass() {
  return std::make_unique<TightenVariableScopePass>();
}
