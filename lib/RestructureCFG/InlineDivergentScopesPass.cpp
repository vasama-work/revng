//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/GenericDomTree.h"

#include "revng/RestructureCFG/InlineDivergentScopesPass.h"
#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"
#include "revng/Support/Assert.h"
#include "revng/Support/GraphAlgorithms.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

// Debug logger
Logger<> InlineDivergentScopesPassLogger("ids");

// TODO: remove this by not using a global variable, or at least improving the
//       initialization
BasicBlock *PlaceholderTarget = nullptr;

static bool isConditional(BasicBlock *Node) {
  auto Successors = llvm::children<Scope<BasicBlock *>>(Node);
  size_t NumSuccessors = std::distance(Successors.begin(), Successors.end());
  if (NumSuccessors >= 2) {
    return true;
  } else {
    return false;
  }
}

/// Helper function which detects
static bool isExit(BasicBlock *Node) {
  auto Successors = llvm::children<Scope<BasicBlock *>>(Node);
  return std::ranges::empty(Successors);
}

/// Helper function which simplifies all the terminators containing `nullptr`s
/// left in this state by the core of the IDS implementation
static void simplifyTerminator(BasicBlock *BB) {
  Instruction *Terminator = BB->getTerminator();

  if (auto *Branch = dyn_cast<BranchInst>(Terminator)) {
    if (Branch->isConditional()) {

      // We want to transform a conditional branch with one of the destination
      // set to `PlaceHolderTarget` to a non conditional branch
      BasicBlock *SingleDestination = nullptr;

      if (Branch->getSuccessor(0) == PlaceholderTarget) {
        SingleDestination = Branch->getSuccessor(1);
        revng_assert(SingleDestination != PlaceholderTarget);
      } else if (Branch->getSuccessor(1) == PlaceholderTarget) {
        SingleDestination = Branch->getSuccessor(0);
        revng_assert(SingleDestination != PlaceholderTarget);
      }

      // If we found a `BranchInst` candidate for promotion, we substitute it
      // with an unconditional branch
      if (SingleDestination) {
        IRBuilder<> Builder(Terminator);
        Builder.CreateBr(SingleDestination);

        // We remove the old conditional branch
        Terminator->eraseFromParent();
      }
    }
  } else if (auto *Switch = dyn_cast<SwitchInst>(Terminator)) {

    // TODO: understand what to do when the `PlaceHolderTarget` is set as the
    //       default case of the `SwitchInst`

    // Handle the simplification when `PlaceHolderTager` is the default
    // destination of the `SwitchInst`
    BasicBlock *DefaultTarget = Switch->getDefaultDest();
    if (DefaultTarget == PlaceholderTarget) {

      // We promote the first case, not pointing to `PlaceHolderTarget`. If we
      // promote a case already pointing to `PlaceHolderTarget`, this would, in
      // turn, cause the `default` case to not be simplified ever.
      for (auto CaseIt = Switch->case_begin(); CaseIt != Switch->case_end();
           ++CaseIt) {
        if (CaseIt->getCaseSuccessor() != PlaceholderTarget) {
          Switch->setDefaultDest(CaseIt->getCaseSuccessor());
          Switch->removeCase(CaseIt);
          break;
        }
      }
    }

    // Handle the simplification when `PlaceHolderTarget` is part the standard
    // `case`s
    for (auto CaseIt = Switch->case_begin(); CaseIt != Switch->case_end();) {
      if (CaseIt->getCaseSuccessor() == PlaceholderTarget) {

        // We do not want to have a situation where the `PlaceHolderTarget` is
        // both the `default` successor of a `switch` and one of its standard
        // case
        CaseIt = Switch->removeCase(CaseIt);
      } else {
        ++CaseIt;
      }
    }

    if (Switch->getNumCases() == 0
        and Switch->getDefaultDest() == PlaceholderTarget) {
      Switch->eraseFromParent();
      revng_abort();
    }
  }
}

/// Implementation class used to run the `IDS` transformation
class InlineDivergentScopesImpl {
  Function &F;

public:
  InlineDivergentScopesImpl(Function &F) : F(F) {}

public:
  bool run() {
    // We keep a boolean variable to track whether the `Function` was modified
    bool FunctionModified = false;

    // TODO: remove the following assignment once we scatter it in the correct
    //       position in the code were the LLVM function is modified
    FunctionModified = true;

    // 1: Perform the identification of the divergent exits and the divergent
    // branches
    // using EdgeDescriptor = revng::detail::EdgeDescriptor<BasicBlock *>;

    llvm::DomTreeOnView<llvm::BasicBlock, Scope> DomTree;

    // Every time we perform a change due to the IDS restructuring, we may have
    // unlocked the potential to perform new IDS closures in nested subtree of
    // the changed
    bool ExitsChanged = true;
    while (ExitsChanged) {

      // The main need for recomputing the `DomTree` is that we insert the new
      // `idb` block, check if we can use the `DomTree` update mechanism even in
      // this situation.
      DomTree.recalculate(F);

      ExitsChanged = false;

      // Collect the exit nodes
      Scope<Function *> ScopeGraph(&F);
      SmallVector<BasicBlock *> Exits;
      for (BasicBlock *BB : llvm::post_order(ScopeGraph)) {
        if (isExit(BB)) {
          Exits.push_back(BB);
        }
      }

      // Determine if the exit is a divergent exit
      for (BasicBlock *Exit : Exits) {

        // TODO: we use the following `std::optional` in order to contain the
        //       divergent node candidate, with all the object needed to perform
        //       the transformation
        using DivergentEdgesDescriptor = std::pair<BasicBlock *,
                                                   SmallSet<BasicBlock *, 4>>;
        using DivergenceDescriptorT = std::pair<DivergentEdgesDescriptor,
                                                BasicBlock *>;

        std::optional<DivergenceDescriptorT> DivergenceDescriptor;

        // TODO: remove me
        size_t Counter = 0;

        // Process each conditional node up until the root
        SmallVector<BasicBlock *> Worklist;
        Worklist.push_back(Exit);

        while (not Worklist.empty()) {
          BasicBlock *Candidate = Worklist.back();
          Worklist.pop_back();
          revng_assert(Worklist.empty());
          if (isConditional(Candidate)) {

            // Check if it is the conditional making it a divergent node
            if (DomTree.dominates(Candidate, Exit)) {
              using ScopeBlock = Scope<BasicBlock *>;
              auto SuccessorsDuplicated = llvm::children<ScopeBlock>(Candidate);
              SmallSet<BasicBlock *, 2> Successors;
              for (auto *Successor : SuccessorsDuplicated) {
                Successors.insert(Successor);
              }
              std::map<BasicBlock *, SmallSet<BasicBlock *, 4>>
                ReachablesFromSuccessor;
              size_t SuccessorsSize = std::distance(Successors.begin(),
                                                    Successors.end());

              for (auto *Successor : Successors) {
                bool ReachesExit = false;
                SmallSet<BasicBlock *, 4> ReachableExits;

                // Explore all the exits that are reachable from each Successor.
                // For having a divergent exit, we need to find a set of
                // successors that reach only the exit under analysis
                for (auto *DFSNode :
                     llvm::depth_first(Scope<BasicBlock *>(Successor))) {
                  if (isExit(DFSNode)) {
                    ReachableExits.insert(DFSNode);
                  }
                }

                ReachablesFromSuccessor[Successor] = ReachableExits;
              }

              bool Ok = true;
              SmallSet<BasicBlock *, 4> DivergentSuccessors;
              for (const auto &[Successor, ReachableExits] :
                   ReachablesFromSuccessor) {
                if (ReachableExits.size() == 1
                    and *ReachableExits.begin() == Exit) {
                  DivergentSuccessors.insert(Successor);
                } else if (ReachableExits.contains(Exit)) {
                  // not ok
                  Ok = false;
                }
              }

              // We proceed only if: 1) there are some divergent exits and some
              // non divergent exits. 2) We are in a "Ok" situation
              if (DivergentSuccessors.size() < SuccessorsSize and Ok) {
                DivergenceDescriptor = { { Candidate, DivergentSuccessors },
                                         Exit };
              }
            }
          }

          // Enqueue the immediate dominator as the next node to consider as
          // `Candidate`
          auto *ImmediateDominatorNode = DomTree.getNode(Candidate)->getIDom();
          if (ImmediateDominatorNode) {
            Worklist.push_back(ImmediateDominatorNode->getBlock());
          }

          // Exit as soon as
          if (DivergenceDescriptor) {
            break;
          }
        }

        // TODO: improve the following
        // After this point, if `Exit` is a divergent exit, the divergent
        // conditional has been already identified
        if (DivergenceDescriptor) {

          // Create the new `BasicBlock` representing the `C'` conditional
          // inserted by the IDS transformation
          BasicBlock *Conditional = DivergenceDescriptor->first.first;
          auto DivergentSuccessors = DivergenceDescriptor->first.second;
          BasicBlock *Exit = DivergenceDescriptor->second;
          LLVMContext &Context = getContext(Conditional);
          BasicBlock *Tail = BasicBlock::Create(Context,
                                                Conditional->getName() + "_idb",
                                                &F);

          revng_assert(Tail->empty());

          // A: We clone the terminator already present in `BasicBlock`
          //    `Conditional`, so that a superset of the correct final
          //    successors are already connected
          Instruction *ConditionalTerminator = Conditional->getTerminator();
          Instruction *TailTerminator = ConditionalTerminator->clone();
          IRBuilder<> TailBuilder(Tail);
          TailBuilder.Insert(TailTerminator);

          // B: We connect the `Conditional` to `Tail`, by making sure that all
          // the
          //    previous slots and cases going to the nondivergent exits, are
          //    now connected to the `Tail` block, in order to preserve the
          //    original semantics. The original paths going to the nondivergent
          //    exits, are preserved by the the fact that the `Terminator` has
          //    been cloned into `Tail`.
          SmallVector<BasicBlock *> NonDivergentSuccessors;
          for (BasicBlock *Successor :
               children<Scope<BasicBlock *>>(Conditional)) {
            if (not DivergentSuccessors.contains(Successor)) {
              NonDivergentSuccessors.push_back(Successor);
            }
          }

          revng_assert(not DivergentSuccessors.empty()
                       and not NonDivergentSuccessors.empty());

          for (BasicBlock *Successor : NonDivergentSuccessors) {
            ConditionalTerminator->replaceSuccessorWith(Successor, Tail);
          }

          // C: We remove from the `Terminator` of `Tail`, all the edges that
          // target
          //    `DivergentSuccessor`, since it will be only reached by
          //    `Conditional`

          // TODO: we use this temporary substitution in order to not invalidate
          //       the successors with a `nullptr`. Maybe remove the global and
          //       do it locally
          for (BasicBlock *Successor : DivergentSuccessors) {
            TailTerminator->replaceSuccessorWith(Successor, PlaceholderTarget);
          }

          simplifyTerminator(Tail);

          // D: We add a `scope_closer` edge between the divergent exit node and
          // the
          //    `Tail` node
          ScopeGraphBuilder SGBuilder(&F);
          SGBuilder.addScopeCloser(Exit, Tail);

          // We performed a change, so we will eventually need to reprocess all
          // the exits, since we may have unlocked new opportunities for IDS. We
          // do not early exit, since we may still process more exit nodes in
          // the same iteration
          ExitsChanged = true;
          break;
        }
      }
    }

    return FunctionModified;
  }
};

char InlineDivergentScopesPass::ID = 0;

static constexpr const char *Flag = "ids";

using Reg = llvm::RegisterPass<InlineDivergentScopesPass>;

static Reg X(Flag,
             "Perform the inline of divergent scopes canonicalization process");

bool InlineDivergentScopesPass::runOnFunction(Function &F) {

  // TODO: this block is now used solely as a placeholder, therefore we need
  //       to do cleanup at the end of this pass
  LLVMContext &Context = getContext(&F);
  PlaceholderTarget = BasicBlock::Create(Context,
                                         "placeholder_destination",
                                         &F);

  // Instantiate and call the `Impl` class
  InlineDivergentScopesImpl IDSImpl(F);
  bool FunctionModified = IDSImpl.run();

  // TODO: improve this cleanup strategy of the `PlaceHolderTarget`
  PlaceholderTarget->eraseFromParent();

  // This pass may transform the CFG by assign some blocks to perform the IDS
  // canonicalization and by redirecting edges on the `ScopeGraph`
  return FunctionModified;
}

void InlineDivergentScopesPass::getAnalysisUsage(llvm::AnalysisUsage &AU)
  const {
  // This pass does not preserve the CFG
}
