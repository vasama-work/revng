#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"

#include <bit>
#include <optional>
#include <tuple>

using namespace llvm;

namespace {

static std::optional<uint64_t> matchConstantPowerOfTwo(const Value *const V) {
  auto *const C = dyn_cast<ConstantInt>(V);
  if (C == nullptr) {
    return std::nullopt;
  }
  if (C->getBitWidth() > 64) {
    return std::nullopt;
  }
  if (C->isNegative()) {
    return std::nullopt;
  }
  uint64_t const Int = C->getZExtValue();
  if (!std::has_single_bit(Int)) {
    return std::nullopt;
  }
  return Int;
}

using MatchTuple = std::tuple<Value*, uint64_t>;

// Matches a multiplication by a constant power of two.
// Returns a potentially non-constant multiplicand and a constant multiplier.
static std::optional<MatchTuple> matchOperands(const MulOperator *const O) {
  Value *const LHS = O->getOperand(0);
  Value *const RHS = O->getOperand(1);

  if (const auto Match = matchConstantPowerOfTwo(RHS)) {
    return MatchTuple{ LHS, *Match };
  }

  if (const auto Match = matchConstantPowerOfTwo(LHS)) {
    return MatchTuple{ RHS, *Match };
  }

  return std::nullopt;
}

struct MyPass : public FunctionPass {
  static char ID;
  MyPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    bool Modified = false;
    for (auto &B : F) {
      for (auto &I : B) {
        auto *const Operator = dyn_cast<MulOperator>(&I);
        if (Operator == nullptr) {
          continue;
        }

        const auto Operands = matchOperands(Operator);
        if (!Operands) {
          continue;
        }

        const auto &[Multiplicand, Multiplier] = *Operands;
        const unsigned Shift = std::countr_zero(Multiplier);
        // The operands of 'mul' are of the same type.
        // It is not possible for the multiplier to have
        // a value not representable by its type.
        assert(Shift < cast<IntegerType>(Multiplicand->getType())->getBitWidth());

        IRBuilder<> Builder(&I);
        Value *const ShiftValue = ConstantInt::get(
          Multiplicand->getType(),
          Shift);
        Value *const ShiftOperation = Builder.CreateShl(
          Multiplicand,
          ShiftValue,
          "",
          Operator->hasNoUnsignedWrap(),
          Operator->hasNoSignedWrap());

        // Replace any uses of the multiplication result with the shift result.
        for (const auto &Use : I.uses()) {
          Use.getUser()->setOperand(Use.getOperandNo(), ShiftOperation);
        }

        Modified = true;
      }
    }
    return Modified;
  }
}; // end of struct Hello
}  // end of anonymous namespace

char MyPass::ID = 0;
static RegisterPass<MyPass> X("mypass", "My Pass");
