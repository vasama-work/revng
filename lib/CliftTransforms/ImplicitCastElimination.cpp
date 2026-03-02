//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <bit>

#include "llvm/ADT/APSInt.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Passes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTIMPLICITCASTELIMINATION
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

// WIP
#include "llvm/ADT/ScopeExit.h"

//#define debug_print(...) (llvm::errs() << __VA_ARGS__)

#ifndef debug_print
#define debug_print(...)
#endif

namespace clift = mlir::clift;
using namespace clift;

namespace {

class Visitor {
  unsigned IntSize = 4;
  PrimitiveType IntType;

  mlir::OpBuilder Builder;

public:
  static void visitExpressionTree(YieldOp Yield) {
    Visitor(Yield.getContext()).visitYieldOp(Yield);
  }

private:
  explicit Visitor(mlir::MLIRContext *Context) : Builder(Context) {
    IntType = PrimitiveType::get(Context, PrimitiveKind::SignedKind, IntSize);
  }

  PrimitiveKind getCommonKind(PrimitiveKind LHS, PrimitiveKind RHS) {
    using enum PrimitiveKind;

    if (LHS == RHS)
      return LHS;

    if (LHS >= NumberKind and RHS >= NumberKind)
      return NumberKind;

    if (LHS >= PointerOrNumberKind and RHS >= PointerOrNumberKind)
      return PointerOrNumberKind;

    return GenericKind;
  }

  ValueType getCommonType(ValueType LHS, ValueType RHS) {
    PrimitiveType L = getUnderlyingIntegerType(LHS);
    PrimitiveType R = getUnderlyingIntegerType(RHS);

    if (R.getSize() > L.getSize())
      std::swap(L, R);

    auto NewKind = L.getSize() > R.getSize() ?
                     L.getKind() :
                     getCommonKind(L.getKind(), R.getKind());

    return PrimitiveType::get(LHS.getContext(),
                              NewKind,
                              std::max(L.getSize(), R.getSize()));
  }

  template<std::convertible_to<ValueType>... RestT>
  ValueType getCommonType(ValueType First, ValueType Second, RestT &&...Rest) {
    return getCommonType(getCommonType(First, Second),
                         std::forward<RestT>(Rest)...);
  }

  void markCastAsImplicit(CastOp Op) {
    Op->setAttr("clift.implicit", mlir::UnitAttr::get(Op.getContext()));
  }

  CastKind getCastKind(ValueType OldType, ValueType NewType) {
    auto OldSize = OldType.getByteSize();
    auto NewSize = NewType.getByteSize();
    if (NewSize > OldSize)
      return CastKind::Extend;
    if (NewSize < OldSize)
      return CastKind::Truncate;
    return CastKind::Bitcast;
  }

  mlir::Value emitCast(ExpressionOpInterface User,
                       mlir::Value Value,
                       ValueType NewType,
                       bool Implicit = false) {
    if (Value.getType() != NewType) {
      auto Kind = getCastKind(Value.getType(), NewType);

      Builder.setInsertionPoint(User);
      auto Op = Builder.create<CastOp>(User->getLoc(), NewType, Value, Kind);

      if (Implicit)
        markCastAsImplicit(Op);

      Value = Op;
    }
    return Value;
  }

  mlir::Value promote(ExpressionOpInterface User,
                      mlir::Value Value,
                      ValueType NewType) {
    revng_assert(getCastKind(Value.getType(), NewType) != CastKind::Truncate);
    return emitCast(User, Value, NewType, /*Implicit=*/true);
  }

  mlir::Value truncate(ExpressionOpInterface User,
                       mlir::Value Value,
                       ValueType NewType) {
    if (Value.getType() != NewType) {
      Builder.setInsertionPoint(User);
      auto Op = Builder.create<CastOp>(User->getLoc(),
                                       NewType,
                                       Value,
                                       getCastKind(Value.getType(),
                                                   NewType));
      Value = Op;
    }
    return Value;
  }

  RecursiveCoroutine<mlir::Value> visitExternalValue(mlir::Value Value) {
    debug_print("non-expression value\n");
    rc_return Value;
  }

  RecursiveCoroutine<mlir::Value> visitCast(CastOp E) {
    debug_print("enter cast<" << E.getKind() << ">\n");
    auto G = llvm::make_scope_exit([&]() {
      debug_print("leave cast<" << E.getKind() << ">\n");
    });

    mlir::Value Result = E->getResult(0);
    mlir::OpOperand &Operand = E->getOpOperand(0);
    ValueType Type = Operand.get().getType();

    mlir::Value NewOperand = rc_recur visit(Operand.get());

    switch (E.getKind()) {
    case CastKind::Extend:
    case CastKind::Bitcast:
    case CastKind::Truncate: {
      auto T1 = getUnderlyingIntegerType(NewOperand.getType());
      auto T2 = getUnderlyingIntegerType(Result.getType());

      if (T1 and T2)
        rc_return NewOperand;
    } break;

    default:
      break;
    }

    Operand.set(truncate(E, NewOperand, Type));
    rc_return Result;
  }

  RecursiveCoroutine<mlir::Value>
  visitRelaxedArithmeticOp(ExpressionOpInterface E) {
    debug_print("enter relaxed arithmetic\n");

    mlir::Value Result = E->getResult(0);
    mlir::OpOperand &LHS = E->getOpOperand(0);
    mlir::OpOperand &RHS = E->getOpOperand(1);

    mlir::Value NewLHS = rc_recur visit(LHS.get());
    mlir::Value NewRHS = rc_recur visit(RHS.get());

    ValueType PromotedType = getCommonType(IntType,
                                           NewLHS.getType(),
                                           NewRHS.getType());

    ValueType NewType = getCommonType(PromotedType, Result.getType());

    if (PromotedType != NewType) {
      NewLHS = emitCast(E, NewLHS, NewType);
      NewRHS = emitCast(E, NewRHS, NewType);
    }

    LHS.set(promote(E, NewLHS, NewType));
    RHS.set(promote(E, NewRHS, NewType));

    Result.setType(NewType);
    debug_print("leave relaxed arithmetic\n");
    rc_return Result;
  }

  RecursiveCoroutine<mlir::Value>
  visitDivisionOrModuloOp(ExpressionOpInterface E) {
    debug_print("enter division\n");

    mlir::Value Result = E->getResult(0);
    mlir::OpOperand &LHS = E->getOpOperand(0);
    mlir::OpOperand &RHS = E->getOpOperand(1);

    mlir::Value NewLHS = rc_recur visit(LHS.get());
    mlir::Value NewRHS = rc_recur visit(RHS.get());

    mlir::Value NewLHSValue = truncate(E, NewLHS, Result.getType());
    mlir::Value NewRHSValue = truncate(E, NewRHS, Result.getType());

    ValueType NewType = getCommonType(IntType,
                                      NewLHSValue.getType(),
                                      NewRHSValue.getType());

    LHS.set(promote(E, NewLHSValue, NewType));
    RHS.set(promote(E, NewRHSValue, NewType));

    Result.setType(NewType);
    debug_print("leave division\n");
    rc_return Result;
  }

  RecursiveCoroutine<mlir::Value> visitShiftLeftOp(ShiftLeftOp E) {
    mlir::Value Result = E->getResult(0);
    mlir::OpOperand &LHS = E->getOpOperand(0);
    mlir::OpOperand &RHS = E->getOpOperand(1);

    ValueType LHSType = LHS.get().getType();
    ValueType RHSType = RHS.get().getType();

    mlir::Value NewLHS = rc_recur visit(LHS.get());
    mlir::Value NewRHS = rc_recur visit(RHS.get());

    mlir::Value NewLHSValue = NewLHS;
    mlir::Value NewRHSValue = truncate(E, NewRHS, RHSType);

    ValueType PromotedLHSType = getCommonType(IntType, NewLHSValue.getType());
    ValueType NewLHSType = getCommonType(PromotedLHSType, LHSType);
    ValueType NewRHSType = getCommonType(IntType, NewRHSValue.getType());

    if (PromotedLHSType != NewLHSType)
      NewLHSValue = emitCast(E, NewLHSValue, NewLHSType);

    LHS.set(promote(E, NewLHSValue, NewLHSType));
    RHS.set(promote(E, NewRHSValue, NewRHSType));

    Result.setType(NewLHSType);
    rc_return Result;
  }

  RecursiveCoroutine<mlir::Value> visitShiftRightOp(ShiftRightOp E) {
    mlir::Value Result = E->getResult(0);
    mlir::OpOperand &LHS = E->getOpOperand(0);
    mlir::OpOperand &RHS = E->getOpOperand(1);

    ValueType LHSType = LHS.get().getType();
    ValueType RHSType = RHS.get().getType();

    mlir::Value NewLHS = rc_recur visit(LHS.get());
    mlir::Value NewRHS = rc_recur visit(RHS.get());

    mlir::Value NewLHSValue = truncate(E, NewLHS, LHSType);
    mlir::Value NewRHSValue = truncate(E, NewRHS, RHSType);

    ValueType NewLHSType = getCommonType(IntType, NewLHSValue.getType());
    ValueType NewRHSType = getCommonType(IntType, NewRHSValue.getType());

    LHS.set(promote(E, NewLHSValue, NewLHSType));
    RHS.set(promote(E, NewRHSValue, NewRHSType));

    Result.setType(NewLHSType);
    rc_return Result;
  }

  RecursiveCoroutine<mlir::Value> visitTernaryOp(TernaryOp E) {
    mlir::Value Result = E->getResult(0);
    mlir::OpOperand &Cnd = E->getOpOperand(0);
    mlir::OpOperand &LHS = E->getOpOperand(1);
    mlir::OpOperand &RHS = E->getOpOperand(2);

    mlir::Type CndType = Cnd.get().getType();

    mlir::Value NewCnd = rc_recur visit(Cnd.get());
    mlir::Value NewLHS = rc_recur visit(LHS.get());
    mlir::Value NewRHS = rc_recur visit(RHS.get());

    ValueType NewType = Result.getType();
    if (NewLHS.getType() != NewRHS.getType()) {
      NewType = getCommonType(IntType,
                              NewLHS.getType(),
                              NewRHS.getType());
    }

    Cnd.set(truncate(E, NewCnd, CndType));
    LHS.set(promote(E, NewLHS, NewType));
    RHS.set(promote(E, NewRHS, NewType));

    Result.setType(NewType);
    rc_return Result;
  }

  RecursiveCoroutine<mlir::Value> visitAssignOp(AssignOp E) {
    debug_print("enter assignment: " << E.getLhs().getType() << "\n");

    mlir::OpOperand &LHS = E->getOpOperand(0);
    mlir::OpOperand &RHS = E->getOpOperand(1);

    mlir::Value OldLHS = LHS.get();
    ValueType OldLHSType = OldLHS.getType();

    mlir::Value NewLHS = rc_recur visit(LHS.get());
    mlir::Value NewRHS = rc_recur visit(RHS.get());

    revng_assert(NewLHS == OldLHS);
    revng_assert(NewLHS.getType() == OldLHSType);

    RHS.set(emitCast(E, NewRHS, LHS.get().getType(), /*Implicit=*/true));

    debug_print("leave assignment: " << E.getLhs().getType() << "\n");
    rc_return E;
  }

  RecursiveCoroutine<mlir::Value> visitCallOp(CallOp E) {
    debug_print("enter call\n");

    mlir::Value OldCallee = E.getFunction();
    ValueType OldCalleeType = OldCallee.getType();

    mlir::Value NewCallee = rc_recur visit(OldCallee);

    revng_assert(NewCallee == OldCallee);
    revng_assert(NewCallee.getType() == OldCalleeType);

    for (unsigned I = 1, C = E->getNumOperands(); I < C; ++I) {
      mlir::OpOperand &Operand = E->getOpOperand(I);
      ValueType Type = Operand.get().getType();
      mlir::Value NewOperand = rc_recur visit(Operand.get());
      Operand.set(emitCast(E, NewOperand, Type, /*Implicit=*/true));
    }
    debug_print("leave call\n");
    rc_return E;
  }

  RecursiveCoroutine<mlir::Value> visitUnhandledOp(ExpressionOpInterface E) {
    debug_print("enter " << E->getName() << "\n");
    for (unsigned I = 0, C = E->getNumOperands(); I < C; ++I) {
      mlir::OpOperand &Operand = E->getOpOperand(I);
      ValueType Type = Operand.get().getType();
      mlir::Value NewOperand = rc_recur visit(Operand.get());
      Operand.set(emitCast(E, NewOperand, Type, /*Implicit=*/true));
    }
    debug_print("leave " << E->getName() << "\n");
    rc_return E->getResult(0);
  }

  RecursiveCoroutine<mlir::Value> visit(mlir::Value Value) {
    mlir::Operation *Op = Value.getDefiningOp();
    if (Op == nullptr)
      return visitExternalValue(Value);
    auto Expr = mlir::cast<ExpressionOpInterface>(Op);

    if (auto E = mlir::dyn_cast<CastOp>(Op))
      return visitCast(E);

    if (mlir::isa<AddOp,
                  SubOp,
                  MulOp,
                  BitwiseNotOp,
                  BitwiseAndOp,
                  BitwiseOrOp,
                  BitwiseXorOp>(Op))
      return visitRelaxedArithmeticOp(Expr);

    if (mlir::isa<DivOp, RemOp>(Op))
      return visitDivisionOrModuloOp(Expr);

    if (auto E = mlir::dyn_cast<ShiftLeftOp>(Op))
      return visitShiftLeftOp(E);

    if (auto E = mlir::dyn_cast<ShiftRightOp>(Op))
      return visitShiftRightOp(E);

    if (auto E = mlir::dyn_cast<TernaryOp>(Op))
      return visitTernaryOp(E);

    if (auto E = mlir::dyn_cast<AssignOp>(Op))
      return visitAssignOp(E);

    if (auto E = mlir::dyn_cast<CallOp>(Op))
      return visitCallOp(E);

    return visitUnhandledOp(Expr);
  }

  void visitYieldOp(YieldOp Yield) {
    mlir::OpOperand &Operand = Yield->getOpOperand(0);
    ValueType Type = Operand.get().getType();
    mlir::Value NewOperand = visit(Operand.get());
    // WIP: Depending on context, an implicit cast might be appropriate.
    Operand.set(emitCast(Yield, NewOperand, Type));
  }
};

template<typename T>
using PassBase = mlir::clift::impl::CliftImplicitCastEliminationBase<T>;

struct SimpleRewriter : mlir::PatternRewriter {
  explicit SimpleRewriter(mlir::MLIRContext *Context) :
    PatternRewriter(Context) {}
};

struct ImplicitCastEliminationPass : PassBase<ImplicitCastEliminationPass> {
  void runOnOperation() override {
    auto const Walker = [](ExpressionRegionOpInterface Op) -> mlir::WalkResult {
      for (mlir::Region &R : Op.getExpressionRegions()) {
        if (auto Yield = getYieldOp(R))
          Visitor::visitExpressionTree(Yield);
      }
      return mlir::WalkResult::skip();
    };
    getOperation()->walk(Walker);

    SimpleRewriter Rewriter(&getContext());
    (void)mlir::runRegionDCE(Rewriter, getOperation().getBody());
  }
};

} // namespace

PassPtr<FunctionOp> clift::createImplicitCastEliminationPass() {
  return std::make_unique<ImplicitCastEliminationPass>();
}
