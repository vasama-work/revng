#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/mlir/Dialect/Clift/Utils/Legalization.h"

namespace clift = mlir::clift;

namespace {

template<typename OpT>
struct PointerResizePattern : mlir::OpRewritePattern<OpT> {
  explicit PointerResizePattern(mlir::MLIRContext *Context,
                                uint64_t TargetPointerSize) :
    mlir::OpRewritePattern<OpT>(Context),
    TargetPointerSize(TargetPointerSize) {}

  uint64_t TargetPointerSize;

  clift::PointerType
  makeTargetPointerType(clift::PointerType OldPointerType) const {
    return clift::PointerType::get(OldPointerType.getPointeeType(),
                                   TargetPointerSize);
  }

  mlir::Value emitCast(mlir::PatternRewriter &Rewriter,
                       mlir::Location Loc,
                       mlir::Value Value,
                       clift::ValueType OldType,
                       clift::ValueType NewType) const {
    auto Kind = OldType.getByteSize() < NewType.getByteSize() ?
      clift::CastKind::Extend :
      clift::CastKind::Truncate;

    return Rewriter.create<clift::CastOp>(Loc, NewType, Value, Kind);
  }

  mlir::LogicalResult replacePointerOperand(mlir::PatternRewriter &Rewriter,
                                            clift::ExpressionOpInterface Op,
                                            unsigned Index = 0) const {
    mlir::Value Operand = Op->getOperand(Index);
    revng_assert(Operand.hasOneUse());

    auto OldType = clift::getPointerType(Operand.getType());
    if (not OldType or OldType.getPointerSize() == TargetPointerSize)
      return mlir::failure();

    auto NewType = makeTargetPointerType(OldType);

    Rewriter.setInsertionPoint(Op);
    Op->setOperand(Index, emitCast(Rewriter,
                                   Op->getLoc(),
                                   Operand,
                                   OldType,
                                   NewType));

    return mlir::success();
  }

  mlir::LogicalResult
  replacePointerResult(mlir::PatternRewriter &Rewriter,
                       clift::ExpressionOpInterface Op) const {
    auto Result = Op->getResult(0);

    auto OldType = clift::getPointerType(Result.getType());
    revng_assert(OldType);

    if (OldType.getPointerSize() == TargetPointerSize)
      return mlir::failure();

    revng_assert(Result.hasOneUse());
    mlir::OpOperand &Use = *Result.use_begin();

    auto NewType = makeTargetPointerType(OldType);

    Rewriter.setInsertionPointAfter(Op);
    Use.set(emitCast(Rewriter,
                     Op->getLoc(),
                     Result,
                     NewType,
                     OldType));

    Op->getOpResult(0).setType(NewType);

    return mlir::success();
  }
};

template<typename OpT>
struct ResizePointerArithmeticPattern : PointerResizePattern<OpT> {
  using PointerResizePattern<OpT>::PointerResizePattern;

  mlir::LogicalResult
  matchAndRewrite(OpT Op, mlir::PatternRewriter &Rewriter) const override {
    unsigned Index = Op.getPointerOperandIndex();
    if (this->replacePointerOperand(Rewriter, Op, Index).failed())
      return mlir::failure();

    auto OldType = clift::getPointerType(Op.getResult().getType());
    auto NewType = Op->getOperand(Index).getType();

    revng_assert(Op.getResult().hasOneUse());
    Op->getOpResult(0).setType(NewType);

    Rewriter.setInsertionPointAfter(Op);
    Op->use_begin()->set(this->emitCast(Rewriter,
                                        Op->getLoc(),
                                        Op.getResult(),
                                        NewType,
                                        OldType));

    return mlir::success();
  }
};

using ResizePtrAddPattern = ResizePointerArithmeticPattern<clift::PtrAddOp>;
using ResizePtrSubPattern = ResizePointerArithmeticPattern<clift::PtrSubOp>;

struct ResizePtrDiffPattern : PointerResizePattern<clift::PtrDiffOp> {
  using PointerResizePattern::PointerResizePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::PtrDiffOp Op,
                  mlir::PatternRewriter &Rewriter) const override {
    if (replacePointerOperand(Rewriter, Op).failed())
      return mlir::failure();

    auto OldType = mlir::cast<clift::ValueType>(Op->getOperand(1).getType());
    auto NewType = mlir::cast<clift::ValueType>(Op->getOperand(0).getType());
    Op->setOperand(1, emitCast(Rewriter,
                               Op->getLoc(),
                               Op->getOperand(0),
                               OldType,
                               NewType));

    return mlir::success();
  }
};

struct ResizeIndirectionPattern : PointerResizePattern<clift::IndirectionOp> {
  using PointerResizePattern::PointerResizePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::IndirectionOp Op,
                  mlir::PatternRewriter &Rewriter) const override {
    return replacePointerOperand(Rewriter, Op);
  }
};

struct ResizeSubscriptPattern : PointerResizePattern<clift::SubscriptOp> {
  using PointerResizePattern::PointerResizePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::SubscriptOp Op,
                  mlir::PatternRewriter &Rewriter) const override {
    return replacePointerOperand(Rewriter, Op);
  }
};

struct ResizeAccessPattern : PointerResizePattern<clift::AccessOp> {
  using PointerResizePattern::PointerResizePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::AccessOp Op,
                  mlir::PatternRewriter &Rewriter) const override {
    if (not Op.isIndirect())
      return mlir::failure();

    return replacePointerOperand(Rewriter, Op);
  }
};

struct ResizeCallPattern : PointerResizePattern<clift::CallOp> {
  using PointerResizePattern::PointerResizePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::CallOp Op,
                  mlir::PatternRewriter &Rewriter) const override {
    return replacePointerOperand(Rewriter, Op);
  }
};

struct ResizeAddressofPattern : PointerResizePattern<clift::AddressofOp> {
  using PointerResizePattern::PointerResizePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::AddressofOp Op,
                  mlir::PatternRewriter &Rewriter) const override {
    return replacePointerResult(Rewriter, Op);
  }
};

struct ResizeDecayCastPattern : PointerResizePattern<clift::CastOp> {
  using PointerResizePattern::PointerResizePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::CastOp Op,
                  mlir::PatternRewriter &Rewriter) const override {
    if (Op.getKind() != clift::CastKind::Decay)
      return mlir::failure();

    return replacePointerResult(Rewriter, Op);
  }
};

//TODO: Use the newer single walk pattern applicator once the LLVM upgrade is
//      done.

#if 0
static clift::FunctionType
convertHelperFunctionType(clift::FunctionType Type,
                          const TargetCImplementation &Target) {
  bool AnyConverted = false;
  auto ConvertType = [&](mlir::Type T) -> mlir::Type {
    if (auto PT = mlir::dyn_cast<clift::PointerType>(T)) {
      if (PT.getPointerSize() != Target.PointerSize) {
        T = clift::PointerType::get(PT.getPointeeType(),
                                    Target.PointerSize);
        AnyConverted = true;
      }
    }
    return T;
  };

  llvm::SmallVector<mlir::Type> ParameterTypes;
  for (mlir::Type T : Type.getArgumentTypes())
    ParameterTypes.push_back(ConvertType(T));
  mlir::Type ReturnType = ConvertType(Type.getReturnType());

  if (not AnyConverted)
    return Type;

  return clift::FunctionType::get(Type.getHandle(),
                                  Type.getName(),
                                  ParameterTypes,
                                  ReturnType);
}

static mlir::LogicalResult
convertHelperCalls(mlir::ModuleOp Module, const TargetCImplementation &Target) {
  for (mlir::Operation &Op : Module.getBody().getOperations()) {
    if (auto F = mlir::dyn_cast<clift::Function>(&Op)) {
      if (not pipeline::locationFromString(revng::ranks::HelperFunction,
                                           F.getHandle()))
        continue;

      auto NewType = convertHelperFunctionType(F.getCliftFunctionType());
      
    }
  }
}

static mlir::LogicalResult
resizePointerUses(mlir::ModuleOp Module, const TargetCImplementation &Target) {
  mlir::MLIRContext *Context = Function.getContext();
  mlir::RewritePatternSet Set(Context);

  Set.add<ResizePtrAddPattern>(Context, Target.PointerSize);
  Set.add<ResizePtrSubPattern>(Context, Target.PointerSize);
  Set.add<ResizePtrDiffPattern>(Context, Target.PointerSize);
  Set.add<ResizeIndirectionPattern>(Context, Target.PointerSize);
  Set.add<ResizeSubscriptPattern>(Context, Target.PointerSize);
  Set.add<ResizeAccessPattern>(Context, Target.PointerSize);
  Set.add<ResizeCallPattern>(Context, Target.PointerSize);
  Set.add<ResizeAddressofPattern>(Context, Target.PointerSize);
  Set.add<ResizeDecayCastPattern>(Context, Target.PointerSize);

  auto Patterns = mlir::FrozenRewritePatternSet(std::move(Set));
  return mlir::applyPatternsAndFoldGreedily(Function, Patterns);
}
#endif

} // namespace

mlir::LogicalResult clift::legalizeForC(clift::FunctionOp Function,
                                        const TargetCImplementation &Target) {
  mlir::MLIRContext *Context = Function.getContext();
  mlir::RewritePatternSet Set(Context);

  Set.add<ResizePtrAddPattern>(Context, Target.PointerSize);
  Set.add<ResizePtrSubPattern>(Context, Target.PointerSize);
  Set.add<ResizePtrDiffPattern>(Context, Target.PointerSize);
  Set.add<ResizeIndirectionPattern>(Context, Target.PointerSize);
  Set.add<ResizeSubscriptPattern>(Context, Target.PointerSize);
  Set.add<ResizeAccessPattern>(Context, Target.PointerSize);
  Set.add<ResizeCallPattern>(Context, Target.PointerSize);
  Set.add<ResizeAddressofPattern>(Context, Target.PointerSize);
  Set.add<ResizeDecayCastPattern>(Context, Target.PointerSize);

  auto Patterns = mlir::FrozenRewritePatternSet(std::move(Set));
  return mlir::applyPatternsAndFoldGreedily(Function, Patterns);
}
