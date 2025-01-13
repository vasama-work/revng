#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/GenericDomTree.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng/mlir/Dialect/Clift/Utils/ImportLLVM.h"
#include "revng/mlir/Dialect/Clift/Utils/ImportModel.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/FunctionTags.h"

namespace clift = mlir::clift;
using namespace clift;

#if 0
#define myprintf(...) fprintf(stderr, __VA_ARGS__)
#define myprintl(fmt, ...) (myprint_ip(), myprintf(fmt "\n", ## __VA_ARGS__))
#else
#define myprintf(...) ((void)0)
#define myprintl(fmt, ...) ((void)0)
#endif

static thread_local int myprint_i = 0;
[[maybe_unused]] static auto indent() {
  ++myprint_i;
  return llvm::make_scope_exit([]() {
    --myprint_i;
  });
}
[[maybe_unused]] static void myprint_ip() {
  for (int i = 0; i < myprint_i; ++i)
    myprintf("    ");
}

namespace {

[[nodiscard]] mlir::OpBuilder::InsertPoint
saveInsertionPointAfter(const mlir::OpBuilder &Builder) {
  mlir::Block *Block = Builder.getInsertionBlock();
  mlir::Block::iterator Point = Builder.getInsertionPoint();

  // End iterator specifies insertion at the start of Block. Non-end iterator
  // specifies insertion *after* the operation referred to by the iterator.
  Point = Point == Block->begin() ? Block->end() : std::prev(Point);

  return mlir::OpBuilder::InsertPoint(Block, Point);
}

void restoreInsertionPointAfter(mlir::OpBuilder &Builder,
                                mlir::OpBuilder::InsertPoint InsertPoint) {
  revng_assert(InsertPoint.isSet());

  mlir::Block *Block = InsertPoint.getBlock();
  mlir::Block::iterator Point = InsertPoint.getPoint();

  // Convert an end iterator back into begin and advance non-end iterators. This
  // is because the builder inserts operations *before* the specified iterator.
  Point = Point == Block->end() ? Block->begin() : std::next(Point);

  Builder.setInsertionPoint(Block, Point);
}


template<typename OpT, typename... ArgsT>
static mlir::OwningOpRef<OpT> createOperation(mlir::MLIRContext *Context,
                                              mlir::Location Location,
                                              ArgsT &&...Args) {
  OpT Op = mlir::OpBuilder(Context).create<OpT>(Location,
                                                std::forward<ArgsT>(Args)...);
  return mlir::OwningOpRef<OpT>(Op);
}


#if 0
class LLVMTypeImporter {
public:
  static clift::ValueType import(mlir::MLIRContext *Context,
                                 const llvm::Type *Type) {
    return LLVMTypeImporter(Context).importType(Type);
  }

private:
  explicit LLVMTypeImporter(mlir::MLIRContext *Context) :
    Context(Context) {}

  mlir::BoolAttr getBoolAttr(bool Value) {
    return mlir::BoolAttr::get(Context, Value);
  }

  clift::ValueType importIntegerType(const llvm::IntegerType *Type) {
    return PrimitiveType::get(Context,
                              PrimitiveKind::GenericKind,
                              getIntegerSize(Type->getBitWidth()),
                              getBoolAttr(false));
  }

  clift::ValueType getVoidType() {
    if (not VoidType) {
      VoidType = PrimitiveType::get(Context,
                                    PrimitiveKind::VoidKind,
                                    0,
                                    getBoolAttr(false));
    }
    return VoidType;
  }

  clift::ValueType importPointerType(const llvm::PointerType *Type) {
    if (not PointerType) {
      revng_assert(Type->getAddressSpace() == 0);

      // WIP: Figure out the pointer size somehow.
      uint64_t PointerSize = 8;

      PointerType = clift::PointerType::get(Context,
                                            getVoidType(),
                                            PointerSize,
                                            getBoolAttr(false));
    }
    return PointerType;
  }

  RecursiveCoroutine<clift::ValueType>
  importArrayType(const llvm::ArrayType *Type) {
    clift::ValueType ElementType = rc_recur importType(Type->getElementType());
    rc_return ArrayType::get(Context, ElementType, Type->getNumElements());
  }

  RecursiveCoroutine<clift::ValueType>
  importFunctionType(const llvm::FunctionType *Type) {
    revng_assert(not Type->isVarArg());

    clift::ValueType ReturnType = importType(Type->getReturnType());

    llvm::SmallVector<clift::ValueType> ParameterTypes;
    ParameterTypes.reserve(Type->getNumParams());

    for (const llvm::Type *T : Type->params())
      ParameterTypes.push_back(rc_recur importType(T));

    auto Attr = FunctionTypeAttr::get(Context,
                                      NextInventedTypeID++,
                                      "",
                                      ReturnType,
                                      ParameterTypes);

    rc_return DefinedType::get(Context, Attr, getBoolAttr(false));
  }

  RecursiveCoroutine<clift::ValueType>
  importStructType(const llvm::StructType *Type) {
    revng_assert(Type->isLiteral());

    llvm::SmallVector<FieldAttr> Fields;
    Fields.reserve(Type->getNumElements());

    uint64_t Offset = 0;
    for (const llvm::Type *T : Type->elements()) {
      clift::ValueType FieldType = rc_recur importType(T);
      Fields.push_back(FieldAttr::get(Context, Offset, FieldType, ""));
      Offset += FieldType.getByteSize();
    }

    auto Attr = StructTypeAttr::get(Context,
                                    NextInventedTypeID++,
                                    "",
                                    Offset,
                                    Fields);

    rc_return DefinedType::get(Context, Attr, getBoolAttr(false));
  }

  RecursiveCoroutine<clift::ValueType> importType(const llvm::Type *Type) {
    if (Type->isVoidTy())
      rc_return getVoidType();

    if (auto *T = llvm::dyn_cast<llvm::PointerType>(Type))
      rc_return importPointerType(T);

    if (auto *T = llvm::dyn_cast<llvm::IntegerType>(Type))
      rc_return importIntegerType(T);

    if (auto *T = llvm::dyn_cast<llvm::ArrayType>(Type))
      rc_return rc_recur importArrayType(T);

    if (auto *T = llvm::dyn_cast<llvm::FunctionType>(Type))
      rc_return rc_recur importFunctionType(T);

    if (auto *T = llvm::dyn_cast<llvm::StructType>(Type))
      rc_return rc_recur importStructType(T);

    revng_abort("Unsupported LLVM type");
  }

  mlir::MLIRContext *Context;

  clift::ValueType VoidType;
  clift::ValueType PointerType;
};
#endif


using ScopeGraphPostDomTree = llvm::PostDomTreeOnView<llvm::BasicBlock, Scope>;

class LLVMCodeImporter {
public:
  static mlir::OwningOpRef<clift::ModuleOp> import(mlir::MLIRContext *Context,
                                                   const model::Binary &Model,
                                                   const llvm::Module *Module) {
    return LLVMCodeImporter(Context, Model).importModule(Module);
  }

private:
  explicit LLVMCodeImporter(mlir::MLIRContext *Context,
                            const model::Binary &Model) :
    Context(Context),
    Model(Model),
    Builder(Context) {}

  /* Type utilities */

  mlir::BoolAttr getBoolAttr(bool Value) {
    return mlir::BoolAttr::get(Context, Value);
  }

  static uint64_t getIntegerSize(unsigned IntegerWidth) {
    // Compute the smallest power-of-two number of bytes capable of representing
    // the type based on its bit width:
    return std::bit_ceil((IntegerWidth + 7) / 8);
  }

  uint64_t getPointerSize() {
    return 8; // WIP
  }

  clift::ValueType makePointerType(clift::ValueType ElementType) {
    return PointerType::get(Context,
                            ElementType,
                            getPointerSize(),
                            getBoolAttr(false));
  }

  clift::ValueType getVoidType() {
    if (not VoidTypeCache) {
      VoidTypeCache = PrimitiveType::get(Context,
                                         PrimitiveKind::VoidKind,
                                         0,
                                         getBoolAttr(false));
    }
    return VoidTypeCache;
  }

  clift::ValueType getVoidPointerType() {
    if (not PointerTypeCache)
      PointerTypeCache = makePointerType(getVoidType());
    return PointerTypeCache;
  }

  clift::ValueType getIntptrType() {
    return PrimitiveType::get(Context,
                              PrimitiveKind::GenericKind,
                              getPointerSize(),
                              getBoolAttr(false));
  }

  clift::ValueType
  getPrimitiveType(uint64_t Size,
                   PrimitiveKind Kind = PrimitiveKind::GenericKind) {
    return PrimitiveType::get(Context, Kind, Size, getBoolAttr(false));
  }

  /* LLVM type import */

  clift::ValueType
  importLLVMIntegerType(const llvm::IntegerType* Type,
                        PrimitiveKind Kind = PrimitiveKind::GenericKind) {
    return getPrimitiveType(getIntegerSize(Type->getBitWidth()), Kind);
  }

  clift::ValueType importLLVMPointerType(const llvm::PointerType *Type) {
    revng_assert(Type->getAddressSpace() == 0);
    return getVoidPointerType();
  }

  RecursiveCoroutine<clift::ValueType>
  importLLVMArrayType(const llvm::ArrayType *Type) {
    auto ElementType = rc_recur importLLVMType(Type->getElementType());
    rc_return ArrayType::get(Context, ElementType, Type->getNumElements());
  }

  RecursiveCoroutine<clift::ValueType>
  importLLVMFunctionType(const llvm::FunctionType *Type) {
    revng_assert(not Type->isVarArg());

    clift::ValueType ReturnType = importLLVMType(Type->getReturnType());

    llvm::SmallVector<clift::ValueType> ParameterTypes;
    ParameterTypes.reserve(Type->getNumParams());

    for (const llvm::Type *T : Type->params())
      ParameterTypes.push_back(rc_recur importLLVMType(T));

    auto Attr = FunctionTypeAttr::get(Context,
                                      NextInventedTypeID++,
                                      "",
                                      ReturnType,
                                      ParameterTypes);

    rc_return DefinedType::get(Context, Attr, getBoolAttr(false));
  }

  RecursiveCoroutine<clift::ValueType>
  importLLVMStructType(const llvm::StructType *Type) {
    revng_assert(Type->isLiteral());

    llvm::SmallVector<FieldAttr> Fields;
    Fields.reserve(Type->getNumElements());

    uint64_t Offset = 0;
    for (const llvm::Type *T : Type->elements()) {
      clift::ValueType FieldType = rc_recur importLLVMType(T);
      Fields.push_back(FieldAttr::get(Context, Offset, FieldType, ""));
      Offset += FieldType.getByteSize();
    }

    auto Attr = StructTypeAttr::get(Context,
                                    NextInventedTypeID++,
                                    "",
                                    Offset,
                                    Fields);

    rc_return DefinedType::get(Context, Attr, getBoolAttr(false));
  }

  RecursiveCoroutine<clift::ValueType> importLLVMType(const llvm::Type *Type) {
    if (Type->isVoidTy())
      rc_return getVoidType();

    if (auto *T = llvm::dyn_cast<llvm::IntegerType>(Type))
      rc_return importLLVMIntegerType(T);

    if (auto *T = llvm::dyn_cast<llvm::PointerType>(Type))
      rc_return importLLVMPointerType(T);

    if (auto *T = llvm::dyn_cast<llvm::ArrayType>(Type))
      rc_return rc_recur importLLVMArrayType(T);

    if (auto *T = llvm::dyn_cast<llvm::FunctionType>(Type))
      rc_return rc_recur importLLVMFunctionType(T);

    if (auto *T = llvm::dyn_cast<llvm::StructType>(Type))
      rc_return rc_recur importLLVMStructType(T);

    revng_abort("Unsupported LLVM type");
  }

  clift::ValueType importModelType(const model::Type &Type) {
    auto EmitError = [&]() -> mlir::InFlightDiagnostic {
      return Context->getDiagEngine().emit(mlir::UnknownLoc::get(Context),
                                           mlir::DiagnosticSeverity::Error);
    };
    return clift::importModelType(EmitError, *Context, Type, Model);
  }

  clift::ValueType importModelType(const model::TypeDefinition &Type) {
    auto EmitError = [&]() -> mlir::InFlightDiagnostic {
      return Context->getDiagEngine().emit(mlir::UnknownLoc::get(Context),
                                           mlir::DiagnosticSeverity::Error);
    };
    return clift::importModelType(EmitError, *Context, Type, Model);
  }

  /* LLVM expression import */

  static uint64_t getIntegerConstant(const llvm::ConstantInt *Constant) {
    // WIP: Revisit ConstantInt to uint64_t conversion.
    return Constant->getZExtValue();
  }

  clift::ValueType emitFunctionDeclaration(const llvm::Function *F) {
    const model::Function *MF = llvmToModelFunction(Model, *F);

    clift::ValueType FunctionType = MF != nullptr
      ? importModelType(*MF->Prototype())
      : importLLVMType(F->getFunctionType());

    Builder.create<FunctionOp>(mlir::UnknownLoc::get(Context),
                               F->getName(),
                               FunctionType);

    return FunctionType;
  }

  clift::ValueType emitVariableDeclaration(const llvm::GlobalVariable *V) {
    // WIP: Import type from the model.
    clift::ValueType VariableType = importLLVMType(V->getType());

    Builder.create<GlobalVariableOp>(mlir::UnknownLoc::get(Context),
                                     V->getName(),
                                     VariableType);

    return VariableType;
  }

  clift::ValueType emitGlobalObject(const llvm::GlobalObject *G) {
    auto [Iterator, Inserted] = SymbolMapping.try_emplace(G);
    if (Inserted) {
      Iterator->second = [&]() -> clift::ValueType {
        mlir::OpBuilder::InsertionGuard Guard(Builder);
        Builder.setInsertionPointToEnd(&CurrentModule.getBody().front());

        if (auto *F = llvm::dyn_cast<llvm::Function>(G))
          return emitFunctionDeclaration(F);

        if (auto *V = llvm::dyn_cast<llvm::GlobalVariable>(G))
          return emitVariableDeclaration(V);

        revng_abort("Unsupported global object kind");
      }();
    }
    return Iterator->second;
  }

  template<typename OpT, typename... ArgsT>
  mlir::Value emitExpr(mlir::Location Loc, ArgsT &&...Args) {
    return Builder.create<OpT>(Loc, std::forward<ArgsT>(Args)...).getResult();
  }

  mlir::Value emitCast(mlir::Location Loc,
                       mlir::Value Value,
                       clift::ValueType TargetType,
                       CastKind Kind = CastKind::Reinterpret) {
    if (Value.getType() != TargetType)
      Value = Builder.create<CastOp>(Loc, TargetType, Value, Kind).getResult();

    return Value;
  }

  mlir::Value emitImplicitCast(mlir::Location Loc,
                               mlir::Value Value,
                               clift::ValueType TargetType) {
    auto SourceType = mlir::cast<clift::ValueType>(Value.getType());

    if (SourceType == TargetType)
      return Value;

    auto UnderlyingSourceType = dealias(SourceType, true);
    auto UnderlyingTargetType = dealias(TargetType, true);

    if (UnderlyingSourceType == UnderlyingTargetType) {
      if (SourceType.isConst() >= TargetType.isConst())
        return emitCast(Loc, Value, TargetType);
    }

    if (auto TI = getUnderlyingIntegerType(UnderlyingTargetType)) {
      if (auto SI = getUnderlyingIntegerType(UnderlyingSourceType)) {
        if (TI.getByteSize() == SI.getByteSize())
          return emitCast(Loc, Value, TargetType);
      }
    }

#if 0
    if (auto Struct = getTypeDefinitionAttr<StructTypeAttr>(SourceType)) {
      if (auto Fields = Struct.getFields(); Fields.size() == 1) {
        if (SourceType.removeConst() == Fields[0].getType().removeConst()) {
          return Builder.create<AggregateOp>(Loc, TargetType, Value);
        }
      }
    }
#endif

    return Value;
  }

  mlir::Value emitIntegerOp(mlir::Location Loc,
                            PrimitiveKind Kind,
                            auto ApplyOperation,
                            std::same_as<mlir::Value> auto... Operands) {
    auto ConvertToKind = [&](mlir::Value& Value, PrimitiveKind Kind) {
      uint64_t Size = getUnderlyingIntegerType(Value.getType()).getSize();
      Value = emitCast(Loc, Value, getPrimitiveType(Size, Kind));
    };

    if (Kind != PrimitiveKind::GenericKind)
      (ConvertToKind(Operands, Kind), ...);

    mlir::Value Result = ApplyOperation(Operands...);

    if (Kind != PrimitiveKind::GenericKind)
      ConvertToKind(Result, PrimitiveKind::GenericKind);

    return Result;
  }

  mlir::Value emitIntegerCast(mlir::Location Loc,
                              mlir::Value Operand,
                              uint64_t Size,
                              PrimitiveKind Kind) {
    uint64_t SrcSize = getUnderlyingIntegerType(Operand.getType()).getSize();

    CastKind Cast = CastKind::Reinterpret;
    if (Size > SrcSize)
      Cast = CastKind::Extend;
    if (Size < SrcSize)
      Cast = CastKind::Truncate;

    return emitIntegerOp(Loc, Kind, [&](mlir::Value Operand) {
      return emitCast(Loc, Operand, getPrimitiveType(Size, Kind), Cast);
    }, Operand);
  }

  RecursiveCoroutine<mlir::Value> emitExpression(const llvm::Value *V) {
    if (auto G = llvm::dyn_cast<llvm::GlobalObject>(V)) {
      auto Op = Builder.create<UseOp>(mlir::UnknownLoc::get(Context),
                                      emitGlobalObject(G),
                                      G->getName());
      rc_return Op.getResult();
    }

    if (auto It = ValueMapping.find(V); It != ValueMapping.end())
      rc_return It->second;

    if (auto U = llvm::dyn_cast<llvm::UndefValue>(V)) {
      mlir::Type Type = importLLVMType(U->getType());
      rc_return Builder.create<UndefOp>(mlir::UnknownLoc::get(Context), Type);
    }

    if (auto C = llvm::dyn_cast<llvm::ConstantInt>(V)) {
      const llvm::IntegerType* T = llvm::cast<llvm::IntegerType>(C->getType());
      rc_return Builder.create<ImmediateOp>(mlir::UnknownLoc::get(Context),
                                            importLLVMIntegerType(T),
                                            getIntegerConstant(C)).getResult();
    }

    if (auto N = llvm::dyn_cast<llvm::ConstantPointerNull>(V)) {
      mlir::Location Loc = mlir::UnknownLoc::get(Context);
      auto Op = Builder.create<ImmediateOp>(Loc, getIntptrType(), 0);
      rc_return emitCast(Loc, Op.getResult(), getVoidPointerType());
    }

    if (auto E = llvm::dyn_cast<llvm::ConstantExpr>(V)) {
      // WIP: Implement direct ConstantExpr import.
      llvm::Instruction *I = E->getAsInstruction();
      mlir::Value Value = emitExpression(I);
      I->deleteValue();
      rc_return Value;
    }

    if (auto I = llvm::dyn_cast<llvm::AllocaInst>(V)) {
      auto Iterator = AllocaMapping.find(I);
      revng_assert(Iterator != AllocaMapping.end());

      rc_return Builder.create<AddressofOp>(mlir::UnknownLoc::get(Context),
                                            Iterator->second.getType(),
                                            Iterator->second);
    }

    if (auto I = llvm::dyn_cast<llvm::LoadInst>(V)) {
      mlir::Value Pointer = rc_recur emitExpression(I->getPointerOperand());

      clift::ValueType ValueType = importLLVMType(V->getType());
      clift::ValueType PointerType = makePointerType(ValueType);

      auto Op1 = Builder.create<CastOp>(mlir::UnknownLoc::get(Context),
                                        PointerType,
                                        Pointer,
                                        CastKind::Reinterpret);

      auto Op2 = Builder.create<IndirectionOp>(mlir::UnknownLoc::get(Context),
                                               ValueType,
                                               Op1.getResult());

      rc_return Op2.getResult();
    }

    if (auto I = llvm::dyn_cast<llvm::StoreInst>(V)) {
      mlir::Value Pointer = rc_recur emitExpression(I->getPointerOperand());
      mlir::Value Value = rc_recur emitExpression(I->getValueOperand());
      auto PointerType = makePointerType(Value.getType());

      auto Op1 = Builder.create<CastOp>(mlir::UnknownLoc::get(Context),
                                        PointerType,
                                        Pointer,
                                        CastKind::Reinterpret);

      auto Op2 = Builder.create<IndirectionOp>(mlir::UnknownLoc::get(Context),
                                               PointerType,
                                               Op1.getResult());

      auto Op3 = Builder.create<AssignOp>(mlir::UnknownLoc::get(Context),
                                          Value.getType(),
                                          Op2.getResult(),
                                          Value);

      rc_return Op3.getResult();
    }

    if (auto I = llvm::dyn_cast<llvm::BinaryOperator>(V)) {
      using Operators = llvm::BinaryOperator::BinaryOps;

      PrimitiveKind Kind = PrimitiveKind::GenericKind;
      switch (I->getOpcode()) {
      case Operators::SDiv:
      case Operators::SRem:
      case Operators::AShr:
        Kind = PrimitiveKind::SignedKind;
        break;
      case Operators::UDiv:
      case Operators::URem:
      case Operators::LShr:
        Kind = PrimitiveKind::UnsignedKind;
        break;
      default:
        break;
      }

      mlir::Value Lhs = rc_recur emitExpression(I->getOperand(0));
      mlir::Value Rhs = rc_recur emitExpression(I->getOperand(1));

      auto *IntegerType = llvm::cast<llvm::IntegerType>(V->getType());
      auto Type = importLLVMIntegerType(IntegerType, Kind);

      mlir::Location Loc = mlir::UnknownLoc::get(Context);
      rc_return emitIntegerOp(Loc, Kind, [&](mlir::Value Lhs, mlir::Value Rhs) {
        switch (I->getOpcode()) {
        case Operators::Add:
          return emitExpr<AddOp>(Loc, Type, Lhs, Rhs);
        case Operators::Sub:
          return emitExpr<SubOp>(Loc, Type, Lhs, Rhs);
        case Operators::Mul:
          return emitExpr<MulOp>(Loc, Type, Lhs, Rhs);
        case Operators::SDiv:
        case Operators::UDiv:
          return emitExpr<DivOp>(Loc, Type, Lhs, Rhs);
        case Operators::SRem:
        case Operators::URem:
          return emitExpr<RemOp>(Loc, Type, Lhs, Rhs);
        case Operators::Shl:
          return emitExpr<ShiftLeftOp>(Loc, Type, Lhs, Rhs);
        case Operators::LShr:
        case Operators::AShr:
          return emitExpr<ShiftRightOp>(Loc, Type, Lhs, Rhs);
        case Operators::And:
          return emitExpr<BitwiseAndOp>(Loc, Type, Lhs, Rhs);
        case Operators::Or:
          return emitExpr<BitwiseOrOp>(Loc, Type, Lhs, Rhs);
        case Operators::Xor:
          return emitExpr<BitwiseXorOp>(Loc, Type, Lhs, Rhs);
        default:
          revng_abort("Unsupported LLVM binary operator.");
        }
      }, Lhs, Rhs);
    }

    if (auto I = llvm::dyn_cast<llvm::ICmpInst>(V)) {
      using enum llvm::ICmpInst::Predicate;

      PrimitiveKind Kind = PrimitiveKind::GenericKind;
      switch (I->getPredicate()) {
      case ICMP_SGT:
      case ICMP_SGE:
      case ICMP_SLT:
      case ICMP_SLE:
        Kind = PrimitiveKind::SignedKind;
        break;
      case ICMP_UGT:
      case ICMP_UGE:
      case ICMP_ULT:
      case ICMP_ULE:
        Kind = PrimitiveKind::UnsignedKind;
        break;
      default:
        break;
      }

      mlir::Value Lhs = rc_recur emitExpression(I->getOperand(0));
      mlir::Value Rhs = rc_recur emitExpression(I->getOperand(1));

      auto *IntegerType = llvm::cast<llvm::IntegerType>(V->getType());
      auto Type = importLLVMIntegerType(IntegerType, Kind);

      auto Loc = mlir::UnknownLoc::get(Context);
      rc_return emitIntegerOp(Loc, Kind, [&](mlir::Value Lhs, mlir::Value Rhs) {
        switch (I->getPredicate()) {
        case ICMP_EQ:
          return emitExpr<EqualOp>(Loc, Type, Lhs, Rhs);
        case ICMP_NE:
          return emitExpr<NotEqualOp>(Loc, Type, Lhs, Rhs);
        case ICMP_SGT:
        case ICMP_UGT:
          return emitExpr<GreaterThanOp>(Loc, Type, Lhs, Rhs);
        case ICMP_SGE:
        case ICMP_UGE:
          return emitExpr<GreaterThanOrEqualOp>(Loc, Type, Lhs, Rhs);
        case ICMP_SLT:
        case ICMP_ULT:
          return emitExpr<LessThanOp>(Loc, Type, Lhs, Rhs);
        case ICMP_SLE:
        case ICMP_ULE:
          return emitExpr<LessThanOrEqualOp>(Loc, Type, Lhs, Rhs);
        default:
          revng_abort("Unsupported LLVM comparison predicate.");
        }
      }, Lhs, Rhs);
    }

    if (auto I = llvm::dyn_cast<llvm::CastInst>(V)) {
      mlir::Value Operand = rc_recur emitExpression(I->getOperand(0));
      mlir::Location Loc = mlir::UnknownLoc::get(Context);

      auto emitIntegerCast = [&](PrimitiveKind Kind) {
        auto *IntegerType = llvm::cast<llvm::IntegerType>(V->getType());
        uint64_t TargetSize = getIntegerSize(IntegerType->getBitWidth());
        return this->emitIntegerCast(Loc, Operand, TargetSize, Kind);
      };

      switch (I->getOpcode()) {
        using Operators = llvm::CastInst::CastOps;
      case Operators::Trunc:
        rc_return emitIntegerCast(PrimitiveKind::GenericKind);
      case Operators::SExt:
        rc_return emitIntegerCast(PrimitiveKind::SignedKind);
      case Operators::ZExt:
        rc_return emitIntegerCast(PrimitiveKind::UnsignedKind);
      case Operators::PtrToInt:
        rc_return emitCast(Loc, Operand, getIntptrType());
      case Operators::IntToPtr:
        rc_return emitCast(Loc, Operand, getVoidPointerType());
      default:
        revng_abort("Unsupported LLVM cast operation.");
      }
    }

    if (auto I = llvm::dyn_cast<llvm::CallInst>(V)) {
      mlir::Location Loc = mlir::UnknownLoc::get(Context);

      mlir::Value Function = rc_recur emitExpression(I->getCalledOperand());

      mlir::Type FunctionType = Function.getType();
      if (auto PT = mlir::dyn_cast<PointerType>(dealias(FunctionType, true)))
        FunctionType = PT.getPointeeType();

      FunctionTypeAttr FunctionAttr = getFunctionTypeAttr(FunctionType);

      mlir::Type CallType = FunctionType;
      if (I->getMetadata(PrototypeMDName)) // WIP
      if (const auto *ModelCallType = getCallSitePrototype(Model, I)) {
        CallType = importModelType(*ModelCallType);
      }

      if (CallType != FunctionType) {
        if (FunctionType != Function.getType()) {
          Function = emitCast(Loc,
                              Function,
                              makePointerType(FunctionType),
                              CastKind::Decay);
        }

        Function = emitCast(Loc, Function, makePointerType(CallType));
      }

      auto ReturnType = getFunctionTypeAttr(CallType).getReturnType();

      llvm::SmallVector<mlir::Value> Arguments;
      for (const llvm::Value *Arg : I->args())
        Arguments.push_back(rc_recur emitExpression(Arg));

      rc_return Builder.create<CallOp>(Loc,
                                       ReturnType,
                                       Function,
                                       Arguments)->getResult(0);
    }

    if (auto I = llvm::dyn_cast<llvm::SelectInst>(V)) {
      mlir::Value Condition = rc_recur emitExpression(I->getCondition());

      mlir::Value True = rc_recur emitExpression(I->getTrueValue());
      mlir::Value False = rc_recur emitExpression(I->getFalseValue());
      revng_assert(True.getType() == False.getType());

      rc_return Builder.create<TernaryOp>(mlir::UnknownLoc::get(Context),
                                          True.getType(),
                                          Condition,
                                          True,
                                          False).getResult();
    }

    if (auto I = llvm::dyn_cast<llvm::FreezeInst>(V))
      rc_return emitExpression(I->getOperand(0));

    revng_abort("Unsupported LLVM instruction.");
  }

  void emitExpressionTreeImpl(mlir::Region &R, auto EmitExpression) {
    revng_assert(R.empty());

    mlir::OpBuilder::InsertionGuard Guard(Builder);
    Builder.setInsertionPointToEnd(&R.emplaceBlock());

    Builder.create<YieldOp>(mlir::UnknownLoc::get(Context), EmitExpression());
  }

  void emitExpressionTree(mlir::Region &R, const llvm::Value *V) {
    emitExpressionTreeImpl(R, [&]() { return emitExpression(V); });
  }

#if 0
  mlir::Type emitEmptyExpressionTree(mlir::Region &R) {
    revng_assert(R.empty());

    mlir::OpBuilder::InsertionGuard Guard(Builder);
    Builder.setInsertionPointToEnd(&R.emplaceBlock());

    mlir::Type Void = getPrimitiveType(0, PrimitiveKind::VoidKind);
    mlir::Value Value = Builder.create<UndefOp>(mlir::UnknownLoc::get(Context),
                                                Void);

    Builder.create<YieldOp>(mlir::UnknownLoc::get(Context), Value);

    return Value.getType();
  }
#endif

  /* LLVM control flow import */

  static mlir::Block::iterator getLabelInsertionPoint(mlir::Block *Block) {
    mlir::Block::iterator Beg = Block->begin();
    mlir::Block::iterator End = Block->end();

    while (Beg != End && mlir::isa<MakeLabelOp>(*Beg))
      ++Beg;

    return Beg;
  }

  mlir::Value emitMakeLabel() {
    mlir::OpBuilder::InsertionGuard Guard(Builder);
    Builder.setInsertionPoint(LabelBlock, getLabelInsertionPoint(LabelBlock));

    char StringBuffer[32];
    snprintf(StringBuffer, sizeof(StringBuffer), "BB_%u", LabelCount++);

    auto Op = Builder.create<MakeLabelOp>(mlir::UnknownLoc::get(Context),
                                          llvm::StringRef(StringBuffer));

    return Op.getResult();
  }

  void emitAssignLabel(mlir::Value Label) {
    Builder.create<AssignLabelOp>(mlir::UnknownLoc::get(Context), Label);
  }

  static bool isScopeGraphEdge(const llvm::BasicBlock *Pred,
                               const llvm::BasicBlock *Succ) {
    for (auto *BB : llvm::children<Scope<const llvm::BasicBlock *>>(Pred)) {
      if (BB == Succ)
        return true;
    }
    return false;
  }

  static bool isUsedOutsideOfBlock(const llvm::Value *V,
                                   const llvm::BasicBlock *BB) {
    for (const llvm::User *U : V->users()) {
      if (auto *I = llvm::dyn_cast<llvm::Instruction>(U)) {
        if (I->getParent() != BB)
          return false;
      }
    }
    return true;
  }

  RecursiveCoroutine<void>
  emitBasicBlock(const llvm::BasicBlock *BB,
                 const llvm::BasicBlock *InnerPostDom,
                 const llvm::BasicBlock *OuterPostDom) {
    myprintl("emitBasicBlock(%p, %p, %p)", BB, InnerPostDom, OuterPostDom);
    auto ggg = indent();

    // Map BB to the MLIR block, emit label if necessary:
    {
      auto [Iterator, Inserted] = BlockMapping.try_emplace(BB);

      // WIP: Just for testing on switch-to-statements output.
      if (Iterator->second.InsertPoint.isSet())
        rc_return;

      revng_assert(not Iterator->second.InsertPoint.isSet());
      Iterator->second.InsertPoint = saveInsertionPointAfter(Builder);

      if (not Inserted and Iterator->second.Label)
        myprintl("assign label 1"), emitAssignLabel(Iterator->second.Label);
    }

    const llvm::Instruction *Terminal = BB->getTerminator();
    bool HasGotoMarker = false;

    for (const llvm::Instruction &I : *BB) {
      if (&I == Terminal)
        break;

      // WIP: Remove this
      if (isCallToHelper(&I))
        continue;

      if (isCallToTagged(&I, FunctionTags::GotoBlockMarker)) {
        HasGotoMarker = true;
        continue;
      }

      // Scope closer markers are just ignored. They only affect the scope graph
      // structure.
      if (isCallToTagged(&I, FunctionTags::ScopeCloserMarker))
        continue;

      if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
        clift::ValueType Type = importLLVMType(Alloca->getAllocatedType());

        llvm::SmallString<16> Name;
        {
          llvm::raw_svector_ostream OutName(Name);
          OutName << "_local_" << LocalCount++;
        }

        auto Op =
          Builder.create<LocalVariableOp>(mlir::UnknownLoc::get(Context),
                                          Type,
                                          Name);

        auto [Iterator, Inserted] = AllocaMapping.try_emplace(Alloca,
                                                              Op.getResult());
        revng_assert(Inserted);

        continue;
      }

      if (I.use_empty()) {
        // Any instruction with no uses can be considered the root of an
        // expression tree. The tree can be emitted in an expression statement.
        auto Op =
          Builder.create<ExpressionStatementOp>(mlir::UnknownLoc::get(Context));

        emitExpressionTree(Op.getExpression(), &I);
      }

#if 0
      if (I.use_empty()) {
        // Any instruction with no uses can be considered the root of an
        // expression tree. The tree can be emitted in an expression statement.
        auto Op = Builder.create<ExpressionStatementOp>(mlir::UnknownLoc::get(Context));
        emitExpressionTree(&I, Op.getExpression());
      } else if (I.hasNUsesOrMore(2) or isUsedOutsideOfBlock(&I, BB)) {
        mlir::Region R;
        mlir::Type Type = emitExpressionTree(&I, R);

        // Any instruction with more than one use, or with a single use outside
        // of this block must be emitted into a local variable initializer.
        auto Op = Builder.create<LocalVariableOp>(mlir::UnknownLoc::get(Context),
                                                  Type,
                                                  ""); // WIP: Name registerN

        // Move the local block into the initializer region.
        Op.getInitializer().push_back(R.getBlocks().remove(R.front()));

        // Map this instruction value to the newly created local variable.
        auto [It, Inserted] = ValueMapping.try_emplace(&I, Op.getResult());
        revng_assert(Inserted);
      }
#endif
    }

    if (Terminal->getNumSuccessors() == 1) {
      const llvm::BasicBlock *Succ = Terminal->getSuccessor(0);

      if (HasGotoMarker) {
        myprintl("goto %p", Succ);
        auto [Iterator, Inserted] = BlockMapping.try_emplace(Succ);

        if (not Iterator->second.Label)
          Iterator->second.Label = emitMakeLabel();

        if (Iterator->second.InsertPoint.isSet()) {
          myprintl("assign label 2");
          mlir::OpBuilder::InsertionGuard Guard(Builder);
          restoreInsertionPointAfter(Builder, Iterator->second.InsertPoint);
          emitAssignLabel(Iterator->second.Label);
        }

        Builder.create<GoToOp>(mlir::UnknownLoc::get(Context),
                               Iterator->second.Label);
      } else {
        rc_recur emitBasicBlock(Succ, InnerPostDom);
      }

      rc_return;
    }

    revng_assert(not HasGotoMarker);

    if (llvm::isa<llvm::UnreachableInst>(Terminal)) {
      // WIP: Emit ~ call to __builtin_trap()
    } else if (auto *Return = llvm::dyn_cast<llvm::ReturnInst>(Terminal)) {
      auto Op = Builder.create<ReturnOp>(mlir::UnknownLoc::get(Context));
      if (const llvm::Value *Value = Return->getReturnValue()) {
        auto FunctionType = CurrentFunction.getFunctionType();
        auto ReturnType = getFunctionTypeAttr(FunctionType).getReturnType();

        emitExpressionTreeImpl(Op.getResult(), [&]() {
          return emitImplicitCast(mlir::UnknownLoc::get(Context),
                                  emitExpression(Value),
                                  ReturnType);
        });
      }
    } else if (auto *Branch = llvm::dyn_cast<llvm::BranchInst>(Terminal)) {
      auto Op = Builder.create<IfOp>(mlir::UnknownLoc::get(Context));
      emitExpressionTree(Op.getCondition(), Branch->getCondition());
      mlir::OpBuilder::InsertionGuard Guard(Builder);

      // Emit true branch:
      revng_assert(Op.getThen().empty());
      Builder.setInsertionPointToEnd(&Op.getThen().emplaceBlock());
      rc_recur emitBasicBlock(Branch->getSuccessor(0), InnerPostDom);

      // Emit false branch:
      revng_assert(Op.getElse().empty());
      Builder.setInsertionPointToEnd(&Op.getElse().emplaceBlock());
      rc_recur emitBasicBlock(Branch->getSuccessor(1), InnerPostDom);
    } else if (auto *Switch = llvm::dyn_cast<llvm::SwitchInst>(Terminal)) {
      llvm::SmallVector<uint64_t> CaseValues;
      CaseValues.reserve(Switch->getNumCases());

      for (auto CH : Switch->cases())
        CaseValues.push_back(getIntegerConstant(CH.getCaseValue()));

      auto Op = Builder.create<SwitchOp>(mlir::UnknownLoc::get(Context),
                                         CaseValues);
      emitExpressionTree(Op.getCondition(), Switch->getCondition());
      mlir::OpBuilder::InsertionGuard Guard(Builder);

      // Emit case blocks:
      for (auto [I, CH] : llvm::enumerate(Switch->cases())) {
        revng_assert(Op.getCaseRegion(I).empty());
        Builder.setInsertionPointToEnd(&Op.getCaseRegion(I).emplaceBlock());
        rc_recur emitBasicBlock(CH.getCaseSuccessor(), InnerPostDom);
      }

      // Emit default block:
      if (const llvm::BasicBlock *Succ = Switch->getDefaultDest();
          Succ != nullptr and isScopeGraphEdge(BB, Succ)) {

        revng_assert(Op.getDefaultCaseRegion().empty());
        Builder.setInsertionPointToEnd(&Op.getDefaultCaseRegion().emplaceBlock());
        rc_recur emitBasicBlock(Succ, InnerPostDom);
      }
    } else {
      revng_abort("Unsupported terminal instruction");
    }
  }

  RecursiveCoroutine<void>
  emitBasicBlock(const llvm::BasicBlock *BB,
                 const llvm::BasicBlock *OuterPostDom) {
    auto ggg = indent();
    while (BB != OuterPostDom) {
      const llvm::BasicBlock *InnerPostDom =
        PostDomTree[BB]->getIDom()->getBlock();

      rc_recur emitBasicBlock(BB, InnerPostDom, OuterPostDom);

      BB = InnerPostDom;
    }
  }

  /* LLVM module import */

  void importFunction(const llvm::Function *F) {
    const model::Function *MF = llvmToModelFunction(Model, *F);

    if (MF == nullptr)
      return;

    auto Op = Builder.create<FunctionOp>(mlir::UnknownLoc::get(Context),
                                         F->getName(),
                                         importModelType(*MF->Prototype()));

    revng_assert(F->arg_size() == Op.getArgumentTypes().size());

    Op.setUniqueHandle(pipeline::locationString(revng::ranks::Function,
                                                MF->Entry()));

    revng_assert(Op.getBody().empty());
    mlir::Block &BodyBlock = Op.getBody().emplaceBlock();

    CurrentFunction = Op;
    LabelBlock = &BodyBlock;
    LabelCount = 0;
    LocalCount = 0;

    // Clear the mappings once this function is emitted.
    auto MappingGuard = llvm::make_scope_exit([&]() {
      BlockMapping.clear();
      AllocaMapping.clear();
      ValueMapping.clear();
    });

    for (const auto [A, T] : llvm::zip(F->args(), Op.getArgumentTypes())) {
      mlir::Value Arg = BodyBlock.addArgument(T,
                                              mlir::UnknownLoc::get(Context));

      ValueMapping.try_emplace(&A, Arg);
    }

    PostDomTree.recalculate(const_cast<llvm::Function&>(*F));

    mlir::OpBuilder::InsertionGuard BuilderGuard(Builder);
    Builder.setInsertionPointToEnd(&BodyBlock);

    emitBasicBlock(&F->getEntryBlock(), /*OuterPostDom=*/nullptr);
  }

  mlir::OwningOpRef<clift::ModuleOp> importModule(const llvm::Module *Module) {
    auto Op = createOperation<clift::ModuleOp>(Context,
                                               mlir::UnknownLoc::get(Context));

    CurrentModule = Op.get();

    revng_assert(Op->getBody().hasOneBlock());
    Builder.setInsertionPointToEnd(&Op->getBody().front());

    for (const llvm::Function &F : Module->functions())
      importFunction(&F);

    return Op;
  }


  mlir::MLIRContext *const Context;
  const model::Binary &Model;
  mlir::OpBuilder Builder;

  clift::ModuleOp CurrentModule;
  clift::FunctionOp CurrentFunction;

  ScopeGraphPostDomTree PostDomTree;

  clift::ValueType VoidTypeCache;
  clift::ValueType PointerTypeCache;

  struct BlockMappingInfo
  {
    // Point where the mapped LLVM IR basic block is emitted. If the block has
    // not yet been visited, this is not set. Note that the LLVM IR basic block
    // is not necessarily emitted at the *start* of any MLIR block.
    mlir::OpBuilder::InsertPoint InsertPoint;

    // The result value of the MakeLabelOp to be used as the target for gotos
    // jumping into the mapped LLVM IR basic block.
    mlir::Value Label;
  };

  llvm::DenseMap<const llvm::GlobalObject *, clift::ValueType> SymbolMapping;
  llvm::DenseMap<const llvm::BasicBlock *, BlockMappingInfo> BlockMapping;
  llvm::DenseMap<const llvm::AllocaInst *, mlir::Value> AllocaMapping;
  llvm::DenseMap<const llvm::Value *, mlir::Value> ValueMapping;

  mlir::Block *LabelBlock = nullptr;
  unsigned LabelCount = 0;
  unsigned LocalCount = 0;

  uint64_t NextInventedTypeID = 1'000'000'000;
};

} // namespace

mlir::OwningOpRef<clift::ModuleOp>
clift::importLLVM(mlir::MLIRContext *Context,
                  const model::Binary &Model,
                  const llvm::Module *Module) {
  return LLVMCodeImporter::import(Context, Model, Module);
}
