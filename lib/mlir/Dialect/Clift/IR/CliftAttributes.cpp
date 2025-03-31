//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "revng/mlir/Dialect/Clift/IR/Clift.h"
#include "revng/mlir/Dialect/Clift/IR/CliftAttributes.h"
#include "revng/mlir/Dialect/Clift/IR/CliftInterfaces.h"
#include "revng/mlir/Dialect/Clift/IR/CliftTypes.h"

// This include should stay here for correct build procedure
//
#define GET_ATTRDEF_CLASSES
#include "revng/mlir/Dialect/Clift/IR/CliftAttributes.cpp.inc"

using namespace mlir::clift;
namespace clift = mlir::clift;

using EmitErrorType = llvm::function_ref<mlir::InFlightDiagnostic()>;

//===-------------------------- Class attributes --------------------------===//

using WalkAttrT = llvm::function_ref<void(mlir::Attribute)>;
using WalkTypeT = llvm::function_ref<void(mlir::Type)>;

using ReplaceAttrT = llvm::ArrayRef<mlir::Attribute>;
using ReplaceTypeT = llvm::ArrayRef<mlir::Type>;

namespace mlir::clift {

class ClassAttrStorage : public mlir::AttributeStorage {
  struct Key {
    llvm::StringRef Handle;
    ClassDefinitionAttr Definition;

    explicit Key(llvm::StringRef Handle) : Handle(Handle) {}

    friend bool operator==(const Key &LHS, const Key &RHS) {
      return LHS.Handle == RHS.Handle;
    }

    [[nodiscard]] llvm::hash_code hashValue() const {
      return llvm::hash_value(Handle);
    }
  };

  Key TheKey;

public:
  using KeyTy = Key;

  const Key &getAsKey() const { return TheKey; }

  static llvm::hash_code hashKey(const Key &Key) { return Key.hashValue(); }

  friend bool operator==(const ClassAttrStorage &LHS, const Key &RHS) {
    return LHS.TheKey == RHS;
  }

  explicit ClassAttrStorage(llvm::StringRef Handle) : TheKey(Handle) {}

  static ClassAttrStorage *
  construct(mlir::StorageUniquer::StorageAllocator &Allocator, const Key &Key) {
    void *Storage = Allocator.allocate<ClassAttrStorage>();
    llvm::StringRef Handle = Allocator.copyInto(Key.Handle);
    auto *S = new (Storage) ClassAttrStorage(Handle);
    S->TheKey.Definition = Key.Definition;
    return S;
  }

  mlir::LogicalResult mutate(mlir::StorageUniquer::StorageAllocator &Allocator,
                             ClassDefinitionAttr Definition,
                             llvm::function_ref<mlir::LogicalResult()> Verify) {
    if (TheKey.Definition)
      return mlir::success(Definition == TheKey.Definition);

    mlir::LogicalResult Result = mlir::failure();
    auto Guard = llvm::make_scope_exit([&]() {
      if (Result.failed())
        TheKey.Definition = nullptr;
    });

    TheKey.Definition = Definition;
    return Result = Verify();
  }

  llvm::StringRef getHandle() const { return TheKey.Handle; }
  ClassDefinitionAttr getDefinition() const { return TheKey.Definition; }
};

template<typename AttrT>
llvm::StringRef ClassAttrImpl<AttrT>::getHandle() const {
  return Base::getImpl()->getHandle();
}

template<typename AttrT>
bool ClassAttrImpl<AttrT>::hasDefinition() const {
  return static_cast<bool>(Base::getImpl()->getDefinition());
}

template<typename AttrT>
ClassDefinitionAttr ClassAttrImpl<AttrT>::getDefinition() const {
  return Base::getImpl()->getDefinition();
}

template<typename AttrT>
void ClassAttrImpl<AttrT>::walkImmediateSubElements(WalkAttrT WalkAttr,
                                                    WalkTypeT WalkType) const {
  if (auto Definition = Base::getImpl()->getDefinition())
    WalkAttr(Definition);
}

template<typename AttrT>
mlir::Attribute
ClassAttrImpl<AttrT>::replaceImmediateSubElements(ReplaceAttrT NewAttrs,
                                                  ReplaceTypeT NewTypes) const {
  revng_abort("Cannot replace sub-elements of a class attribute.");
}

template class ClassAttrImpl<StructAttr>;
template class ClassAttrImpl<UnionAttr>;

} // namespace mlir::clift

//===------------------------------ FieldAttr -----------------------------===//

mlir::LogicalResult FieldAttr::verify(EmitErrorType EmitError,
                                      uint64_t Offset,
                                      clift::ValueType ElementType,
                                      llvm::StringRef Name) {
  if (not isObjectType(ElementType)) {
    return EmitError() << "Struct and union field types must be object types. "
                       << "Field at offset " << Offset << " is not.";
  }

  return mlir::success();
}

//===---------------------------- EnumFieldAttr ---------------------------===//

mlir::LogicalResult EnumFieldAttr::verify(EmitErrorType EmitError,
                                          uint64_t RawValue,
                                          llvm::StringRef Name) {
  return mlir::success();
}

//===------------------------------ EnumAttr ------------------------------===//

mlir::LogicalResult EnumAttr::verify(EmitErrorType EmitError,
                                     llvm::StringRef Handle,
                                     llvm::StringRef Name,
                                     clift::ValueType UnderlyingType,
                                     llvm::ArrayRef<EnumFieldAttr> Fields) {
  auto [DealiasedType, HasConst] = decomposeTypedef(UnderlyingType);

  auto PrimitiveType = mlir::dyn_cast<clift::PrimitiveType>(DealiasedType);
  if (not PrimitiveType or HasConst or PrimitiveType.isConst())
    return EmitError() << "Underlying type of enum must be a non-const "
                          "primitive type";

  const uint64_t BitWidth = PrimitiveType.getSize() * 8;

  if (Fields.empty())
    return EmitError() << "enum requires at least one field";

  uint64_t MinValue = 0;
  uint64_t MaxValue = 0;
  bool IsSigned = false;

  switch (PrimitiveType.getKind()) {
  case PrimitiveKind::UnsignedKind:
    MaxValue = llvm::APInt::getMaxValue(BitWidth).getZExtValue();
    break;
  case PrimitiveKind::SignedKind:
    MinValue = llvm::APInt::getSignedMinValue(BitWidth).getSExtValue();
    MaxValue = llvm::APInt::getSignedMaxValue(BitWidth).getSExtValue();
    IsSigned = true;
    break;
  default:
    return EmitError() << "enum underlying type must be an integral type";
  }

  uint64_t LastValue = 0;
  bool CheckEqual = false;

  for (const auto &Field : Fields) {
    const uint64_t Value = Field.getRawValue();

    const auto UsingSigned = [&](auto Callable, const auto... V) {
      return IsSigned ? Callable(static_cast<int64_t>(V)...) : Callable(V...);
    };

    const auto CheckSigned =
      [EmitError](const auto Value,
                  const auto MinValue,
                  const auto MaxValue) -> mlir::LogicalResult {
      if (Value < MinValue)
        return EmitError() << "enum field " << Value
                           << " is less than the min value of the "
                              "underlying type "
                           << MinValue;

      if (Value > MaxValue)
        return EmitError() << "enum field " << Value
                           << " is greater than the max value of the "
                              "underlying type "
                           << MaxValue;

      return mlir::success();
    };

    const mlir::LogicalResult R = UsingSigned(CheckSigned,
                                              Value,
                                              MinValue,
                                              MaxValue);

    if (failed(R))
      return R;

    if (Value < LastValue || (CheckEqual && Value == LastValue))
      return EmitError() << "enum fields must be strictly ordered by their "
                            "unsigned values";

    LastValue = Value;
    CheckEqual = true;
  }

  return mlir::success();
}

//===----------------------------- TypedefAttr ----------------------------===//

mlir::LogicalResult TypedefAttr::verify(EmitErrorType EmitError,
                                        llvm::StringRef Handle,
                                        llvm::StringRef Name,
                                        clift::ValueType UnderlyingType) {
  return mlir::success();
}

//===----------------------------- StructAttr -----------------------------===//

static mlir::LogicalResult verifyStructSize(EmitErrorType EmitError,
                                            uint64_t Size) {
  if (Size == 0)
    return EmitError() << "struct type cannot have a size of zero";

  return mlir::success();
}

static mlir::LogicalResult verifyStructImpl(EmitErrorType EmitError,
                                            ClassDefinitionAttr Definition) {
  if (verifyStructSize(EmitError, Definition.getSize()).failed())
    return mlir::failure();

  if (not Definition.getFields().empty()) {
    uint64_t LastEndOffset = 0;

    llvm::SmallSet<llvm::StringRef, 16> NameSet;
    for (const auto &Field : Definition.getFields()) {
      if (Field.getOffset() < LastEndOffset)
        return EmitError() << "Fields of structs must be ordered by offset, "
                              "and "
                              "they cannot overlap";

      LastEndOffset = Field.getOffset() + Field.getType().getByteSize();

      if (not Field.getName().empty()) {
        if (not NameSet.insert(Field.getName()).second)
          return EmitError() << "struct field names must be empty or unique";
      }
    }

    if (LastEndOffset > Definition.getSize())
      return EmitError() << "offset + size of field of struct type is greater "
                            "than the struct type size.";
  }

  return mlir::success();
}

mlir::LogicalResult StructAttr::verify(EmitErrorType EmitError,
                                       llvm::StringRef Handle) {
  return mlir::success();
}

mlir::LogicalResult StructAttr::verify(EmitErrorType EmitError,
                                       llvm::StringRef Handle,
                                       ClassDefinitionAttr Definition) {
  auto Attr = Base::get(Definition.getContext(), Handle);

  mlir::LogicalResult Result = mlir::success();
  auto Verify = [&]() {
    Result = verifyStructImpl(EmitError, Definition);
    // We don't actually want to mutate here, so return failure.
    return mlir::failure();
  };

  mlir::LogicalResult R = Attr.Base::mutate(Definition, Verify);
  revng_assert(R.failed() or Attr.getDefinition() == Definition);

  return Result;
}

mlir::LogicalResult StructAttr::verify(EmitErrorType EmitError,
                                       llvm::StringRef Handle,
                                       llvm::StringRef Name,
                                       uint64_t Size,
                                       llvm::ArrayRef<FieldAttr> Fields) {
  if (Fields.empty())
    return verifyStructSize(EmitError, Size);

  // The context must be inferred from a field type in order to construct the
  // class definition attribute.
  auto Definition = ClassDefinitionAttr::get(Fields.front().getContext(),
                                             Name,
                                             Size,
                                             Fields);

  return verify(EmitError, Handle, Definition);
}

StructAttr StructAttr::get(MLIRContext *Context, llvm::StringRef Handle) {
  return Base::get(Context, Handle);
}

StructAttr StructAttr::getChecked(EmitErrorType EmitError,
                                  MLIRContext *Context,
                                  llvm::StringRef Handle) {
  return Base::get(Context, Handle);
}

StructAttr StructAttr::get(MLIRContext *Context,
                           llvm::StringRef Handle,
                           ClassDefinitionAttr Definition) {
  auto Attr = Base::get(Context, Handle);
  auto R = Attr.Base::mutate(Definition, [] { return mlir::success(); });
  revng_assert(R.succeeded()
               and "Attempted to mutate the definition of an already defined "
                   "struct attribute.");
  return Attr;
}

StructAttr StructAttr::getChecked(EmitErrorType EmitError,
                                  MLIRContext *Context,
                                  llvm::StringRef Handle,
                                  ClassDefinitionAttr Definition) {
  auto Attr = Base::get(Context, Handle);

  auto Verify = [&]() -> mlir::LogicalResult {
    return verifyStructImpl(EmitError, Definition);
  };

  if (Attr.Base::mutate(Definition, Verify).failed())
    return {};

  return Attr;
}

StructAttr StructAttr::get(MLIRContext *Context,
                           llvm::StringRef Handle,
                           llvm::StringRef Name,
                           uint64_t Size,
                           llvm::ArrayRef<FieldAttr> Fields) {
  return get(Context,
             Handle,
             ClassDefinitionAttr::get(Context, Name, Size, Fields));
}

StructAttr StructAttr::getChecked(EmitErrorType EmitError,
                                  MLIRContext *Context,
                                  llvm::StringRef Handle,
                                  llvm::StringRef Name,
                                  uint64_t Size,
                                  llvm::ArrayRef<FieldAttr> Fields) {
  return getChecked(EmitError,
                    Context,
                    Handle,
                    ClassDefinitionAttr::get(Context, Name, Size, Fields));
}

//===------------------------------ UnionAttr -----------------------------===//

static uint64_t getUnionSize(llvm::ArrayRef<FieldAttr> Fields) {
  uint64_t Max = 0;
  for (auto const &Field : Fields)
    Max = std::max(Max, Field.getType().getByteSize());
  return Max;
}

mlir::LogicalResult UnionAttr::verify(EmitErrorType EmitError,
                                      llvm::StringRef Handle) {
  return mlir::success();
}

mlir::LogicalResult UnionAttr::verify(EmitErrorType EmitError,
                                      llvm::StringRef Handle,
                                      ClassDefinitionAttr Definition) {
  return verify(EmitError,
                Handle,
                Definition.getName(),
                Definition.getSize(),
                Definition.getFields());
}

mlir::LogicalResult UnionAttr::verify(EmitErrorType EmitError,
                                      llvm::StringRef Handle,
                                      llvm::StringRef Name,
                                      llvm::ArrayRef<FieldAttr> Fields) {
  if (Fields.empty())
    return EmitError() << "union types must have at least one field";

  llvm::SmallSet<llvm::StringRef, 16> NameSet;
  for (const auto &Field : Fields) {
    if (Field.getOffset() != 0)
      return EmitError() << "union field offsets must be zero";

    if (not Field.getName().empty()) {
      if (not NameSet.insert(Field.getName()).second)
        return EmitError() << "union field names must be empty or unique";
    }
  }

  return mlir::success();
}

UnionAttr UnionAttr::get(MLIRContext *Context, llvm::StringRef Handle) {
  return Base::get(Context, Handle);
}

UnionAttr UnionAttr::getChecked(EmitErrorType EmitError,
                                MLIRContext *Context,
                                llvm::StringRef Handle) {
  return Base::get(Context, Handle);
}

UnionAttr UnionAttr::get(MLIRContext *Context,
                         llvm::StringRef Handle,
                         ClassDefinitionAttr Definition) {
  auto Attr = Base::get(Context, Handle);
  auto R = Attr.Base::mutate(Definition, [] { return mlir::success(); });
  revng_assert(R.succeeded()
               and "Attempted to mutate the definition of an already defined "
                   "union attribute.");
  return Attr;
}

UnionAttr UnionAttr::getChecked(EmitErrorType EmitError,
                                MLIRContext *Context,
                                llvm::StringRef Handle,
                                ClassDefinitionAttr Definition) {
  if (verify(EmitError, Handle, Definition).failed())
    return {};

  return get(Context, Handle, Definition);
}

UnionAttr UnionAttr::get(MLIRContext *Context,
                         llvm::StringRef Handle,
                         llvm::StringRef Name,
                         llvm::ArrayRef<FieldAttr> Fields) {
  return get(Context,
             Handle,
             ClassDefinitionAttr::get(Context,
                                      Name,
                                      getUnionSize(Fields),
                                      Fields));
}

UnionAttr UnionAttr::getChecked(EmitErrorType EmitError,
                                MLIRContext *Context,
                                llvm::StringRef Handle,
                                llvm::StringRef Name,
                                llvm::ArrayRef<FieldAttr> Fields) {
  return getChecked(EmitError,
                    Context,
                    Handle,
                    ClassDefinitionAttr::get(Context,
                                             Name,
                                             getUnionSize(Fields),
                                             Fields));
}

//===---------------------------- CliftDialect ----------------------------===//

void CliftDialect::registerAttributes() {
  addAttributes<StructAttr, UnionAttr,
  // Include the list of auto-generated attributes
#define GET_ATTRDEF_LIST
#include "revng/mlir/Dialect/Clift/IR/CliftAttributes.cpp.inc"
                /* End of auto-generated list */>();
}

/// Parse an attribute registered to this dialect
mlir::Attribute CliftDialect::parseAttribute(mlir::DialectAsmParser &Parser,
                                             mlir::Type Type) const {
  return {};
}

/// Print an attribute registered to this dialect
void CliftDialect::printAttribute(mlir::Attribute Attr,
                                  mlir::DialectAsmPrinter &Printer) const {
  revng_abort("cannot print attribute");
}
