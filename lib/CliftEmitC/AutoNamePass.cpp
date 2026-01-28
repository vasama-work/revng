//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Pass/Pass.h"

#include "revng/Clift/ModuleVisitor.h"
#include "revng/CliftTransforms/Passes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTAUTONAME
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

// Helper class for mutating the attribute dictionary of a function parameter.
// All attributes associated with a given function parameter are stored in a
// dictionary attribute, which is by its nature immutable. Changing individual
// function parameter attributes is difficult and inefficient. This class allows
// changes to all function parameter attribute dictionaries to be aggregated and
// applied all at once.
class ArgumentAttributeMutator {
  clift::FunctionOp Function;
  llvm::SmallVector<mlir::NamedAttrList> AttrLists;

public:
  explicit ArgumentAttributeMutator(clift::FunctionOp Op) : Function(Op) {
    for (unsigned I = 0; I < Op.getArgCount(); ++I) {
      AttrLists.emplace_back(Op.getArgAttrs(I).getDictionaryAttr());
    }
  }

  void set(unsigned Index, llvm::StringRef Name, mlir::Attribute Attr) {
    AttrLists[Index].set(Name, Attr);
  }

  void setString(unsigned Index, llvm::StringRef Name, llvm::StringRef Value) {
    set(Index, Name, mlir::StringAttr::get(Function.getContext(), Value));
  }

  void commit() {
    llvm::SmallVector<mlir::Attribute> ArgAttrs;
    for (const mlir::NamedAttrList &AttrList : AttrLists)
      ArgAttrs.push_back(AttrList.getDictionary(Function.getContext()));

    Function.setArgAttrsAttr(mlir::ArrayAttr::get(Function.getContext(),
                                                  ArgAttrs));
  }
};

class NameImporter : public clift::ModuleVisitor<NameImporter> {
  llvm::DenseMap<llvm::StringRef, uint64_t> NameCounters;

  std::string getName(llvm::StringRef Prefix) {
    auto [Iterator, Inserted] = NameCounters.try_emplace(Prefix, 0);

    std::string Name;
    {
      llvm::raw_string_ostream Out(Name);
      Out << Prefix;
      Out << '_';
      Out << Iterator->second++;
    }
    return Name;
  }

public:
  //===---------------------- ModuleVisitor interface ---------------------===//

  mlir::LogicalResult visitType(mlir::Type Type) {
    if (auto T = mlir::dyn_cast<clift::FunctionType>(Type)) {
      if (T.getName().empty())
        T.getMutableName().setValue(getName("function_type"));
    }

    return mlir::success();
  }

  mlir::LogicalResult visitAttr(mlir::Attribute Attr) {
    if (auto T = mlir::dyn_cast<clift::StructAttr>(Attr)) {
      if (T.getName().empty())
        T.getMutableName().setValue(getName("struct"));

      if (importFieldNames(T).failed())
        return mlir::failure();
    }

    if (auto T = mlir::dyn_cast<clift::UnionAttr>(Attr)) {
      if (T.getName().empty())
        T.getMutableName().setValue(getName("union"));

      if (importFieldNames(T).failed())
        return mlir::failure();
    }

    if (auto T = mlir::dyn_cast<clift::TypedefAttr>(Attr)) {
      if (T.getName().empty())
        T.getMutableName().setValue(getName("typedef"));
    }

    if (auto T = mlir::dyn_cast<clift::EnumAttr>(Attr)) {
      if (T.getName().empty())
        T.getMutableName().setValue(getName("enum"));

      for (auto E : T.getFields()) {
        if (E.getName().empty())
          E.getMutableName().setValue(getName("enumerator"));
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult visitNestedOp(mlir::Operation *Op) {
    if (auto S = mlir::dyn_cast<clift::MakeLabelOp>(Op))
      return visitMakeLabelOp(S);

    if (auto S = mlir::dyn_cast<clift::LocalVariableOp>(Op))
      return visitLocalVariableOp(S);

    return mlir::success();
  }

  mlir::LogicalResult visitModuleLevelOp(mlir::Operation *Op) {
    if (auto F = mlir::dyn_cast<clift::FunctionOp>(Op))
      return visitFunctionOp(F);

    return mlir::success();
  }

private:
  //===------------------------- Type name import -------------------------===//

  template<typename AttrT>
  mlir::LogicalResult importFieldNames(AttrT Attr) {
    for (auto F : Attr.getFields()) {
      if (F.getName().empty())
        F.getMutableName().setValue(getName("field"));
    }

    return mlir::success();
  }

  //===----------------------- Operation name import ----------------------===//

  mlir::LogicalResult visitMakeLabelOp(clift::MakeLabelOp Op) {
    if (Op.getName().empty())
      Op.setName(getName("label"));

    return mlir::success();
  }

  mlir::LogicalResult visitLocalVariableOp(clift::LocalVariableOp Op) {
    if (Op.getName().empty())
      Op.setName(getName("var"));

    return mlir::success();
  }

  mlir::LogicalResult visitFunctionOp(clift::FunctionOp Op) {
    ArgumentAttributeMutator Attrs(Op);
    for (unsigned I = 0; I < Op.getArgCount(); ++I) {
      auto ArgAttrs = Op.getArgAttrs(I);
      if (ArgAttrs.getStringOrEmpty("clift.name").empty())
        Attrs.setString(I, "clift.name", getName("arg"));
    }
    Attrs.commit();

    return mlir::success();
  }
};

struct AutoNamePass : clift::impl::CliftAutoNameBase<AutoNamePass> {
  void runOnOperation() override {
    auto R = NameImporter::visit(getOperation());
    revng_assert(R.succeeded());
  }
};

} // namespace

clift::PassPtr<mlir::ModuleOp> clift::createAutoNamePass() {
  return std::make_unique<AutoNamePass>();
}
