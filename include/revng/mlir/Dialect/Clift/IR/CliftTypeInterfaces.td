//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_INTERFACE_3
#define MLIR_CLIFT_INTERFACE_3

include "revng/mlir/Dialect/Clift/IR/CliftTypeInterfaces2.td"

def Clift_ClassType : Clift_TypeInterface<"ClassType", [Clift_DefinedType]> {
  let methods = [
    InterfaceMethod<
      "Returns true if the type is not incomplete.",
      "bool", "isComplete", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_type.isComplete();
      }]
    >,
    InterfaceMethod<
      "Returns the field attributes of this class type",
      "llvm::ArrayRef<mlir::clift::FieldAttr>", "getFields", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_type.getFields();
      }]
    >,
  ];
}

#endif

