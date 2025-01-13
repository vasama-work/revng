//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_PASSES
#define MLIR_CLIFT_PASSES

include "mlir/Pass/PassBase.td"

def CliftImportLLVM : Pass<"import-llvm", "mlir::ModuleOp"> {
  let summary = "";
  let constructor = "mlir::clift::createImportLLVMPass()";
  let dependentDialects = ["clift::CliftDialect"];

  let options = [
    Option<"LLVMIRPath", "llvm", "std::string", /*default*/"\"\"", "LLVM IR Path">,
    Option<"Model", "model", "mlir::clift::ModelOptionType", /*default=*/"nullptr", "Model path">
  ];
}

def CliftVerifyC : Pass<"verify-c", "clift::ModuleOp"> {
  let summary = "Verify that the Clift semantics correspond to C semantics " #
                "in the specified target implementation.";
  let constructor = "mlir::clift::createVerifyCPass()";
  let dependentDialects = ["clift::CliftDialect"];
}

def CliftEmitC : Pass<"emit-c", "clift::ModuleOp"> {
  let summary = "Emit C code (optionally with PTML tags) from the Clift code.";
  let constructor = "mlir::clift::createEmitCPass()";
  let dependentDialects = ["clift::CliftDialect"];

  let options = [
    Option<"Tagless", "tagless", "bool", /*default=*/"false", "Emit without PTML tags">,
    Option<"Model", "model", "mlir::clift::ModelOptionType", /*default=*/"nullptr", "Model path">,
    Option<"Output", "output", "std::string", /*default*/"\"-\"", "C output path">
  ];
}

#endif // MLIR_CLIFT_PASSES
