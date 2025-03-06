//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_PASSES
#define MLIR_CLIFT_PASSES

include "mlir/Pass/PassBase.td"

class Clift_Pass<string arg, string operation> : Pass<arg, operation> {
  let dependentDialects = ["clift::CliftDialect"];
}

def CliftBeautify : Clift_Pass<"beautify", "clift::FunctionOp"> {
  let summary = "Beautify Clift";
  let constructor = "mlir::clift::createBeautifyPass()";
}

def CliftVerifyC : Clift_Pass<"verify-c", "clift::ModuleOp"> {
  let summary = "Verify that the Clift semantics correspond to C semantics " #
                "in the specified target implementation.";
  let constructor = "mlir::clift::createVerifyCPass()";
}

def CliftEmitC : Clift_Pass<"emit-c", "clift::ModuleOp"> {
  let summary = "Emit C code (optionally with PTML tags) from the Clift code.";
  let constructor = "mlir::clift::createEmitCPass()";

  let options = [
    Option<"Tagless", "tagless", "bool", /*default=*/"false", "Emit without PTML tags">,
    Option<"Model", "model", "mlir::clift::ModelOptionType", /*default=*/"nullptr", "Model path">,
    Option<"Output", "output", "std::string", /*default*/"\"-\"", "C output path">
  ];
}

#endif // MLIR_CLIFT_PASSES
